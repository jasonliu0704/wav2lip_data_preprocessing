import os
import sys
import subprocess
import concurrent.futures
import argparse
import re
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Segment videos into 10-second clips with multi-GPU support")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("presenter_name", type=str, help="Name of the presenter folder")
    parser.add_argument("--n_processes", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU IDs for encoding")
    parser.add_argument("--segment_length", type=int, default=10, help="Length of each segment in seconds")
    return parser.parse_args()

def get_video_duration(video_path):
    """Extract video duration using ffprobe instead of grep/sed"""
    try:
        # Use ffprobe to get duration directly in seconds
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "json", 
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error getting duration for {video_path}: {result.stderr.decode()}")
            return 0
            
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return int(duration)
    except Exception as e:
        print(f"Failed to get video duration: {str(e)}")
        return 0

def segment(args):
    """Process a video file, splitting it into segments"""
    name_video, gpu_id, segment_length = args
    
    video_path = os.path.join(input_video_path, name_video)
    
    # Setup output paths
    split_video_path = os.path.join(output_video_path, name_video.split('.')[0])
    split_audio_path = os.path.join(output_audio_path, name_video.split('.')[0])
    
    # Skip if already processed
    if os.path.exists(split_video_path) and len(os.listdir(split_video_path)) > 0:
        print(f"Skipping {name_video} as segments already exist")
        return
    
    # Create output directories
    os.makedirs(split_video_path, exist_ok=True)
    os.makedirs(split_audio_path, exist_ok=True)
    
    # Get video duration
    duration_video = get_video_duration(video_path)
    if duration_video == 0:
        print(f"Skipping {name_video} - couldn't determine duration")
        return
        
    # Create segments
    segment_time = list(range(0, duration_video, segment_length))
    if len(segment_time) < 2:  # Add end time if video is shorter than segment_length
        segment_time.append(duration_video)
    
    processed_segments = 0
    failed_segments = 0
    
    for i in range(len(segment_time) - 1):
        start_time = segment_time[i]
        duration = min(segment_length, segment_time[i+1] - start_time)
        
        # Define output paths for this segment
        small_video_path = os.path.join(split_video_path, f'{start_time:04d}_{start_time+duration:04d}.mp4')
        small_audio_path = os.path.join(split_audio_path, f'{start_time:04d}_{start_time+duration:04d}.wav')
        
        # Skip if both files exist
        if os.path.exists(small_video_path) and os.path.exists(small_audio_path):
            processed_segments += 1
            continue
            
        try:
            # Hardware encoder settings if GPU is available
            encoder_args = f"-c:v h264_nvenc -preset p7" if gpu_id is not None else "-c:v libx264 -preset fast"
            gpu_args = f" -hwaccel cuda -hwaccel_device {gpu_id}" if gpu_id is not None else ""
            
            # Extract segment with ffmpeg
            vid_command = (
                f"ffmpeg{gpu_args} -nostdin -y -ss {start_time} -i {video_path} "
                f"-t {duration} -filter:v fps=25 {encoder_args} -b:v 4M "
                f"{small_video_path}"
            )
            
            vid_result = subprocess.run(vid_command, shell=True, stderr=subprocess.PIPE)
            if vid_result.returncode != 0:
                print(f"Error creating video segment {start_time}-{start_time+duration} for {name_video}")
                failed_segments += 1
                continue
                
            # Extract audio from the segment
            aud_command = f"ffmpeg -nostdin -y -i {small_video_path} -ar 16000 {small_audio_path}"
            aud_result = subprocess.run(aud_command, shell=True, stderr=subprocess.PIPE)
            
            if aud_result.returncode != 0:
                print(f"Error extracting audio for segment {start_time}-{start_time+duration} in {name_video}")
                failed_segments += 1
            else:
                processed_segments += 1
                print(f"Processed {small_video_path} on {'GPU '+str(gpu_id) if gpu_id is not None else 'CPU'}")
                
        except Exception as e:
            print(f"Error processing segment {start_time}-{start_time+duration} in {name_video}: {str(e)}")
            failed_segments += 1
    
    return {
        "video": name_video,
        "total_segments": len(segment_time) - 1,
        "processed": processed_segments,
        "failed": failed_segments
    }

if __name__ == "__main__":
    args = parse_args()
    
    input_video_path = os.path.join(args.dataset_path, args.presenter_name, 'videos_crop')
    output_video_path = os.path.join(args.dataset_path, args.presenter_name, 'videos_segment')
    output_audio_path = os.path.join(args.dataset_path, args.presenter_name, 'audios_segment')
    
    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_audio_path, exist_ok=True)
    
    source_files = [f for f in os.listdir(input_video_path) if os.path.isfile(os.path.join(input_video_path, f))]
    print(f"Found {len(source_files)} videos to process")
    
    # Configure GPU assignments
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else None
    
    # Create task distribution
    tasks = []
    if gpu_ids:
        # Distribute videos across GPUs
        for i, video in enumerate(source_files):
            gpu_idx = i % len(gpu_ids)
            tasks.append((video, gpu_ids[gpu_idx], args.segment_length))
        
        # Use ThreadPoolExecutor for GPU tasks
        executor_class = concurrent.futures.ThreadPoolExecutor
    else:
        # CPU-only mode
        tasks = [(video, None, args.segment_length) for video in source_files]
        executor_class = concurrent.futures.ProcessPoolExecutor
    
    # Process videos
    start_time = datetime.now()
    results = []
    
    with executor_class(max_workers=args.n_processes) as executor:
        futures = {executor.submit(segment, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            video_name = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    total_segments = sum(r.get("total_segments", 0) for r in results if r)
    processed_segments = sum(r.get("processed", 0) for r in results if r)
    failed_segments = sum(r.get("failed", 0) for r in results if r)
    
    print(f"Segmentation completed in {duration:.2f} seconds")
    print(f"Total videos: {len(source_files)}")
    print(f"Total segments: {total_segments}")
    print(f"Processed segments: {processed_segments}")
    print(f"Failed segments: {failed_segments}")

# python3 3_segment.py /mnt/ nvme2/ 8
