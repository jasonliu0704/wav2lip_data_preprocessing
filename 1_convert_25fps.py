import os
import concurrent.futures
import subprocess
import sys
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Convert videos to 25fps with multi-GPU support")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("presenter_name", type=str, help="Name of the presenter folder")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of parallel CPU processes")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU IDs (e.g. '0,1,2,3')")
    return parser.parse_args()

def convert_25fps(args):
    name_video, gpu_id = args
    video = os.path.join(input_video_path, name_video)
    new_video = os.path.join(output_video_path, name_video)
    
    # Skip if output already exists
    if os.path.exists(new_video):
        print(f"Skipping {name_video} as it already exists")
        return
    
    # Use GPU acceleration if available
    if gpu_id is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} ffmpeg -y -hwaccel cuda -i {video} -filter:v fps=25 -c:v h264_nvenc -preset p7 -b:v 50M -c:a copy {new_video}"
    else:
        cmd = f"ffmpeg -y -i {video} -filter:v fps=25 -b:v 50M {new_video}"
    
    print(f"Processing {name_video} on {'GPU '+str(gpu_id) if gpu_id is not None else 'CPU'}")
    subprocess.call(cmd, shell=True)
    return name_video

if __name__ == "__main__":
    args = parse_args()
    
    input_video_path = os.path.join(args.dataset_path, args.presenter_name, 'full_voice')
    output_video_path = os.path.join(args.dataset_path, args.presenter_name, 'full_voice_25fps')
    
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    
    source_videos = [x for x in os.listdir(input_video_path) if os.path.isfile(os.path.join(input_video_path, x))]
    
    # Configure GPU assignments
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else None
    
    # Create task distribution
    tasks = []
    if gpu_ids:
        # Distribute videos across GPUs
        for i, video in enumerate(source_videos):
            gpu_idx = i % len(gpu_ids)
            tasks.append((video, gpu_ids[gpu_idx]))
        
        # Use ThreadPoolExecutor for GPU tasks to avoid CUDA context issues
        executor_class = concurrent.futures.ThreadPoolExecutor
        n_workers = min(args.n_processes, len(source_videos))
    else:
        # CPU-only mode
        tasks = [(video, None) for video in source_videos]
        executor_class = concurrent.futures.ProcessPoolExecutor
        n_workers = args.n_processes
    
    # Execute conversion
    with executor_class(max_workers=n_workers) as executor:
        results = list(executor.map(convert_25fps, tasks))
    
    print(f"Completed converting {len(results)} videos to 25fps")