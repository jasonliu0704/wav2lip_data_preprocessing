import os
import concurrent.futures
import cv2
import dlib
import subprocess
import sys
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Crop videos to face regions with multi-GPU support")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("presenter_name", type=str, help="Name of the presenter folder")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU IDs (e.g. '0,1,2,3')")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale factor for face crop (default: 1.5)")
    return parser.parse_args()

def crop_video(args):
    name_video, gpu_id, scale_factor = args
    vid_path = os.path.join(input_video_path, name_video)
    out_path = os.path.join(output_video_path, name_video)
    
    # Skip if output already exists
    if os.path.exists(out_path):
        print(f"Skipping {name_video} as it already exists")
        return
    
    try:
        detector = dlib.get_frontal_face_detector()

        # Load the video
        cap = cv2.VideoCapture(vid_path)
        
        if not cap.isOpened():
            print(f"Error opening video file {vid_path}")
            return

        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read frames until we get a good one or run out
        max_frames_to_check = 100
        frame_count = 0
        face_detected = False
        
        while frame_count < max_frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Try to detect face
            faces = detector(frame)
            if len(faces) > 0:
                face_detected = True
                break
                
            frame_count += 1
            
        cap.release()
        
        if not face_detected:
            print(f"No face detected in {name_video} after checking {frame_count} frames")
            return

        # Get the bounding box of the first face detected
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Calculate center of face
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate size of square crop
        crop_size = int(max(w, h) * scale_factor)
        
        # Make sure crop doesn't exceed frame boundaries
        crop_x = max(0, center_x - crop_size // 2)
        crop_y = max(0, center_y - crop_size // 2)
        
        # Ensure we don't crop beyond video dimensions
        if crop_x + crop_size > width:
            crop_x = max(0, width - crop_size)
        if crop_y + crop_size > height:
            crop_y = max(0, height - crop_size)
            
        # If crop size is still too big, reduce it
        crop_size = min(crop_size, width - crop_x, height - crop_y)
        
        # Use ffmpeg to crop the video based on the adjusted bounding box
        gpu_cmd = f"-hwaccel cuda -hwaccel_device {gpu_id}" if gpu_id is not None else ""
        encoder = f"-c:v h264_nvenc -preset p7" if gpu_id is not None else "-c:v libx264 -preset fast"
        
        command = f"ffmpeg -y {gpu_cmd} -i {vid_path} -filter:v \"crop={crop_size}:{crop_size}:{crop_x}:{crop_y}\" {encoder} -b:v 4M -c:a copy {out_path}"
        print(f"Processing {name_video} with crop {crop_size}x{crop_size} at ({crop_x},{crop_y}) on {'GPU '+str(gpu_id) if gpu_id is not None else 'CPU'}")
        
        result = subprocess.run(command, shell=True, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error processing {name_video}: {result.stderr.decode()}")
            
    except Exception as e:
        print(f"Error processing {name_video}: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    
    input_video_path = os.path.join(args.dataset_path, args.presenter_name, 'full_voice_25fps')
    output_video_path = os.path.join(args.dataset_path, args.presenter_name, 'videos_crop')
    
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
        
    source_dir = [x for x in os.listdir(input_video_path) if os.path.isfile(os.path.join(input_video_path, x))]
    print(f"Found {len(source_dir)} videos to process")
    
    # Configure GPU assignments
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else None
    
    # Create task distribution
    tasks = []
    if gpu_ids:
        # Distribute videos across GPUs
        for i, video in enumerate(source_dir):
            gpu_idx = i % len(gpu_ids)
            tasks.append((video, gpu_ids[gpu_idx], args.scale))
        
        # Use ThreadPoolExecutor for GPU tasks
        executor_class = concurrent.futures.ThreadPoolExecutor
        n_workers = min(args.n_processes, len(source_dir))
    else:
        # CPU-only mode
        tasks = [(video, None, args.scale) for video in source_dir]
        executor_class = concurrent.futures.ProcessPoolExecutor
        n_workers = args.n_processes
    
    # Execute cropping
    with executor_class(max_workers=n_workers) as executor:
        results = list(executor.map(crop_video, tasks))
    
    print(f"Completed processing {len(source_dir)} videos")
    