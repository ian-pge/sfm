import argparse
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import os

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration: {e}")
        sys.exit(1)

def extract_frames_fixed(video_path, output_dir, num_frames, downscale_factor, start_number=0):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    duration = get_video_duration(video_path)
    if duration <= 0:
        print("Error: Invalid video duration.")
        sys.exit(1)
        
    target_fps = num_frames / duration
    print(f"Mode: Fixed (Target FPS: {target_fps:.4f} for {num_frames} frames)")
    
    filters = [f"fps={target_fps}"]
    if downscale_factor > 1:
        filters.append(f"scale=iw/{downscale_factor}:-1")
    
    filter_str = ",".join(filters)
    output_pattern = images_dir / "frame_%05d.png"
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", filter_str,
        "-vsync", "0",
        "-q:v", "1",
        "-start_number", str(start_number),
        str(output_pattern)
    ]
    
    print("Running ffmpeg...")
    try:
        subprocess.run(cmd, check=True)
        # Count extracted frames
        current_images = sorted(list(images_dir.glob("frame_*.png")))
        count = 0
        for img in current_images:
            try:
                idx = int(img.stem.split("_")[-1])
                if idx >= start_number:
                    count += 1
            except:
                pass
        return count
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        sys.exit(1)

def extract_with_features(video_path, output_dir, overlap_thresh=0.50, downscale_factor=1, start_number=0):
    # 1. Setup Output Directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    # 2. Initialize ORB Feature Detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # 3. Initialize Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_keypoints = None
    prev_descriptors = None
    
    saved_count = start_number
    frame_count = 0
    frames_extracted_this_session = 0
    
    # Statistics Collection
    stats_overlap = []
    stats_matches = []
    stats_keypoints = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"--- STARTING EXTRACTION ({video_path.name}) ---")
    print(f"Target: Save a new frame when feature overlap drops below {overlap_thresh*100}%")

    while True:
        ret, frame_full = cap.read()
        if not ret:
            break
            
        process_frame = cv2.resize(frame_full, (640, 360))
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        should_save = False
        current_overlap = 0.0
        match_count = 0
        prev_count = 0

        if prev_descriptors is None:
            should_save = True
            current_overlap = 0.0 # First frame
        else:
            if descriptors is not None and len(descriptors) > 0:
                matches = bf.match(prev_descriptors, descriptors)
                
                match_count = len(matches)
                prev_count = len(prev_descriptors)
                
                if prev_count > 0:
                    current_overlap = match_count / prev_count
                else:
                    current_overlap = 0
                
                if current_overlap < overlap_thresh:
                    should_save = True
            else:
                pass

        if should_save:
            # Save Frame
            if downscale_factor > 1:
                h, w = frame_full.shape[:2]
                new_w, new_h = w // downscale_factor, h // downscale_factor
                save_frame = cv2.resize(frame_full, (new_w, new_h))
            else:
                save_frame = frame_full

            filename = images_dir / f"frame_{saved_count:05d}.png"
            cv2.imwrite(str(filename), save_frame)
            
            # Timestamp calculation
            elapsed_time = frame_count / fps if fps > 0 else 0
            
            # LOGGING (Console)
            print(f"[Saved {saved_count:05d}] T={elapsed_time:06.1f}s | Overlap: {current_overlap*100:04.1f}% | Matches: {match_count:03d} | RefKpts: {prev_count:03d}")
            
            # Store stats
            if prev_descriptors is not None: # Don't log stats for first frame as overlap is meaningless
                stats_overlap.append(current_overlap)
                stats_matches.append(match_count)
                stats_keypoints.append(prev_count)

            # Update Memory
            prev_keypoints = keypoints
            prev_descriptors = descriptors
            
            saved_count += 1
            frames_extracted_this_session += 1

        frame_count += 1

    cap.release()
    print(f"\n--- SUMMARY {video_path.name} ---")
    print(f"Total Scanned Frames: {frame_count}")
    print(f"Total Saved Frames:   {frames_extracted_this_session}")
    
    if len(stats_overlap) > 0:
        avg_overlap = sum(stats_overlap) / len(stats_overlap)
        min_overlap = min(stats_overlap)
        max_overlap = max(stats_overlap)
        avg_matches = sum(stats_matches) / len(stats_matches)
        
        print(f"Overlap Trigger stats: Avg={avg_overlap*100:.1f}%, Min={min_overlap*100:.1f}%, Max={max_overlap*100:.1f}%")
        print(f"Average Matches per saved frame: {avg_matches:.1f}")
    else:
        print("No stats collected (maybe only 1 frame saved).")
        
    print(f"--------------------------------------------------\n")
    return frames_extracted_this_session

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", nargs='+', required=True, help="Path to input video file(s)")
    
    # Mode args
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive ORB feature extraction (Parallax based)")
    parser.add_argument("--overlap", type=float, default=0.50, help="Adaptive: Feature overlap threshold (0.0 - 1.0). Default 0.50. Lower = more spacing.")
    
    parser.add_argument("--num_frames", type=int, help="Fixed: Number of frames to extract per video")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale factor (e.g. 2 for half size). Default 1 (no downscale).")
    parser.add_argument("--output", help="Output dataset directory. Defaults to datasets/<first_video_name>")
    
    args = parser.parse_args()
    
    # Setup Output
    video_paths = [Path(p).resolve() for p in args.video]
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        root_dir = Path(__file__).parent.parent
        output_dir = root_dir / "datasets" / video_paths[0].stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    global_frame_count = 0
    
    for video in video_paths:
        if not video.exists():
            print(f"Error: {video} not found")
            continue
            
        print(f"\n--- Processing {video.name} ---")
        
        if args.adaptive:
            count = extract_with_features(video, output_dir, args.overlap, args.downscale, start_number=global_frame_count)
        else:
            if args.num_frames is None:
                print("Error: --num_frames is required for fixed mode")
                sys.exit(1)
            count = extract_frames_fixed(video, output_dir, args.num_frames, args.downscale, start_number=global_frame_count)
            
        global_frame_count += count
        
    print(f"\nAll Done. Total frames extracted: {global_frame_count}")

if __name__ == "__main__":
    main()
