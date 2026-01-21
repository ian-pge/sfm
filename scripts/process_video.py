import argparse
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import os
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import numpy_image_to_torch
from tqdm import tqdm

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

def extract_precise_geometry(video_path, output_dir, overlap_thresh=0.60, downscale_factor=1, start_number=0):
    # 1. Setup Output Directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CHANGE 1: Use ALIKED (Deep Learning Features) ---
    # aliked-n16, aliked-n16rot, aliked-n32, etc. 
    extractor = ALIKED(max_num_keypoints=1024, detection_threshold=0.01).eval().to(device)
    
    # --- CHANGE 2: LightGlue Matcher ---
    matcher = LightGlue(features='aliked').eval().to(device)
    
    prev_feats = None
    
    # To store the corners of the previous frame (for geometry calc)
    # 1. Get correct aspect ratio
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_w / original_h

    # 2. Calculate height dynamically based on width of 640
    w_proc = 640
    h_proc = int(w_proc / aspect_ratio)

    prev_corners = np.float32([[0, 0], [0, h_proc], [w_proc, h_proc], [w_proc, 0]]).reshape(-1, 1, 2)
    
    saved_count = start_number
    frame_count = 0
    saved_this_session = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 24.0 # Fallback
    
    # Conservative stride: Analyse ~10 frames per second
    stride = max(1, int(fps / 10))
    
    print(f"--- STARTING PRECISE EXTRACTION ({video_path.name}) ---")
    print("Using ALIKED + LightGlue + Homography + Polygon Intersection")
    print(f"  - Original Resolution: {original_w}x{original_h}")
    print(f"  - Downscale for Save:  {downscale_factor}x (Output: {original_w//downscale_factor}x{original_h//downscale_factor})")
    print(f"  - Analysis Resolution: {w_proc}x{h_proc}")
    print(f"  - Frame Rate:          {fps:.2f} fps")
    print(f"  - Analysis Stride:     {stride} frames (Checking every {stride/fps*1000:.1f} ms)")
    
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Frames", unit="frame")

    while True:
        # Optimization: Skip frames according to stride
        if frame_count % stride != 0:
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_count += 1
            pbar.update(1)
            continue

        ret, frame_full = cap.read()
        if not ret:
            break
            
        # Processing resolution (keep this consistent)
        process_frame = cv2.resize(frame_full, (w_proc, h_proc))
        
        # Prepare for LightGlue (RGB + Tensor)
        frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = numpy_image_to_torch(frame_rgb).to(device)
        
        # Extract
        # ALIKED extraction
        with torch.no_grad():
            feats = extractor.extract(frame_tensor)
        
        should_save = False
        calculated_overlap = 0.0
        
        if prev_feats is None:
            should_save = True
            calculated_overlap = 0.0
        else:
            # --- CHANGE 3: Geometric Verification ---
            # Match
            with torch.no_grad():
                matches = matcher({'image0': prev_feats, 'image1': feats})
                
            # Filter matches
            matches0 = matches['matches0'][0] # indices of matches in image1 for each kp in image0
            valid = matches0 > -1
            
            # We need at least 4 points to find a Homography
            if valid.sum() > 10:
                mkpts0 = prev_feats['keypoints'][0][valid].cpu().numpy()
                mkpts1 = feats['keypoints'][0][matches0[valid]].cpu().numpy()
                
                # Find Homography (Matrix M maps Current -> Previous)
                # We use RANSAC to ignore outliers
                # Reshape for findHomography
                dst_pts = mkpts1.reshape(-1, 1, 2)
                src_pts = mkpts0.reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # 1. Project current frame corners into previous frame space
                    curr_corners = np.float32([[0, 0], [0, h_proc], [w_proc, h_proc], [w_proc, 0]]).reshape(-1, 1, 2)
                    projected_corners = cv2.perspectiveTransform(curr_corners, M)
                    
                    try:
                        # 1. Calculate Intersection Area
                        ret_area, intersection_contour = cv2.intersectConvexConvex(prev_corners, projected_corners)
                        
                        # 2. Calculate Areas of both frames individualy
                        area_prev = w_proc * h_proc
                        area_curr = cv2.contourArea(projected_corners)
                        
                        # 3. Calculate UNION (Total footprint of both frames)
                        union_area = area_prev + area_curr - ret_area
                        
                        # 4. IoU Calculation (Standard Computer Vision Metric)
                        # Protect against divide by zero
                        if union_area > 0:
                            calculated_overlap = ret_area / union_area
                        else:
                            calculated_overlap = 0.0
                        
                        if calculated_overlap < overlap_thresh:
                            should_save = True
                            
                    except Exception as e:
                        should_save = True
                        calculated_overlap = 0.0

                else:
                    # Homography failed (too much movement/blur), force save
                    should_save = True
                    calculated_overlap = 0.0
            else:
                # Not enough matches, scene changed completely
                should_save = True
                calculated_overlap = 0.0

        if should_save:
            # Save Logic
            if downscale_factor > 1:
                h, w = frame_full.shape[:2]
                save_frame = cv2.resize(frame_full, (w // downscale_factor, h // downscale_factor))
            else:
                save_frame = frame_full
                
            filename = images_dir / f"frame_{saved_count:05d}.png"
            cv2.imwrite(str(filename), save_frame)
            
            elapsed_time = frame_count / fps if fps > 0 else 0
            # print(f"[Saved {saved_count:05d}] T={elapsed_time:06.1f}s | Geo-Overlap: {calculated_overlap*100:05.1f}%")
            pbar.set_postfix(saved=saved_count, overlap=f"{calculated_overlap*100:.1f}%")
            
            # Update References
            prev_feats = feats
            saved_count += 1
            saved_this_session += 1
            
        frame_count += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    return saved_this_session

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", nargs='+', required=True, help="Path to input video file(s)")
    
    # Mode args
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive ALIKED+LightGlue geometric feature extraction")
    parser.add_argument("--overlap", type=float, default=0.60, help="Adaptive: Geometric overlap threshold (0.0 - 1.0). Default 0.60. Lower = more spacing.")
    
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
    
    print(f"Output directory: {output_dir}")
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    global_frame_count = 0
    
    for video in video_paths:
        if not video.exists():
            print(f"Error: {video} not found")
            continue
            
        print(f"\n--- Processing {video.name} ---")
        
        if args.adaptive:
            count = extract_precise_geometry(video, output_dir, args.overlap, args.downscale, start_number=global_frame_count)
        else:
            if args.num_frames is None:
                print("Error: --num_frames is required for fixed mode")
                sys.exit(1)
            count = extract_frames_fixed(video, output_dir, args.num_frames, args.downscale, start_number=global_frame_count)
            
        global_frame_count += count
        
    print(f"\nAll Done. Total frames extracted: {global_frame_count}")

if __name__ == "__main__":
    main()
