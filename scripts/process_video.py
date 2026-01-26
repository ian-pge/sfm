import argparse
import subprocess
import sys
import os
import warnings

# Suppress Warnings (Qt, Python, etc.)
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
warnings.filterwarnings("ignore")

from pathlib import Path
import cv2
import numpy as np
import os
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import numpy_image_to_torch
from tqdm.auto import tqdm
import time
from ultralytics import YOLO

def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def extract_frames_fixed(video_path, output_dir, num_frames, downscale_factor, start_number=0, video_idx=0):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    duration = get_video_duration(video_path)
    if duration <= 0:
        print("Error: Invalid video duration.")
        sys.exit(1)
        
    target_fps = num_frames / duration
    print(f"ℹ️  Mode: Fixed (Target FPS: {target_fps:.4f} for {num_frames} frames)")
    
    filters = [f"fps={target_fps}"]
    if downscale_factor > 1:
        filters.append(f"scale=iw/{downscale_factor}:-1")
    
    filter_str = ",".join(filters)
    output_pattern = images_dir / f"frame_%05d_video_{video_idx}.png"
    
    cmd = [
        "ffmpeg", "-i", str(video_path), "-vf", filter_str,
        "-vsync", "0", "-q:v", "1", "-start_number", str(start_number),
        str(output_pattern)
    ]
    
    print("🎬 Running ffmpeg...")
    try:
        subprocess.run(cmd, check=True)
        current_images = sorted(list(images_dir.glob(f"frame_*_video_{video_idx}.png")))
        count = sum(1 for img in current_images if int(img.stem.split("_")[1]) >= start_number)
        return count, {}
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        sys.exit(1)

def extract_precise_geometry(video_path, output_dir, overlap_thresh=0.60, downscale_factor=1, start_number=0, video_idx=0, use_yolo=False, show_gui=False):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load Models
    extractor = ALIKED(max_num_keypoints=1024, detection_threshold=0.01).eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    yolo_model = None
    if use_yolo:
        print("🚗 Loading YOLOv8 Medium Segmentation...")
        yolo_model = YOLO("yolov8m-seg.pt") 
    
    prev_feats = None
    
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_w / original_h

    w_proc = 640
    h_proc = int(w_proc / aspect_ratio)

    prev_corners = np.float32([[0, 0], [0, h_proc], [w_proc, h_proc], [w_proc, 0]]).reshape(-1, 1, 2)
    
    saved_count = start_number
    frame_count = 0
    saved_this_session = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    
    # Dynamic Stride Variables
    base_stride = max(1, int(fps / 10)) 
    current_stride = base_stride
    max_stride = int(fps) 
    
    stats = {
        "processed_frames": 0,
        "saved_frames": 0,
        "skipped_overlap": 0,
        "fallback_background": 0, # Renamed from safety_saves to reflect logic
        "skipped_no_features": 0,
        "total_matches_on_car": 0,
        "valid_car_detections": 0
    }

    if show_gui:
        cv2.startWindowThread()
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    print(f"🚀 --- STARTING SMART EXTRACTION ({video_path.name}) ---")
    
    # ℹ️ Video Input Stats
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    print(f"  ℹ️  Video Info:")
    print(f"      • Resolution:   {original_w}x{original_h}")
    print(f"      • FPS:          {fps:.2f}")
    print(f"      • Total Frames: {total_frames}")
    print(f"      • Duration:     {duration_sec:.2f}s")
    
    sys.stdout.flush()

    pbar = tqdm(total=total_frames, desc="🎥 Processing", unit="frame", dynamic_ncols=True, mininterval=0.5, leave=False)
    
    with pbar:
        while True:
            # Dynamic Stride Skip
            if frame_count % current_stride != 0:
                if not cap.grab(): break
                frame_count += 1
                pbar.update(1)
                continue

            ret, frame_full = cap.read()
            if not ret: break
            
            stats["processed_frames"] += 1
            
            process_frame = cv2.resize(frame_full, (w_proc, h_proc))
            frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            frame_tensor = numpy_image_to_torch(frame_rgb).to(device)
            
            with torch.no_grad():
                feats = extractor.extract(frame_tensor)
            
            should_save = False
            calculated_overlap = 0.0
            car_detected = False
            combined_mask = None 
            
            # Variables for Visualization
            current_kpts_all = feats['keypoints'][0].cpu().numpy() # For Vis
            matched_kpts_curr = None # For Vis
            
            if prev_feats is None:
                should_save = True
                # Initial check for stats
                if use_yolo:
                    res = yolo_model(process_frame, classes=[2, 7], verbose=False, retina_masks=False, conf=0.10)
                    if res[0].masks is not None:
                         car_detected = True
                         stats["valid_car_detections"] += 1
                    else:
                         stats["fallback_background"] += 1
            else:
                with torch.no_grad():
                    matches = matcher({'image0': prev_feats, 'image1': feats})
                
                matches0 = matches['matches0'][0]
                valid = matches0 > -1
                
                if valid.sum() > 10:
                    mkpts0 = prev_feats['keypoints'][0][valid].cpu().numpy()
                    mkpts1 = feats['keypoints'][0][matches0[valid]].cpu().numpy()
                    
                    # ---------------------------------------------------------
                    # 🧩 LOGIC: YOLO Masking or Background Fallback
                    # ---------------------------------------------------------
                    if use_yolo:
                        results = yolo_model(process_frame, classes=[2, 7], verbose=False, retina_masks=False, conf=0.05)
                        
                        if results[0].masks is not None and results[0].masks.data.shape[0] > 0:
                            masks = results[0].masks.data
                            areas = torch.sum(masks, dim=(1, 2))
                            largest_idx = torch.argmax(areas)
                            
                            combined_mask = masks[largest_idx].cpu().numpy().astype(bool)
                            if combined_mask.shape[:2] != (h_proc, w_proc):
                                combined_mask = cv2.resize(combined_mask.astype(np.uint8), (w_proc, h_proc)).astype(bool)
                            
                            # Mask Dilation (5 pixels)
                            kernel_size = 5 
                            kernel = np.ones((kernel_size, kernel_size), np.uint8)
                            mask_uint8 = combined_mask.astype(np.uint8)
                            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
                            combined_mask = dilated_mask.astype(bool)
                            
                            # Filter Points -> Only Keep Car Points
                            int_pts = np.round(mkpts1).astype(int)
                            int_pts[:, 0] = np.clip(int_pts[:, 0], 0, w_proc - 1)
                            int_pts[:, 1] = np.clip(int_pts[:, 1], 0, h_proc - 1)
                            
                            is_on_car = combined_mask[int_pts[:, 1], int_pts[:, 0]]
                            
                            if is_on_car.sum() > 8:
                                mkpts0 = mkpts0[is_on_car]
                                mkpts1 = mkpts1[is_on_car]
                                car_detected = True
                                stats["valid_car_detections"] += 1
                                stats["total_matches_on_car"] += is_on_car.sum()
                            else:
                                car_detected = False
                        else:
                            car_detected = False

                        # FALLBACK LOGIC:
                        if not car_detected:
                            # We do NOT force save. We do NOT filter points.
                            # We keep ALL mkpts0 and mkpts1 (Background tracking)
                            # Just record stat that we are in fallback mode
                            stats["fallback_background"] += 1
                            # combined_mask remains None, so GUI knows.
                    
                    # ---------------------------------------------------------
                    # 📐 GEOMETRIC OVERLAP (Homography)
                    # ---------------------------------------------------------
                    # Set Matched Points for Vis (Green)
                    matched_kpts_curr = mkpts1
                    
                    dst_pts = mkpts1.reshape(-1, 1, 2)
                    src_pts = mkpts0.reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        curr_corners = np.float32([[0, 0], [0, h_proc], [w_proc, h_proc], [w_proc, 0]]).reshape(-1, 1, 2)
                        projected_corners = cv2.perspectiveTransform(curr_corners, M)
                        try:
                            ret_area, intersection_contour = cv2.intersectConvexConvex(prev_corners, projected_corners)
                            area_prev = w_proc * h_proc
                            area_curr = cv2.contourArea(projected_corners)
                            union_area = area_prev + area_curr - ret_area
                            
                            if union_area > 0:
                                calculated_overlap = ret_area / union_area
                            
                            if calculated_overlap < overlap_thresh:
                                should_save = True
                            else:
                                stats["skipped_overlap"] += 1
                        except:
                            should_save = True
                    else:
                            should_save = True 
                else:
                    should_save = True
                    stats["skipped_no_features"] += 1

            # Dynamic Stride Update
            if should_save:
                current_stride = base_stride
            else:
                if calculated_overlap > 0.95:
                    current_stride = min(current_stride + 1, max_stride)
                elif calculated_overlap < (overlap_thresh + 0.1):
                    current_stride = base_stride
            
            # --- VISUALIZATION ---
            if show_gui:
                vis_img = process_frame.copy()
                
                # 1. Draw Mask (Red Overlay)
                if use_yolo and combined_mask is not None:
                    overlay = np.zeros_like(vis_img)
                    overlay[combined_mask] = [0, 0, 255] # Red tint
                    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
                
                # 2. Draw ALL Keypoints (Red Dots)
                for kp in current_kpts_all:
                    cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

                # 3. Draw MATCHED Keypoints (Green Dots)
                if matched_kpts_curr is not None:
                     for kp in matched_kpts_curr:
                         # Draw slightly larger green dot
                         cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

                # Status Text
                status_color = (0, 255, 0) # Green (Skipping)
                if should_save: status_color = (0, 0, 255) # Red (Saving)

                txt = f"Ovlp:{calculated_overlap:.2f} Stride:{current_stride}"
                cv2.putText(vis_img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                if use_yolo and not car_detected:
                     # Changed text to indicate Fallback logic is active
                     cv2.putText(vis_img, "NO CAR - BG FALLBACK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Result", vis_img)
                cv2.waitKey(1)

            if should_save:
                if downscale_factor > 1:
                    h, w = frame_full.shape[:2]
                    save_frame = cv2.resize(frame_full, (w // downscale_factor, h // downscale_factor))
                else:
                    save_frame = frame_full
                    
                filename = images_dir / f"frame_{saved_count:05d}_video_{video_idx}.png"
                cv2.imwrite(str(filename), save_frame)
                
                pbar.set_postfix_str(f"💾 Saved: {saved_count} | 📐 Ov: {calculated_overlap:.2f} | ⏩ Stride: {current_stride}")
                
                prev_feats = feats
                saved_count += 1
                saved_this_session += 1
                stats["saved_frames"] += 1
                
            frame_count += 1
            pbar.update(1)
        
    cap.release()
    if show_gui: cv2.destroyAllWindows()
    return saved_this_session, stats

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", nargs='+', required=True, help="Video file(s)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive ALIKED+LightGlue")
    parser.add_argument("--yolo", action="store_true", help="Use YOLOv8-Medium Segmentation")
    parser.add_argument("--gui", action="store_true", help="Show GUI")
    parser.add_argument("--overlap", type=float, default=0.80, help="Overlap threshold")
    parser.add_argument("--num_frames", type=int, help="Fixed mode count")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale")
    parser.add_argument("--output", help="Output dir")
    
    args = parser.parse_args()
    
    raw_video_paths = [Path(p).resolve() for p in args.video]
    video_paths = []
    for p in raw_video_paths:
        if p.exists(): video_paths.append(p)
             
    if not video_paths: sys.exit(1)

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        root_dir = Path(__file__).parent.parent
        output_dir = root_dir / "datasets" / video_paths[0].stem
    
    print(f"📂 Output directory: {output_dir}")
    global_frame_count = 0
    all_stats_summary = []
    
    for video_idx, video in enumerate(video_paths):
        print(f"\n🎞️  --- Processing {video.name} ---")
        start_time = time.time()
        
        if args.adaptive:
            count, v_stats = extract_precise_geometry(video, output_dir, args.overlap, args.downscale, start_number=global_frame_count, video_idx=video_idx, use_yolo=args.yolo, show_gui=args.gui)
        else:
            if args.num_frames is None:
                print("Error: --num_frames is required for fixed mode")
                sys.exit(1)
            count, v_stats = extract_frames_fixed(video, output_dir, args.num_frames, args.downscale, start_number=global_frame_count, video_idx=video_idx)
            
        end_time = time.time()
        duration_proc = end_time - start_time
        
        # --- STATS LOGIC ---
        avg_matches = 0
        if v_stats.get("valid_car_detections", 0) > 0:
            avg_matches = v_stats["total_matches_on_car"] / v_stats["valid_car_detections"]

        summary_entry = {
            "name": video.name,
            "extracted": count,
            "fallback": v_stats.get("fallback_background", 0),
            "skipped_overlap": v_stats.get("skipped_overlap", 0),
            "avg_matches": avg_matches
        }
        all_stats_summary.append(summary_entry)

        print(f"\n📊 --- DEBUG STATS: {video.name} ---")
        print(f"  Frames Processed (Analyzed): {v_stats.get('processed_frames', 0)}")
        print(f"  ✅ Saved Total:              {count}")
        print(f"  ❌ Skipped (Overlap):        {v_stats.get('skipped_overlap', 0)}")
        print(f"  ⚠️ Fallback to BG:           {v_stats.get('fallback_background', 0)}")
        
        if args.yolo and args.adaptive:
            det_count = v_stats.get("valid_car_detections", 0)
            print(f"  🚗 Valid Car Detections:     {det_count}")
            if det_count > 0:
                print(f"  🔢 Avg Features on Car:      {avg_matches:.1f}")
        
        print("------------------------------------------------------------")
        global_frame_count += count
    
    print(f"\n✅ All Done. Total frames extracted: {global_frame_count}")

    # --- SUMMARY TABLE ---
    print(f"\n{'='*80}")
    print(f"📈 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Video Name':<25} | {'Saved':<6} | {'Fallback':<10} | {'Overlap':<7} | {'Avg Feats'}")
    print(f"{'-'*25}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}")
    
    for s in all_stats_summary:
        print(f"{s['name']:<25} | {s['extracted']:<6} | {s['fallback']:<10} | {s['skipped_overlap']:<7} | {s['avg_matches']:.1f}")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()