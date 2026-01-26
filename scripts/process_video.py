import argparse
import subprocess
import sys
import os
import warnings

# Suppress Qt Warnings
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
os.environ["QT_QPA_PLATFORM"] = "xcb" # Sometimes helps with thread issues

from pathlib import Path
import cv2
import numpy as np
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
    # (Kept unchanged for brevity, but you can copy your previous version here if needed)
    # This function is not the main focus of your request.
    pass 

def extract_precise_geometry(video_path, output_dir, overlap_thresh=0.60, downscale_factor=1, start_number=0, video_idx=0, use_yolo=False, show_gui=False):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load Geometry Models
    extractor = ALIKED(max_num_keypoints=1024, detection_threshold=0.01).eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    # Load YOLO
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
    
    base_stride = max(1, int(fps / 10)) 
    current_stride = base_stride
    max_stride = int(fps) 
    
    stats = {
        "processed_frames": 0,
        "saved_frames": 0,
        "skipped_overlap": 0,
        "fallback_background": 0, 
        "skipped_no_features": 0,
        "total_matches_on_car": 0,
        "valid_car_detections": 0
    }

    if show_gui:
        cv2.startWindowThread()
        # Small floating window (fits image size exactly, usually 640x480)
        cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("Result", w_proc, h_proc)

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
    
    proc_start_time = time.time()
    
    with pbar:
        while True:
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
            
            # Variables for Vis
            current_kpts_all = feats['keypoints'][0].cpu().numpy()
            matched_kpts_curr = None 
            
            if prev_feats is None:
                should_save = True
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

                        if not car_detected:
                            stats["fallback_background"] += 1
                    
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

            if should_save:
                current_stride = base_stride
            else:
                if calculated_overlap > 0.95:
                    current_stride = min(current_stride + 1, max_stride)
                elif calculated_overlap < (overlap_thresh + 0.1):
                    current_stride = base_stride
            
            # --- SAVE ---
            if should_save:
                if downscale_factor > 1:
                    h, w = frame_full.shape[:2]
                    save_frame = cv2.resize(frame_full, (w // downscale_factor, h // downscale_factor))
                else:
                    save_frame = frame_full
                    
                filename = images_dir / f"frame_{saved_count:05d}_video_{video_idx}.png"
                cv2.imwrite(str(filename), save_frame)

                pbar.set_postfix_str(f"💾 {saved_count} | 📐 {calculated_overlap:.2f}")
                
                prev_feats = feats
                saved_count += 1
                saved_this_session += 1
                stats["saved_frames"] += 1

            # --- VISUALIZATION ---
            if show_gui:
                vis_img = process_frame.copy()
                if use_yolo and combined_mask is not None:
                    overlay = np.zeros_like(vis_img)
                    overlay[combined_mask] = [0, 0, 255]
                    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
                
                for kp in current_kpts_all:
                    cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

                if matched_kpts_curr is not None:
                     for kp in matched_kpts_curr:
                         cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

                # Colors (BGR)
                # Default: Dark Blue (139, 0, 0)
                # Flash:   Cyan      (255, 255, 0)
                text_color = (139, 0, 0) 
                if should_save: 
                    text_color = (255, 255, 0)

                # Top Block (One info per line, spacing 30px)
                # 1. Overlap
                cv2.putText(vis_img, f"Overlap: {calculated_overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                # 2. Stride
                cv2.putText(vis_img, f"Stride: {current_stride}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                elapsed_proc = time.time() - proc_start_time
                proc_fps = stats["processed_frames"] / elapsed_proc if elapsed_proc > 0 else 0
                progress_pct = (frame_count / total_frames) * 100
                
                # 3. Speed
                cv2.putText(vis_img, f"Speed: {proc_fps:.1f} fps", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                # 4. Kept
                cv2.putText(vis_img, f"Kept: {saved_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                # 5. Time
                cv2.putText(vis_img, f"Time: {elapsed_proc:.0f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                # 6. Progress
                cv2.putText(vis_img, f"Prog: {progress_pct:.1f}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                if use_yolo and not car_detected:
                     cv2.putText(vis_img, "NO CAR - BG FALLBACK", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Video Stats Overlay (Bottom Left)
                h_vis, w_vis = vis_img.shape[:2]
                duration_sec = total_frames / fps if fps > 0 else 0
                
                # Input Video Stats
                stats_text = f"{original_w}x{original_h} | {fps:.2f}fps | {total_frames} frms | {duration_sec:.1f}s"
                cv2.putText(vis_img, stats_text, (10, h_vis - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Result", vis_img)
                cv2.waitKey(1)

            frame_count += 1
            pbar.update(1)
        
    cap.release()
    if show_gui: cv2.destroyAllWindows()
    return saved_this_session, stats

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", nargs='+', required=True, help="Video file(s)")
    
    parser.add_argument("--no-adaptive", action="store_false", dest="adaptive", help="Disable adaptive mode")
    parser.add_argument("--no-yolo", action="store_false", dest="yolo", help="Disable YOLO segmentation")

    parser.set_defaults(adaptive=True, yolo=True)
    
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
            # Pass use_siglip argument
            count, v_stats = extract_precise_geometry(
                video, output_dir, args.overlap, args.downscale, 
                start_number=global_frame_count, video_idx=video_idx, 
                use_yolo=args.yolo, show_gui=args.gui
            )
        else:
            if args.num_frames is None:
                print("Error: --num_frames is required for fixed mode")
                sys.exit(1)
            # Fixed mode doesn't support SigLIP in this snippet, you can add it if needed
            count, v_stats = extract_frames_fixed(video, output_dir, args.num_frames, args.downscale, start_number=global_frame_count, video_idx=video_idx)
            
        end_time = time.time()
        duration_proc = end_time - start_time
        
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
        print(f"  ✅ Saved Total:              {count}")
        print(f"  ⚠️ Fallback to BG:           {v_stats.get('fallback_background', 0)}")
        
        print("------------------------------------------------------------")
        global_frame_count += count
    
    print(f"\n✅ All Done. Total frames extracted: {global_frame_count}")

    print(f"\n{'='*80}")
    print(f"📈 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Video Name':<25} | {'Saved':<6} | {'Fallback':<10} | {'Overlap':<7}")
    print(f"{'-'*25}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}")
    
    for s in all_stats_summary:
        print(f"{s['name']:<25} | {s['extracted']:<6} | {s['fallback']:<10} | {s['skipped_overlap']:<7}")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()