import argparse
import numpy as np
import torch
import cv2
import glob
import os
from pathlib import Path
from multiprocessing import Queue, Process
import time
import sys
import shutil
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# DPVO imports
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import video_stream

import rerun as rr

def pose_vec_to_mat(vec):
    tx, ty, tz, qx, qy, qz, qw = vec
    q = np.array([qx, qy, qz, qw])
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def get_calibration(width, height, fov=80):
    f = max(width, height) * 0.5 / np.tan(np.deg2rad(fov / 2))
    cx = width / 2
    cy = height / 2
    return [f, f, cx, cy]

def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0

def filter_trajectory_robust(poses, tstamps, K, overlap_ratio=0.9):
    """
    Monte Carlo Volumetric Overlap Check.
    Robust to rotation, translation (parallax), and forward/backward motion (zoom).
    """
    if len(poses) == 0:
        return [], []
        
    kept_indices = [0]
    last_pose_w2c = pose_vec_to_mat(poses[0])
    
    # --- CONFIGURATION ---
    # We simulate a "Cloud" of points to represent the scene volume
    n_points = 100 
    depth_min = 0.5   # Near clip (meters)
    depth_max = 15.0  # Far clip (meters) - increased for robustness
    
    # Image dimensions (approx from K)
    w = K[2] * 2
    h = K[3] * 2
    
    # 1. Generate Random 3D Points in Camera Space (Normalized)
    # x,y in range [-1, 1] for Normalized Device Coords, then scaled by depth/K
    # We essentially back-project random pixels at random depths
    u_rand = np.random.uniform(0, w, n_points)
    v_rand = np.random.uniform(0, h, n_points)
    d_rand = np.random.uniform(depth_min, depth_max, n_points)
    
    # Back-project pixels to 3D Camera coordinates (Frame A)
    # X = (u - cx) * z / fx
    # Y = (v - cy) * z / fy
    # Z = z
    X_local = (u_rand - K[2]) * d_rand / K[0]
    Y_local = (v_rand - K[3]) * d_rand / K[1]
    Z_local = d_rand
    
    # Homogeneous coords [N, 4]
    # This "cloud" represents the view of the LAST SAVED KEYFRAME
    points_cam_fixed = np.vstack((X_local, Y_local, Z_local, np.ones(n_points))).T

    for i in range(1, len(poses)):
        curr_pose_w2c = pose_vec_to_mat(poses[i])
        
        # Calculate Relative Pose: T_current_from_last
        # T_rel = T_curr * inv(T_last)
        T_rel = curr_pose_w2c @ np.linalg.inv(last_pose_w2c)
        
        # Transform points from Last Keyframe (Cam A) to Current Frame (Cam B)
        # points_cam_b = (T_rel @ points_cam_a.T).T
        points_cam_b = points_cam_fixed @ T_rel.T
        
        # Extract new coordinates
        x_b = points_cam_b[:, 0]
        y_b = points_cam_b[:, 1]
        z_b = points_cam_b[:, 2]
        
        # --- VISIBILITY CHECK ---
        
        # 1. Check Depth (Points must be in front of camera)
        valid_depth = z_b > 0.1
        
        # 2. Project valid points to Pixels in Current Frame
        # u = fx * x/z + cx
        if np.sum(valid_depth) == 0:
            current_overlap = 0.0
        else:
            u_b = (K[0] * x_b[valid_depth] / z_b[valid_depth]) + K[2]
            v_b = (K[1] * y_b[valid_depth] / z_b[valid_depth]) + K[3]
            
            # 3. Check Image Bounds (0 < u < w, 0 < v < h)
            in_view_u = (u_b >= 0) & (u_b < w)
            in_view_v = (v_b >= 0) & (v_b < h)
            visible_count = np.sum(in_view_u & in_view_v)
            
            # Calculate Overlap Ratio
            current_overlap = visible_count / n_points
        
        # Also check for resolution loss (Zooming OUT)
        # If we zoom out, overlap is 100%, but resolution drops.
        # Check ratio of average depth.
        # If z_b.mean() > z_fixed.mean() * 1.2 -> We moved back significantly
        avg_depth_ratio = np.mean(z_b[valid_depth]) / np.mean(Z_local) if np.any(valid_depth) else 1.0
        
        is_zoomed_out = avg_depth_ratio > 1.15 # 15% depth increase
        
        # DECISION: Save frame if we lost points OR zoomed out
        if current_overlap < overlap_ratio or is_zoomed_out:
            kept_indices.append(i)
            last_pose_w2c = curr_pose_w2c
            
            # NOTE: For true robustness, we do NOT regenerate points here.
            # We keep comparing against the "Last Saved Frame's" volume.
            # But effectively, since 'last_pose_w2c' updated, T_rel is now relative to this new frame.
            # So we are implicitly checking "Does the NEW frame cover the VOLUME of the LAST saved frame?"
            # This is correct.

    return poses[kept_indices], tstamps[kept_indices]


@torch.no_grad()
def run_dpvo_and_extract(network_path, video_path, output_dir, start_number=0, skip=0, stride=1, opts=[], video_idx=0, overlap_thresh=0.9, config_file="default.yaml"):
    
    # Load config
    config_path = Path(f"third_party/DPVO/config/{config_file}")
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return 0

    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(opts)

    queue = Queue(maxsize=8)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"  ℹ️  Video Info: {w}x{h} | {fps:.2f} FPS | {total_frames} frames")
    
    calib = get_calibration(w, h)
    calib_file = output_dir / "temp_calib.txt"
    np.savetxt(calib_file, np.array(calib).reshape(1, 4))
    
    target_short = 480.0
    short_dim = min(w, h)
    
    if short_dim > target_short:
        resize_factor = target_short / short_dim
        print(f"  📉 Auto-Resizing to {int(w*resize_factor)}x{int(h*resize_factor)} (Scale: {resize_factor:.3f})")
    else:
        resize_factor = 1.0 
    
    process = Process(target=video_stream, args=(queue, str(video_path), str(calib_file), stride, skip, resize_factor))
    process.start()
    
    slam = None
    pbar = tqdm(total=total_frames, desc=f"🧠 Tracking {video_path.name}", unit="frame", dynamic_ncols=True, leave=False)
    
    last_t = 0
    
    while True:
        (t, image, intrinsics) = queue.get()
        if t < 0: break 

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network_path, ht=H, wd=W, viz=False)
        
        slam(t, image, intrinsics)
        
        progres = t - last_t
        if progres > 0:
            pbar.update(progres * stride)
            last_t = t
        
    process.join()
    pbar.close()
    
    print(f"  🔍 Full Trajectory Analysis (Robust 3D Overlap Filtering)...")
    
    # Get FULL trajectory
    poses, tstamps = slam.terminate()
    
    # -------------------------------------------------------------
    # 🧠 NEW: ROBUST 3D VOLUME FILTERING
    # -------------------------------------------------------------
    print(f"  ⚖️  Filtering frames with {overlap_thresh*100:.0f}% volumetric overlap...")
    K_list = calib # [fx, fy, cx, cy]
    
    poses_kept, tstamps_kept = filter_trajectory_robust(poses, tstamps, K_list, overlap_ratio=overlap_thresh)
    
    print(f"  ✨ Filtered: {len(poses)} -> {len(poses_kept)} keyframes")
    
    # Update for saving
    poses = poses_kept
    tstamps = tstamps_kept
    
    if len(poses) == 0:
        print("Warning: No frames kept.")
        return 0

    kept_indices = np.array(tstamps).astype(int)
    kept_indices = np.sort(kept_indices)
    
    # Save Trajectory
    traj_path = output_dir / f"trajectory_video_{video_idx}.txt"
    with open(traj_path, "w") as f:
        for i in range(len(poses)):
            T_w2c = pose_vec_to_mat(poses[i])
            T_c2w = np.linalg.inv(T_w2c)
            tx, ty, tz = T_c2w[:3, 3]
            R_c2w = T_c2w[:3, :3]
            
            # Mat to Quat 
            tr = np.trace(R_c2w)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R_c2w[2,1] - R_c2w[1,2]) / S
                qy = (R_c2w[0,2] - R_c2w[2,0]) / S
                qz = (R_c2w[1,0] - R_c2w[0,1]) / S
            elif (R_c2w[0,0] > R_c2w[1,1]) and (R_c2w[0,0] > R_c2w[2,2]):
                S = np.sqrt(1.0 + R_c2w[0,0] - R_c2w[1,1] - R_c2w[2,2]) * 2
                qw = (R_c2w[2,1] - R_c2w[1,2]) / S
                qx = 0.25 * S
                qy = (R_c2w[0,1] + R_c2w[1,0]) / S
                qz = (R_c2w[0,2] + R_c2w[2,0]) / S
            elif R_c2w[1,1] > R_c2w[2,2]:
                S = np.sqrt(1.0 + R_c2w[1,1] - R_c2w[0,0] - R_c2w[2,2]) * 2
                qw = (R_c2w[0,2] - R_c2w[2,0]) / S
                qx = (R_c2w[0,1] + R_c2w[1,0]) / S
                qy = 0.25 * S
                qz = (R_c2w[1,2] + R_c2w[2,1]) / S
            else:
                S = np.sqrt(1.0 + R_c2w[2,2] - R_c2w[0,0] - R_c2w[1,1]) * 2
                qw = (R_c2w[1,0] - R_c2w[0,1]) / S
                qx = (R_c2w[0,2] + R_c2w[2,0]) / S
                qy = (R_c2w[1,2] + R_c2w[2,1]) / S
                qz = 0.25 * S
            
            f.write(f"{tstamps[i]} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    # Rerun Logging
    rrd_path = output_dir / f"trajectory_video_{video_idx}.rrd"
    rr.init(f"dpvo_video_{video_idx}", spawn=False)
    
    all_pos, all_q = [], []

    for i in range(len(poses)):
        T_w2c = pose_vec_to_mat(poses[i])
        T_c2w = np.linalg.inv(T_w2c)
        tx, ty, tz = T_c2w[:3, 3]
        
        # Recalculate or reuse quat (simplified for brevity)
        R_c2w = T_c2w[:3, :3]
        tr = np.trace(R_c2w)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R_c2w[2,1] - R_c2w[1,2]) / S
            qy = (R_c2w[0,2] - R_c2w[2,0]) / S
            qz = (R_c2w[1,0] - R_c2w[0,1]) / S
        elif (R_c2w[0,0] > R_c2w[1,1]) and (R_c2w[0,0] > R_c2w[2,2]):
            S = np.sqrt(1.0 + R_c2w[0,0] - R_c2w[1,1] - R_c2w[2,2]) * 2
            qw = (R_c2w[2,1] - R_c2w[1,2]) / S
            qx = 0.25 * S
            qy = (R_c2w[0,1] + R_c2w[1,0]) / S
            qz = (R_c2w[0,2] + R_c2w[2,0]) / S
        elif R_c2w[1,1] > R_c2w[2,2]:
            S = np.sqrt(1.0 + R_c2w[1,1] - R_c2w[0,0] - R_c2w[2,2]) * 2
            qw = (R_c2w[0,2] - R_c2w[2,0]) / S
            qx = (R_c2w[0,1] + R_c2w[1,0]) / S
            qy = 0.25 * S
            qz = (R_c2w[1,2] + R_c2w[2,1]) / S
        else:
            S = np.sqrt(1.0 + R_c2w[2,2] - R_c2w[0,0] - R_c2w[1,1]) * 2
            qw = (R_c2w[1,0] - R_c2w[0,1]) / S
            qx = (R_c2w[0,2] + R_c2w[2,0]) / S
            qy = (R_c2w[1,2] + R_c2w[2,1]) / S
            qz = 0.25 * S

        all_pos.append([tx, ty, tz])
        all_q.append([qx, qy, qz, qw])
        
    all_pos = np.array(all_pos)
    rr.log("world/trajectory", rr.LineStrips3D([all_pos], colors=[[0, 0, 255]]))
    rr.save(rrd_path)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # 🚀 FAST LINEAR SCAN SAVING
    # -------------------------------------------------------------
    
    original_indices = [skip + t * stride + (stride - 1) for t in kept_indices]
    indices_to_save = set(original_indices)
    max_idx = max(original_indices) if original_indices else 0
    
    print(f"  💾 Saving {len(original_indices)} keyframes (Fast Linear Scan)...")
    
    pbar_extract = tqdm(total=max_idx+1, desc="💾 Saving Keyframes", unit="frame", dynamic_ncols=True, leave=False)
    
    cap = cv2.VideoCapture(str(video_path))
    current_idx = 0
    saved_count = start_number
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_idx in indices_to_save:
            out_path = images_dir / f"frame_{saved_count:05d}_video_{video_idx}.png"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
            
        current_idx += 1
        pbar_extract.update(1)
        
        if current_idx > max_idx:
            break
            
    pbar_extract.close()
    cap.release()
    
    return len(original_indices)


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes using DPVO")
    parser.add_argument("--video", nargs='+', required=True, help="Path to input video file(s) or folder(s)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--model", default="third_party/DPVO/dpvo.pth", help="Path to DPVO model")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--opts", nargs='+', default=[], help="DPVO config options")
    parser.add_argument("--overlap", type=float, default=0.9, help="Overlap threshold (0.0 - 1.0). Default 0.9. Lower = Fewer frames.")
    parser.add_argument("--fast", action="store_true", help="Fast config")
    
    args = parser.parse_args()
    
    # Setup Output
    raw_video_paths = [Path(p).resolve() for p in args.video]
    video_paths = []
    
    for p in raw_video_paths:
        if p.is_dir():
             valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
             found_videos = sorted([
                 f for f in p.rglob("*") 
                 if f.suffix.lower() in valid_extensions and f.is_file()
             ])
             video_paths.extend(found_videos)
        elif p.exists():
             video_paths.append(p)
             
    if not video_paths:
        print("❌ Error: No valid videos found.")
        sys.exit(1)
        
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        root_dir = Path(__file__).parent.parent
        output_dir = root_dir / "datasets" / video_paths[0].stem
    
    print(f"📂 Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        fallback = Path("third_party/DPVO/dpvo.pth")
        if fallback.exists():
            model_path = fallback
        else:
            print("Error: Model not found.")
            sys.exit(1)
    
    global_frame_count = 0
    stats_list = []
    
    print(f"{'='*60}")
    print(f"🚀 PROCESSING {len(video_paths)} VIDEOS WITH DPVO")
    print(f"{'='*60}")
    
    for video_idx, video in enumerate(video_paths):
        print(f"\n🎞️  --- Processing {video.name} ---")
        start_time = time.time()
        
        try:
            cfg_name = "fast.yaml" if args.fast else "default.yaml"
            count = run_dpvo_and_extract(str(model_path), video, output_dir, start_number=global_frame_count, stride=args.stride, opts=args.opts, video_idx=video_idx, overlap_thresh=args.overlap, config_file=cfg_name)
            
            end_time = time.time()
            duration_proc = end_time - start_time
            fps_proc = count / duration_proc if duration_proc > 0 else 0
            
            vid_duration = get_video_duration(video)
                 
            stats = {
                "name": video.name,
                "duration": vid_duration,
                "extracted": count,
                "time": duration_proc,
                "fps": fps_proc
            }
            stats_list.append(stats)
            
            print(f"\n📊 --- RECAP: {video.name} ---")
            print(f"  Extracted: {count} frames")
            print(f"  Time taken: {duration_proc:.2f}s")
            print("------------------------------------------------------------")
            
            global_frame_count += count
            
        except Exception as e:
            print(f"❌ Error processing {video.name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Final Recap
    print(f"\n{'='*60}")
    print(f"📈 FINAL STATISTICS RECAP")
    print(f"{'='*60}")
    print(f"{'Video Name':<30} | {'Extracted':<10} | {'Time (s)':<10}")
    print(f"{'-'*30}-+-{'-'*10}-+-{'-'*10}")
    
    total_time = 0
    for s in stats_list:
        print(f"{s['name']:<30} | {s['extracted']:<10} | {s['time']:<10.2f}")
        total_time += s['time']
        
    print(f"{'-'*60}")
    print(f"{'TOTAL':<30} | {global_frame_count:<10} | {total_time:<10.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()