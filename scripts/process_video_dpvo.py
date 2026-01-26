
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
from dpvo.stream import video_stream
from dpvo.utils import Timer
import rerun as rr

# We need to make sure DPVO config is loaded with default values
# cfg.merge_from_file("third_party/DPVO/config/default.yaml") # We will load this dynamically

def calculate_overlap(pose1, pose2, K, h, w):
    """
    Calculate geometric overlap (IoU) between two frames given their poses and intrinsics.
    pose1, pose2: 4x4 matrices (World to Camera) or (Camera to World)?
    DPVO returns poses as World-to-Camera (inverse of camera trajectory) or Camera-to-World?
    DPVO .terminate() returns: Poses [N, 7] (x y z qx qy qz qw) which are World-to-Camera (inv).
    
    We want to project corners of Image 1 into Image 2.
    P1 = K * T_w2c_1 * P_world
    P2 = K * T_w2c_2 * P_world
    
    P2 = K * T_w2c_2 * (T_w2c_1)^-1 * inv(K) * uv1
    T_rel = T_w2c_2 * inv(T_w2c_1)  (maps from Cam1 to Cam2)
    """
    
    # 1. Construct Relative Pose (Cam1 -> Cam2)
    # Input poses are likely [N, 7]. We need 4x4.
    # But this function takes 4x4.
    
    T1 = np.linalg.inv(pose1) # C1 -> W
    T2 = pose2 # W -> C2
    
    # T_1_to_2 = T2 * T1
    T_rel = T2 @ T1
    
    # 2. Project 4 corners of Image 1 into Image 2
    # Assume z=1 (unit plane) for corners, scale doesn't matter for homography if planar, 
    # but here we are doing full 3D projection? No, we don't have depth.
    # We can approximate infinite depth rot-only or planar?
    # Actually, DPVO keeps scale consistent.
    # But without depth map, we can't project exactly.
    # HOWEVER, for "overlap", SfM usually assumes points are at some median Scene Depth.
    # Let's assume a Proxy Geometry (Sphere or Plane at d=1). 
    # Or simpler: Just overlap of Frustums?
    # The existing process_video uses Homography from feature matches. We don't have matches here for every frame pair easily exposed.
    # We have poses.
    # Let's use a "projected frustum" overlap at median depth 1.0 (since visual scale is arbitrary in monocular)
    
    # Corners in Cam1 (Normalized coordinates at z=1)
    # We strip K for rotation logic usually, but here let's keep it simple.
    
    fx, fy, cx, cy = K[0], K[1], K[2], K[3]
    
    # Pixel corners
    corners_pix = np.array([
        [0, 0], 
        [w, 0], 
        [w, h], 
        [0, h]
    ], dtype=np.float32)
    
    # Back-project to Cam1 3D rays (z=1)
    corners_cam1 = np.zeros((4, 3))
    corners_cam1[:, 0] = (corners_pix[:, 0] - cx) / fx
    corners_cam1[:, 1] = (corners_pix[:, 1] - cy) / fy
    corners_cam1[:, 2] = 1.0
    
    # Homogenize
    corners_cam1_h = np.hstack([corners_cam1, np.ones((4, 1))]) # 4x4
    
    # Transform to Cam2
    corners_cam2_h = (T_rel @ corners_cam1_h.T).T
    corners_cam2 = corners_cam2_h[:, :3]
    
    # Check if behind camera
    if np.any(corners_cam2[:, 2] <= 0):
        # Determine overlap based on visibility?
        # If corners cross behind, overlap is weird. 
        # Simpler metric: Rotation angle + Translation direction?
        # Let's rely on projection clipping.
        return 0.0 # Safety fallback
        
    # Project to Pixel2
    corners_pix2 = np.zeros((4, 2))
    corners_pix2[:, 0] = fx * corners_cam2[:, 0] / corners_cam2[:, 2] + cx
    corners_pix2[:, 1] = fy * corners_cam2[:, 1] / corners_cam2[:, 2] + cy
    
    # Calculate IoU of Polygon(corners_pix2) with Rect(0,0,w,h)
    
    # Use OpenCV for polygon intersection
    rect_poly = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    proj_poly = corners_pix2.astype(np.float32)
    
    try:
        if not cv2.isContourConvex(proj_poly):
             # Depending on distortion, might not be convex, but usually is for pinhole
             pass
             
        ret_area, intersection = cv2.intersectConvexConvex(rect_poly, proj_poly)
        
        area_rect = w * h
        area_proj = cv2.contourArea(proj_poly)
        
        union_area = area_rect + area_proj - ret_area
        
        if union_area <= 0: return 0.0
        
        iou = ret_area / union_area
        return iou
        
    except:
        return 0.0

def pose_vec_to_mat(vec):
    # vec: x y z qx qy qz qw
    # output: 4x4 matrix
    tx, ty, tz, qx, qy, qz, qw = vec
    
    # Quaternion to Rot Matrix
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
    """
    Generate a simple calibration matrix based on image size and approximate FOV.
    """
    f = max(width, height) * 0.5 / np.tan(np.deg2rad(fov / 2))
    cx = width / 2
    cy = height / 2
    return [f, f, cx, cy]

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        # print(f"Error getting duration: {e}")
        return 0

@torch.no_grad()
def run_dpvo_and_extract(network_path, video_path, output_dir, start_number=0, skip=0, stride=1, opts=[], video_idx=0, overlap_thresh=0.9, config_file="default.yaml"):
    
    # Load config
    # Assuming we are running from workspace root
    config_path = Path(f"third_party/DPVO/config/{config_file}")
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return 0

    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list(opts)

    # Initialize DPVO
    queue = Queue(maxsize=8)
    
    # Probe video for size
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"  ℹ️  Video Info: {w}x{h} | {fps:.2f} FPS | {duration:.2f}s | {total_frames} frames")
    
    calib = get_calibration(w, h)
    calib_file = output_dir / "temp_calib.txt"
    np.savetxt(calib_file, np.array(calib).reshape(1, 4))
    
    # Determine resize factor
    # User requested ~480 pixels resolution for speed, but high accuracy config.
    # We calculate scale to make the shorter dimension ~480.
    target_short = 480.0
    short_dim = min(w, h)
    
    if short_dim > target_short:
        resize_factor = target_short / short_dim
        print(f"  📉 Auto-Resizing to {int(w*resize_factor)}x{int(h*resize_factor)} (Scale: {resize_factor:.3f})")
    else:
        resize_factor = 1.0 # Already small enough
    
    process = Process(target=video_stream, args=(queue, str(video_path), str(calib_file), stride, skip, resize_factor))
    process.start()
    
    slam = None
    
    # Progress bar for DPVO tracking
    pbar = tqdm(total=total_frames, desc=f"🧠 Tracking {video_path.name}", unit="frame", dynamic_ncols=True, leave=False)
    
    last_t = 0
    
    # Main loop
    while True:
        (t, image, intrinsics) = queue.get()
        if t < 0: break # End of stream

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
    
    # Process Termination & Keyframe Extraction
    print(f"  🔍 Identifying keyframes (Full Trajectory Analysis)...")
    
    # slam.terminate() returns (poses, tstamps)
    # poses: [N, 7] (x, y, z, qx, qy, qz, qw) - World to Camera (likely, as per standard VO)
    # tstamps: [N] frame indices
    
    poses, tstamps = slam.terminate()
    
    # Filter based on overlap
    # We iterate and keep frames if overlap < threshold
    
    kept_indices = []
    
    # Always keep first frame
    if len(poses) > 0:
        kept_indices.append(tstamps[0])
        last_kept_pose = pose_vec_to_mat(poses[0])
        # We need intrinsics for overlap calc. We passed `calib` to stream process, and generated `intrinsics` tensor.
        # Let's reuse the initial calib we generated.
        # calib from get_calibration is [fx, fy, cx, cy]
        # w, h are known.
        
        K = calib
        
        for i in range(1, len(poses)):
            curr_pose = pose_vec_to_mat(poses[i])
            
            # Calculate overlap with last KEPT pose
            # Note: DPVO poses are World-to-Camera? Or Camera-to-World?
            # DPVO internal: poses are stored as Lie Algebra...
            # The .terminate() converts them.
            # Usually SLAM returns trajectory (Cam to World).
            # But let's check DPVO viewer:
            # point = pose.inv() * point_world ? No.
            # Let's assume Poses are World-to-Camera (Transformation to move World to this Camera).
            # So T_w2c
            
            iou = calculate_overlap(last_kept_pose, curr_pose, K, h, w)
            
            # print(f"Frame {tstamps[i]} IoU: {iou:.3f}")
            
            if iou < overlap_thresh:
                kept_indices.append(tstamps[i])
                last_kept_pose = curr_pose
                
    else:
        print("Warning: No poses returned by DPVO.")

    kept_indices = np.array(kept_indices).astype(int)
    kept_indices = np.sort(kept_indices)
    
    print(f"  ✨ Filtered: {len(poses)} -> {len(kept_indices)} keyframes (Threshold: {overlap_thresh})")
    
    # Save Trajectory (TUM Format: timestamp tx ty tz qx qy qz qw)
    traj_path = output_dir / f"trajectory_video_{video_idx}.txt"
    print(f"  💾 Saving trajectory to {traj_path}")
    
    with open(traj_path, "w") as f:
        for i in range(len(poses)):
            # DPVO poses are [N, 7] (tx, ty, tz, qx, qy, qz, qw)
            # Assuming these are World-to-Camera (W2C).
            # We want to save Camera-to-World (C2W) for visualization (The path of the camera).
            
            # W2C matrix
            T_w2c = pose_vec_to_mat(poses[i])
            
            # C2W matrix (Inverse)
            T_c2w = np.linalg.inv(T_w2c)
            
            # Extract Translation
            tx, ty, tz = T_c2w[:3, 3]
            
            # Extract Rotation (Matrix -> Quaternion)
            # Simple conversion or use scipy/trimesh if available? 
            # Let's implement simple Mat->Quat to avoid heavy deps inside the loop if possible, 
            # Or just use the original quaternion if we assume it's just inverse sign?
            # Inverse of quaternion q is conjugate q* (if unit).
            # q = (qx, qy, qz, qw). q* = (-qx, -qy, -qz, qw).
            # BUT this only works if the standard is the same.
            # Rotation matrix inversion R^T is accurate.
            
            # Let's rely on T_c2w rotation matrix to Quaternion
            R_c2w = T_c2w[:3, :3]
            
            # Implementation of matrix to quaternion (Hamilton convention usually)
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
            
            ts = tstamps[i]
            # Write TUM format: timestamp tx ty tz qx qy qz qw
            f.write(f"{ts} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    # --- Rerun Logging ---
    rrd_path = output_dir / f"trajectory_video_{video_idx}.rrd"
    print(f"  💾 Saving Rerun recording to {rrd_path}")
    
    rr.init(f"dpvo_video_{video_idx}", spawn=False)
    
    # Collect positions and quats for bulk logging
    all_pos = []
    all_q = []
    all_ts = []

    for i in range(len(poses)):
        T_w2c = pose_vec_to_mat(poses[i])
        T_c2w = np.linalg.inv(T_w2c)
        tx, ty, tz = T_c2w[:3, 3]
        
        # Rot matrix to Quat
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
        all_q.append([qx, qy, qz, qw]) # Rerun xyzw
        all_ts.append(tstamps[i])
        
    all_pos = np.array(all_pos)
    
    # Log Trajectory Path
    rr.log("world/trajectory", rr.LineStrips3D([all_pos], colors=[[0, 0, 255]]))
    
    # Log Cameras
    for i in range(len(all_pos)):
        rr.set_time("timeline", duration=all_ts[i])
        
        rr.log(
            "world/camera", 
            rr.Transform3D(
                translation=all_pos[i], 
                rotation=rr.Quaternion(xyzw=all_q[i])
            )
        )
        rr.log(
            "world/camera/view",
            rr.Pinhole(resolution=[w, h], focal_length=float(K[0]))
        )

    # Log Overview Frustums (Subsampled)
    stride_vis = max(1, len(all_pos) // 100) # Ensure ~100 frames max for overview
    rr.log(
        "world/all_cameras",
        rr.Arrows3D(
            origins=all_pos[::stride_vis],
            vectors=np.array([
                pose_vec_to_mat([0,0,0] + all_q[i]).T[:3, 2] * 0.5 # Approximate forward vector 
                for i in range(0, len(all_q), stride_vis)
            ]),
            colors=[[255, 0, 0]]
        )
    )
    
    rr.save(rrd_path)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    saved_count = start_number
    
    # Indices are sorted, so we can just read through
    # Note: tstamps are 't' indices from DPVO stream.
    # We need to convert them to original video frame indices.
    # Logic in stream.py:
    # for _ in range(stride): cap.read()
    # queue.put(t, image)
    # So t=0 corresponds to the (stride-1)-th frame if skip=0.
    # frame_idx = skip + t * stride + (stride - 1)
    
    # Pre-calculate target indices
    original_indices = [skip + t * stride + (stride - 1) for t in kept_indices]
    
    print(f"  💾 Saving {len(original_indices)} keyframes (Optimized Seeking)...")
    
    pbar_extract = tqdm(total=len(original_indices), desc="💾 Saving Keyframes", unit="frame", dynamic_ncols=True, leave=False)
    
    for idx in original_indices:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            out_path = images_dir / f"frame_{saved_count:05d}_video_{video_idx}.png"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
        else:
            print(f"Warning: Could not read frame {idx}")
            
        pbar_extract.update(1)
        
    pbar_extract.close()
    cap.release()
    
    return len(original_indices)


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes using DPVO")
    parser.add_argument("--video", nargs='+', required=True, help="Path to input video file(s) or folder(s)")
    parser.add_argument("--output", help="Output directory (defaults to datasets/<first_video_name>)")
    parser.add_argument("--model", default="third_party/DPVO/dpvo.pth", help="Path to DPVO model checkpoint")
    parser.add_argument("--stride", type=int, default=1, help="Stride for DPVO tracking")
    parser.add_argument("--opts", nargs='+', default=[], help="DPVO config options")
    parser.add_argument("--overlap", type=float, default=0.9, help="Overlap threshold (0.0 - 1.0) for filtering. Default 0.9. Lower = more spacing.")
    parser.add_argument("--fast", action="store_true", help="Use fast DPVO configuration (less accurate but faster)")
    
    args = parser.parse_args()
    
    # Setup Output
    raw_video_paths = [Path(p).resolve() for p in args.video]
    video_paths = []
    
    for p in raw_video_paths:
        if p.is_dir():
             # Recursively find videos in folder
             valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
             found_videos = sorted([
                 f for f in p.rglob("*") 
                 if f.suffix.lower() in valid_extensions and f.is_file()
             ])
             video_paths.extend(found_videos)
             print(f"📂 Found {len(found_videos)} videos in {p.name}")
        elif p.exists():
             video_paths.append(p)
        else:
             print(f"⚠️ Warning: {p} does not exist")
             
    if not video_paths:
        print("❌ Error: No valid videos found in provided paths.")
        sys.exit(1)
        
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        root_dir = Path(__file__).parent.parent
        output_dir = root_dir / "datasets" / video_paths[0].stem
    
    print(f"📂 Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Model Check
    model_path = Path(args.model)
    if not model_path.exists():
        fallback = Path("third_party/DPVO/dpvo.pth")
        if fallback.exists():
            model_path = fallback
        else:
            print("Error: Model not found.")
            sys.exit(1)
            
    print(f"🧠 Model: {model_path}")
    
    global_frame_count = 0
    stats_list = []
    
    print(f"{'='*60}")
    print(f"🚀 PROCESSING {len(video_paths)} VIDEOS WITH DPVO")
    print(f"{'='*60}")
    
    for video_idx, video in enumerate(video_paths):
        if not video.exists():
            print(f"❌ Error: {video} not found")
            continue
            
        print(f"\n🎞️  --- Processing {video.name} ---")
        start_time = time.time()
        
        try:
            cfg_name = "fast.yaml" if args.fast else "default.yaml"
            count = run_dpvo_and_extract(str(model_path), video, output_dir, start_number=global_frame_count, stride=args.stride, opts=args.opts, video_idx=video_idx, overlap_thresh=args.overlap, config_file=cfg_name)
            
            # --- Stats Collection ---
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
            
            # --- Single Video Recap ---
            print(f"\n📊 --- RECAP: {video.name} ---")
            print(f"  Extracted: {count} frames")
            print(f"  Time taken: {duration_proc:.2f}s")
            print(f"  Speed:      {fps_proc:.2f} fps")
            print("------------------------------------------------------------")
            
            global_frame_count += count
            
        except Exception as e:
            print(f"❌ Error processing {video.name}: {e}")
            import traceback
            traceback.print_exc()
            
    # --- Final Recap ---
    print(f"\n{'='*60}")
    print(f"📈 FINAL STATISTICS RECAP")
    print(f"{'='*60}")
    print(f"{'Video Name':<30} | {'Extracted':<10} | {'Time (s)':<10} | {'FPS':<6}")
    print(f"{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    
    total_time = 0
    for s in stats_list:
        print(f"{s['name']:<30} | {s['extracted']:<10} | {s['time']:<10.2f} | {s['fps']:<6.2f}")
        total_time += s['time']
        
    print(f"{'-'*60}")
    print(f"{'TOTAL':<30} | {global_frame_count:<10} | {total_time:<10.2f} | {global_frame_count/total_time if total_time > 0 else 0:<6.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
