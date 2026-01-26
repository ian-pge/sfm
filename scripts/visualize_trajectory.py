
import argparse
import numpy as np
import rerun as rr
import time
from pathlib import Path

def quat_to_rotmat(q):
    """
    Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def read_tum_trajectory(filepath):
    """
    Reads a TUM format trajectory file.
    Format: timestamp tx ty tz qx qy qz qw
    Returns: positions (N, 3), quaternions (N, 4) [xyzw]
    """
    positions = []
    quats = []
    timestamps = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = list(map(float, line.split()))
            # parts: [timestamp, tx, ty, tz, qx, qy, qz, qw]
            
            timestamps.append(parts[0])
            positions.append(parts[1:4])
            # Rerun expects [x, y, z, w] for quaternions
            quats.append(parts[4:8])
            
    return np.array(timestamps), np.array(positions), np.array(quats)

def viz_trajectory(traj_path):
    print(f"Reading trajectory from {traj_path}...")
    timestamps, positions, quats = read_tum_trajectory(traj_path)
    
    if len(positions) == 0:
        print("Error: No positions found in file.")
        return

    # Initialize Rerun
    rr.init("trajectory_viz", spawn=False)
    
    # Standard Practice: The SDK logs data. The Viewer reads it.
    # We will log to file to decouple the two.
    output_rrd = str(Path(traj_path).with_suffix(".rrd"))
    print(f"Logging visualization data to: {output_rrd}")
    rr.save(output_rrd)

    # Log the entire path as a line strip (background)
    rr.log("world/trajectory", rr.LineStrips3D([positions], colors=[[0, 0, 255]]))
    
    # Log Coordinate Arrows to visualize orientation
    # We can log them all at once using Arrows3D or Transform3D?
    # Rerun Transform3D is usually for the logging context.
    # We want to visualize the camera frustums.
    
    # Let's log 'Frames'
    for i in range(len(positions)):
        t = timestamps[i]
        pos = positions[i]
        q = quats[i]
        
        # Log a transform at the timestamp?
        # Simulating time playback:
        rr.set_time("timeline", duration=t)
        
        # Log the Camera Pose
        # Rerun uses (x, y, z, w) for quaternion
        rr.log(
            "world/camera", 
            rr.Transform3D(
                translation=pos, 
                rotation=rr.Quaternion(xyzw=q)
            )
        )
        
        # Log a frustum geometry AT the camera
        # Standard Pinhole (ViewCoordinates.RUB)
        rr.log(
            "world/camera/view",
            rr.Pinhole(
                resolution=[1920, 1080], # Dummy high-res
                focal_length=1000,       # Dummy focal
            )
        )

    # Also Log all frames as a static cloud of frustums for overview
    # We do this by logging to a static timeline context or just independent entities
    # To keep it cleaning, let's just make a separate "all_frames" entity
    
    # Filter for overview (every 5th frame)
    stride = 5
    rr.log(
        "world/all_cameras",
        rr.Arrows3D(
            origins=positions[::stride],
            vectors=np.array([quat_to_rotmat(q) @ np.array([0, 0, 1]) for q in quats[::stride]]) * 0.5,
            colors=[[255, 0, 0]]
        )
    )
    
    print("Done. Data saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Trajectory (TUM Format) with Rerun")
    parser.add_argument("--input", required=True, help="Path to trajectory.txt file")
    args = parser.parse_args()
    
    viz_trajectory(args.input)
