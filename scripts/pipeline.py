import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import torch

# Add external scripts to path for SuperGluePretrainedNetwork (just in case, though ALIKED doesn't use it)
current_dir = Path(__file__).parent
external_dir = current_dir / "external"
if external_dir.exists():
    sys.path.append(str(external_dir))


from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction


# Configuration for different feature extractors and matchers
FEATURE_CONFIGS = {
    "aliked": {
        "feature": "aliked-n16",
        "matcher": "aliked+lightglue",
    },
    "superpoint": {
        "feature": "superpoint_max",
        "matcher": "superpoint+lightglue",
    },
    "disk": {
        "feature": "disk",
        "matcher": "disk+lightglue",
    },
    "sift": {
        "feature": "sift",
        "matcher": "adalam",
    },
}


def setup_paths(args):
    dataset_path = Path(args.dataset)
    images_path = dataset_path / "images"
    cameras_path = dataset_path / "cameras.txt"

    if not images_path.exists():
        print(f"Error: Images directory not found at {images_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    return dataset_path, images_path, cameras_path, output_path


def run_feature_extraction(images_path, output_path, feature_type="aliked"):
    config = FEATURE_CONFIGS[feature_type]
    feature_conf = extract_features.confs[config["feature"]]
    feature_path = output_path / "features.h5"

    print(f"Extracting features to {feature_path} using {feature_type}...")
    # HLOC uses auto-detection. Verifying device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Feature Extraction Device: {device}")
    extract_features.main(feature_conf, images_path, feature_path=feature_path)
    return feature_path


def generate_sequential_pairs(images_path, pairs_path, window_size=10):
    images = sorted([p.name for p in images_path.iterdir() if p.is_file()])
    print(f"Found {len(images)} images for sequential matching.")

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, min(i + 1 + window_size, len(images))):
            pairs.append((images[i], images[j]))

    with open(pairs_path, "w") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")


def run_matching(output_path, feature_path, images_path, feature_type="aliked", matching_type="sequential"):
    config = FEATURE_CONFIGS[feature_type]
    matcher_conf = match_features.confs[config["matcher"]]
    match_path = output_path / "matches.h5"
    pairs_path = output_path / "pairs.txt"

    if matching_type == "sequential":
        print(f"Generating sequential pairs to {pairs_path}...")
        generate_sequential_pairs(images_path, pairs_path, window_size=10)
    elif matching_type == "exhaustive":
        print(f"Generating exhaustive pairs to {pairs_path}...")
        pairs_from_exhaustive.main(pairs_path, features=feature_path)
    elif matching_type == "retrieval":
        print(f"Extracting global features for retrieval...")
        global_conf = extract_features.confs["netvlad"]
        global_features_path = output_path / "global_features.h5"
        extract_features.main(global_conf, images_path, feature_path=global_features_path)
        
        print(f"Generating retrieval pairs to {pairs_path}...")
        pairs_from_retrieval.main(global_features_path, pairs_path, num_matched=20)
    else:
        print(f"Unknown matching type: {matching_type}")
        sys.exit(1)

    print(f"Matching features into {match_path} using {config['matcher']}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Matching Device: {device}")
    match_features.main(
        matcher_conf, pairs_path, features=feature_path, matches=match_path
    )
    return match_path, pairs_path


def parse_cameras_txt(cameras_path):
    cameras = {}
    with open(cameras_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = (model, width, height, params)
    return cameras


def update_camera_intrinsics(database_path, cameras_path):
    cameras = parse_cameras_txt(cameras_path)
    if not cameras:
        print("No cameras found in cameras.txt")
        return

    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    for cam_id, (model_name, w, h, params) in cameras.items():
        params_blob = np.array(params, dtype=np.float64).tobytes()
        model_map = {
            "SIMPLE_PINHOLE": 0,
            "PINHOLE": 1,
            "SIMPLE_RADIAL": 2,
            "RADIAL": 3,
            "OPENCV": 4,
        }
        model_id = model_map.get(model_name, 1)

        print(f"Updating Camera {cam_id} to {model_name} ({w}x{h}) in DB...")
        c.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model_id, w, h, params_blob, cam_id),
        )

    conn.commit()
    conn.close()


def run_mapping(
    output_path,
    dataset_path,
    images_path,
    cameras_path,
    feature_path,
    match_path,
    pairs_path,
    camera_model="SIMPLE_RADIAL",
):
    database_path = output_path / "database.db"

    if database_path.exists():
        database_path.unlink()

    print("Creating database...")
    reconstruction.create_empty_db(database_path)

    if cameras_path.exists():
        print(f"Using manual intrinsics from {cameras_path}...")
        reconstruction.import_images(
            images_path,
            database_path,
            camera_mode=reconstruction.pycolmap.CameraMode.SINGLE,
        )
        print("Updating intrinsics from cameras.txt...")
        update_camera_intrinsics(database_path, cameras_path)
    else:
        print(f"Using automatic intrinsics with model {camera_model}...")
        options = {"camera_model": camera_model}
        reconstruction.import_images(
            images_path,
            database_path,
            camera_mode=reconstruction.pycolmap.CameraMode.SINGLE,
            options=options,
        )

    print("Importing features and matches...")
    image_ids = reconstruction.get_image_ids(database_path)
    with reconstruction.pycolmap.Database.open(str(database_path)) as db:
        reconstruction.import_features(image_ids, db, feature_path)
        reconstruction.import_matches(
            image_ids, db, pairs_path, match_path, min_match_score=None
        )

    print("Running geometric verification...")
    reconstruction.estimation_and_geometric_verification(database_path, pairs_path)

    sparse_output = output_path / "sparse"
    sparse_output.mkdir(exist_ok=True)

    print("Running GLOMAP mapper...")

    cmd = [
        "glomap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_path),
        "--output_path",
        str(sparse_output),
    ]
    subprocess.run(cmd, check=True)

    return sparse_output


def export_reconstruction(sparse_path, output_path):
    print(f"Exporting reconstruction from {sparse_path} to {output_path}...")

    # We need to find the specific reconstruction folder (often '0')
    # If sparse_path is 'output/sparse', the actual model might be in 'output/sparse/0'
    # But hloc/glomap might save directly. Let's check if 'points3D.bin' exists directly or in subdir.

    model_path = sparse_path
    if (
        not (model_path / "points3D.bin").exists()
        and (model_path / "0" / "points3D.bin").exists()
    ):
        model_path = model_path / "0"

    if not (model_path / "points3D.bin").exists():
        print(f"Error: Could not find points3D.bin in {sparse_path} or subdirectories.")
        return

    try:
        recon = reconstruction.pycolmap.Reconstruction(model_path)
    except Exception as e:
        print(f"Error loading reconstruction: {e}")
        return

    points = []
    colors = []

    for p3d in recon.points3D.values():
        points.append(p3d.xyz)
        colors.append(p3d.color / 255.0)

    if not points:
        print("Warning: No points in reconstruction to export.")
        return

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(points))
    # pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    ply_path = output_path / "sparse.ply"
    glb_path = output_path / "sparse.glb"

    print(f"Saving {ply_path}...")
    # o3d.io.write_point_cloud(str(ply_path), pcd)
    
    # Use Trimesh for PLY export instead
    pcd_trimesh = trimesh.PointCloud(vertices=points, colors=(np.array(colors) * 255).astype(np.uint8))
    pcd_trimesh.export(str(ply_path))

    print(f"Saving {glb_path}...")
    try:
        # Open3D doesn't support GLB point cloud export well, so we use Trimesh
        # Trimesh expects colors as (n, 4) uint8 usually, or (n, 3)
        colors_uint8 = (np.array(colors) * 255).astype(np.uint8)

        # Create trimesh PointCloud
        pcd_trimesh = trimesh.PointCloud(vertices=points, colors=colors_uint8)

        # Create Scene to hold points and cameras
        scene = trimesh.Scene()
        scene.add_geometry(pcd_trimesh)

        # Add Cameras
        # We need to compute frustums.
        # Size of frustum? Let's infer proper scale from the scene extent.
        if points:
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            scene_scale = np.linalg.norm(bbox_max - bbox_min)
            cam_scale = scene_scale * 0.05  # 5% of scene size
        else:
            cam_scale = 1.0

        frustum_lines = []

        # Canonical frustum in camera frame (looking down +Z or -Z? COLMAP is +Z usually, wait.
        # COLMAP: Camera local frame: +X right, +Y down, +Z forward.
        # But we verify this via the rotation.
        # Let's simple create a pyramid: origin (0,0,0) -> 4 corners at Z=1 (scaled)

        # Corners at distance 1
        w, h = (
            1.0,
            0.75,
        )  # aspect ratio guess, or use actual if we had per-camera dims easily accessible here
        # We could look up camera_id from image and get dims, but for visualization generic is fine.

        corners = (
            np.array([[0, 0, 0], [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1]])
            * cam_scale
        )

        # Edges: 0-1, 0-2, 0-3, 0-4, 1-2, 2-3, 3-4, 4-1
        edges = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

        for image in recon.images.values():
            # Image pose is World-to-Camera (R, t)
            # Accessed via cam_from_world in newer pycolmap

            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation

            # C = -R^T * t
            # R_inv = R^T
            R_wc = R.T
            t_wc = -R_wc @ t

            # Transform frustum to world
            # Point_world = R_wc * Point_cam + t_wc
            transformed_corners = (R_wc @ corners.T).T + t_wc

            # Create lines for this camera
            # We can't batch these easily into one lineset unless we merge valid vertices.
            # Trimesh Path3D can handle disjoint lines.

            # Create a list of line segments
            for start, end in edges:
                frustum_lines.append(
                    [transformed_corners[start], transformed_corners[end]]
                )

        if frustum_lines:
            # Create a Path3D or just flatten lines for a specialized GL lineset?
            # Trimesh load_path with 'entities' is complex.
            # Easiest: Create a Path3D from all segments.
            # Or simply trimesh.load_path(np.array(frustum_lines))

            # Note: trimesh.load_path requires (n, 2, 3) for separate lines?
            # Let's try creating a Path3D directly.
            camera_vis = trimesh.load_path(np.array(frustum_lines))
            camera_vis.colors = np.array(
                [[255, 0, 0, 255]] * len(camera_vis.entities)
            )  # Red
            scene.add_geometry(camera_vis)

        scene.export(str(glb_path))
    except Exception as e:
        print(f"Failed to export GLB with Trimesh: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction Pipeline using HLOC + GLOMAP"
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to dataset containing 'images' folder"
    )
    parser.add_argument("--output", default="output", help="Path to output directory")
    parser.add_argument(
        "--stage",
        choices=["all", "features", "matching", "mapping", "export"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--camera_model",
        default="SIMPLE_RADIAL",
        help="Camera model for auto-intrinsics (e.g. PINHOLE, SIMPLE_RADIAL, OPENCV)",
    )
    parser.add_argument(
        "--feature_type",
        choices=list(FEATURE_CONFIGS.keys()),
        default="aliked",
        help="Feature extractor usage (default: aliked). Use 'sift' for classic robust features, or 'superpoint'/'disk' for deep features.",
    )
    parser.add_argument(
        "--matching_type",
        choices=["sequential", "exhaustive", "retrieval"],
        default="sequential",
        help="Matching strategy (default: sequential). Use 'exhaustive' for small datasets, 'retrieval' for large ones.",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ Error: No GPU detected! This pipeline requires a GPU.")
        print("   Please ensure you have installed CUDA-enabled PyTorch.")
        print("   Try running: pixi install")
        sys.exit(1)

    dataset_path, images_path, cameras_path, output_path = setup_paths(args)

    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")

    feature_path = output_path / "features.h5"
    match_path = output_path / "matches.h5"
    pairs_path = output_path / "pairs.txt"

    if args.stage in ["all", "features"]:
        feature_path = run_feature_extraction(
            images_path, output_path, feature_type=args.feature_type
        )

    if args.stage in ["all", "matching"]:
        if not feature_path.exists():
            print("Features file not found. Run 'features' stage first.")
            sys.exit(1)
        match_path, pairs_path = run_matching(
            output_path,
            feature_path,
            images_path,
            feature_type=args.feature_type,
            matching_type=args.matching_type,
        )

    if args.stage in ["all", "mapping"]:
        if not match_path.exists():
            print("Matches file not found. Run 'matching' stage first.")
            sys.exit(1)
        run_mapping(
            output_path,
            dataset_path,
            images_path,
            cameras_path,
            feature_path,
            match_path,
            pairs_path,
            camera_model=args.camera_model,
        )

    if args.stage in ["all", "export"]:
        sparse_output = output_path / "sparse"
        export_reconstruction(sparse_output, output_path)

        sparse_output = output_path / "sparse"
        export_reconstruction(sparse_output, output_path)


if __name__ == "__main__":
    main()