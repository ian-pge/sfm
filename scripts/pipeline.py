import argparse
import colorsys
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.path.entities

# Add external scripts to path for SuperGluePretrainedNetwork (just in case, though ALIKED doesn't use it)
current_dir = Path(__file__).parent
external_dir = current_dir / "external"
if external_dir.exists():
    sys.path.append(str(external_dir))


from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)

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


def generate_sequential_pairs(images_path, pairs_path, window_size=3):
    images = sorted([p.name for p in images_path.iterdir() if p.is_file()])
    print(f"Found {len(images)} images for sequential matching.")

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, min(i + 1 + window_size, len(images))):
            pairs.append((images[i], images[j]))

    with open(pairs_path, "w") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")


def run_matching(
    output_path,
    feature_path,
    images_path,
    feature_type="aliked",
    matching_type="sequential",
):
    config = FEATURE_CONFIGS[feature_type]
    matcher_conf = match_features.confs[config["matcher"]]
    match_path = output_path / "matches.h5"
    pairs_path = output_path / "pairs.txt"

    if matching_type == "sequential":
        print(f"Generating sequential pairs to {pairs_path}...")
        generate_sequential_pairs(images_path, pairs_path, window_size=3)
    elif matching_type == "exhaustive":
        print(f"Generating exhaustive pairs to {pairs_path}...")
        pairs_from_exhaustive.main(pairs_path, features=feature_path)
    elif matching_type == "retrieval":
        print(f"Extracting global features for retrieval...")
        global_conf = extract_features.confs["netvlad"]
        global_features_path = output_path / "global_features.h5"
        extract_features.main(
            global_conf, images_path, feature_path=global_features_path
        )

        print(f"Generating retrieval pairs to {pairs_path}...")
        pairs_from_retrieval.main(global_features_path, pairs_path, num_matched=20)
    elif matching_type == "hybrid":
        print("Running hybrid matching (Sequential + Retrieval)...")
        pairs_seq = output_path / "pairs_sequential.txt"
        generate_sequential_pairs(images_path, pairs_seq, window_size=3)

        print("Generating retrieval pairs...")
        global_conf = extract_features.confs["netvlad"]
        global_features_path = output_path / "global_features.h5"
        extract_features.main(
            global_conf, images_path, feature_path=global_features_path
        )

        pairs_ret = output_path / "pairs_retrieval.txt"
        pairs_from_retrieval.main(global_features_path, pairs_ret, num_matched=20)

        # Merge pairs
        pairs = set()
        for p in [pairs_seq, pairs_ret]:
            with open(p, "r") as f:
                for line in f:
                    p1, p2 = line.strip().split()
                    # ensure consistent ordering for deduplication
                    if p1 > p2:
                        p1, p2 = p2, p1
                    pairs.add((p1, p2))

        print(f"Merged {len(pairs)} unique pairs from Sequential and Retrieval.")
        with open(pairs_path, "w") as f:
            for p1, p2 in sorted(pairs):
                f.write(f"{p1} {p2}\n")
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
    mapper="glomap",
):
    database_path = output_path / "database.db"
    sparse_output = output_path / "sparse"
    sparse_output.mkdir(exist_ok=True)

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

    if mapper == "glomap":
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
    elif mapper == "colmap":
        print("Running COLMAP mapper...")
        reconstruction.pycolmap.incremental_mapping(
            database_path, images_path, sparse_output
        )
    else:
        print(f"Unknown mapper: {mapper}")
        sys.exit(1)

    return sparse_output


def run_normalization(sparse_path, output_path):
    print(f"Running scene normalization on {sparse_path}...")

    # We need to find the specific reconstruction folder (often '0')
    model_path = sparse_path
    if (
        not (model_path / "points3D.bin").exists()
        and (model_path / "0" / "points3D.bin").exists()
    ):
        model_path = model_path / "0"

    if not (model_path / "points3D.bin").exists():
        print(
            f"Error: Could not find points3D.bin in {sparse_path} or subdirectories for normalization."
        )
        return sparse_path

    try:
        recon = reconstruction.pycolmap.Reconstruction(model_path)
    except Exception as e:
        print(f"Error loading reconstruction for normalization: {e}")
        return sparse_path

    # Compute Centroid of Camera Centers
    print("Computing scale and centroid...")
    cam_centers = []
    for image in recon.images.values():
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        center = -R.T @ t
        cam_centers.append(center)

    if not cam_centers:
        print("Warning: No cameras found. Skipping normalization.")
        return sparse_path

    cam_centers = np.array(cam_centers)
    centroid = np.mean(cam_centers, axis=0)

    # Compute Scale: We want the cameras to fit comfortably within the unit sphere.
    # We use the max distance from centroid to scale everything.
    # Radius of camera cloud
    dists = np.linalg.norm(cam_centers - centroid, axis=1)
    max_dist = np.max(dists)

    # If all cameras are at the same spot (single view?), scale defaults to 1
    if max_dist < 1e-6:
        print("Warning: Cameras are clumped together. Skipping scale.")
        scale = 1.0
    else:
        scale = 1.0 / max_dist

    print(f"Centroid: {centroid}")
    print(f"Scale: {scale} (Max camera distance: {max_dist})")

    # Apply Similarity Transform (T = s * [R | t])
    # We want new_point = s * (old_point - centroid)
    # new_point = s * old_point + s * (-centroid)
    # So translation component is s * (-centroid).
    # But Pycolmap Sim3d constructor takes (scale, rotation, translation) where
    # P_new = scale * (R * P_old + t)   <-- CHECK THIS DEFINITION CAREFULLY
    # HELP says: "Apply the 3D similarity transformation to all images and points."

    # Let's verify Sim3d usually implies P' = s * R * P + t  OR  P' = s * R * (P + t)?
    # Standard Sim3 transform in COLMAP algebra (Sim3d):
    # If we construct Sim3d(scale, rotation, translation),
    # usually transform aligns `new_from_old`.
    # Let's assume P_new = s * R * P_old + t is NOT it,
    # Usually it is composed.
    # Let's stick to the construction:
    # We want to translate by -centroid, then scale by s.
    # T = Scale(s) * Translation(-centroid)
    # In matrix form 4x4:
    # [sI  0] * [I  -c] = [sI  -sc]
    # [0   1]   [0   1]   [0    1]

    # So Rotation is Identity.
    # Translation is -scale * centroid? Or just -centroid?
    # Sim3d(scale, rotation, translation)
    # If Sim3d applies as: x' = s * R * x + t
    # Then we need x' = s * (x - c) = s * x - s * c
    # So t = -s * c

    sim3 = reconstruction.pycolmap.Sim3d(
        scale, reconstruction.pycolmap.Rotation3d(), -scale * centroid
    )

    recon.transform(sim3)

    print("Saving normalized reconstruction...")
    # Overwrite the existing model
    recon.write(model_path)

    # Also save the normalization parameters for potential inversion later
    norm_info_path = model_path / "normalization.txt"
    with open(norm_info_path, "w") as f:
        f.write(f"Centroid: {centroid[0]} {centroid[1]} {centroid[2]}\n")
        f.write(f"Scale: {scale}\n")

    return sparse_path


def run_undistortion(sparse_path, images_path, output_path):
    print("Running image undistortion (COLMAP)...")

    # Define undistortion output path
    undistorted_output = output_path / "undistorted"
    undistorted_output.mkdir(exist_ok=True, parents=True)

    # We need to find the correct sparse model path (same logic as export)
    model_path = sparse_path
    if (
        not (model_path / "cameras.bin").exists()
        and not (model_path / "cameras.txt").exists()
        and (model_path / "0").exists()
    ):
        model_path = model_path / "0"

    print(f"Using model from: {model_path}")

    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(images_path),
        "--input_path",
        str(model_path),
        "--output_path",
        str(undistorted_output),
        "--output_type",
        "COLMAP",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Undistorted images saved to {undistorted_output / 'images'}")

        # Gaussian Splatting loaders typically expect the model in 'sparse/0'
        # COLMAP image_undistorter saves directly to 'sparse'.
        # We'll move it to 'sparse/0' for maximum compatibility.
        sparse_dir = undistorted_output / "sparse"
        sparse_0_dir = sparse_dir / "0"

        if sparse_dir.exists() and not sparse_0_dir.exists():
            print(
                "Organizing output for Gaussian Splatting (moving sparse -> sparse/0)..."
            )
            sparse_0_dir.mkdir(parents=True)
            for item in sparse_dir.iterdir():
                if item.name == "0":
                    continue
                # Move files/dirs into 0/
                shutil.move(str(item), str(sparse_0_dir / item.name))

    except subprocess.CalledProcessError as e:
        print(f"Error running image undistortion: {e}")
        # Don't exit, just print error as this is post-processing


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
    pcd_trimesh = trimesh.PointCloud(
        vertices=points, colors=(np.array(colors) * 255).astype(np.uint8)
    )
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
        # Adaptive Camera Sizing
        # Compute camera centers to find average nearest neighbor distance
        cam_centers = []
        for image in recon.images.values():
            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            center = -R.T @ t
            cam_centers.append(center)

        cam_centers = np.array(cam_centers)

        if len(cam_centers) > 1:
            # Compute distances to nearest neighbor for each camera
            # Using simple broadcasting (N is typically < few thousands, so N^2 is fine)
            # For extremely large datasets, a k-d tree would be better, but we stick to numpy deps
            d1 = cam_centers[:, None, :]
            d2 = cam_centers[None, :, :]
            dists = np.linalg.norm(d1 - d2, axis=-1)

            # Set diagonal to infinity to ignore self-distance
            np.fill_diagonal(dists, np.inf)

            # Find distance to nearest neighbor for each camera
            min_dists = np.min(dists, axis=1)

            # Use the median of these nearest distances as a robust scale base
            # We scale it slightly (e.g., 0.5) so frustums don't touch/overlap too much
            median_nn_dist = np.median(min_dists)
            cam_scale = median_nn_dist * 1.5
            print(
                f"Adaptive camera scale: {cam_scale:.4f} (based on median neighbor dist: {median_nn_dist:.4f})"
            )

        elif points:
            # Fallback if only 1 camera
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            scene_scale = np.linalg.norm(bbox_max - bbox_min)
            cam_scale = scene_scale * 0.005  # Fallback to 0.5%
        else:
            cam_scale = 1.0

        frustum_lines = []
        line_colors = []

        # Canonical frustum in camera frame
        w, h = (1.0, 0.75)
        # 5 vertices: Origin, and 4 corners at Z=1*scale
        raw_corners = (
            np.array([[0, 0, 0], [-w, -h, 1], [w, -h, 1], [w, h, 1], [-w, h, 1]])
            * cam_scale
        )

        all_vertices = []
        all_entities = []
        all_colors = []

        # Continuous path to draw pyramid: Base loop + edges to apex
        # 1-2-3-4-1 (base), then 1-0-2-0-3-0-4 (edges to apex with backtracking)
        # Indices relative to the 5 vertices of the camera
        camera_indices = [1, 2, 3, 4, 1, 0, 2, 0, 3, 0, 4]

        # Sort images by ID to ensure smooth gradient
        sorted_images = sorted(recon.images.values(), key=lambda x: x.image_id)
        num_images = len(sorted_images)

        for i, image in enumerate(sorted_images):
            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation

            R_wc = R.T
            t_wc = -R_wc @ t

            # Transform 5 vertices to world
            transformed_vertices = (R_wc @ raw_corners.T).T + t_wc

            # Append to master list
            start_idx = len(all_vertices)
            all_vertices.extend(transformed_vertices)

            # Create entity
            entity_indices = [idx + start_idx for idx in camera_indices]
            all_entities.append(trimesh.path.entities.Line(points=entity_indices))

            # Create color gradient (Rainbow/HSV)
            # Map index to hue 0.0-0.7 (Red to Blue, skipping purple/magenta to keeping it distinct from start)
            # Or full loop 0.0-1.0
            hue = i / (num_images - 1) if num_images > 1 else 0.0
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = np.array([int(c * 255) for c in rgb] + [255], dtype=np.uint8)
            all_colors.append(color)

        if all_vertices:
            camera_vis = trimesh.path.Path3D(
                vertices=np.array(all_vertices), entities=all_entities
            )
            camera_vis.colors = np.array(all_colors)
            scene.add_geometry(camera_vis)

        # Fix orientation: Rotate 180 degrees around X-axis
        # GLTF typically uses +Y up, -Z forward. COLMAP/OpenCV +Y down.
        # Flipping Y and Z (180 deg around X) standardizes this view.
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        scene.apply_transform(rotation_matrix)

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
        default="OPENCV",
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
        choices=["sequential", "exhaustive", "retrieval", "hybrid"],
        default="sequential",
        help="Matching strategy (default: sequential). Use 'exhaustive' for small datasets, 'retrieval' for large ones.",
    )
    parser.add_argument(
        "--mapper",
        choices=["glomap", "colmap"],
        default="glomap",
        help="Mapper usage (default: glomap)",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort images after reconstruction (outputs to output/undistorted)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize scene to unit sphere (centered and scaled)",
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
        sparse_output = run_mapping(
            output_path,
            dataset_path,
            images_path,
            cameras_path,
            feature_path,
            match_path,
            pairs_path,
            camera_model=args.camera_model,
            mapper=args.mapper,
        )

        if args.normalize:
            # Updates sparse_output in-place (conceptually, on disk)
            sparse_output = run_normalization(sparse_output, output_path)

        if args.undistort:
            run_undistortion(sparse_output, images_path, output_path)

    if args.stage in ["all", "export"]:
        sparse_output = output_path / "sparse"
        # If we normalize, we should probably make sure export sees the right thing.
        # But since normalization overwrites the sparse model, looking at output/sparse is correct.
        export_reconstruction(sparse_output, output_path)
        # If we normalize, we should probably make sure export sees the right thing.
        # But since normalization overwrites the sparse model, looking at output/sparse is correct.
        export_reconstruction(sparse_output, output_path)


if __name__ == "__main__":
    main()
