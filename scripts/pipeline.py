import argparse
import colorsys
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import h5py
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


def setup_aliked_masking():
    """
    Monkey-patch ALIKED to support masking.
    The mask is expected to be passed in the 'data' dictionary.
    """
    from lightglue.aliked import ALIKED

    # Avoid double patching
    if getattr(ALIKED, "_is_patched_for_masking", False):
        return

    original_forward = ALIKED.forward

    def masked_forward(self, data):
        image = data["image"]
        # Handle grayscale if needed (ALIKED expects RGB usually)
        if image.shape[1] == 1:
            from kornia.color import grayscale_to_rgb
            image = grayscale_to_rgb(image)

        feature_map, score_map = self.extract_dense_map(image)

        # --- MASK INJECTION START ---
        if "mask" in data and data["mask"] is not None:
            mask = data["mask"]
            # mask: Bx1xHxW. score_map: Bx1xH'xW'
            
            # Interpolate mask to match score_map dimensions if needed
            if mask.shape[-2:] != score_map.shape[-2:]:
                mask = torch.nn.functional.interpolate(
                    mask, size=score_map.shape[-2:], mode="nearest"
                )

            # Apply mask: Set scores of masked areas (1) to -infinity
            # Ensure mask is on the same device
            mask = mask.to(score_map.device)
            score_map = score_map.masked_fill(mask > 0.5, float("-inf"))
        # --- MASK INJECTION END ---

        keypoints, kptscores, scoredispersitys = self.dkd(
            score_map, image_size=data.get("image_size")
        )
        descriptors, offsets = self.desc_head(feature_map, keypoints)

        _, _, h, w = image.shape
        wh = torch.tensor([w - 1, h - 1], device=image.device)
        return {
            "keypoints": wh * (torch.stack(keypoints) + 1) / 2.0,
            "descriptors": torch.stack(descriptors),
            "keypoint_scores": torch.stack(kptscores),
        }

    ALIKED.forward = masked_forward
    ALIKED._is_patched_for_masking = True
    print("✅ ALIKED monkey-patched for masking support.")


def run_feature_extraction(images_path, output_path, feature_type="aliked", use_mask=False, dataset_path=None, keypoints_viz=False):
    config = FEATURE_CONFIGS[feature_type]
    feature_conf = extract_features.confs[config["feature"]]
    feature_path = output_path / "features.h5"

    print(f"Extracting features to {feature_path} using {feature_type}...")
    
    # If masking OR visualization is enabled and we are using ALIKED
    if (use_mask or keypoints_viz) and feature_type == "aliked":
        if use_mask:
            print("🎭 Masking enabled. Using custom extraction loop.")
            setup_aliked_masking()
            if dataset_path:
                mask_path = dataset_path / "masks" / "window"
                if not mask_path.exists():
                    print(f"❌ Error: Mask directory not found at {mask_path}")
                    sys.exit(1)
            else:
                print("❌ Error: dataset_path required for masking")
                sys.exit(1)
        else:
            print("🎨 Visualization enabled (No Mask). Using custom extraction loop.")
            mask_path = None # No mask

        # Inject viz flag into conf
        feature_conf["keypoints_viz"] = keypoints_viz
            
        custom_extract_features(feature_conf, images_path, mask_path, feature_path)
    else:
        # Standard HLOC extraction
        extract_features.main(feature_conf, images_path, feature_path=feature_path)
        
    return feature_path


def custom_extract_features(conf, image_dir, mask_dir, feature_path):
    """
    Custom version of hloc.extract_features.main to include masks.
    """
    import h5py
    import torch
    from hloc import extractors, logger
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.io import list_h5_names, read_image
    from hloc.extract_features import ImageDataset, resize_image
    from tqdm import tqdm
    import cv2  # Ensure cv2 is available for mask reading

    logger.info(f"Extracting features with masks from {mask_dir}...")

    # We reuse hloc's ImageDataset for images, but we'll manually load masks in the loop
    dataset = ImageDataset(image_dir, conf["preprocessing"])
    
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    
    if len(dataset.names) == 0:
        logger.info("Skipping the extraction (all found in output).")
        return feature_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    # Use a simple loader
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        
        # Load Mask
        current_mask_path = None
        mask = None
        
        if mask_dir:
            current_mask_path = mask_dir / name
            
            # If exact match doesn't exist, try replacing extension with common ones
            if not current_mask_path.exists():
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                    test_path = mask_dir / (Path(name).stem + ext)
                    if test_path.exists():
                        current_mask_path = test_path
                        break
            
            if current_mask_path and current_mask_path.exists():
                # Read mask
                # Masks are often 1-channel or 3-channel. We want 1-channel binary-like.
                mask_np = cv2.imread(str(current_mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_np is not None:
                    # Resize mask to match original image size (data['original_size'])
                    # data['image'] is already resized by ImageDataset potentially.
                    # data['original_size'] is the size BEFORE preprocessing resize.
                    
                    # However, the 'image' tensor passed to the model has been resized.
                    # We should match the 'image' tensor size.
                    
                    image_tensor_shape = data["image"].shape[-2:] # H, W
                    mask_resized = cv2.resize(mask_np, (image_tensor_shape[1], image_tensor_shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Normalize to 0-1 and convert to tensor
                    # 255 = mask, 0 = background?
                    # User said "masques des vitres de SAM". Normally SAM outputs binary masks.
                    # We accept any non-zero as mask.
                    mask_tensor = torch.from_numpy(mask_resized).float() / 255.0
                    mask_tensor = (mask_tensor > 0.5).float() # Binary
                    
                    # Add to batch (B=1) and channel (C=1)
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Add to input
                    # Note: data['image'] is a batch from DataLoader, so it's [B, C, H, W]
                    # We need to construct the input dict for the model
                    image_input = data["image"].to(device, non_blocking=True)
                    
                    pred_input = {"image": image_input, "mask": mask_tensor}
                    
                    # Run Model
                    with torch.no_grad():
                        pred = model(pred_input)
                else:
                    if mask_dir:
                        logger.warning(f"Could not read mask: {current_mask_path}")
                    with torch.no_grad():
                        pred = model({"image": data["image"].to(device, non_blocking=True)})
        else:
            if mask_dir:
                logger.warning(f"Mask not found for {name} at {mask_dir}")
            with torch.no_grad():
                pred = model({"image": data["image"].to(device, non_blocking=True)})

        # --- End Custom Logic, continue with HLOC saving ---
        
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        # --- Visualization Logic ---
        if conf.get("keypoints_viz", False):
            viz_dir = feature_path.parent / "keypoints_viz"
            viz_dir.mkdir(exist_ok=True, parents=True)
            
            # Prepare image
            # data["image"] is [B, C, H, W] tensor (normalized 0-1)
            img_tensor = data["image"][0].cpu() # C, H, W
            if img_tensor.shape[0] == 3:
                img_np = img_tensor.permute(1, 2, 0).numpy() * 255 # H, W, 3
                img_np = img_np.astype(np.uint8)
                img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_np = img_tensor[0].numpy() * 255
                img_np = img_np.astype(np.uint8)
                img_vis = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

            # Draw Mask Overlay (if present)
            if mask is not None:
                # Mask tensor is on GPU, detach
                # Mask tensor we created was [1, 1, H, W]
                m_vis = mask_tensor[0, 0].cpu().numpy() # H, W
                m_vis = cv2.resize(m_vis, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask (e.g. Red)
                overlay = img_vis.copy()
                overlay[m_vis > 0.5] = [0, 0, 255] # Red in BGR
                cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0, img_vis)
            
            # Draw Keypoints
            if "keypoints" in pred:
                kpts = pred["keypoints"]
                for x, y in kpts:
                    # Scale keypoints back to current image size for visualization?
                    # pred["keypoints"] are scaled to ORIGINAL size.
                    # But img_vis is the PREPROCESSED size (resized).
                    # Wait, data["image"] is resized.
                    # We need keypoints in the coordinate system of img_vis.
                    
                    # Inverse scale from pred keys logic
                    # pred["keypoints"] = (raw + 0.5) * scales - 0.5
                    # raw = (pred + 0.5) / scales - 0.5
                    
                    x_vis = (x + 0.5) / scales[0] - 0.5
                    y_vis = (y + 0.5) / scales[1] - 0.5
                    
                    cv2.circle(img_vis, (int(x_vis), int(y_vis)), 2, (0, 255, 0), -1) # Green
            
            # Save
            # Maintain original filename but png/jpg
            save_name = Path(name).stem + ".jpg" # force jpg for vis
            cv2.imwrite(str(viz_dir / save_name), img_vis)
        # ---------------------------

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error("Out of disk space.")
                    del grp, fd[name]
                raise error
        del pred

    logger.info("Finished exporting features (custom masked).")
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
    matching_type="hybrid",
    window_size=10,
    num_matched=30,
):
    config = FEATURE_CONFIGS[feature_type]
    matcher_conf = match_features.confs[config["matcher"]]
    match_path = output_path / "matches.h5"
    pairs_path = output_path / "pairs.txt"

    if matching_type == "sequential":
        print(f"Generating sequential pairs to {pairs_path}...")
        generate_sequential_pairs(images_path, pairs_path, window_size=window_size)
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
        pairs_from_retrieval.main(
            global_features_path, pairs_path, num_matched=num_matched
        )
    elif matching_type == "hybrid":
        print("Running hybrid matching (Sequential + Retrieval)...")
        pairs_seq = output_path / "pairs_sequential.txt"
        generate_sequential_pairs(images_path, pairs_seq, window_size=window_size)

        print("Generating retrieval pairs...")
        global_conf = extract_features.confs["netvlad"]
        global_features_path = output_path / "global_features.h5"
        extract_features.main(
            global_conf, images_path, feature_path=global_features_path
        )

        import bucket_matcher  # Dynamic import unless I put it top-level

        pairs_ret = output_path / "pairs_retrieval.txt"

        # Check if we have labeled frames
        images_all = list(images_path.iterdir())
        suffixes = ["_fl", "_fr", "_bl", "_br"]
        has_labels = any(p.stem.endswith(tuple(suffixes)) for p in images_all)

        if has_labels:
            print(
                f"ℹ️  Labeled frames detected. Using FILTERED bucket matching (NetVLAD + Buckets)."
            )
            bucket_matcher.generate_bucketed_retrieval_pairs(
                global_features_path, pairs_ret, num_matched=num_matched
            )
        else:
            print(f"ℹ️  No labeled frames detected. Using classical Global Retrieval.")
            pairs_from_retrieval.main(
                global_features_path, pairs_ret, num_matched=num_matched
            )

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
            "--skip_pruning",
            "1",
            "--Thresholds.min_inlier_num",
            "10",
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


def analyze_failures(database_path, registered_image_ids):
    if not database_path.exists():
        return

    print("\n" + "=" * 60)
    print("📉 FAILURE ANALYSIS")
    print("=" * 60)

    try:
        conn = sqlite3.connect(database_path)
        c = conn.cursor()

        # Get all images
        c.execute("SELECT image_id, name FROM images")
        all_images = {row[0]: row[1] for row in c.fetchall()}

        # Identify deleted (unregistered) images
        deleted_ids = set(all_images.keys()) - set(registered_image_ids)

        if not deleted_ids:
            print("🎉 No frames were deleted! (100% registration)")
            conn.close()
            return

        print(f"Found {len(deleted_ids)} deleted frames. Analyzing reasons...\n")
        print(f"{'Image Name':<50} | {'KPs':<8} | {'Pairs':<6} | {'Reason'}")
        print("-" * 90)

        # Count verified pairs for each image
        # pair_id = (image_id1 << 31) + image_id2
        image_pair_counts = {id: 0 for id in deleted_ids}
        c.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows >= 15"
        )  # Filter for usable matches
        for (pair_id,) in c.fetchall():
            id2 = pair_id & 2147483647
            id1 = pair_id >> 31

            if id1 in image_pair_counts:
                image_pair_counts[id1] += 1
            if id2 in image_pair_counts:
                image_pair_counts[id2] += 1

        for image_id in sorted(deleted_ids):
            name = all_images[image_id]
            # Use just the basename for cleaner output if name is a path
            # name = Path(name).name
            # (User requested "original image name", assuming full relative path or name in DB is desired)

            # Get Keypoints count
            c.execute("SELECT rows FROM keypoints WHERE image_id=?", (image_id,))
            res = c.fetchone()
            num_kps = res[0] if res else 0

            num_pairs = image_pair_counts.get(image_id, 0)

            reason = "Unknown"
            if num_kps < 100:
                reason = "Low Keypoints (<100)"
            elif num_pairs == 0:
                reason = "No Valid Pairs"
            elif num_pairs < 3:
                reason = "Few Matches"
            else:
                reason = "Structure/Geometry Fail"

            print(f"{name:<50} | {num_kps:<8} | {num_pairs:<6} | {reason}")

        conn.close()
    except Exception as e:
        print(f"Error analyzing database: {e}")
    print("-" * 90)


def print_summary(
    start_time,
    output_path,
    num_images_initial=0,
    feature_type="aliked",
    mapper="glomap",
):
    duration = time.time() - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    # Gather stats
    num_sparse_points = 0
    num_registered_images = 0
    num_db_images = 0
    registered_image_ids = set()

    # 1. Check sparse model (final result)
    # Check both direct 'sparse' or 'sparse/0'
    model_path = output_path / "sparse"
    if (
        not (model_path / "points3D.bin").exists()
        and (model_path / "0" / "points3D.bin").exists()
    ):
        model_path = model_path / "0"

    if (model_path / "points3D.bin").exists():
        try:
            recon = reconstruction.pycolmap.Reconstruction(model_path)
            num_sparse_points = len(recon.points3D)
            num_registered_images = len(recon.images)
            registered_image_ids = set(recon.images.keys())
        except:
            pass

    # 2. Check Database (input size)
    db_path = output_path / "database.db"
    if db_path.exists():
        try:
            cx = sqlite3.connect(db_path)
            c = cx.cursor()
            c.execute("SELECT count(*) FROM images")
            num_db_images = c.fetchone()[0]
            cx.close()
        except:
            pass

    print("\n" + "=" * 60)
    print("✨ RECONSTRUCTION SUMMARY ✨")
    print("=" * 60)
    print(f"⏱️  Total Time:       {minutes:02d}m {seconds:02d}s")
    print(f"📂 Output Directory: {output_path}")
    print("-" * 60)
    print(f"📸 Images Processed: {num_db_images} (Init: {num_images_initial})")
    print(f"✅ Registered:       {num_registered_images} / {num_db_images}")
    print(f"🌟 3D Points:        {num_sparse_points:,}")
    print("-" * 60)
    print(f"🧩 Feature Type:     {feature_type}")
    print(f"🗺️  Mapper:           {mapper}")
    print("=" * 60)
    if num_registered_images == 0:
        print("❌ Warning: No images were registered. Reconstruction failed.")
    elif num_registered_images < num_db_images * 0.5:
        print("⚠️  Warning: Less than 50% of images registered.")
    else:
        print("🚀 Success! Reconstruction looks good.")
    print("\n")

    # Run Failure Analysis
    if num_registered_images < num_db_images:
        analyze_failures(output_path / "database.db", registered_image_ids)


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
        choices=["sequential", "exhaustive", "retrieval", "hybrid"],
        default="hybrid",
        help="Matching strategy (default: hybrid). Use 'exhaustive' for small datasets, 'retrieval' for large ones.",
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
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Alias for --matching_type hybrid. Adds global retrieval loop closure to sequential matching.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of sequential images to match (default: 3).",
    )
    parser.add_argument(
        "--retrieval_num",
        type=int,
        default=20,
        help="Number of candidates for Global Retrieval (default: 30).",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Use masks for ALIKED feature extraction (ignores areas where mask > 0.5)",
    )
    parser.add_argument(
        "--keypoints_viz",
        action="store_true",
        help="Visualize keypoints and masks (saves to output/keypoints_viz/)",
    )

    args = parser.parse_args()

    start_time = time.time()

    # Handle the --hybrid alias
    if args.hybrid:
        args.matching_type = "hybrid"

    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ Error: No GPU detected! This pipeline requires a GPU.")
        print("   Please ensure you have installed CUDA-enabled PyTorch.")
        print("   Try running: pixi install")
        sys.exit(1)

    dataset_path, images_path, cameras_path, output_path = setup_paths(args)

    # Count initial images
    initial_images = sorted([p for p in images_path.iterdir() if p.is_file()])
    num_initial_images = len(initial_images)

    print("\n" + "=" * 60)
    print("🚀  S F M   P I P E L I N E   S T A R T")
    print("=" * 60)
    print(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Dataset:    {dataset_path}")
    print(f"📁 Output:     {output_path}")
    print(f"📸 Images:     {num_initial_images}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU:        {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU:        Not Detected (Expect slow performance)")

    print("-" * 60)
    print("⚙️  Configuration:")
    print(f"   • Feature Type:   {args.feature_type}")
    print(f"   • Matching:       {args.matching_type}")
    print(f"   • Mapper:         {args.mapper}")
    print(f"   • Camera Model:   {args.camera_model}")
    print(f"   • Window Size:    {args.window_size}")
    print(f"   • Retrieval Num:  {args.retrieval_num}")
    
    flags = []
    if args.mask: flags.append("🎭 Masking")
    if args.keypoints_viz: flags.append("🎨 Visualization")
    if args.undistort: flags.append("📐 Undistort")
    if args.normalize: flags.append("🌐 Normalize")
    
    print(f"   • Active Flags:   {', '.join(flags) if flags else 'None'}")
    print("=" * 60 + "\n")

    feature_path = output_path / "features.h5"
    match_path = output_path / "matches.h5"
    pairs_path = output_path / "pairs.txt"

    if args.stage in ["all", "features"]:
        feature_path = run_feature_extraction(
            images_path,
            output_path,
            feature_type=args.feature_type,
            use_mask=args.mask,
            dataset_path=dataset_path,
            keypoints_viz=args.keypoints_viz,
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
            window_size=args.window_size,
            num_matched=args.retrieval_num,
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

        # Export immediately after reconstruction/normalization (before slow undistortion)
        print("Exporting reconstruction immediately...")
        export_reconstruction(sparse_output, output_path)

        if args.undistort:
            run_undistortion(sparse_output, images_path, output_path)

    if args.stage == "export":
        sparse_output = output_path / "sparse"
        # If we normalize, we should probably make sure export sees the right thing.
        # But since normalization overwrites the sparse model, looking at output/sparse is correct.
        export_reconstruction(sparse_output, output_path)

    print_summary(
        start_time,
        output_path,
        num_images_initial=num_initial_images,
        feature_type=args.feature_type,
        mapper=args.mapper,
    )


if __name__ == "__main__":
    main()
