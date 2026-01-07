# Reconstruction Pipeline: HLOC + GLOMAP

This pipeline automates the creation of a sparse 3D reconstruction from a set of images using modern deep learning features (ALIKED + LightGlue by default) and a global structure-from-motion mapper (GLOMAP).

> **Note**: This pipeline uses everything compiled from source with CUDA support for optimal performance.


## Overview

The pipeline performs the following steps:
1.  **Feature Extraction**: Extracts keypoints using **ALIKED** (default), SuperPoint, DISK, or SIFT.
2.  **Matching**: Matches keypoints between image pairs using **LightGlue** (or Adalam for SIFT). Supports Sequential, Exhaustive, or Retrieval-based matching strategies.
3.  **Database Creation**: Imports intrinsics and matches into a COLMAP database (`database.db`).
4.  **Geometric Verification**: Verifies matches to filter outliers (Crucial for GLOMAP).
5.  **Reconstruction**: estimating camera poses and 3D points using **GLOMAP**.

## data Preparation

**Important**: Ensure your images are properly ordered (e.g., sequential frame numbers). The pipeline relies on correct image ordering for accurate matching and reconstruction.

Your dataset should look like this:

```
/path/to/dataset/
├── images/
│   ├── frame_00001.png
│   ├── frame_00002.png
│   └── ...
└── cameras.txt
```

### `cameras.txt`
This file defines the camera intrinsics in standard COLMAP format. If provided, the pipeline will use it (Manual Mode).
If missing, the pipeline will infer intrinsics automatically using the specified `--camera_model`.

Example for a single PINHOLE camera (optional):
```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 1920 1080 1266.9 1267.62 960 540
```

## Usage

Run the pipeline using `scripts/pipeline.py`:

```bash
# Default sequential matching (good for video)
python3 scripts/pipeline.py \
    --dataset /path/to/dataset \
    --output /path/to/output_folder \
    --camera_model PINHOLE

# Exhaustive matching (good for small, unordered datasets)
python3 scripts/pipeline.py \
    --dataset /path/to/dataset \
    --matching_type exhaustive

# Retrieval matching (good for large, unordered datasets)
python3 scripts/pipeline.py \
    --dataset /path/to/dataset \
    --matching_type retrieval
```

### Arguments
- `--dataset`: Path to the dataset root (containing `images/`).
- `--output`: Directory where results will be saved.
- `--camera_model`: Camera model for auto-intrinsics (e.g., `PINHOLE`, `SIMPLE_RADIAL`, `OPENCV`). Defaults to `SIMPLE_RADIAL`. Ignored if `cameras.txt` is present.
- `--matching_type`: Strategy for pairing images:
  - `sequential` (default): Matches consecutive frames. Good for video.
  - `exhaustive`: Matches every image with every other image. Good for small datasets.
  - `retrieval`: Uses global descriptors (NetVLAD) to find overlapping pairs. Good for large datasets.
- `--feature_type`: Local feature extractor: `aliked` (default), `superpoint`, `disk`, `sift`.
- `--stage`: (Optional) Run specific stage: `features`, `matching`, `mapping`, `export` or `all` (default).

## Output

The results will be saved in the `--output` directory:
- `features.h5`: Extracted features.
- `matches.h5`: Match data.
- `pairs.txt`: List of image pairs.
- `database.db`: SQLite database with all data.
- **`sparse.ply` / `sparse.glb`**: Exported point clouds for visualization.
- **`sparse/0/`**: The final sparse reconstruction files:
    - `cameras.bin`
    - `images.bin`
    - `points3D.bin`
