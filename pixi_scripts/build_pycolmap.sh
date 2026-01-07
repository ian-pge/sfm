#!/bin/bash
set -e

# Build PyCOLMAP from the colmap repository (integrated bindings)
cd colmap

# Ensure dependencies are installed
# scikit-build-core is now in pixi.toml

# Build and install
# We use existing colmap installation found in the environment
# Use CMAKE_ARGS to ensure CUDA is enabled if applicable, though pycolmap setup should detect it.
# We also set CUDAToolkit_ROOT to help CMake find CUDA.
export CUDAToolkit_ROOT=$CONDA_PREFIX
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCOLMAP_CUDA_ENABLED=ON -DGENERATE_STUBS=OFF" pip install . --no-build-isolation -v
