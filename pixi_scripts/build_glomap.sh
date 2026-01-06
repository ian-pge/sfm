#!/bin/bash
set -e

echo "Building GLOMAP..."

# We assume running inside pixi environment, so cmake, ninja, cxx-compiler are available.
# We also assume dependencies (colmap, ceres, eigen, etc.) are installed in the environment.

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: This script should be run within a pixi environment (pixi run build-glomap)."
    exit 1
fi

# Clone if not exists
if [ ! -d "glomap" ]; then
    git clone https://github.com/colmap/glomap.git
else
    echo "glomap directory exists, pulling updates..."
    cd glomap && git pull && cd ..
fi

cd glomap

# Clean build
rm -rf build
mkdir build
cd build

# Configure
# We point CMAKE_INSTALL_PREFIX to the pixi environment
export CUDAToolkit_ROOT="$CONDA_PREFIX"
cmake .. -GNinja \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)"

# Build and Install
ninja
ninja install

echo "✅ GLOMAP built and installed to $CONDA_PREFIX"
