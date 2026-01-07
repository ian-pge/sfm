#!/bin/bash
set -e

echo "Building COLMAP with CUDA support..."

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: This script should be run within a pixi environment."
    exit 1
fi

# Clone COLMAP (using commit compatible with Glomap)
if [ ! -d "colmap" ]; then
    git clone https://github.com/colmap/colmap.git
    cd colmap
    git checkout b6b7b54eca6078070f73a3f0a084f79c629a6f10
    cd ..
else
    echo "colmap directory exists, using it."
fi

cd colmap

rm -rf build
mkdir build
cd build

cmake .. -GNinja \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    -DGUI_ENABLED=OFF \
    -DOPENGL_ENABLED=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF

ninja
ninja install

echo "✅ COLMAP built and installed to $CONDA_PREFIX"
