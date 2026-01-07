#!/bin/bash
set -e

echo "Building Ceres Solver with CUDA support..."

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: This script should be run within a pixi environment."
    exit 1
fi

if [ ! -d "ceres-solver" ]; then
    git clone https://github.com/ceres-solver/ceres-solver.git
fi

cd ceres-solver
echo "Switching to master branch..."
git checkout master
git pull origin master
cd ..

cd ceres-solver

rm -rf build
mkdir build
cd build

# Check if cuDSS is available
if [ -f "$CONDA_PREFIX/include/cudss.h" ]; then
    echo "✅ cuDSS found!"
    USE_CUDSS="ON"
else
    echo "⚠️ cuDSS not found in $CONDA_PREFIX/include, disabling."
    USE_CUDSS="OFF"
fi

cmake .. -GNinja \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA=ON \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DUSE_CUDSS=$USE_CUDSS

ninja
ninja install

echo "✅ Ceres Solver built and installed to $CONDA_PREFIX"
