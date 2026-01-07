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

echo "Applying patches for system COLMAP linkage..."
# Patch 1: Make COLMAP visible globally in FindDependencies.cmake
if ! grep -q "find_package(colmap REQUIRED)" glomap/cmake/FindDependencies.cmake; then
    sed -i '/find_package(OpenMP REQUIRED COMPONENTS C CXX)/a \
\
if(NOT FETCH_COLMAP)\
    find_package(colmap REQUIRED)\
endif()' glomap/cmake/FindDependencies.cmake
fi

# Patch 2: Prevent duplicate finding in thirdparty/CMakeLists.txt
sed -i 's/find_package(COLMAP REQUIRED)/if(NOT TARGET colmap::colmap)\n    find_package(COLMAP REQUIRED)\nendif()/' glomap/thirdparty/CMakeLists.txt

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
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    -DFETCH_COLMAP=OFF

# Build and Install
ninja
ninja install

echo "✅ GLOMAP built and installed to $CONDA_PREFIX"
