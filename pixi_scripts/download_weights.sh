#!/bin/bash
set -e

# Define cache directory matching the python code expectations or standard torch cache
CACHE_DIR="/home/zed/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"

echo "Downloading weights to $CACHE_DIR..."

# MegaLoc
if [ ! -f "$CACHE_DIR/megaloc.torch" ]; then
    wget -q -O "$CACHE_DIR/megaloc.torch" \
      "https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch" && echo "✅ MegaLoc Downloaded"
else
    echo "✅ MegaLoc already exists"
fi

# ALIKED
if [ ! -f "$CACHE_DIR/aliked-n16.pth" ]; then
    wget -q -O "$CACHE_DIR/aliked-n16.pth" \
      "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth" && echo "✅ ALIKED Downloaded"
else
    echo "✅ ALIKED already exists"
fi

# LightGlue
if [ ! -f "$CACHE_DIR/aliked_lightglue_v0-1_arxiv.pth" ]; then
    wget -q -O "$CACHE_DIR/aliked_lightglue_v0-1_arxiv.pth" \
      "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth" && echo "✅ LightGlue Weights Downloaded"
else
    echo "✅ LightGlue Weights already exist"
fi

# SuperGluePretrainedNetwork (cloned repo)
mkdir -p scripts/external
if [ ! -d "scripts/external/SuperGluePretrainedNetwork" ]; then
    echo "Cloning SuperGluePretrainedNetwork..."
    git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git scripts/external/SuperGluePretrainedNetwork
else
    echo "✅ SuperGluePretrainedNetwork already exists"
fi
