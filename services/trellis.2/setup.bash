#!/bin/bash

# Initialize the environment
uv init --python 3.10
uv venv --python=3.10

# Build backend for the --no-build-isolation source builds below (fresh uv venvs lack setuptools)
uv pip install "setuptools<81" wheel

# PyTorch (TRELLIS.2 upstream pins torch 2.6.0 + CUDA 12.4)
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Basic (upstream setup.sh --basic, minus demo-only gradio/tensorboard;
# plain pillow instead of pillow-simd, which needs system libjpeg-dev via sudo apt.
# transformers is required by trellis2's BiRefNet rembg and DINOv3 cond model
# but undeclared upstream; pinned to 4.x — v5 removes DINOv3ViTModel.layer
# which DinoV3FeatureExtractor accesses directly. "httpx[socks]" is for
# huggingface_hub downloads behind a SOCKS proxy)
uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh pandas lpips zstandard kornia timm pillow "transformers==4.57.1" "httpx[socks]"
uv pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Flash-Attention (takes much time to build, just wait)
export MAX_JOBS=$(nproc)
uv pip install flash-attn==2.7.3 --no-build-isolation

# TRELLIS.2 CUDA extensions (clone to /tmp/extensions, per upstream setup.sh)
mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
uv pip install /tmp/extensions/nvdiffrast --no-build-isolation

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
uv pip install /tmp/extensions/nvdiffrec --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh
uv pip install /tmp/extensions/CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM
uv pip install /tmp/extensions/FlexGEMM --no-build-isolation

# o-voxel local package (bundled in the deps/trellis.2 submodule)
(cd ../../deps/trellis.2 && git submodule update --init --recursive)
uv pip install ../../deps/trellis.2/o-voxel --no-build-isolation

# FastAPI server dependencies
uv pip install fastapi uvicorn python-multipart requests pillow

# Model weight preheat (start-services.sh launches services with HF_HUB_OFFLINE=1,
# so microsoft/TRELLIS.2-4B, the DINO conditioning model and BiRefNet rembg
# must all be cached; from_pretrained constructs and downloads all of them)
if [ -f ".weights-preheated" ]; then
    echo "Model weights already preheated, skipping download"
else
    PYTHONPATH=../../deps/trellis.2 uv run -- python -c "\
from trellis2.pipelines import Trellis2ImageTo3DPipeline; \
Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')" \
        && touch .weights-preheated
fi
