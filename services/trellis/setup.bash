# !/bin/bash

# Initialize the environment
uv init
uv venv --python=3.11

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Basic
uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers wheel psutil zmq "httpx[socks]" "pyglet<2"
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Xformers
uv pip install xformers==0.0.33 --index-url https://download.pytorch.org/whl/cu128

# Flash-Attention (token much time to build, just wait)
uv pip install flash-attn --no-build-isolation

# KAOLIN
uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

# NVDiffRast
mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
uv pip install /tmp/extensions/nvdiffrast --no-build-isolation

# Diffoctreerast
mkdir -p /tmp/extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
git -C /tmp/extensions/diffoctreerast submodule update --init --recursive
uv pip install /tmp/extensions/diffoctreerast --no-build-isolation

# mipgaussian
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
uv pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# spconv
uv pip install spconv-cu120

# FastAPI and Uvicorn for the web server
uv pip install fastapi uvicorn python-multipart
