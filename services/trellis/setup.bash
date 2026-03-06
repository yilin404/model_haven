# !/bin/bash

uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Basic
uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers wheel psutil
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Xformers
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

# Flash-Attention (token much time to build, just wait)
uv pip install flash-attn==2.7.3 --no-build-isolation

# KAOLIN
uv pip install kaolin

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