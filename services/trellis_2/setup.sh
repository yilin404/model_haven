# !/bin/bash

uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Basic
uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard wheel psutil
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
sudo apt install -y libjpeg-dev
uv pip install pillow-simd
uv pip install kornia timm

# Flash-Attention (token much time to build, just wait)
uv pip install flash-attn==2.7.3 --no-build-isolation

# NVDiffRast
mkdir -p /tmp/extensions
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
uv pip install /tmp/extensions/nvdiffrast --no-build-isolation

# NVDiffRec
mkdir -p /tmp/extensions
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
uv pip install /tmp/extensions/nvdiffrec --no-build-isolation

# CuMesh
mkdir -p /tmp/extensions
git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
uv pip install /tmp/extensions/CuMesh --no-build-isolation

# FlexGEMM
mkdir -p /tmp/extensions
git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
uv pip install /tmp/extensions/FlexGEMM --no-build-isolation

# O-Vexel
mkdir -p /tmp/extensions
cp -r ../../deps/trellis_2/o-voxel /tmp/extensions/o-voxel
uv pip install /tmp/extensions/o-voxel --no-build-isolation
