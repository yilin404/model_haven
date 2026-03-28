#!/bin/bash

# Initialize the environment
uv init
uv venv --python=3.10

# Basic
uv pip install torch==2.1.0 torchvision==0.16.0 torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

uv pip install -e ../../deps/graspgen

# Install PointNet dependency (automated script handles CUDA environment)
# Set CUDA compilation environment variables
export CC=/usr/bin/g++
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.6"

# Navigate to pointnet2_ops directory and install
uv pip install --no-build-isolation ../../deps/graspgen/pointnet2_ops
