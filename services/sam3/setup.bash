#!/bin/bash

# Initialize the environment
uv init --python 3.12
uv venv --python=3.12

# PyTorch with CUDA 12.8
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# other dependencies
uv pip install einops pycocotools psutil

# Install SAM3 from local submodule (auto-installs timm, numpy<2, ftfy, regex, iopath, huggingface_hub)
uv pip install -e ../../deps/sam3

# FastAPI server
uv pip install fastapi uvicorn python-multipart

# Client & utilities
uv pip install requests pillow

# Model Checkpoints
if [ -f "checkpoints/.download-complete" ]; then
    echo "Model weights already exist, skipping download"
else
    uv pip install modelscope
    uv run -- modelscope download --model facebook/sam3 --local_dir "./checkpoints"
    find ./checkpoints \( -name '._____temp' -o -name '.msc' -o -name '.mv' \) -delete
    touch checkpoints/.download-complete
fi
