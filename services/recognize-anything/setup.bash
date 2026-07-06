#!/bin/bash

# Initialize the environment
uv init --python 3.10
uv venv --python=3.10

# PyTorch with CUDA 12.1 (same stack as GraspGen; best compat with timm==0.4.12)
uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# RAM++ dependencies (closed-set inference does NOT need CLIP / pycocoevalcap).
# Pin transformers==4.38.2 + numpy<2: newer transformers requires torch>=2.4
# (conflicts with our torch 2.1.0) and torch 2.1 is incompatible with numpy 2.x.
uv pip install "timm==0.4.12" "transformers==4.38.2" "numpy<2" "fairscale==0.4.4" Pillow scipy

# Install RAM (Recognize Anything) from local submodule
uv pip install -e ../../deps/recognize-anything

# FastAPI server
uv pip install fastapi uvicorn python-multipart

# Client & utilities
uv pip install requests pillow

# Model Checkpoint (huggingface_hub handles LFS + resume reliably; curl truncated on large files)
if [ -f "checkpoints/.download-complete" ]; then
    echo "Model weights already exist, skipping download"
else
    mkdir -p checkpoints
    uv pip install huggingface_hub
    uv run -- python -c "from huggingface_hub import hf_hub_download; hf_hub_download('xinyu1205/recognize-anything-plus-model', 'ram_plus_swin_large_14m.pth', local_dir='./checkpoints')"
    touch checkpoints/.download-complete
fi
