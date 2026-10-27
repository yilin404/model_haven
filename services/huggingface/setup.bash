#!/bin/bash

# Initialize the environment
uv init --python 3.11
uv venv --python=3.11

# PyTorch with CUDA 12.8
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# HuggingFace core (transformers>=5.17: Qwen3-VL encoder used by Qwen-Image-2.1)
uv pip install -U "transformers>=5.17" accelerate sentencepiece protobuf
uv pip install modelscope
uv pip install "httpx[socks]"

# Diffusers (FLUX / Qwen-Image-2.1; QwenImage21Pipeline needs >=0.38)
uv pip install -U "diffusers>=0.38"

# FastAPI server
uv pip install fastapi uvicorn python-multipart

# Client & utilities
uv pip install requests pillow
