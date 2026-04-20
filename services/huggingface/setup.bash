#!/bin/bash

# Initialize the environment
uv init
uv venv --python=3.11

# PyTorch with CUDA 12.8
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# HuggingFace core
uv pip install transformers accelerate sentencepiece protobuf

# Diffusers (FLUX image generation)
uv pip install -U diffusers

# FastAPI server
uv pip install fastapi uvicorn python-multipart

# Client & utilities
uv pip install requests pillow
