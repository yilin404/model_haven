#!/bin/bash
set -euo pipefail

DEFAULT_MODEL_NAME="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
MODEL_NAME="${DA3_MODEL_NAME:-$DEFAULT_MODEL_NAME}"

if [ ! -d ".venv" ]; then
    uv venv --python=3.11
fi

# Match the CUDA/PyTorch stack used by the newer model_haven services.
uv pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install \
    xformers==0.0.32.post2 \
    --index-url https://download.pytorch.org/whl/cu128

# Install the pinned local upstream source and the service/test utilities.
uv pip install -e ../../deps/depth-anything-3
uv pip install fastapi uvicorn requests pillow pydantic pytest "httpx[socks]"

# start-services.sh runs with HF_HUB_OFFLINE=1, so warm the shared Hub cache now.
echo "Caching model: $MODEL_NAME"
uv run -- hf download "$MODEL_NAME"

echo "Depth Anything V3 environment is ready"
