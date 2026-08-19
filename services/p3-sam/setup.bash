#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# P3-SAM service environment (uv venv, torch cu128 stack like newer services)
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    uv venv --python=3.10
fi

# PyTorch with CUDA 12.8 (matches the newer model_haven services)
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Sonata (PointTransformerV3 backbone) hard deps: spconv + torch_scatter.
# spconv wheels are torch-version independent; if spconv-cu124 misbehaves on
# this driver stack try spconv-cu126 / spconv-cu120.
uv pip install spconv-cu124
# torch_scatter: the PyG prebuilt wheel for pt28cu128 aborts with std::bad_alloc
# at import time against torch 2.8.0+cu128, so build from source (same strategy
# EmbodiedGen uses for its affordance stack). CUDA_HOME must point at the 12.8
# toolkit explicitly - /usr/local/cuda may symlink to an older toolkit and the
# build then aborts with CUDA_MISMATCH.
export FORCE_CUDA=1
export MAX_JOBS=8
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
uv pip install --no-build-isolation --no-deps --force-reinstall --no-cache-dir git+https://github.com/rusty1s/pytorch_scatter.git
unset FORCE_CUDA MAX_JOBS

# Python deps required by P3-SAM demo (auto_mask.py) and the sonata package
#   - timm: sonata/model.py imports timm.layers.DropPath; its deps would drag
#     torchvision (and in turn a newer torch) into the env, so install it
#     without deps and pin the matching cu128 torchvision explicitly.
#   - httpx[socks]: huggingface_hub 1.x uses httpx; SOCKS proxies need socksio
uv pip install numpy scipy scikit-learn fpsample numba trimesh tqdm addict omegaconf easydict huggingface_hub "httpx[socks]"
uv pip install timm --no-deps
uv pip install "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128

# FastAPI server
uv pip install fastapi uvicorn python-multipart

# Client & utilities
uv pip install requests

# ---------------------------------------------------------------------------
# Weight preheat: start-services.sh runs with HF_HUB_OFFLINE=1, so everything
# must be in its final location before the first request.
#   1. P3-SAM checkpoint -> shared HF cache (resolved via hf_hub_download)
#   2. sonata backbone   -> ~/.cache/sonata/ckpt (sonata.load uses local_dir)
# ---------------------------------------------------------------------------
# 国内网络走 hf-mirror；如处代理环境需把镜像域名加入 no_proxy（同 recognize-anything）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export no_proxy="${no_proxy:-},hf-mirror.com"
export NO_PROXY="${NO_PROXY:-},hf-mirror.com"

uv run -- hf download tencent/Hunyuan3D-Part p3sam/p3sam.safetensors

uv run -- python - <<'EOF'
import os
from huggingface_hub import hf_hub_download

target = os.path.expanduser("~/.cache/sonata/ckpt")
hf_hub_download(repo_id="facebook/sonata", filename="sonata.pth", local_dir=target)
print(f"sonata checkpoint ready under {target}")
EOF

echo "P3-SAM environment is ready"
