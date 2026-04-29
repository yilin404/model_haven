#!/bin/bash

# Initialize the environment
uv init --python 3.11
uv venv --python=3.11

# PyTorch CUDA 12.1
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# nvidia-pyindex and cuda-python (for CUDA runtime API access)
uv pip install pip appdirs

# Build dependencies for sam3d_objects (needed when using --no-build-isolation)
uv pip install hatchling hatch-requirements-txt editables
uv pip install "setuptools<81"

# Install sam3d_objects core dependencies (without [dev] — dev tools only)
# --no-build-isolation: use current venv instead of isolated build env (avoids chasing undeclared build deps)
# --index-strategy unsafe-best-match: allow cross-index version matching (cuda-python etc.)
uv pip install --no-build-isolation -e ../../deps/sam-3d-objects \
  --extra-index-url https://pypi.ngc.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match

# Install pytorch3d (separate step — upstream has broken torch dependency)
# --force-reinstall --no-cache: must recompile against current torch to match ABI
uv pip install --no-build-isolation --force-reinstall --no-cache -e '../../deps/sam-3d-objects[p3d]' \
  --extra-index-url https://pypi.ngc.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match

# Install inference dependencies (kaolin needs special find-links URL)
uv pip install --no-build-isolation -e '../../deps/sam-3d-objects[inference]' \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# Patch hydra (required by official setup.md — fixes hydra 1.3.2 instantiate() bug)
uv run ../../deps/sam-3d-objects/patching/hydra

# FastAPI server dependencies
uv pip install fastapi uvicorn python-multipart requests pillow

# Model Checkpoints
if [ -f "checkpoints/.download-complete" ]; then
    echo "Model weights already exist, skipping download"
else
    uv pip install modelscope
    uv run -- modelscope download --model facebook/sam-3d-objects --local_dir "." --include "checkpoints/*"
    find ./checkpoints \( -name '.____temp' -o -name '.msc' -o -name '.mv' \) -delete
    touch checkpoints/.download-complete
fi
