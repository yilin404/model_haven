#!/bin/bash

# Initialize the environment (Python 3.10 required by particulate)
uv init --python 3.10
uv venv --python=3.10

# PyTorch with CUDA 12.4
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

# torch-scatter (match torch+CUDA version)
uv pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Core deps (from particulate requirements.txt + transitive imports)
uv pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3 \
  mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d \
  simple_parsing arrgh open3d vtk omegaconf diffusers scipy pygltflib \
  huggingface_hub requests pillow psutil matplotlib networkx tqdm scikit-learn "httpx[socks]"
# Build toolchain needed for PVCNN CUDA kernels and torch-scatter
uv pip install ninja cmake

# FastAPI server
uv pip install fastapi uvicorn python-multipart

# Download Particulate model checkpoint
if [ -f "checkpoints/.download-complete" ]; then
    echo "Particulate checkpoint already exists, skipping download"
else
    mkdir -p checkpoints
    uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='rayli/Particulate', filename='model.pt', local_dir='checkpoints')
"
    touch checkpoints/.download-complete
fi

# Download PartField model checkpoint to correct path in deps tree
# get_partfield_model() constructs path via os.path.dirname(__file__) from partfield_utils.py
# So the checkpoint MUST be at deps/particulate/PartField/model/model_objaverse.ckpt
# Guard checks actual file existence, not just marker
PARTFIELD_CKPT="../../deps/particulate/PartField/model/model_objaverse.ckpt"
if [ -f "$PARTFIELD_CKPT" ]; then
    echo "PartField checkpoint already exists, skipping download"
else
    mkdir -p ../../deps/particulate/PartField/model
    uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='mikaelaangel/partfield-ckpt', filename='model_objaverse.ckpt', local_dir='../../deps/particulate/PartField/model')
"
fi
