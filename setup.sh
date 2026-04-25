#!/bin/bash
set -e

echo "=== Setting up VLA shared memory test environment ==="

# Create and activate venv
uv venv .venv --python 3.10
source .venv/bin/activate

# Install in order — numpy must come first
uv pip install numpy==1.26.1

# torch: use PyTorch's official index (has aarch64+cu126 wheels from 2.6.0+)
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# cupy v13: last version that supports numpy 1.26 (v14+ requires numpy 2)
uv pip install "cupy-cuda12x>=13.0.0,<14.0.0"

# other deps
uv pip install posix-ipc

echo ""
echo "=== Verifying install ==="
python - << 'PYEOF'
import torch
import cupy as cp
import numpy as np
import posix_ipc

print(f"numpy   : {np.__version__}")
print(f"torch   : {torch.__version__}")
print(f"cupy    : {cp.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device  : {props.name}")
    print(f"unified : {bool(props.is_integrated)}")
PYEOF

echo ""
echo "=== Done. Run: python test_shm.py ==="
