#!/usr/bin/env bash
# PyTorch for Genesis 1.2+ (requires torch>=2.8; cu124 wheels often stop at torch 2.6).
# Usage (from repo root, rm75 env active):
#   source env.sh
#   bash rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh

set -euo pipefail

CUDA="${CUDA:-cu126}"
INDEX="https://download.pytorch.org/whl/${CUDA}"

echo "Using: $(which python)"
echo "Installing torch>=2.8.0 from ${INDEX} ..."
python -m pip install -U "torch>=2.8.0" "torchvision>=0.23.0" --index-url "$INDEX"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
