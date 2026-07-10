#!/usr/bin/env bash
# Install PyTorch with CUDA for bundled Genesis (requires torch>=2.8, see Genesis/genesis/__init__.py).
# Usage:
#   bash scripts/env/install_torch_cuda_for_genesis.sh
#   CUDA=cu124 bash scripts/env/install_torch_cuda_for_genesis.sh
#   /path/to/conda/envs/genesis/bin/python -m pip install ...   # or activate env first

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
# cu126: torch>=2.8 (Genesis warns on torch<2.8). cu124 wheels top out around torch 2.6 for manylinux.
CUDA="${CUDA:-cu126}"
INDEX="https://download.pytorch.org/whl/${CUDA}"

echo "Using: $PYTHON"
echo "PyTorch wheel index: $INDEX"
echo "Installing torch>=2.8.0 torchvision (CUDA ${CUDA}) ..."

"$PYTHON" -m pip install -U "torch>=2.8.0" "torchvision>=0.23.0" --index-url "$INDEX"

"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
PY
