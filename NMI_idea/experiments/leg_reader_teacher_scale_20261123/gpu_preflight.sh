#!/usr/bin/env bash
set -euo pipefail

py="${LEG_SCAN_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
if [[ ! -x "$py" ]]; then
    printf '找不到可执行的 Python：%s\n请把 LEG_SCAN_PYTHON 指向有 CUDA PyTorch 的 Python。\n' "$py" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >&2 || true
fi

exec "$py" - <<'PY'
import json
import sys

import numpy
import scipy
import torch

if not torch.cuda.is_available():
    sys.exit(f"CUDA 初始化失败：{sys.executable}, torch={torch.__version__}, 构建版本={torch.version.cuda}")

device = torch.device("cuda:0")
torch.manual_seed(41)
model = torch.nn.Linear(32, 8).to(device)
optimizer = torch.optim.Adam(model.parameters())
x = torch.ones((64, 32), device=device)
before = model.weight.detach().clone()
loss = model(x).square().mean()
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
torch.cuda.synchronize(device)
assert loss.isfinite().item() and not torch.equal(before, model.weight)
print(json.dumps({
    "python": sys.executable,
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "device": torch.cuda.get_device_name(device),
    "torch_cuda_visible_devices": torch.cuda.device_count(),
    "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "cuda_optimizer_step_verified": True,
}, ensure_ascii=False))
PY
