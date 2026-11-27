#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_python="${LEG_SCAN_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
experiment_workers="${LEG_SCAN_WORKERS:-24}"
if [[ ! -s "$experiment_dir/preflight/cuda_step.json" ]]; then
    bash "$experiment_dir/../gpu_preflight.sh" > "$experiment_dir/preflight/cuda_step.json" \
        2> "$experiment_dir/preflight/driver.txt"
fi
export PYTHONDONTWRITEBYTECODE=1
export LEG_ENABLE_DIRD=1
export LEG_ENABLE_MATERIAL_ATLAS=1
"$experiment_python" -u "$experiment_dir/src/run_online.py" train --workers "$experiment_workers"
"$experiment_python" -u "$experiment_dir/src/run_online.py" evaluate --workers "$experiment_workers"
