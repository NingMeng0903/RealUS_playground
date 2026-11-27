#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="/media/camp/EXT_DRIVE/RealUS_playground:/media/camp/EXT_DRIVE/RealUS_playground/ird_playground:/media/camp/EXT_DRIVE/RealUS_playground/rm75_control:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${TASK_DIR}"
/media/camp/EXT_DRIVE/envs/genesis/bin/python -m pytest -p no:dash -q phase2_boundary_fix/test_boundary_derivatives.py
