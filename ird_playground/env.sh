#!/usr/bin/env bash
# Usage: source /media/camp/EXT_DRIVE/RealUS_playground/ird_playground/env.sh
#
# Training uses the Among_US **genesis** conda env (PyTorch + CUDA).
# Override with IRD_ENV=/path/to/env if needed.

IRD_ENV="${IRD_ENV:-/media/camp/EXT_DRIVE/envs/genesis}"
RM75_CONTROL_ROOT="${RM75_CONTROL_ROOT:-/media/camp/EXT_DRIVE/RealUS_playground/rm75_control}"
IRD_PLAYGROUND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${IRD_ENV}/bin" ]; then
  echo "env not found: ${IRD_ENV} (set IRD_ENV; default=Among_US genesis)" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${IRD_ENV}/bin:${PATH}"

CONDA_BASE="${CONDA_BASE:-/home/camp/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${IRD_ENV}" 2>/dev/null || true
fi

export IRD_PLAYGROUND_ROOT
export RM75_CONTROL_ROOT
export PYTHONPATH="${IRD_PLAYGROUND_ROOT}:${RM75_CONTROL_ROOT}:${PYTHONPATH:-}"

echo "ird_playground env: $(which python)  [Among_US genesis / IRD_ENV]"
echo "IRD_PLAYGROUND_ROOT=${IRD_PLAYGROUND_ROOT}"
echo "RM75_CONTROL_ROOT=${RM75_CONTROL_ROOT}"
