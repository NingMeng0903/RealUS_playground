#!/usr/bin/env bash
# Usage: source /media/camp/EXT_DRIVE/RealUS_playground/us_framegrab/env.sh

US_FRAMEGRAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
US_FRAMEGRAB_ENV="${US_FRAMEGRAB_ENV:-/media/camp/EXT_DRIVE/envs/camera_calib}"

if [ ! -d "${US_FRAMEGRAB_ENV}/bin" ]; then
  echo "camera_calib env not found: ${US_FRAMEGRAB_ENV}" >&2
  echo "Create: conda create -y -p ${US_FRAMEGRAB_ENV} python=3.10" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${US_FRAMEGRAB_ENV}/bin:${PATH}"

CONDA_BASE="${CONDA_BASE:-/home/camp/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${US_FRAMEGRAB_ENV}" 2>/dev/null || true
fi

export US_FRAMEGRAB_ROOT
export PYTHONNOUSERSITE=1
export PYTHONPATH="${US_FRAMEGRAB_ROOT}/src:${PYTHONPATH:-}"

echo "us_framegrab env: $(which python)"
echo "US_FRAMEGRAB_ROOT=${US_FRAMEGRAB_ROOT}"
