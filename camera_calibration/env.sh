#!/usr/bin/env bash
# Usage: source /media/camp/EXT_DRIVE/RealUS_playground/camera_calibration/env.sh

CAM_CALIB_ENV="/media/camp/EXT_DRIVE/envs/camera_calib"

if [ ! -d "${CAM_CALIB_ENV}/bin" ]; then
  echo "camera_calib env not found: ${CAM_CALIB_ENV}" >&2
  echo "Create: conda create -y -p ${CAM_CALIB_ENV} python=3.10" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${CAM_CALIB_ENV}/bin:${PATH}"

CONDA_BASE="${CONDA_BASE:-/home/camp/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CAM_CALIB_ENV}" 2>/dev/null || true
fi

export CAMERA_CALIB_ROOT="/media/camp/EXT_DRIVE/RealUS_playground/camera_calibration"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CAMERA_CALIB_ROOT}/src:${PYTHONPATH:-}"

echo "camera_calib env: $(which python)"
echo "CAMERA_CALIB_ROOT=${CAMERA_CALIB_ROOT}"
