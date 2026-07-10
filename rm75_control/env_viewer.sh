#!/usr/bin/env bash
# Genesis viewer / digital twin — use Among_US genesis env (NOT rm75).
#
#   source env_viewer.sh
#   python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer

GENESIS_ENV="/media/camp/EXT_DRIVE/envs/genesis"

if [ ! -d "${GENESIS_ENV}/bin" ]; then
  echo "genesis env not found: ${GENESIS_ENV}" >&2
  echo "Among_US uses: conda activate /media/camp/EXT_DRIVE/envs/genesis" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${GENESIS_ENV}/bin:${PATH}"

CONDA_BASE="${CONDA_BASE:-/home/camp/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${GENESIS_ENV}" 2>/dev/null || true
fi

export RM75_CONTROL_ROOT="/media/camp/EXT_DRIVE/RealUS_playground/rm75_control"
export RM_API2_PYTHON="/media/camp/EXT_DRIVE/RM_API2/Python"
export REALUS_PROJECT_ROOT="${REALUS_PROJECT_ROOT:-/media/camp/EXT_DRIVE/RealUS_playground}"
export AMONGUS_PROJECT_ROOT="${AMONGUS_PROJECT_ROOT:-${REALUS_PROJECT_ROOT}}"
# Robotic_Arm only needed for rm75_control package __init__ import chain; not used by viewer runtime.
# RealUS src enables track/anatomy/canonical overlays on the twin.
export PYTHONPATH="${REALUS_PROJECT_ROOT}/src:${RM75_CONTROL_ROOT}:${RM_API2_PYTHON}:${PYTHONPATH:-}"

echo "viewer env: $(which python)  (genesis / Among_US)"
echo "RM75_CONTROL_ROOT=${RM75_CONTROL_ROOT}"
echo "REALUS_PROJECT_ROOT=${REALUS_PROJECT_ROOT}"
