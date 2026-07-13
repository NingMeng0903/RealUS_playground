# RealUS playground prelude (perception / genesis_ue_sync / UE bridges / anatomy).
# Usage: cd /media/camp/EXT_DRIVE/RealUS_playground && source env.sh

export REALUS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AMONGUS_PROJECT_ROOT="${REALUS_PROJECT_ROOT}"
export REPO="${REALUS_PROJECT_ROOT}"
export SRC="${REALUS_PROJECT_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}"

# Reuse Among_US genesis conda env
if [ -x /media/camp/EXT_DRIVE/envs/genesis/bin/python ]; then
  export PY=/media/camp/EXT_DRIVE/envs/genesis/bin/python
  # shellcheck disable=SC1091
  source /media/camp/EXT_DRIVE/envs/genesis/bin/activate 2>/dev/null || true
fi

# RealSense Cam publisher: prefer genesis $PY; fall back to camera_calib if pyrealsense2 missing.
export REALUS_CAMERA_PY="${REALUS_CAMERA_PY:-${PY:-python}}"
if [ -n "${PY:-}" ] && ! "${PY}" -c "import pyrealsense2" 2>/dev/null; then
  if [ -x /media/camp/EXT_DRIVE/envs/camera_calib/bin/python ]; then
    export REALUS_CAMERA_PY=/media/camp/EXT_DRIVE/envs/camera_calib/bin/python
    echo "  REALUS_CAMERA_PY=${REALUS_CAMERA_PY} (pyrealsense2 not in genesis; using camera_calib)"
  else
    echo "  WARN: pyrealsense2 missing in genesis. Run: pip install -r perception/requirements.txt" >&2
  fi
fi

export AMONGUS_BLENDER_BIN="${AMONGUS_BLENDER_BIN:-/media/camp/EXT_DRIVE/blender/blender-4.5.8-linux-x64/blender}"
export REALUS_BEDLAM_UNREAL_ROOT="${REALUS_BEDLAM_UNREAL_ROOT:-/media/camp/EXT_DRIVE/Among_US/assets/humans/bedlam2/unreal}"
export REALUS_BEDLAM_RETARGET_ROOT="${REALUS_BEDLAM_RETARGET_ROOT:-/media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam2_retargeting}"

export SESSION_DIR="${SESSION_DIR:-${REALUS_PROJECT_ROOT}/outputs/ue_sessions/realus_fullflow_v1}"
mkdir -p "${SESSION_DIR}"
export AMONGUS_SESSION_DIR="${SESSION_DIR}"

export AMONGUS_GENESIS_CANONICAL_ZMQ_BIND="${AMONGUS_GENESIS_CANONICAL_ZMQ_BIND:-tcp://127.0.0.1:5599}"
export AMONGUS_GENESIS_CANONICAL_STATE_JSONL="${AMONGUS_GENESIS_CANONICAL_STATE_JSONL:-${SESSION_DIR}/genesis_canonical.jsonl}"
export AMONGUS_UE_DRIVE_HUMAN_BONES="${AMONGUS_UE_DRIVE_HUMAN_BONES:-1}"
export AMONGUS_UE_HIDE_BEDLAM_CAMERA_RIG="${AMONGUS_UE_HIDE_BEDLAM_CAMERA_RIG:-1}"
export AMONGUS_UE_SPAWN_HUMAN_ANCHOR_MARKER="${AMONGUS_UE_SPAWN_HUMAN_ANCHOR_MARKER:-0}"

export REALUS_CAMERA_CALIB_BUNDLE="${REALUS_CAMERA_CALIB_BUNDLE:-${REALUS_PROJECT_ROOT}/camera_calibration/calibration_results/genesis_bundle.yaml}"
export CAMERA_CALIB_BUNDLE="${CAMERA_CALIB_BUNDLE:-${REALUS_CAMERA_CALIB_BUNDLE}}"
export REALUS_CAMERAS_YAML="${REALUS_CAMERAS_YAML:-${REALUS_PROJECT_ROOT}/camera_calibration/configs/cameras.yaml}"
export REALUS_SMPLX_OUTPUT_ROOT="${REALUS_SMPLX_OUTPUT_ROOT:-${REALUS_PROJECT_ROOT}/smplx_outputs}"
mkdir -p "${REALUS_SMPLX_OUTPUT_ROOT}"

# UE editor: ROS in LD_LIBRARY_PATH breaks Vulkan; wrong VK_ICD path also crashes UE.
# Ubuntu ICD lives under /usr/share/vulkan (NOT /etc/vulkan — often stale in .bashrc).
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  _realus_ld_clean=""
  IFS=':' read -ra _realus_ld_parts <<< "${LD_LIBRARY_PATH}"
  for _realus_p in "${_realus_ld_parts[@]}"; do
    [ -z "${_realus_p}" ] && continue
    case "${_realus_p}" in
      /opt/ros/*|*/opt/ros/*) continue ;;
    esac
    _realus_ld_clean="${_realus_ld_clean:+${_realus_ld_clean}:}${_realus_p}"
  done
  export LD_LIBRARY_PATH="${_realus_ld_clean}"
  unset _realus_ld_clean _realus_ld_parts _realus_p
fi

echo "RealUS env ready: REPO=${REPO}"
echo "  PY=${PY:-python}"
echo "  SESSION_DIR=${SESSION_DIR}"
echo "  CAMERA_CALIB_BUNDLE=${CAMERA_CALIB_BUNDLE}"
echo "  SMPLX_OUTPUT_ROOT=${REALUS_SMPLX_OUTPUT_ROOT}"
