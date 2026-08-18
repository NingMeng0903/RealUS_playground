#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$ROOT/../../rm75_control/control/joint_admittance_8dof/solver}"
CMEEL="${CMEEL_PREFIX:-/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix}"
export CMEEL_PREFIX="$CMEEL"
PYBIND11_DIR="${PYBIND11_DIR:-/usr/lib/cmake/pybind11}"
BUILD="$ROOT/build"
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_PREFIX_PATH="$CMEEL;$PYBIND11_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEST"
cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
echo "installed _qpik_kernel to $DEST"
