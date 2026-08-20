#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CMEEL="${CMEEL_PREFIX:-/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix}"
export CMEEL_PREFIX="$CMEEL"
SIMDE_DIR="$ROOT/third_party/simde"
if [[ ! -f "$SIMDE_DIR/simde/simde-math.h" ]]; then
  mkdir -p "$ROOT/third_party"
  echo "downloading simde headers..."
  curl -fsSL https://github.com/simd-everywhere/simde/archive/refs/tags/v0.8.2.tar.gz \
    | tar xz -C "$ROOT/third_party"
  rm -rf "$SIMDE_DIR"
  mv "$ROOT/third_party/simde-0.8.2" "$SIMDE_DIR"
fi
BUILD="$ROOT/build"
cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_PREFIX_PATH="$CMEEL" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"
echo "built $BUILD/wbc_rt"
