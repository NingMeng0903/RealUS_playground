#!/usr/bin/env bash
# Decimate RM75 collision DAE -> low-poly STL via Blender (headless).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER="${BLENDER:-/media/camp/EXT_DRIVE/blender/blender}"
exec "${BLENDER}" --background --python "${ROOT}/scripts/blender_simplify_collision_meshes.py"
