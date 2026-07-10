#!/usr/bin/env bash
set -euo pipefail
# Symlink repo BEDLAM starter-pack engine content into the Unreal Engine tree so
# /Engine/PS/Bedlam/... resolves (see assets/humans/bedlam2/unreal/engine_content/PS).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_PS="${REPO_ROOT}/assets/humans/bedlam2/unreal/engine_content/PS"

if [[ ! -d "${SRC_PS}" ]]; then
  echo "Missing BEDLAM engine content: ${SRC_PS}" >&2
  exit 1
fi

resolve_ue_root() {
  if [[ -n "${UNREAL_ENGINE_ROOT:-}" ]]; then
    cd "$(readlink -f "${UNREAL_ENGINE_ROOT}")" && pwd
    return
  fi
  if [[ -n "${UNREAL_EDITOR_CMD:-}" ]]; then
    local bin
    bin="$(readlink -f "${UNREAL_EDITOR_CMD}")"
    # .../Engine/Binaries/Linux/UnrealEditor-Cmd -> UE root (Linux -> Binaries -> Engine -> root)
    cd "$(dirname "${bin}")/../../.." && pwd
    return
  fi
  echo "Set UNREAL_ENGINE_ROOT (UnrealEngine-5.3.2 directory) or UNREAL_EDITOR_CMD." >&2
  exit 1
}

UE_ROOT="$(resolve_ue_root)"
DST_PS="${UE_ROOT}/Engine/Content/PS"

if [[ -L "${DST_PS}" ]]; then
  echo "Already symlink: ${DST_PS} -> $(readlink -f "${DST_PS}")"
  exit 0
fi
if [[ -e "${DST_PS}" ]]; then
  echo "Refusing to overwrite existing path (not a symlink): ${DST_PS}" >&2
  exit 1
fi

mkdir -p "$(dirname "${DST_PS}")"
ln -s "${SRC_PS}" "${DST_PS}"
echo "Linked ${DST_PS} -> ${SRC_PS}"
