#!/usr/bin/env bash
# Clone the exact upstream source revisions used by the benchmark adapters.
set -euo pipefail
benchmark_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$benchmark_root"
mkdir -p ref_code_library

clone_reference() {
  local name="$1" url="$2" revision="$3" target="ref_code_library/$1"
  if [ -d "$target/.git" ]; then
    local actual
    actual="$(git -C "$target" rev-parse HEAD)"
    if [ "$actual" != "$revision" ]; then
      echo "$target is at $actual; adapter was tested at $revision. Existing checkout kept." >&2
      return 1
    fi
  elif [ -e "$target" ]; then
    echo "Existing non-git path kept: $target" >&2
    return 1
  else
    git clone "$url" "$target"
    git -C "$target" checkout --detach "$revision"
  fi
  echo "$name ready at $revision"
}

clone_reference MV4DHMR https://github.com/Att100/MV4DHMR.git 87d61ffb8e0bd1b91abd4344376b86c9bacfca56
clone_reference HybrIK https://github.com/jeffffffli/HybrIK.git c281eeeb3c0689a4d619a06ed0c4488e791eda76
clone_reference pytorch3d https://github.com/facebookresearch/pytorch3d.git 88e182f989c80836f4bd744e0d9cb1852762ce01
