#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
REPAIR_ROOT=$(dirname "$PWD")
g++ -O3 -std=c++17 tet_overlap.cpp -o tet_overlap
"$PYTHON" check_overlap_kernel.py
for SIDE in left right; do
 "$PYTHON" audit.py --asset "$REPAIR_ROOT/harmonic/shaft_roi_v2/atlas_${SIDE}.npz" --side "$SIDE" --tag shaft_roi_v2 --volume-weights "$REPAIR_ROOT/harmonic/shaft_roi_v2/attachments_${SIDE}.npz" --pose-degrees 0 15 30 45 60 75 90 > "shaft_roi_v2_${SIDE}_run.log" 2>&1
 "$PYTHON" material_overlap.py --asset "$REPAIR_ROOT/harmonic/shaft_roi_v2/atlas_${SIDE}.npz" --tag "shaft_roi_v2_${SIDE}" > "shaft_roi_v2_${SIDE}_material.log" 2>&1
 "$PYTHON" roi_contract.py --asset "$REPAIR_ROOT/harmonic/shaft_roi_v2/atlas_${SIDE}.npz" --attachments "$REPAIR_ROOT/harmonic/shaft_roi_v2/attachments_${SIDE}.npz" --source-skin "$REPAIR_ROOT/harmonic/ownership_smooth/owned_smooth_${SIDE}_skin.npz" --tag "shaft_roi_v2_${SIDE}" > "shaft_roi_v2_${SIDE}_contract.log" 2>&1
done
"$PYTHON" make_report.py
