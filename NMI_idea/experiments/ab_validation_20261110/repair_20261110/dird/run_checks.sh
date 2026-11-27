#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_PY=/media/camp/EXT_DRIVE/envs/genesis/bin/python
TASK_IRD=/media/camp/EXT_DRIVE/RealUS_playground/ird_playground
export PYTHONPATH="${TASK_IRD}:${TASK_IRD}/../rm75_control:${TASK_IRD}/..:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${TASK_DIR}"
"${TASK_PY}" -m pytest -p no:dash -q test_calibration_and_training.py \
 "${TASK_IRD}/tests/test_phase3_neural.py" "${TASK_IRD}/tests/test_eval_paths.py" \
 "${TASK_IRD}/tests/test_grouping_performance.py" ../../dird/test_departure_continuity.py
"${TASK_PY}" validate_qp_state.py --backend python
"${TASK_PY}" validate_qp_state.py --backend native
if [[ "${FULL_QP:-0}" == 1 ]]; then
 "${TASK_PY}" qp_replay.py --backend native --tcp-mode urdf --label gated_repeat
 "${TASK_PY}" qp_replay.py --backend native --tcp-mode urdf \
  --seed-json initialization_sensitivity_seed.json --label branch4_gated_repeat
fi
