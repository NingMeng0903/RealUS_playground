#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
TASK_IRD=/media/camp/EXT_DRIVE/RealUS_playground/ird_playground
export PYTHONPATH="${TASK_IRD}:/media/camp/EXT_DRIVE/RealUS_playground/rm75_control:${PYTHONPATH:-}"
cd "${TASK_DIR}"
mkdir -p logs
"${TASK_PYTHON}" -m pytest -p no:dash test_so3_exp_regression.py -q > logs/so3_regression_reproduced.txt 2>&1
"${TASK_PYTHON}" -m pytest -p no:dash test_departure_continuity.py -q > logs/departure_regression_reproduced.txt 2>&1
"${TASK_PYTHON}" -m pytest -p no:dash -q \
  "${TASK_IRD}/tests/test_robot_contract.py" \
  "${TASK_IRD}/tests/test_signed_operator.py" \
  "${TASK_IRD}/tests/test_differentiable_guidance.py" \
  "${TASK_IRD}/tests/test_srs_label.py" \
  "${TASK_IRD}/tests/test_srs_trajectory_dp.py" \
  "${TASK_IRD}/tests/test_ellipsoid_conditioning.py" > logs/production_tests_reproduced.txt 2>&1
"${TASK_PYTHON}" benchmark.py > logs/benchmark_reproduced.txt 2>&1
"${TASK_PYTHON}" field_validation.py > logs/field_validation_reproduced.txt 2>&1
"${TASK_PYTHON}" frame_and_query_audit.py > logs/frame_query_reproduced.txt 2>&1
if [[ "${RUN_GPU:-1}" == 1 ]]; then
  "${TASK_PYTHON}" gpu_and_zero_rotation_audit.py > logs/gpu_audit_reproduced.txt 2>&1
fi
