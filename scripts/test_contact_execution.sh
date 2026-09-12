#!/usr/bin/env bash
# Offline only: no controller, robot socket, rail, or image stream is started.
set -euo pipefail
CONTACT_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CONTACT_TEST_ROOT"
source rm75_control/env.sh
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="${CONTACT_TEST_ROOT}:${CONTACT_TEST_ROOT}/rm75_control:${CONTACT_TEST_ROOT}/src:${PYTHONPATH:-}"
case "${1:-focused}" in
  focused)
    CONTACT_TEST_FILES=(
      peirastic/tests/test_contact_qp_bounded_solver.py
      peirastic/tests/test_contact_qp_lease_gap.py
      peirastic/tests/test_contact_qp_execution_policy.py
      peirastic/tests/test_contact_qp_execution_runtime.py
      peirastic/tests/test_contact_qp_transient_feedback.py
      peirastic/tests/test_contact_qp_active.py
      peirastic/tests/test_contact_qp_active_history.py
      peirastic/tests/test_contact_qp_command_budget_active.py
      peirastic/tests/test_contact_qp_continuous_visual.py
    ) ;;
  --full) CONTACT_TEST_FILES=(peirastic/tests/test_contact_qp*.py) ;;
  *) printf '%s\n' 'Usage: bash scripts/test_contact_execution.sh [focused|--full]' >&2; exit 2 ;;
esac
/usr/bin/cmake --build rm75_control/native/wbc_rt/build -j2
/usr/bin/ctest --test-dir rm75_control/native/wbc_rt/build --output-on-failure
python -m pytest "${CONTACT_TEST_FILES[@]}" \
  rm75_control/tests/test_rocking_envelope.py \
  rm75_control/tests/test_wbc_rt_aborted_reply.py \
  rm75_control/tests/test_wbc_rt_notifications.py \
  rm75_control/tests/test_wbc_rt_facade.py \
  rm75_control/tests/test_final_qpik_send_chain.py -q
bash scripts/run_icra_tank.sh validate
