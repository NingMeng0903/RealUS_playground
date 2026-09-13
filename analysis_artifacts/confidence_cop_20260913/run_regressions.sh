#!/usr/bin/env bash
# Offline only; no controller or sensor transport is started.
set -euo pipefail
FUSION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FUSION_REPO"
source rm75_control/env.sh
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$FUSION_REPO:$FUSION_REPO/rm75_control:$FUSION_REPO/src:${PYTHONPATH:-}"
python -m pytest peirastic/tests/test_contact_qp_*.py \
  peirastic/tests/test_contact_capabilities.py peirastic/tests/test_torque_tilt.py -q "$@"
