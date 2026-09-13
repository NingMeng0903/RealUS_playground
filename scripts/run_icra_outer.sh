#!/usr/bin/env bash
# Shared no-tank profile for controller, confidence, preflight, and recording.
set -euo pipefail
OUTER_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTER_ACQUISITION="${ICRA_SCRIPT_DIR:-/media/camp/EXT_DRIVE/ICRA_YM/script}"
OUTER_PYTHON="${ICRA_RECORD_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
OUTER_CONFIG="${CONTACT_QP_CONFIG:-${OUTER_REPO}/peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml}"
export REALUS_PROJECT_ROOT="$OUTER_REPO"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="${OUTER_REPO}:${OUTER_REPO}/rm75_control:${OUTER_REPO}/src:${PYTHONPATH:-}"
OUTER_ACTION="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$OUTER_REPO"
case "$OUTER_ACTION" in
  help|-h|--help)
    cat <<'HELP'
Usage: bash scripts/run_icra_outer.sh ACTION [options]
  controller   Start controller in its own terminal.
  ultrasound   Start ultrasound with the registered saved crop.
  confidence   Start the versioned weak-side confidence worker.
  gamepad      Start teaching/teleoperation gamepad.
  check        Read-only preflight of fresh, compatible confidence frames.
  validate     Offline profile validation; no device access.
  record       Preflight, then record at 5 mm/s with the fixed 4 N force profile.
Use one terminal per long-running component; restart controller and confidence
worker after an update. CONTACT_QP_CONFIG overrides the shared no-tank profile.
Historical energy profiles must be migrated; this entry never selects them.
HELP
    exit 0
    ;;
  controller|ultrasound|confidence|gamepad|check|validate|record) ;;
  *) echo "Unknown action: $OUTER_ACTION" >&2; exit 2 ;;
esac
# Reject stale profiles before any acquisition or robot-facing entry is opened.
"$OUTER_PYTHON" - "$OUTER_CONFIG" <<'PY'
import sys
from peirastic.contact_qp.runtime_config import load_study_config
cfg = load_study_config(sys.argv[1])
if ((cfg.get('qp') or {}).get('allocation_policy') != 'delay_kf_cop_v1'
        or 'energy' in cfg or cfg.get('energy_constraint_enabled', False)):
    raise SystemExit('Migrate CONTACT_QP_CONFIG to active_probe50_delay_kf_cop.yaml: '
                     'the default outer entry requires the no-tank delayed-KF policy.')
PY
case "$OUTER_ACTION" in
  controller|gamepad)
    exec bash "$OUTER_ACQUISITION/run.sh" "$OUTER_ACTION" "$@"
    ;;
  ultrasound)
    exec bash "$OUTER_ACQUISITION/run.sh" ultrasound --no-auto-crop-on-startup "$@"
    ;;
  confidence)
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_features --feature-config "$OUTER_CONFIG" "$@"
    ;;
  validate)
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_run --validate-only --config "$OUTER_CONFIG" "$@"
    ;;
  check)
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_check --config "$OUTER_CONFIG" "$@"
    ;;
  record)
    "$OUTER_PYTHON" -m peirastic.apps.contact_qp_check --config "$OUTER_CONFIG"
    exec bash "$OUTER_ACQUISITION/run.sh" record --force-profile icra --speed-m-s 0.005 --keep-raw "$@" --contact-qp-config "$OUTER_CONFIG"
    ;;
esac
