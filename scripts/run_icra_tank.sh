#!/usr/bin/env bash
# Each long-running component uses its own terminal. record preflights images.
set -euo pipefail
TANK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TANK_ACQUISITION="${ICRA_SCRIPT_DIR:-/media/camp/EXT_DRIVE/ICRA_YM/script}"
TANK_PYTHON="${ICRA_RECORD_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
TANK_CONFIG="${CONTACT_QP_CONFIG:-${TANK_REPO}/peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml}"
export REALUS_PROJECT_ROOT="$TANK_REPO"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="${TANK_REPO}:${TANK_REPO}/rm75_control:${TANK_REPO}/src:${PYTHONPATH:-}"
TANK_ACTION="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$TANK_REPO"
case "$TANK_ACTION" in
  controller|gamepad)
    exec bash "$TANK_ACQUISITION/run.sh" "$TANK_ACTION" "$@"
    ;;
  ultrasound)
    exec bash "$TANK_ACQUISITION/run.sh" ultrasound --no-auto-crop-on-startup "$@"
    ;;
  confidence)
    exec "$TANK_PYTHON" -m peirastic.apps.contact_qp_features --feature-config "$TANK_CONFIG" "$@"
    ;;
  validate)
    exec "$TANK_PYTHON" -m peirastic.apps.contact_qp_run --validate-only --config "$TANK_CONFIG" "$@"
    ;;
  check)
    exec "$TANK_PYTHON" -m peirastic.apps.contact_qp_check --config "$TANK_CONFIG" "$@"
    ;;
  record)
    "$TANK_PYTHON" -m peirastic.apps.contact_qp_check --config "$TANK_CONFIG"
    exec bash "$TANK_ACQUISITION/run.sh" record --force-profile icra --speed-m-s 0.005 --keep-raw "$@" --contact-qp-config "$TANK_CONFIG"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: bash scripts/run_icra_tank.sh ACTION [options]
  controller   Start controller (restart an existing process to load changes).
  ultrasound   Start ultrasound UI using the saved crop, without auto-recropping.
  confidence   Start confidence worker with the same tank configuration.
  gamepad      Start teaching/teleoperation gamepad.
  check        Read-only check: three fresh, compatible confidence frames.
  validate     Offline configuration check; no device access.
  record       Check confidence first, then record at 5 mm/s and keep raw data.
Use a separate terminal for each long-running component.
CONTACT_QP_CONFIG can select a different shared worker/record configuration.
HELP
    ;;
  *) echo "Unknown action: $TANK_ACTION" >&2; exit 2 ;;
esac
