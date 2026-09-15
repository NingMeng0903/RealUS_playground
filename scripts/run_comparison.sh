#!/usr/bin/env bash
# One comparison run on the phantom Lissajous path. Shuffle --mode across repeats.
set -euo pipefail
OUTER_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTER_ACQUISITION="${ICRA_SCRIPT_DIR:-/media/camp/EXT_DRIVE/ICRA_YM/script}"
OUTER_PYTHON="${ICRA_RECORD_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
OUTER_CONFIG="${CONTACT_QP_CONFIG:-${OUTER_REPO}/peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml}"
DATA_ROOT="${COMPARISON_DATA_ROOT:-/media/camp/PEI_T7/icra 2027_contact/Comparison Study}"
export REALUS_PROJECT_ROOT="$OUTER_REPO"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="${OUTER_REPO}:${OUTER_REPO}/rm75_control:${OUTER_REPO}/src:${PYTHONPATH:-}"
cd "$OUTER_REPO"

MODE=""
ACTION="scan"
PASSTHROUGH=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    controller|ultrasound|confidence|check|validate|scan)
      ACTION="$1"
      shift
      ;;
    help|-h|--help)
      ACTION="help"
      shift
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [ "$ACTION" = "help" ]; then
  cat <<'HELP'
Usage: bash scripts/run_comparison.sh --mode ultrapoc|admittance_1d|ac2d|tafac [scan options]
       bash scripts/run_comparison.sh controller|ultrasound|confidence|check|validate

UltraPoC plans the Lissajous once. The other three laws must --reuse-plan that
directory so they share the same polyline (including any normal-offset noise).

  bash scripts/run_comparison.sh --mode ultrapoc --normal-offset-deg 20
  bash scripts/run_comparison.sh --mode admittance_1d --reuse-plan ".../ultrapoc/00N"
  bash scripts/run_comparison.sh --mode ac2d --reuse-plan ".../ultrapoc/00N"
  bash scripts/run_comparison.sh --mode tafac --reuse-plan ".../ultrapoc/00N"

Do not pass --normal-offset-deg on the reused runs. Shuffle the three
baselines across repeats (gel / heating). Path speed is 10 mm/s for all
four modes. Other scan options (--no-us, --force 4, --lissajous-yaw-deg)
are forwarded.
HELP
  exit 0
fi

"$OUTER_PYTHON" - "$OUTER_CONFIG" <<'PY'
import sys
from peirastic.contact_qp.runtime_config import load_study_config
cfg = load_study_config(sys.argv[1])
if ((cfg.get('qp') or {}).get('allocation_policy') != 'delay_kf_cop_v1'
        or 'energy' in cfg or cfg.get('energy_constraint_enabled', False)):
    raise SystemExit('Migrate CONTACT_QP_CONFIG to active_probe50_delay_kf_cop.yaml')
PY

case "$ACTION" in
  controller)
    exec bash "$OUTER_ACQUISITION/run.sh" controller "${PASSTHROUGH[@]}"
    ;;
  ultrasound)
    exec bash "$OUTER_ACQUISITION/run.sh" ultrasound --no-auto-crop-on-startup "${PASSTHROUGH[@]}"
    ;;
  confidence)
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_features --feature-config "$OUTER_CONFIG" "${PASSTHROUGH[@]}"
    ;;
  validate)
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_run --validate-only --config "$OUTER_CONFIG" "${PASSTHROUGH[@]}"
    ;;
  check)
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_check --config "$OUTER_CONFIG" "${PASSTHROUGH[@]}"
    ;;
  scan)
    if [ -z "$MODE" ]; then
      echo "need --mode ultrapoc|admittance_1d|ac2d|tafac" >&2
      exit 2
    fi
    if [ "$MODE" = "ultrapoc" ]; then
      "$OUTER_PYTHON" -m peirastic.apps.contact_qp_check --config "$OUTER_CONFIG"
    fi
    exec "$OUTER_PYTHON" -m peirastic.DEMO.phathom_scanning.s_scan \
      --mode "$MODE" \
      --pattern lissajous \
      --speed-m-s 0.010 \
      --data-root "$DATA_ROOT" \
      --contact-qp-config "$OUTER_CONFIG" \
      "${PASSTHROUGH[@]}"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
