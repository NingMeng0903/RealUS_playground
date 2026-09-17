#!/usr/bin/env bash
# UltraPoC ablations on one newly planned biased Lissajous.
# full: detect + plan with 20° normal offset under Ablation Study/full.
# The other four: --reuse-plan that full run. Shuffle --mode across repeats.
set -euo pipefail
OUTER_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTER_ACQUISITION="${ICRA_SCRIPT_DIR:-/media/camp/EXT_DRIVE/ICRA_YM/script}"
OUTER_PYTHON="${ICRA_RECORD_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
OUTER_CONFIG="${CONTACT_QP_CONFIG:-${OUTER_REPO}/peirastic/config/contact_qp/active_probe50_delay_kf_cop.yaml}"
DATA_ROOT="${ABLATION_DATA_ROOT:-/media/camp/PEI_T7/icra 2027_contact/Ablation Study}"
PLAN=""
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
    --reuse-plan)
      PLAN="${2:-}"
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
Usage: bash scripts/run_ablation.sh --mode full|no_cop|fixed_alpha|no_us_orient|no_moment [scan options]
       bash scripts/run_ablation.sh controller|ultrasound|confidence|check|validate

Runs land under Ablation Study/<mode>/00N, not Comparison Study.

  full            complete controller; omit --reuse-plan to detect+plan 20°,
                  or --reuse-plan a previous full/00N to repeat that polyline
  no_cop          reuse that plan; hold v_n to v_f (no CoP pairing)
  fixed_alpha     reuse that plan; keep alpha=1
  no_us_orient    reuse that plan; drop image omega, keep image alpha_d
  no_moment       reuse that plan; set omega_tau=0

  bash scripts/run_ablation.sh --mode full
  bash scripts/run_ablation.sh --mode full --reuse-plan ".../Ablation Study/full/001"
  bash scripts/run_ablation.sh --mode no_cop --reuse-plan ".../Ablation Study/full/001"
  bash scripts/run_ablation.sh --mode fixed_alpha --reuse-plan ".../Ablation Study/full/001"
  bash scripts/run_ablation.sh --mode no_us_orient --reuse-plan ".../Ablation Study/full/001"
  bash scripts/run_ablation.sh --mode no_moment --reuse-plan ".../Ablation Study/full/001"

Do not pass --normal-offset-deg on the reused runs. Shuffle modes across
repeats (gel / heating). Path speed is 10 mm/s. Other scan options
(--no-us, --force 4) are forwarded.
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
    exec "$OUTER_PYTHON" -m peirastic.apps.contact_qp_check --config "$OUTER_CONFIG"
    ;;
  scan)
    case "$MODE" in
      full|no_cop|fixed_alpha|no_us_orient|no_moment) ;;
      *)
        echo "need --mode full|no_cop|fixed_alpha|no_us_orient|no_moment" >&2
        exit 2
        ;;
    esac
    "$OUTER_PYTHON" -m peirastic.apps.contact_qp_check --config "$OUTER_CONFIG"
    SCAN=(
      "$OUTER_PYTHON" -m peirastic.DEMO.phathom_scanning.s_scan
      --mode ultrapoc
      --ablation "$MODE"
      --pattern lissajous
      --speed-m-s 0.010
      --data-root "$DATA_ROOT"
      --contact-qp-config "$OUTER_CONFIG"
    )
    if [ -n "$PLAN" ]; then
      if [ ! -f "$PLAN/plan.json" ]; then
        echo "reuse plan missing: $PLAN/plan.json" >&2
        exit 2
      fi
      SCAN+=(--reuse-plan "$PLAN")
    elif [ "$MODE" = "full" ]; then
      SCAN+=(--normal-offset-deg 20)
    else
      echo "need --reuse-plan pointing to Ablation Study/full/00N" >&2
      exit 2
    fi
    exec "${SCAN[@]}" "${PASSTHROUGH[@]}"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
