"""Emit the permanent pivot-offset / hinge-seating diagnostic for V7 (baseline)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import build_pose_map_v1
from projects.genesis_ue_sync.anatomy_retarget.pose_pivot_diag_v10 import (
    evaluate_right_multiply_baseline_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--v7-shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", default="213328")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    label = str(args.subject)
    value = load_whole_chain_rest_fit_v1(
        args.v7_shadow.expanduser().resolve() / f"subject_{label}",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    asset = materialize_subject(operator, betas=np.asarray(value.betas), gender="male").rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=args.oracle.expanduser().resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    with np.load(args.capture_213328.expanduser().resolve(), allow_pickle=False) as data:
        pose_213328 = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        )
    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": pose_213328,
    }
    report = evaluate_right_multiply_baseline_v10(
        value,
        pose_map,
        source_asset=asset,
        calibration=calibration,
        poses=poses,
    )
    report["subject_label"] = label
    report["v7_shadow"] = str(args.v7_shadow.expanduser().resolve())
    report["elapsed_seconds_wall"] = float(time.perf_counter() - started)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"PosePivotDiagV10 composition=right_multiply_bind "
        f"passed={report['passed']} max_err_m={report['max_predicted_error_m']:.4f} "
        f"max_hinge_m={report['max_hinge_seating_error_m']:.4f} -> {output}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
