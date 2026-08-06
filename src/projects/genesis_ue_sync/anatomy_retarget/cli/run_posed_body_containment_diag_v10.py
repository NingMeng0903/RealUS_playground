"""V7 vs V10 posed bone-mesh containment diagnosis (grouped absolute metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_gates_v10 import (
    evaluate_posed_body_containment_v10,
    evaluate_terminal_pose_regression_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import build_pose_map_v1
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import build_pose_map_v10
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v10_artifacts import (
    load_chain_retarget_v10_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--v7-baseline", type=Path, required=True)
    parser.add_argument("--v10-shadow", type=Path, required=True)
    parser.add_argument("--subject", default="213328")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    label = str(args.subject).strip()
    v7 = load_whole_chain_rest_fit_v1(
        args.v7_baseline.expanduser().resolve() / f"subject_{label}",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    v10, _manifest = load_chain_retarget_v10_subject(
        args.v10_shadow.expanduser().resolve() / f"subject_{label}"
    )
    asset = materialize_subject(
        operator, betas=np.asarray(v10.betas), gender="male"
    ).rigged_asset
    oracle = args.oracle.expanduser().resolve()
    digest = operator.runtime_digest(validate=False)
    pose_map = build_pose_map_v10(
        v10,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=digest,
    )
    v7_pose_map = build_pose_map_v1(
        v7,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=digest,
    )
    poses = {"tpose": np.zeros((55, 3), dtype=np.float32)}
    for lab, path in (
        ("213328", args.capture_213328),
        ("213712", args.capture_213712),
    ):
        with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
            poses[f"pose_{lab}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
    body = evaluate_posed_body_containment_v10(
        v10,
        pose_map,
        asset=asset,
        smplx_model=model,
        poses=poses,
        baseline_value=v7,
        baseline_pose_map=v7_pose_map,
    )
    terminal = evaluate_terminal_pose_regression_v10(
        v10,
        pose_map,
        asset=asset,
        smplx_model=model,
        poses=poses,
    )
    report = {
        "subject": label,
        "v7_baseline": str(args.v7_baseline.expanduser().resolve()),
        "v10_shadow": str(args.v10_shadow.expanduser().resolve()),
        "posed_body_containment_v10": body,
        "terminal_pose_regression_v10": terminal,
        "passed": bool(body.get("passed") and terminal.get("passed")),
    }
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"posed_body_containment_diag passed={report['passed']} "
        f"body={body['passed']} terminal={terminal['passed']} -> {out}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
