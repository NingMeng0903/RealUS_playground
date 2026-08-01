"""Run independent two-beta T-pose containment checks for a shadow chain fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    evaluate_rest_containment_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    build_whole_chain_rest_fit_v1,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite containment report: {output}")
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(), operator=operator, required_scope="lower_chain"
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    captures = {
        "213328": args.capture_213328.resolve(),
        "213712": args.capture_213712.resolve(),
    }
    subjects = {}
    for label, capture in captures.items():
        with np.load(capture, allow_pickle=False) as data:
            betas = np.asarray(data["shapes"]).reshape(-1)[:10]
        value = build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas,
            subject_label=label,
            capture_sha256=_sha(capture),
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
        skin, faces = smplx_body_surface_v7(
            model, betas=betas, pose_axis_angle=np.zeros((55, 3), dtype=np.float64)
        )
        subjects[label] = evaluate_rest_containment_v1(
            value, asset=asset, skin_vertices=skin, skin_faces=faces
        )
    report = {
        "schema_version": 1,
        "artifact_kind": "ChainRestContainmentMatrixV1",
        "passed": bool(all(item["passed"] for item in subjects.values())),
        "subjects": subjects,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
