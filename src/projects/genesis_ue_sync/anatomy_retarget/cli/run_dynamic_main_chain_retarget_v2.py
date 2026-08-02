#!/usr/bin/env python3
"""Build exact saved Male Dynamic Main-Chain Retarget V2 subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_retarget_v2 import (
    build_dynamic_main_chain_retarget_v2,
    save_dynamic_main_chain_subject_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v2 import (
    run_dynamic_main_chain_validation_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import build_pose_map_v1
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 matrix: {output}")
    operator_path = args.operator.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    oracle = args.oracle.expanduser().resolve()
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    operator = load_source_operator(operator_path, mmap=True)
    calibration = load_anatomical_calibration_v1(
        calibration_path, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(model_path)
    captures = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    betas: dict[str, np.ndarray] = {}
    poses: dict[str, np.ndarray] = {}
    for label, capture in captures.items():
        with np.load(capture, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
            poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
    output.mkdir(parents=True)
    subjects: dict[str, object] = {}
    passed = True
    for label in ("213328", "213712"):
        value = build_dynamic_main_chain_retarget_v2(
            operator,
            calibration,
            betas=betas[label],
            subject_label=label,
            capture_sha256=_sha256(captures[label]),
            smplx_model=model,
            smplx_model_sha256=model_sha,
            recorded_poses=poses,
        )
        asset = materialize_subject(operator, betas=betas[label], gender="male").rigged_asset
        pose_map = build_pose_map_v1(
            value,
            asset=asset,
            calibration=calibration,
            oracle_path=oracle,
            source_operator_digest=operator.runtime_digest(validate=False),
        )
        validation = run_dynamic_main_chain_validation_v2(
            value,
            pose_map,
            asset=asset,
            smplx_model=model,
            recorded_poses=poses,
        )
        subject_path = output / f"subject_{label}"
        save_dynamic_main_chain_subject_v2(
            subject_path,
            value,
            provenance={
                "operator_manifest_sha256": _sha256(operator_path / "manifest.json"),
                "calibration_manifest_sha256": _sha256(calibration_path / "manifest.json"),
                "oracle_sha256": _sha256(oracle),
                "capture_sha256s": {
                    key: _sha256(path) for key, path in captures.items()
                },
                "smplx_model_sha256": model_sha,
            },
        )
        check_path = output / f"subject_{label}_dynamic_v2.json"
        check_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        subjects[label] = {
            "path": subject_path.name,
            "dynamic_check": check_path.name,
            "passed": bool(validation["passed"]),
            "build_seconds": value.build_report["elapsed_seconds"],
        }
        passed = passed and bool(validation["passed"])
    manifest = {
        "schema_version": 2,
        "artifact_kind": "DynamicMainChainMatrixV2",
        "subjects": subjects,
        "passed": bool(passed),
        "accepted_scope": "full_main_chain_shadow_v2",
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"DynamicMainChainMatrixV2 passed={str(passed).lower()} -> {output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
