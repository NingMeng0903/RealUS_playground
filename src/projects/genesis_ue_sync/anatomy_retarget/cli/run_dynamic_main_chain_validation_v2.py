#!/usr/bin/env python3
"""Run identity-bound posed-male-skin validation for one whole-chain subject."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import _content_digest
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v2 import (
    run_dynamic_main_chain_validation_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_map_content_digest,
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
    build_whole_chain_rest_fit_v1,
    load_whole_chain_rest_fit_v1,
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subject", type=Path)
    source.add_argument("--build-current-subject", choices=("213328", "213712"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 validation: {output}")
    operator_path = args.operator.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    oracle_path = args.oracle.expanduser().resolve()
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    capture_paths = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    operator = load_source_operator(operator_path, mmap=True)
    calibration = load_anatomical_calibration_v1(
        calibration_path,
        operator=operator,
        required_scope="full_main_chain",
    )
    model = load_smplx_model_v7(model_path)
    betas: dict[str, np.ndarray] = {}
    poses: dict[str, np.ndarray] = {}
    for label, path in capture_paths.items():
        with np.load(path, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"]).reshape(-1)[:10]
            poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
    if args.subject is not None:
        subject_path = args.subject.expanduser().resolve()
        value = load_whole_chain_rest_fit_v1(
            subject_path,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
            recheck=False,
        )
        source = {
            "mode": "loaded_exact_subject",
            "subject_path": str(subject_path),
            "subject_manifest_sha256": _sha256(subject_path / "manifest.json"),
            "subject_npz_sha256": _sha256(
                subject_path / "whole_chain_rest_fit_subject_v1.npz"
            ),
        }
    else:
        label = str(args.build_current_subject)
        value = build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas[label],
            subject_label=label,
            capture_sha256=_sha256(capture_paths[label]),
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        source = {"mode": "current_in_memory_build"}
    asset = materialize_subject(
        operator, betas=value.betas, gender="male"
    ).rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle_path,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    validation = run_dynamic_main_chain_validation_v2(
        value,
        pose_map,
        asset=asset,
        smplx_model=model,
        recorded_poses=poses,
    )
    package = {
        "schema_version": 2,
        "artifact_kind": "IdentityBoundDynamicMainChainValidationV2",
        "passed": bool(validation["passed"]),
        "subject_label": str(value.subject_label),
        "subject_content_digest": _content_digest(value),
        "pose_map_content_digest": pose_map_content_digest(pose_map),
        "source": source,
        "provenance": {
            "smplx_gender": "male",
            "smplx_model_sha256": model_sha,
            "capture_sha256": {
                label: _sha256(path) for label, path in capture_paths.items()
            },
            "operator_manifest_sha256": _sha256(operator_path / "manifest.json"),
            "calibration_manifest_sha256": _sha256(
                calibration_path / "manifest.json"
            ),
            "oracle_sha256": _sha256(oracle_path),
            "implementation_sha256": {
                path.name: _sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1]
                    / "dynamic_main_chain_validation_v2.py",
                    Path(__file__).resolve().parents[1] / "pose_map_v1.py",
                    Path(__file__).resolve().parents[1] / "whole_chain_rest_fit_v1.py",
                    Path(__file__).resolve().parents[1] / "chain_rest_fit_v1.py",
                )
            },
        },
        "validation": validation,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(package, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"IdentityBoundDynamicMainChainValidationV2 passed={package['passed']} -> {output}",
        flush=True,
    )
    return 0 if package["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
