"""Build and check Dynamic Main-Chain V5 from frozen node2_004 subjects."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_retarget_v5 import (
    build_dynamic_main_chain_retarget_v5,
    save_dynamic_main_chain_subject_v5,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v5 import (
    check_dynamic_main_chain_retarget_v5,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--node2-004-root", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V5 matrix: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(),
        operator=operator,
        require_complete=False,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    captures = {
        "213328": args.capture_213328.resolve(),
        "213712": args.capture_213712.resolve(),
    }
    subjects = {}
    all_pass = True
    for label in ("213328", "213712"):
        subject, pose_map, asset = build_dynamic_main_chain_retarget_v5(
            operator=operator,
            calibration=calibration,
            whole_chain_subject_dir=args.node2_004_root.resolve() / f"subject_{label}",
            smplx_model=model,
            smplx_model_sha256=model_sha,
            oracle_path=args.oracle.resolve(),
        )
        checker = check_dynamic_main_chain_retarget_v5(
            subject,
            pose_map,
            operator=operator,
            asset=asset,
            calibration=calibration,
            smplx_model=model,
            captures=captures,
            model_path=model_path,
        )
        subject_dir = output / f"subject_{label}"
        save_dynamic_main_chain_subject_v5(
            subject_dir, subject, checker_report=checker
        )
        check_path = output / f"subject_{label}_check_v5.json"
        check_path.write_text(
            json.dumps(checker, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        subjects[label] = {
            "passed": bool(checker.get("passed")),
            "path": str(subject_dir),
            "checker": str(check_path),
            "decision": checker.get("decision"),
        }
        all_pass = all_pass and bool(checker.get("passed"))
    manifest = {
        "schema_version": 5,
        "artifact_kind": "DynamicMainChainMatrixV5",
        "baseline_commit": "142ece5f0bc646978ae3e8c9add76deea71c26a2",
        "branch_baseline_commit": "31133afba2ced3f4de01df7328d487859c7f9b05",
        "rest_bind_authority": "whole_chain_node2_004",
        "v4_solver_used": False,
        "passed": all_pass,
        "decision": (
            "accepted_for_user_genesis_review" if all_pass else "rejected_for_redesign"
        ),
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "subjects": subjects,
        "elapsed_seconds": float(time.perf_counter() - started),
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"DynamicMainChainMatrixV5 passed={all_pass} -> {output}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
