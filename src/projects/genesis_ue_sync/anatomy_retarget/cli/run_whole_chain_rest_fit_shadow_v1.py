"""Build the authenticated two-beta full-main-chain shadow matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    evaluate_rest_containment_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_chain_validation_v1 import (
    run_dynamic_chain_validation_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    check_pose_map_v1,
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
    BASELINE_COMMIT,
    FROZEN_CAPTURE_SHA256,
    WHOLE_CHAIN_MATRIX_KIND,
    build_whole_chain_rest_fit_v1,
    check_whole_chain_rest_fit_v1,
    save_whole_chain_rest_fit_v1,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite whole-chain matrix: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    try:
        operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
        if operator.runtime_digest(validate=False) != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("whole-chain matrix requires the frozen 142 operator")
        calibration = load_anatomical_calibration_v1(
            args.calibration.expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        oracle = args.oracle.expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError("whole-chain matrix requires the frozen Blender oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        captures = {
            "213328": args.capture_213328.expanduser().resolve(),
            "213712": args.capture_213712.expanduser().resolve(),
        }
        capture_sha256s = {label: _sha256(path) for label, path in captures.items()}
        if capture_sha256s != FROZEN_CAPTURE_SHA256:
            raise ValueError("whole-chain matrix capture digests differ from the frozen pair")
        betas: dict[str, np.ndarray] = {}
        recorded: dict[str, np.ndarray] = {}
        for label, capture in captures.items():
            with np.load(capture, allow_pickle=False) as data:
                betas[label] = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
                recorded[f"pose_{label}"] = easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                )

        subjects: dict[str, dict] = {}
        for label in ("213328", "213712"):
            subject_started = time.perf_counter()
            value = build_whole_chain_rest_fit_v1(
                operator,
                calibration,
                betas=betas[label],
                subject_label=label,
                capture_sha256=capture_sha256s[label],
                smplx_model=model,
                smplx_model_sha256=model_sha,
            )
            rest_report = check_whole_chain_rest_fit_v1(
                value,
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
            )
            asset = materialize_subject(
                operator, betas=betas[label], gender="male"
            ).rigged_asset
            pose_map = build_pose_map_v1(
                value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            pose_report = check_pose_map_v1(pose_map, value, source_asset=asset)
            dynamic_report = run_dynamic_chain_validation_v1(
                value,
                pose_map,
                asset=asset,
                calibration=calibration,
                recorded_poses=recorded,
            )
            skin, skin_faces = smplx_body_surface_v7(
                model,
                betas=betas[label],
                pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
            )
            containment_report = evaluate_rest_containment_v1(
                value,
                asset=asset,
                skin_vertices=skin,
                skin_faces=skin_faces,
            )
            reports = {
                "pose_map": pose_report,
                "dynamic": dynamic_report,
                "containment": containment_report,
            }
            if not rest_report["passed"] or not all(
                report["passed"] for report in reports.values()
            ):
                raise ValueError(f"whole-chain subject {label} failed automatic gates")
            subject_path = temporary / f"subject_{label}"
            save_whole_chain_rest_fit_v1(
                subject_path,
                value,
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
                capture_sha256s=capture_sha256s,
                blender_oracle_sha256=oracle_sha,
                validation_reports=reports,
            )
            for name, report in {"rest_fit": rest_report, **reports}.items():
                _write_json(temporary / f"subject_{label}_{name}_check.json", report)
            subjects[label] = {
                "path": f"subject_{label}",
                "passed": True,
                "build_seconds": rest_report["build_seconds"],
                "end_to_end_seconds": float(time.perf_counter() - subject_started),
            }
        matrix = {
            "schema_version": 1,
            "artifact_kind": WHOLE_CHAIN_MATRIX_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "accepted_scope": "full_main_chain_shadow",
            "source_operator_digest": operator.runtime_digest(validate=False),
            "calibration_digest": _calibration_content_digest(calibration),
            "blender_oracle_sha256": oracle_sha,
            "smplx_gender": "male",
            "smplx_model_sha256": model_sha,
            "capture_sha256s": capture_sha256s,
            "subjects": subjects,
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
            "tube_transport_application_count": 1,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(temporary / "manifest.json", matrix)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"{WHOLE_CHAIN_MATRIX_KIND} passed=true subjects=2 "
        f"seconds={matrix['elapsed_seconds']:.3f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
