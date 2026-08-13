"""Build V11 shadow: anchored rest (prefit hinge restore) + hybrid V10 pose."""

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

from projects.genesis_ue_sync.anatomy_retarget.anchored_rest_fit_v11 import (
    ANCHORED_REST_V11_METHOD,
    CARRY_MESH_PRESETS,
    HINGE_RESTORE_PRESETS,
    build_anchored_rest_fit_v11,
)
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
from projects.genesis_ue_sync.anatomy_retarget.chain_gates_v10 import (
    evaluate_joint_contact_nonregress_v10,
    evaluate_knee_pose_containment_v10,
    evaluate_posed_body_containment_v10,
    evaluate_terminal_pose_regression_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_gates_v11 import (
    evaluate_lr_symmetry_v11,
    evaluate_pose_invariant_distance_v11,
    evaluate_rest_anatomical_anchor_v11,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    POSE_MAP_V10_COMPOSITION,
    apply_pose_map_global_v10,
    build_pose_map_v10,
    check_pose_map_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_pivot_diag_v10 import (
    evaluate_pose_pivot_diag_v10,
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
    load_whole_chain_rest_fit_v1,
)


WHOLE_CHAIN_V11_MATRIX_KIND = "WholeChainRestFitMatrixV11"


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


def _save_subject_shadow(path: Path, value, *, reports: dict) -> None:
    path.mkdir(parents=True, exist_ok=False)
    skip = {
        "source_operator_digest",
        "calibration_digest",
        "source_subject_digest",
        "smplx_model_sha256",
        "capture_sha256",
        "subject_label",
        "build_report",
    }
    arrays = {
        name: np.asarray(field)
        for name, field in value.__dict__.items()
        if name not in skip and field is not None
    }
    np.savez_compressed(path / "whole_chain_rest_fit_subject_v11.npz", **arrays)
    manifest = {
        "schema_version": 11,
        "artifact_kind": "WholeChainRestFitSubjectV11",
        "subject_label": value.subject_label,
        "source_operator_digest": value.source_operator_digest,
        "calibration_digest": value.calibration_digest,
        "source_subject_digest": value.source_subject_digest,
        "smplx_model_sha256": value.smplx_model_sha256,
        "capture_sha256": value.capture_sha256,
        "build_report": value.build_report,
        "validation_reports": reports,
        "pose_map_composition": POSE_MAP_V10_COMPOSITION,
        "rest_method": ANCHORED_REST_V11_METHOD,
        "publishable": False,
        "trusted_latest_updated": False,
    }
    _write_json(path / "manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--v7-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hinge-restore",
        choices=sorted(HINGE_RESTORE_PRESETS),
        default="full",
        help="which hinge origins to restore to prefit",
    )
    parser.add_argument(
        "--carry-mesh",
        choices=sorted(CARRY_MESH_PRESETS),
        default="none",
        help=(
            "V12a: which controllers' mesh follows the restored C_bone so the "
            "bind and the geometry describe the same joint; 'none' is V11"
        ),
    )
    parser.add_argument(
        "--subjects",
        default="213328,213712",
        help="Comma-separated subject labels (default both frozen captures)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V11 matrix: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    try:
        operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
        if operator.runtime_digest(validate=False) != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("V11 matrix requires the frozen 142 operator")
        calibration = load_anatomical_calibration_v1(
            args.calibration.expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        oracle = args.oracle.expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError("V11 matrix requires the frozen Blender oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        captures = {
            "213328": args.capture_213328.expanduser().resolve(),
            "213712": args.capture_213712.expanduser().resolve(),
        }
        capture_sha256s = {label: _sha256(path) for label, path in captures.items()}
        if capture_sha256s != FROZEN_CAPTURE_SHA256:
            raise ValueError("V11 matrix capture digests differ from the frozen pair")
        v7_baseline = args.v7_baseline.expanduser().resolve()
        recorded: dict[str, np.ndarray] = {}
        for label, capture in captures.items():
            with np.load(capture, allow_pickle=False) as data:
                recorded[f"pose_{label}"] = easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                )
        joint_domains = FrozenJointMaterialDomainsV7.freeze(
            source_bind_vertices=np.asarray(
                operator.template_asset.vertices_rest, dtype=np.float64
            ),
            faces=np.asarray(operator.template_asset.faces, dtype=np.int32),
            domains=operator.fixed_material_domains,
        )
        subject_labels = [s.strip() for s in str(args.subjects).split(",") if s.strip()]
        subjects: dict[str, dict] = {}
        for label in subject_labels:
            subject_started = time.perf_counter()
            v7_value = load_whole_chain_rest_fit_v1(
                v7_baseline / f"subject_{label}",
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
                recheck=False,
            )
            asset = materialize_subject(
                operator, betas=np.asarray(v7_value.betas), gender="male"
            ).rigged_asset
            value = build_anchored_rest_fit_v11(
                v7_value,
                asset=asset,
                calibration=calibration,
                restore_controllers=HINGE_RESTORE_PRESETS[args.hinge_restore],
                carry_mesh_controllers=CARRY_MESH_PRESETS[args.carry_mesh],
            )
            pose_map = build_pose_map_v10(
                value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            pose_report = check_pose_map_v10(pose_map, value, source_asset=asset)
            # Same-composition V7 hybrid baseline (V7 rest + V10 FK) for body/knee.
            v7_pose_map_v10 = build_pose_map_v10(
                v7_value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            # Historical contact baseline uses V7 right-multiply flex.
            v7_pose_map_rm = build_pose_map_v1(
                v7_value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            poses = {
                "tpose": np.zeros((55, 3), dtype=np.float32),
                "pose_213328": recorded["pose_213328"],
                "pose_213712": recorded["pose_213712"],
            }
            posed_globals = {
                name: apply_pose_map_global_v10(
                    pose_map, source_asset=asset, pose_axis_angle=pose
                )
                for name, pose in poses.items()
            }
            pivot_report = evaluate_pose_pivot_diag_v10(
                pose_map,
                source_asset=asset,
                calibration=calibration,
                poses=poses,
                composition=POSE_MAP_V10_COMPOSITION,
                posed_globals=posed_globals,
            )
            skin, skin_faces = smplx_body_surface_v7(
                model,
                betas=np.asarray(value.betas),
                pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
            )
            containment_report = evaluate_rest_containment_v1(
                value,
                asset=asset,
                skin_vertices=skin,
                skin_faces=skin_faces,
            )
            terminal_report = evaluate_terminal_pose_regression_v10(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses,
            )
            knee_report = evaluate_knee_pose_containment_v10(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses,
                baseline_value=v7_value,
                baseline_pose_map=v7_pose_map_v10,
                baseline_composition=POSE_MAP_V10_COMPOSITION,
            )
            body_report = evaluate_posed_body_containment_v10(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses,
                baseline_value=v7_value,
                baseline_pose_map=v7_pose_map_v10,
                baseline_composition=POSE_MAP_V10_COMPOSITION,
            )
            v7_flex, _ = pose_whole_chain_vertices(
                v7_value,
                v7_pose_map_rm,
                source_asset=asset,
                pose_axis_angle=recorded["pose_213328"],
            )
            contact_report = evaluate_joint_contact_nonregress_v10(
                value,
                domains=joint_domains,
                baseline_vertices=np.asarray(v7_value.vertices_final, dtype=np.float64),
                pose_map=pose_map,
                source_asset=asset,
                flex_pose_axis_angle=recorded["pose_213328"],
                baseline_flex_vertices=np.asarray(v7_flex, dtype=np.float64),
            )
            anchor_report = evaluate_rest_anatomical_anchor_v11(
                value,
                asset=asset,
                calibration=calibration,
                baseline_value=v7_value,
            )
            symmetry_report = evaluate_lr_symmetry_v11(value, asset=asset)
            ed_report = evaluate_pose_invariant_distance_v11(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses,
            )
            hard_reports = {
                "pose_map_v10": pose_report,
                "pose_pivot_diag_v10": pivot_report,
                "containment": containment_report,
                "terminal_pose_regression_v10": terminal_report,
                "knee_pose_containment_v10": knee_report,
                "posed_body_containment_v10": body_report,
                "joint_contact_nonregress_v10": contact_report,
                "rest_anatomical_anchor_v11": anchor_report,
                "lr_symmetry_v11": symmetry_report,
                "pose_invariant_distance_v11": ed_report,
            }
            must_pass = {
                "pose_map_v10": pose_report,
                "pose_pivot_diag_v10": pivot_report,
                "terminal_pose_regression_v10": terminal_report,
                "joint_contact_nonregress_v10": contact_report,
                "rest_anatomical_anchor_v11": anchor_report,
                "lr_symmetry_v11": symmetry_report,
                "knee_pose_containment_v10": knee_report,
                "posed_body_containment_v10": body_report,
                "pose_invariant_distance_v11": ed_report,
            }
            failed = [
                name for name, report in must_pass.items() if not report.get("passed")
            ]
            if failed:
                _write_json(
                    temporary / f"subject_{label}_FAILED_reports.json",
                    hard_reports,
                )
                raise ValueError(
                    f"V11 subject {label} failed automatic gates: {failed}"
                )
            subject_path = temporary / f"subject_{label}"
            _save_subject_shadow(subject_path, value, reports=hard_reports)
            for name, report in hard_reports.items():
                _write_json(temporary / f"subject_{label}_{name}_check.json", report)
            flex_knee = knee_report["cells"]["pose_213328"]
            body_flex = body_report["cells"]["pose_213328"]["groups"]
            subjects[label] = {
                "path": f"subject_{label}",
                "passed": True,
                "end_to_end_seconds": float(time.perf_counter() - subject_started),
                "max_predicted_pivot_error_m": pivot_report["max_predicted_error_m"],
                "max_hinge_seating_error_m": pivot_report["max_hinge_seating_error_m"],
                "flex_candidate_worst_outside_m": flex_knee["candidate_worst_outside_m"],
                "flex_baseline_worst_outside_m": flex_knee["baseline_worst_outside_m"],
                "forearm_L_area_inside": body_flex["forearm_L"][
                    "mean_candidate_area_inside"
                ],
                "shank_L_area_inside": body_flex["shank_L"]["mean_candidate_area_inside"],
                "patella_L_area_inside": body_flex["patella_L"][
                    "mean_candidate_area_inside"
                ],
                "forearm_L_delta_vs_hybrid": body_flex["forearm_L"]["area_inside_delta"],
                "shank_L_delta_vs_hybrid": body_flex["shank_L"]["area_inside_delta"],
                "patella_L_delta_vs_hybrid": body_flex["patella_L"]["area_inside_delta"],
            }
        matrix = {
            "schema_version": 11,
            "artifact_kind": WHOLE_CHAIN_V11_MATRIX_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "accepted_scope": "full_main_chain_shadow_v11",
            "source_operator_digest": operator.runtime_digest(validate=False),
            "calibration_digest": _calibration_content_digest(calibration),
            "blender_oracle_sha256": oracle_sha,
            "smplx_gender": "male",
            "smplx_model_sha256": model_sha,
            "capture_sha256s": capture_sha256s,
            "subjects": subjects,
            "terminal_policy": "identity_142_hand_foot",
            "pose_map_composition": POSE_MAP_V10_COMPOSITION,
            "rest_method": ANCHORED_REST_V11_METHOD,
            "hinge_restore_preset": str(args.hinge_restore),
            "carry_mesh_preset": str(args.carry_mesh),
            "v7_baseline": str(v7_baseline),
            "publishable": False,
            "trusted_latest_updated": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(temporary / "manifest.json", matrix)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"{WHOLE_CHAIN_V11_MATRIX_KIND} passed=true subjects={len(subjects)} "
        f"seconds={matrix['elapsed_seconds']:.3f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
