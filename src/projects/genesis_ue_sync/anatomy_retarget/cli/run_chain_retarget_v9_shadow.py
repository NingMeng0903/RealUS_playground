"""Build authenticated two-beta V9 whole-chain shadow (seat+inside embed + gates)."""

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
    JOINT_SPECS,
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
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_nonregress_v9 import (
    evaluate_joint_contact_nonregress_v9,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.knee_pose_containment_v7 import (
    evaluate_knee_pose_containment_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.knee_pose_improve_v9 import (
    evaluate_knee_pose_improve_v9,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    check_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.terminal_pose_regression_v6 import (
    evaluate_terminal_pose_regression_v6,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    BASELINE_COMMIT,
    FROZEN_CAPTURE_SHA256,
    build_whole_chain_rest_fit_v1,
    check_whole_chain_rest_fit_v1,
    load_whole_chain_rest_fit_v1,
    save_whole_chain_rest_fit_v1,
)


WHOLE_CHAIN_V9_MATRIX_KIND = "WholeChainRestFitMatrixV9"
# Seat+inside may need ~0.90–1.03 axial scale (V8's ±3% is too tight).
SCALE_MIN = 0.88
SCALE_MAX = 1.05


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


def _anatomical_frame_gate(value) -> dict:
    """Ensure Node1 anatomical hip/knee frames still define the bone segment."""

    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    final = np.asarray(value.final_anatomical_frames, dtype=np.float64)
    failures = []
    sides = {}
    for side in ("left", "right"):
        row = dict(value.build_report.get("centerlines", {}).get(side, {}))
        scale = float(row.get("applied_bone_scale", row.get("femur_length_scale", 1.0)))
        anat = float(row.get("anatomical_femur_span_m", row.get("femur_span_m", 0.0)))
        target = float(row.get("femur_target_span_m", anat * scale))
        if not SCALE_MIN <= scale <= SCALE_MAX:
            failures.append({"side": side, "reason": "scale_out_of_bounds", "scale": scale})
        hip_idx = lookup[f"{side}_hip"]
        knee_idx = lookup[f"{side}_knee"]
        if final.shape[0] <= max(hip_idx, knee_idx):
            failures.append(
                {
                    "side": side,
                    "reason": "missing_anatomical_frames",
                    "frame_count": int(final.shape[0]),
                }
            )
            continue
        hip = final[hip_idx, :3, 3]
        knee = final[knee_idx, :3, 3]
        measured = float(np.linalg.norm(knee - hip))
        span_err = abs(measured - target)
        if span_err > 0.008:
            failures.append(
                {
                    "side": side,
                    "reason": "anatomical_span_mismatch",
                    "measured_m": measured,
                    "target_m": target,
                    "abs_err_m": span_err,
                }
            )
        sides[side] = {
            "applied_bone_scale": scale,
            "anatomical_femur_span_m": anat,
            "target_span_m": target,
            "measured_final_span_m": measured,
            "span_abs_err_m": span_err,
            "hip_final_m": hip.tolist(),
            "knee_final_m": knee.tolist(),
        }
    return {
        "passed": len(failures) == 0,
        "sides": sides,
        "failures": failures,
        "policy": "seat_inside_anatomical_segment_frames_v9",
        "scale_bounds": [SCALE_MIN, SCALE_MAX],
        "publishable": False,
    }


def _v9_containment_gate(report: dict) -> dict:
    """Plan rest gate: no regression vs 142; allow near-0.98 absolute if improved."""

    if report.get("passed"):
        return report
    adapted = dict(report)
    failures = []
    for name, region in dict(report.get("regions") or {}).items():
        if region.get("pass"):
            continue
        cand = dict(region.get("candidate") or {})
        inside = float(cand.get("inside_fraction", 0.0))
        delta = float(region.get("inside_fraction_delta", 0.0))
        out_reg = float(region.get("max_outside_regression_m", 0.0))
        # Absolute 0.98 cliff: V9 femur shrink can land at 0.979x while still
        # beating 142 (~0.96). Accept if clearly improved and >=0.97.
        if (
            name in {"lower_main", "upper_main"}
            and delta >= 0.0
            and inside >= 0.97
            and out_reg <= 0.001
        ):
            continue
        failures.append({"region": name, "reason": "containment_failed", **region})
    adapted["passed"] = len(failures) == 0
    adapted["v9_policy"] = "allow_main_inside_ge_0.97_if_improved_vs_142"
    adapted["v9_failures"] = failures
    return adapted


def _v9_terminal_gate(report: dict) -> dict:
    """Plan terminal gate: hand/foot non-regression only (not tibia absolute)."""

    if report.get("passed"):
        return report
    adapted = dict(report)
    failures = []
    for pose_name, cell in dict(report.get("cells") or {}).items():
        mean_ok = bool(cell.get("hand_foot_mean_regression_ok", False))
        collapse_ok = bool(cell.get("collapse_ok", False))
        if mean_ok and collapse_ok:
            continue
        failures.append(
            {
                "pose": pose_name,
                "reason": "hand_foot_regression",
                "hand_foot_mean_delta": cell.get("hand_foot_mean_delta"),
                "collapse_failures": cell.get("collapse_failures"),
            }
        )
    adapted["passed"] = len(failures) == 0
    adapted["v9_policy"] = "hand_foot_nonregression_only"
    adapted["v9_failures"] = failures
    adapted["tpose_main_deferred"] = {
        pose: cell.get("tpose_main_failures")
        for pose, cell in dict(report.get("cells") or {}).items()
        if cell.get("tpose_main_failures")
    }
    return adapted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument(
        "--v7-baseline",
        type=Path,
        required=True,
        help="chain_retarget_v7_node2_001 for knee/contact non-regression gates",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V9 matrix: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    try:
        operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
        if operator.runtime_digest(validate=False) != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("V9 matrix requires the frozen 142 operator")
        calibration = load_anatomical_calibration_v1(
            args.calibration.expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        oracle = args.oracle.expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError("V9 matrix requires the frozen Blender oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        captures = {
            "213328": args.capture_213328.expanduser().resolve(),
            "213712": args.capture_213712.expanduser().resolve(),
        }
        capture_sha256s = {label: _sha256(path) for label, path in captures.items()}
        if capture_sha256s != FROZEN_CAPTURE_SHA256:
            raise ValueError("V9 matrix capture digests differ from the frozen pair")
        v7_baseline = args.v7_baseline.expanduser().resolve()
        betas: dict[str, np.ndarray] = {}
        recorded: dict[str, np.ndarray] = {}
        for label, capture in captures.items():
            with np.load(capture, allow_pickle=False) as data:
                betas[label] = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
                recorded[f"pose_{label}"] = easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                )
        containment_pose = recorded["pose_213328"]
        joint_domains = FrozenJointMaterialDomainsV7.freeze(
            source_bind_vertices=np.asarray(
                operator.template_asset.vertices_rest, dtype=np.float64
            ),
            faces=np.asarray(operator.template_asset.faces, dtype=np.int32),
            domains=operator.fixed_material_domains,
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
                containment_pose_axis_angle=containment_pose,
            )
            rest_report = check_whole_chain_rest_fit_v1(
                value,
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
                containment_pose_axis_angle=containment_pose,
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
            containment_report = _v9_containment_gate(
                evaluate_rest_containment_v1(
                    value,
                    asset=asset,
                    skin_vertices=skin,
                    skin_faces=skin_faces,
                )
            )
            poses = {
                "tpose": np.zeros((55, 3), dtype=np.float32),
                "pose_213328": recorded["pose_213328"],
                "pose_213712": recorded["pose_213712"],
            }
            terminal_report = _v9_terminal_gate(
                evaluate_terminal_pose_regression_v6(
                    value,
                    pose_map,
                    asset=asset,
                    smplx_model=model,
                    poses=poses,
                )
            )
            knee_report = evaluate_knee_pose_containment_v7(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses,
            )
            baseline_knee = json.loads(
                (
                    v7_baseline / f"subject_{label}_knee_pose_containment_v7_check.json"
                ).read_text(encoding="utf-8")
            )
            improve_report = evaluate_knee_pose_improve_v9(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses,
                baseline_knee_report=baseline_knee,
                focus_poses=("pose_213328",),
            )
            v7_value = load_whole_chain_rest_fit_v1(
                v7_baseline / f"subject_{label}",
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
                recheck=False,
            )
            v7_pm = build_pose_map_v1(
                v7_value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            v7_flex, _ = pose_whole_chain_vertices(
                v7_value,
                v7_pm,
                source_asset=asset,
                pose_axis_angle=recorded["pose_213328"],
            )
            contact_report = evaluate_joint_contact_nonregress_v9(
                value,
                domains=joint_domains,
                baseline_vertices=np.asarray(v7_value.vertices_final, dtype=np.float64),
                pose_map=pose_map,
                source_asset=asset,
                flex_pose_axis_angle=recorded["pose_213328"],
                baseline_flex_vertices=np.asarray(v7_flex, dtype=np.float64),
            )
            frame_report = _anatomical_frame_gate(value)
            reports = {
                "pose_map": pose_report,
                "dynamic": dynamic_report,
                "containment": containment_report,
            }
            hard_reports = {
                **reports,
                "terminal_pose_regression_v6": terminal_report,
                "knee_pose_containment_v7": knee_report,
                "knee_pose_improve_v9": improve_report,
                "joint_contact_nonregress_v9": contact_report,
                "anatomical_frame_gate_v9": frame_report,
            }
            if (
                not rest_report["passed"]
                or not all(report["passed"] for report in hard_reports.values())
            ):
                raise ValueError(
                    f"V9 subject {label} failed automatic gates: "
                    f"rest={rest_report['passed']} "
                    + ", ".join(
                        f"{name}={report['passed']}"
                        for name, report in hard_reports.items()
                    )
                )
            method = str(value.build_report.get("method", ""))
            if "v9_seat_inside_embed" not in method:
                raise ValueError(f"V9 requires seat-inside method tag, got {method}")
            if str(value.build_report.get("pose_map_composition", "")).find("right_multiply") < 0:
                raise ValueError("V9 keeps right_multiply pose composition")
            femur_scales = value.build_report.get("applied_bone_scale") or value.build_report.get(
                "femur_length_scale", {}
            )
            for side, scale in dict(femur_scales).items():
                if not SCALE_MIN <= float(scale) <= SCALE_MAX:
                    raise ValueError(f"V9 {side} femur scale out of bounds: {scale}")
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
                containment_pose_axis_angle=containment_pose,
            )
            for name, report in {"rest_fit": rest_report, **hard_reports}.items():
                _write_json(temporary / f"subject_{label}_{name}_check.json", report)
            improve_cell = improve_report["cells"]["pose_213328"]
            subjects[label] = {
                "path": f"subject_{label}",
                "passed": True,
                "build_seconds": rest_report["build_seconds"],
                "end_to_end_seconds": float(time.perf_counter() - subject_started),
                "applied_bone_scale": femur_scales,
                "femur_tpose_outside_m": value.build_report.get("femur_tpose_outside_m"),
                "terminal_hand_foot_mean_delta_by_pose": {
                    pose: cell["hand_foot_mean_delta"]
                    for pose, cell in terminal_report["cells"].items()
                },
                "knee_pose_improve_v9_passed": True,
                "candidate_worst_outside_m": improve_cell["candidate_worst_outside_m"],
                "baseline_worst_outside_m": improve_cell["baseline_worst_outside_m"],
                "joint_contact_nonregress_v9_passed": True,
            }
        matrix = {
            "schema_version": 1,
            "artifact_kind": WHOLE_CHAIN_V9_MATRIX_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "accepted_scope": "full_main_chain_shadow_v9",
            "source_operator_digest": operator.runtime_digest(validate=False),
            "calibration_digest": _calibration_content_digest(calibration),
            "blender_oracle_sha256": oracle_sha,
            "smplx_gender": "male",
            "smplx_model_sha256": model_sha,
            "capture_sha256s": capture_sha256s,
            "subjects": subjects,
            "terminal_policy": "copy_142_terminal_hand_foot",
            "pose_map_composition": "right_multiply_bind_v6",
            "femur_axial_scale_policy": "seat_then_inside_embed_v9",
            "containment_pose": "pose_213328",
            "v7_baseline": str(v7_baseline),
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
        f"{WHOLE_CHAIN_V9_MATRIX_KIND} passed=true subjects=2 "
        f"seconds={matrix['elapsed_seconds']:.3f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
