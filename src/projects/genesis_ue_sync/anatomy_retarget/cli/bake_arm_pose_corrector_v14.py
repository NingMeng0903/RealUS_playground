"""Fit and bake a bounded left-arm PoseCorrectorV14 from explicit geometry NPZs.

The command consumes a previously compiled V14 subject and a directory of
pose/skin/joints NPZs.  ``--fit-pose`` is the only source of training names;
all other discovered names are regression exports.  No AMASS loader or
implicit pose source is used here.

The resulting directory contains the original source pack inside the saved
compiled package, a pickle-free ``pose_corrector.npz``, per-pose
before/after Genesis-compatible NPZs, and a fail-closed diagnostics manifest.
An unsupported regression pose is exported with an empty after array and an
explicit ``status=unsupported``; it is never replaced by a zero correction.
"""

from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
from ..consistent_runtime_v14 import load_compiled_subject
from ..pose_corrector_v14 import PoseCorrectionSupportError, PoseCorrectorV14
from ..v8_artifacts import load_source_operator
from ..arm_pose_fit_v14 import (
    ARM_CONTROLLER_IDS,
    ARM_MAX_ROTATION_NORM,
    ARM_MAX_TRANSLATION_NORM,
    ARM_RBF_RADIUS,
    ARM_SELECTED_JOINT_IDS,
    ArmPoseGeometryV14,
    ArmPoseFitResultV14,
    cache_arm_pose_frames_v14,
    corrector_twists_for_pose_v14,
    evaluate_arm_twists_v14,
    fit_arm_pose_corrector_v14,
    load_pose_geometry_directory,
    write_json,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1"


def _names(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        tokens = value if isinstance(value, (list, tuple)) else [value]
        for token in tokens:
            result.extend(part.strip() for part in str(token).split(",") if part.strip())
    if len(set(result)) != len(result):
        raise ValueError(f"pose names contain duplicates: {result}")
    return result


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_strict_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _zero_corrector(poses: dict[str, np.ndarray]) -> PoseCorrectorV14:
    values = np.asarray([poses[name] for name in poses], dtype=np.float64)
    zeros = np.zeros((len(values), len(ARM_CONTROLLER_IDS), 6), dtype=np.float64)
    return PoseCorrectorV14.fit(
        values,
        ARM_SELECTED_JOINT_IDS,
        ARM_CONTROLLER_IDS,
        zeros,
        radius=ARM_RBF_RADIUS,
        max_rotation_norm=ARM_MAX_ROTATION_NORM,
        max_translation_norm=ARM_MAX_TRANSLATION_NORM,
    )


def _base_vertices(compiled: Any, pose: np.ndarray) -> np.ndarray:
    """Evaluate the saved compiled package without a corrector."""
    base = copy(compiled)
    base.corrector = None
    return np.asarray(base.apply_pose(pose), dtype=np.float32)


def _save_pose_npz(
    path: Path,
    *,
    compiled: Any,
    geometry: ArmPoseGeometryV14,
    frame: Any,
    before: np.ndarray,
    after: np.ndarray,
    status: str,
    role: str,
    correction_twists: np.ndarray | None,
    error: str | None,
) -> None:
    asset = compiled.source_asset
    if geometry.source_vertices is not None and geometry.source_vertices.shape == before.shape:
        source_vertices = np.asarray(geometry.source_vertices, dtype=np.float32)
    else:
        source_vertices = np.asarray(before, dtype=np.float32)
    if geometry.vertex_tissue is not None and len(geometry.vertex_tissue) == len(before):
        tissue = np.asarray(geometry.vertex_tissue)
    else:
        tissue = np.zeros(len(before), dtype=np.int8)
    output_after = np.asarray(after, dtype=np.float32)
    metadata = {
        "artifact_kind": "ArmPoseGeometryV14",
        "status": status,
        "role": role,
        "pose_name": geometry.name,
        "error": error,
        "unsupported_is_not_zero_fallback": status == "unsupported",
        "source_global_cached": True,
        "base_target_local_cached": True,
        "controller_ids": ARM_CONTROLLER_IDS.tolist(),
        "selected_joint_ids": ARM_SELECTED_JOINT_IDS.tolist(),
    }
    arrays = {
        "faces": np.asarray(asset.faces, dtype=np.int32),
        "source_vertices": source_vertices,
        "before_vertices": np.asarray(before, dtype=np.float32),
        "candidate_vertices": output_after,
        "after_vertices": output_after,
        "skin_vertices": np.asarray(geometry.skin_vertices, dtype=np.float32),
        "skin_faces": np.asarray(geometry.skin_faces, dtype=np.int32),
        "smplx_joints": (
            np.asarray(geometry.smplx_joints, dtype=np.float32)
            if geometry.smplx_joints is not None else np.zeros((55, 3), dtype=np.float32)
        ),
        "pose": np.asarray(geometry.pose, dtype=np.float32),
        "vertex_tissue": tissue,
        "source_global": np.asarray(frame.source_global, dtype=np.float64),
        "base_target_local": np.asarray(frame.base_target_local, dtype=np.float64),
        "correction_twists": (
            np.asarray(correction_twists, dtype=np.float64)
            if correction_twists is not None else np.empty((0, 6), dtype=np.float64)
        ),
        "correction_controller_ids": np.asarray(ARM_CONTROLLER_IDS, dtype=np.int64),
        "pose_name": np.asarray(geometry.name),
        "subject": np.asarray(str(getattr(compiled, "provenance", {}).get("subject", ""))),
        "status": np.asarray(status),
        "role": np.asarray(role),
        "metadata_json": np.asarray(json.dumps(_strict_json(metadata), sort_keys=True, allow_nan=False)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True, help="saved CompiledAnatomyV14 directory")
    parser.add_argument("--geometry-dir", type=Path, required=True, help="directory containing explicit pose/skin/joints NPZs")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--fit-pose", action="append", nargs="+", required=True, help="training pose name(s), repeat or comma-separate")
    parser.add_argument("--regression-pose", action="append", nargs="+", help="optional regression pose name(s), repeat or comma-separate")
    parser.add_argument("--subject", help="filter geometry NPZs by their scalar subject field")
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--maxfev", type=int, default=300)
    parser.add_argument(
        "--optimizer",
        choices=("powell", "slsqp", "powell_slsqp"),
        default="powell_slsqp",
        help="bounded pose optimizer; powell_slsqp warm-starts normalized SLSQP from Powell",
    )
    parser.add_argument(
        "--slsqp-maxiter",
        type=int,
        default=60,
        help="maximum normalized SLSQP iterations per non-neutral fit pose",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    output = Path(args.output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "artifact_kind": "ArmPoseCorrectorBakeV14",
        "schema_version": 14,
        "status": "running",
        "publishable": False,
        "anatomical_passed": False,
        "acceptance_scope": {
            "fit_metrics": "frozen FIT arm domains plus every left-hand bone vertex",
            "skin_threshold": "vertex signed-distance excess over 1 mm",
            "joint_threshold": "frozen FIT domain penetration over 0.5 mm; direct-contact gap only where required",
            "triangle_surface_audit": "not performed by this pose fitter; complete triangle intersection and whole-mesh review are independent gates",
            "interpretation": "a pose metrics_after.passed value is scoped to the listed arm FIT queries and does not certify the full triangle mesh or publishability",
        },
        "compiled_input": str(Path(args.compiled).resolve()),
        "geometry_dir": str(Path(args.geometry_dir).resolve()),
        "fit_pose_names": [],
        "regression_pose_names": [],
        "fit_source": "explicit_geometry_dir_only_no_AMASS",
        "selected_joint_ids": ARM_SELECTED_JOINT_IDS.tolist(),
        "controller_ids": ARM_CONTROLLER_IDS.tolist(),
        "rbf_radius": ARM_RBF_RADIUS,
        "max_rotation_norm_rad": ARM_MAX_ROTATION_NORM,
        "max_translation_norm_m": ARM_MAX_TRANSLATION_NORM,
        "maxfev_per_non_neutral_pose": int(args.maxfev),
        "optimizer": str(args.optimizer),
        "slsqp_maxiter_per_non_neutral_pose": int(args.slsqp_maxiter),
        "poses": {},
        "failures": [],
    }
    try:
        fit_names = _names(args.fit_pose)
        if "tpose" not in fit_names:
            raise ValueError("--fit-pose must include tpose")
        geometries = load_pose_geometry_directory(args.geometry_dir, subject=args.subject)
        missing = [name for name in fit_names if name not in geometries]
        if missing:
            raise ValueError(f"requested fit poses are absent from geometry-dir: {missing}")
        requested_regression = _names(args.regression_pose)
        if requested_regression:
            missing = [name for name in requested_regression if name not in geometries]
            if missing:
                raise ValueError(f"requested regression poses are absent from geometry-dir: {missing}")
            regression_names = requested_regression
        else:
            regression_names = [name for name in geometries if name not in fit_names]
        overlap = sorted(set(fit_names) & set(regression_names))
        if overlap:
            raise ValueError(f"pose names cannot be both fit and regression: {overlap}")
        report["fit_pose_names"] = fit_names
        report["regression_pose_names"] = regression_names

        compiled = load_compiled_subject(args.compiled)
        operator = load_source_operator(args.operator)
        calibration = load_anatomical_calibration_v1(args.calibration, operator=operator)
        fit_poses = {name: geometries[name].pose for name in fit_names}
        all_names = fit_names + [name for name in regression_names if name not in fit_names]
        all_poses = {name: geometries[name].pose for name in all_names}
        frames = cache_arm_pose_frames_v14(compiled, all_poses, geometries)

        try:
            fit_result = fit_arm_pose_corrector_v14(
                compiled,
                fit_poses,
                {name: geometries[name] for name in fit_names},
                calibration,
                maxfev=int(args.maxfev),
                optimizer=str(args.optimizer),
                slsqp_maxiter=int(args.slsqp_maxiter),
            )
        except Exception as exc:
            # Preserve a valid zero RBF artifact and the exact setup failure;
            # this is an explicit rejected bake, not a successful fallback.
            fit_result = ArmPoseFitResultV14(
                corrector=_zero_corrector(fit_poses),
                local_twists=np.zeros((len(fit_names), len(ARM_CONTROLLER_IDS), 6), dtype=np.float64),
                fit_pose_names=tuple(fit_names),
                reports={
                    name: {
                        "status": "rejected",
                        "optimizer": "none",
                        "error": f"fit setup failed: {type(exc).__name__}: {exc}",
                        "chosen_twists": np.zeros((len(ARM_CONTROLLER_IDS), 6)).tolist(),
                        "zero_candidate_preserved": True,
                    }
                    for name in fit_names
                },
                frames={name: frames[name] for name in fit_names},
                score_ids=np.empty(0, dtype=np.int64),
            )
            report["failures"].append({"stage": "fit", "error": f"{type(exc).__name__}: {exc}"})

        corrected = copy(compiled)
        corrected.corrector = fit_result.corrector
        corrected.save(output / "compiled")
        report["corrector_saved"] = str((output / "compiled" / "pose_corrector.npz").resolve())
        report["source_pack_saved"] = str((output / "compiled" / "source_pack").resolve())
        report["fit_reports"] = fit_result.reports
        for pose_name, pose_report in fit_result.reports.items():
            if pose_report.get("status") == "rejected":
                report["failures"].append({
                    "pose": pose_name,
                    "stage": "fit",
                    "status": "rejected",
                    "error": pose_report.get("optimizer_message", pose_report.get("error")),
                })

        # Regression names are scored after fitting and never enter the RBF
        # centers.  Unsupported queries are retained as explicit statuses.
        regression_poses = {name: geometries[name].pose for name in regression_names}
        if regression_poses:
            from ..arm_pose_fit_v14 import evaluate_arm_regression_v14

            report["regression_reports"] = evaluate_arm_regression_v14(
                fit_result,
                regression_poses,
                {name: geometries[name] for name in regression_names},
                compiled,
                calibration,
            )
        else:
            report["regression_reports"] = {}

        base_compiled = copy(compiled)
        base_compiled.corrector = None
        for name in all_names:
            geometry = geometries[name]
            frame = frames[name]
            subject_tag = args.subject or next(
                (
                    str(geometry.metadata.get("subject"))
                    for geometry in geometries.values()
                    if geometry.metadata.get("subject") not in {None, ""}
                ),
                "compiled",
            )
            path = output / f"subject_{subject_tag}_{name}.npz"
            try:
                before = np.asarray(base_compiled.apply_pose(geometry.pose), dtype=np.float32)
            except Exception as exc:
                before = np.empty((0, 3), dtype=np.float32)
                report["failures"].append({"pose": name, "stage": "before", "error": f"{type(exc).__name__}: {exc}"})
            status = "supported"
            role = "fit" if name in fit_names else "regression"
            correction_twists: np.ndarray | None = None
            after = np.empty((0, 3), dtype=np.float32)
            error: str | None = None
            try:
                correction_twists = corrector_twists_for_pose_v14(fit_result.corrector, geometry.pose)
                after = np.asarray(corrected.apply_pose(geometry.pose), dtype=np.float32)
            except Exception as exc:
                if isinstance(exc, PoseCorrectionSupportError):
                    status = "unsupported"
                else:
                    status = "rejected"
                error = f"{type(exc).__name__}: {exc}"
                correction_twists = None
                after = np.empty((0, 3), dtype=np.float32)
            _save_pose_npz(
                path,
                compiled=compiled,
                geometry=geometry,
                frame=frame,
                before=before,
                after=after,
                status=status,
                role=role,
                correction_twists=correction_twists,
                error=error,
            )
            report["poses"][name] = {
                "role": role,
                "status": status,
                "output": str(path.resolve()),
                "correction_twists": None if correction_twists is None else correction_twists.tolist(),
                "error": error,
            }
            if status != "supported":
                report["failures"].append({"pose": name, "stage": "after", "status": status, "error": error})
        report["status"] = "complete"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["all_fit_and_regression_exports_written"] = True
        write_json(output / "report.json", report)
        write_json(output / "manifest.json", report)
        print(json.dumps(_strict_json({"status": report["status"], "output": str(output), "failures": report["failures"]}), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "rejected"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["failures"].append({"stage": "setup", "error": f"{type(exc).__name__}: {exc}"})
        write_json(output / "report.json", report)
        write_json(output / "manifest.json", report)
        print(f"bake rejected: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
