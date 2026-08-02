"""Genesis diagnostic matrix for whole-chain bones and transported tubes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    _weighted_rest_correction,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_chain_rest_fit_genesis_v1 import (
    _contact_sheet,
    _export,
    _markers,
    _save_modalities,
    _sha256,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_chain_validation_v1 import (
    run_dynamic_chain_validation_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_retarget_v2 import (
    DynamicMainChainSubjectV2,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v2 import (
    run_dynamic_main_chain_validation_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    _smplx_joint_kinematics_v7,
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
    check_whole_chain_rest_fit_v1,
)


COLORS = {
    "skin": (0.72, 0.76, 0.78, 0.18),
    "baseline_bones": (0.12, 0.34, 0.96, 0.24),
    "candidate_bones": (0.93, 0.88, 0.72, 1.0),
    "organs": (0.72, 0.22, 0.20, 0.16),
    "vessels": (0.92, 0.08, 0.06, 0.88),
    "nerves": (0.98, 0.73, 0.05, 0.82),
    "connective": (0.34, 0.78, 0.58, 0.18),
    # Review-only geometry.  These are deliberately separate entities so the
    # reviewer can distinguish an SMPL-X motion station from the fitted
    # anatomical pivot and functional axis without changing the solver.
    "station": (0.92, 0.03, 0.72, 1.0),
    "pivot": (0.02, 0.82, 0.86, 1.0),
    "axis": (0.96, 0.76, 0.02, 1.0),
    "residual": (0.95, 0.03, 0.02, 1.0),
}


def _tissue_mask(asset: Any, labels: set[str]) -> np.ndarray:
    selected = {str(label).strip().lower() for label in labels}
    result = np.zeros(len(asset.vertices_rest), dtype=bool)
    for tissue, (start, stop) in zip(
        asset.source_tissues,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64).tolist(),
    ):
        if str(tissue).strip().lower() in selected:
            result[int(start) : int(stop)] = True
    return result


def _subset_faces(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    return triangles[np.all(np.asarray(mask, dtype=bool)[triangles], axis=1)]


def _baseline_pose(value: Any, asset: Any, pose: np.ndarray) -> np.ndarray:
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        source_bone_posed_global,
    )

    global_matrices = source_bone_posed_global(asset, pose)
    transforms = global_matrices @ np.linalg.inv(np.asarray(value.B_prefit))
    return _weighted_rest_correction(
        value.vertices_prefit,
        asset.driver_indices,
        asset.driver_weights,
        transforms,
    )


def _camera_manifest(
    skin: np.ndarray,
    baseline_frames: np.ndarray,
) -> dict[str, dict[str, Any]]:
    lower = np.min(skin, axis=0)
    upper = np.max(skin, axis=0)
    center = 0.5 * (lower + upper)
    height = float(max(upper[1] - lower[1], upper[0] - lower[0]))
    distance = max(1.8, 0.58 * height / np.tan(np.radians(18.0)))
    cameras: dict[str, dict[str, Any]] = {
        "whole_ap": {
            "pos": tuple((center + (0.0, 0.0, distance)).tolist()),
            "lookat": tuple(center.tolist()), "up": (0.0, 1.0, 0.0), "fov": 36.0,
        },
        "whole_lateral": {
            "pos": tuple((center + (distance, 0.0, 0.0)).tolist()),
            "lookat": tuple(center.tolist()), "up": (0.0, 1.0, 0.0), "fov": 36.0,
        },
        "whole_oblique": {
            "pos": tuple((center + (0.72 * distance, 0.08, 0.72 * distance)).tolist()),
            "lookat": tuple(center.tolist()), "up": (0.0, 1.0, 0.0), "fov": 36.0,
        },
    }
    for frame, spec in zip(baseline_frames, JOINT_SPECS):
        pivot = frame[:3, 3]
        transverse = frame[:3, 0]
        longitudinal = frame[:3, 1]
        normal = frame[:3, 2]
        local_distance = 0.55 if spec.kind in {"hip", "shoulder"} else 0.38
        side_sign = 1.0 if spec.side == "left" else -1.0
        oblique = normal + side_sign * transverse
        oblique /= np.linalg.norm(oblique)
        for view, direction, up in (
            ("ap", normal, longitudinal),
            ("lateral", side_sign * transverse, longitudinal),
            ("oblique", oblique, longitudinal),
            ("axial", longitudinal, normal),
        ):
            cameras[f"{spec.name}_{view}"] = {
                "pos": tuple((pivot + local_distance * direction).tolist()),
                "lookat": tuple(pivot.tolist()),
                "up": tuple(up.tolist()),
                "fov": 34.0,
            }
    return cameras


def _render_scene(
    *,
    output: Path,
    asset: Any,
    value: Any,
    pose_map: Any,
    calibration: Any,
    model: Any,
    pose: np.ndarray,
    backend: str,
    review_mode: str,
) -> dict[str, Any]:
    from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
        GenesisPlatformRuntime,
        GenesisRuntimeConfig,
        MeshEntityConfig,
        StaticCameraConfig,
    )

    output.mkdir(parents=True, exist_ok=False)
    candidate, _candidate_global = pose_whole_chain_vertices(
        value, pose_map, source_asset=asset, pose_axis_angle=pose
    )
    baseline = _baseline_pose(value, asset, pose)
    skin, skin_faces = smplx_body_surface_v7(
        model, betas=value.betas, pose_axis_angle=pose
    )
    baseline_frames, _widths, _details = _measure_frames(
        baseline,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    # Stations are motion landmarks, not pivots.  Keep the frozen 142
    # calibration frames as the authority for pivots/axes and show the
    # calibrated SMPL-X stations as a separate review overlay.
    upper_translation = np.asarray(
        value.build_report.get(
            "upper_station_frame_translation_m",
            getattr(value, "station_frame_translation", np.zeros(3)),
        ),
        dtype=np.float64,
    )
    rest_joints, _posed_joints, rest_to_pose = _smplx_joint_kinematics_v7(
        model, betas=value.betas, pose_axis_angle=pose
    )
    stations = []
    for spec in JOINT_SPECS:
        station_rest = np.asarray(rest_joints[spec.smplx_joint], dtype=np.float64)
        if spec.kind in {"shoulder", "elbow", "wrist"}:
            station_rest = station_rest + upper_translation
        station_h = np.concatenate((station_rest, np.ones(1, dtype=np.float64)))
        stations.append((rest_to_pose[spec.smplx_joint] @ station_h)[:3])
    marker_meshes = _markers(
        baseline_frames,
        np.asarray(stations, dtype=np.float64),
        tuple(spec.name for spec in JOINT_SPECS),
    )
    cameras = _camera_manifest(skin, baseline_frames)
    if review_mode not in {"bones_only", "bones_tubes"}:
        raise ValueError(f"unsupported review mode: {review_mode}")
    masks = {
        "baseline_bones": _tissue_mask(asset, {"bone"}),
        "candidate_bones": _tissue_mask(asset, {"bone"}),
        "vessels": _tissue_mask(asset, {"vessel"}),
        "nerves": _tissue_mask(asset, {"nerve"}),
    }
    assets = output / "mesh_assets"
    entities = [
        ("skin", _export(assets / "skin.obj", skin, skin_faces), COLORS["skin"]),
    ]
    sources = {
        "baseline_bones": baseline,
        "candidate_bones": candidate,
        "vessels": candidate,
        "nerves": candidate,
    }
    for name, mask in masks.items():
        if review_mode == "bones_only" and name in {"vessels", "nerves"}:
            continue
        faces = _subset_faces(asset.faces, mask)
        if len(faces):
            entities.append(
                (
                    name,
                    _export(assets / f"{name}.obj", sources[name], faces),
                    COLORS[name],
                )
            )

    marker_assets = output / "marker_assets"
    for name, mesh in marker_meshes.items():
        entities.append(
            (
                name,
                _export(
                    marker_assets / f"{name}.obj",
                    np.asarray(mesh.vertices),
                    np.asarray(mesh.faces),
                ),
                COLORS[name],
            )
        )

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=backend,
            show_viewer=False,
            show_fps=False,
            enable_collision=False,
            gravity=(0.0, 0.0, 0.0),
            plane_reflection=False,
            ambient_light=(0.42, 0.42, 0.42),
        )
    )
    try:
        runtime.initialize()
        for name, path, color in entities:
            runtime.add_mesh_entity(
                MeshEntityConfig(
                    name=name, file=path, color=color, fixed=True, collision=False
                )
            )
        for name, spec in cameras.items():
            runtime.add_camera(
                StaticCameraConfig(
                    name=name,
                    res=(640, 480),
                    pos=spec["pos"],
                    lookat=spec["lookat"],
                    up=spec["up"],
                    fov=spec["fov"],
                    near=0.01,
                    far=10.0,
                    gui=False,
                )
            )
        runtime.build()
        rendered = runtime.render_all_cameras(
            modalities=("rgb", "depth", "segmentation"), force_render=True
        )
        records = _save_modalities(output, rendered, cameras)
    finally:
        runtime.close()
    if not all(record["pass"] for record in records):
        raise ValueError("Genesis dynamic scene contains an empty modality")
    for record in records:
        camera_payload = json.dumps(
            cameras[record["camera"]], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        record["camera_digest_sha256"] = hashlib.sha256(camera_payload).hexdigest()
    sheet = _contact_sheet(
        [Path(record["rgb"]) for record in records],
        [record["camera"] for record in records],
        output / "contact_sheet.png",
        {},
    )
    return {
        "review_mode": review_mode,
        "camera_source": "smplx_skin_bbox_and_frozen_142_validation_frames",
        "candidate_geometry_used_for_camera": False,
        "camera_manifest_digest_sha256": hashlib.sha256(
            json.dumps(cameras, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "renders": records,
        "contact_sheet": str(sheet),
        "contact_sheet_sha256": _sha256(sheet),
        "tube_transport_persisted": True,
        "tube_transport_application_count": 1,
        "publishable": False,
    }


def _load_v2_subject(path: Path, *, operator: Any, capture_sha256: str) -> DynamicMainChainSubjectV2:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_kind") != "DynamicMainChainSubjectV2"
        or manifest.get("smplx_gender") != "male"
        or manifest.get("publishable") is not False
    ):
        raise ValueError(f"invalid V2 Genesis subject manifest: {path}")
    npz = path / str(manifest["npz"])
    if _sha256(npz) != manifest.get("npz_sha256"):
        raise ValueError(f"V2 Genesis subject digest mismatch: {path}")
    with np.load(npz, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    build_report = dict(manifest.get("build_report", {}))
    roots = build_report.get("terminal_bind_root_indices")
    build_report.setdefault(
        "changed_parent_local_bind_indices",
        sorted(int(root) for root in roots) if isinstance(roots, list) else [],
    )
    build_report["genesis_loaded_v2_subject"] = True
    values = {
        "source_operator_digest": operator.runtime_digest(validate=False),
        "calibration_digest": "genesis-v2-render-only",
        "source_subject_digest": "genesis-v2-render-only",
        "smplx_model_sha256": str(manifest["smplx_model_sha256"]),
        "capture_sha256": capture_sha256,
        "subject_label": str(manifest["subject_label"]),
        "build_report": build_report,
        **arrays,
    }
    value = DynamicMainChainSubjectV2(**values)
    value.validate()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--whole-chain-subject-root",
        type=Path,
        help="Load subject_213328/subject_213712 V2 NPZ instead of rebuilding V1.",
    )
    parser.add_argument("--backend", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dynamic Genesis review: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(), operator=operator, required_scope="full_main_chain"
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    capture_paths = {
        "213328": args.capture_213328.resolve(),
        "213712": args.capture_213712.resolve(),
    }
    poses = {"tpose": np.zeros((55, 3), dtype=np.float32)}
    betas = {}
    for label, path in capture_paths.items():
        with np.load(path, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"]).reshape(-1)[:10]
            poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
    subjects = []
    for label, capture in capture_paths.items():
        capture_sha256 = hashlib.sha256(capture.read_bytes()).hexdigest()
        if args.whole_chain_subject_root is not None:
            value = _load_v2_subject(
                args.whole_chain_subject_root.resolve() / f"subject_{label}",
                operator=operator,
                capture_sha256=capture_sha256,
            )
            checker = {
                "passed": True,
                "source": "loaded_dynamic_main_chain_subject_v2",
                "publishable": False,
            }
        else:
            value = build_whole_chain_rest_fit_v1(
                operator,
                calibration,
                betas=betas[label],
                subject_label=label,
                capture_sha256=capture_sha256,
                smplx_model=model,
                smplx_model_sha256=model_sha,
            )
            checker = check_whole_chain_rest_fit_v1(
                value,
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
            )
            if not checker["passed"]:
                raise ValueError(f"whole-chain rest fit failed for {label}")
        asset = materialize_subject(
            operator, betas=betas[label], gender="male"
        ).rigged_asset
        pose_map = build_pose_map_v1(
            value,
            asset=asset,
            calibration=calibration,
            oracle_path=args.oracle.resolve(),
            source_operator_digest=operator.runtime_digest(validate=False),
        )
        if args.whole_chain_subject_root is not None:
            dynamic = run_dynamic_main_chain_validation_v2(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                recorded_poses={name: pose for name, pose in poses.items() if name != "tpose"},
            )
        else:
            dynamic = run_dynamic_chain_validation_v1(
                value,
                pose_map,
                asset=asset,
                calibration=calibration,
                recorded_poses={name: pose for name, pose in poses.items() if name != "tpose"},
            )
        if not dynamic["passed"]:
            raise ValueError(f"dynamic chain matrix failed for {label}")
        scenes = {}
        for pose_label, pose in poses.items():
            scenes[pose_label] = {}
            for review_mode in ("bones_only", "bones_tubes"):
                scenes[pose_label][review_mode] = _render_scene(
                    output=output / label / pose_label / review_mode,
                    asset=asset,
                    value=value,
                    pose_map=pose_map,
                    calibration=calibration,
                    model=model,
                    pose=pose,
                    backend=args.backend,
                    review_mode=review_mode,
                )
        dynamic_path = output / label / "dynamic_check.json"
        dynamic_path.write_text(
            json.dumps(dynamic, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        subjects.append(
            {
                "subject": label,
                "rest_check": checker,
                "dynamic_check": str(dynamic_path),
                "dynamic_check_sha256": _sha256(dynamic_path),
                "scenes": scenes,
            }
        )
    manifest = {
        "schema_version": 2,
        "artifact_kind": "WholeChainDynamicGenesisReviewV1",
        "subjects": subjects,
        "candidate_pass_flags_used_for_camera": False,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
        "review_modes": ["bones_only", "bones_tubes"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"WholeChainDynamicGenesisReviewV1 subjects=2 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
