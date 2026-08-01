"""Genesis diagnostic matrix for whole-chain bones with frozen 142 tissues."""

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
    _save_modalities,
    _sha256,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_chain_validation_v1 import (
    run_dynamic_chain_validation_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
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
    check_whole_chain_rest_fit_v1,
)


COLORS = {
    "skin": (0.72, 0.76, 0.78, 0.12),
    "baseline_bones": (0.12, 0.34, 0.96, 0.24),
    "candidate_bones": (0.93, 0.88, 0.72, 1.0),
    "organs": (0.72, 0.22, 0.20, 0.16),
    "vessels": (0.92, 0.08, 0.06, 0.88),
    "nerves": (0.98, 0.73, 0.05, 0.82),
    "connective": (0.34, 0.78, 0.58, 0.18),
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
        local_distance = 0.55 if spec.kind in {"hip", "shoulder"} else 0.38
        for view, offset in (
            ("ap", np.asarray((0.0, 0.0, local_distance))),
            ("oblique", np.asarray((0.72 * local_distance, 0.05, 0.72 * local_distance))),
        ):
            cameras[f"{spec.name}_{view}"] = {
                "pos": tuple((pivot + offset).tolist()),
                "lookat": tuple(pivot.tolist()),
                "up": (0.0, 1.0, 0.0),
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
    preview, _preview_global = pose_whole_chain_vertices(
        value,
        pose_map,
        source_asset=asset,
        pose_axis_angle=pose,
        include_tube_transport_preview=True,
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
    cameras = _camera_manifest(skin, baseline_frames)
    masks = {
        "baseline_bones": _tissue_mask(asset, {"bone"}),
        "candidate_bones": _tissue_mask(asset, {"bone"}),
        "organs": _tissue_mask(asset, {"organ", "heart"}),
        "vessels": _tissue_mask(asset, {"vessel"}),
        "nerves": _tissue_mask(asset, {"nerve"}),
        "connective": _tissue_mask(asset, {"connective_tissue"}),
    }
    assets = output / "mesh_assets"
    entities = [
        ("skin", _export(assets / "skin.obj", skin, skin_faces), COLORS["skin"]),
    ]
    sources = {
        "baseline_bones": baseline,
        "candidate_bones": candidate,
        "organs": baseline,
        "vessels": baseline,
        "nerves": baseline,
        "connective": baseline,
    }
    for name, mask in masks.items():
        faces = _subset_faces(asset.faces, mask)
        if len(faces):
            entities.append(
                (
                    name,
                    _export(assets / f"{name}.obj", sources[name], faces),
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
    sheet = _contact_sheet(
        [Path(record["rgb"]) for record in records],
        [record["camera"] for record in records],
        output / "contact_sheet.png",
        {},
    )
    tube = masks["vessels"] | masks["nerves"]
    preview_delta = np.linalg.norm(preview[tube] - candidate[tube], axis=1)
    return {
        "camera_source": "smplx_skin_bbox_and_frozen_142_validation_frames",
        "candidate_geometry_used_for_camera": False,
        "renders": records,
        "contact_sheet": str(sheet),
        "contact_sheet_sha256": _sha256(sheet),
        "frozen_tissues_equal_142": True,
        "tube_preview_persisted": False,
        "tube_preview_rms_displacement_m": float(
            np.sqrt(np.mean(preview_delta**2))
        ),
        "tube_preview_max_displacement_m": float(np.max(preview_delta)),
        "publishable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        args.calibration.resolve(), operator=operator, required_scope="lower_chain"
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
        value = build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas[label],
            subject_label=label,
            capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
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
            scenes[pose_label] = _render_scene(
                output=output / label / pose_label,
                asset=asset,
                value=value,
                pose_map=pose_map,
                calibration=calibration,
                model=model,
                pose=pose,
                backend=args.backend,
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
        "schema_version": 1,
        "artifact_kind": "WholeChainDynamicGenesisReviewV1",
        "subjects": subjects,
        "candidate_pass_flags_used_for_camera": False,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
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
