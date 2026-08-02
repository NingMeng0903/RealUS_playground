"""Layered Genesis truth renders for skin, bones, tubes, and outside masks."""

from __future__ import annotations

import argparse
import hashlib
import json
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
)


COLORS = {
    "skin": (0.90, 0.58, 0.43, 0.18),
    "baseline": (0.12, 0.34, 0.96, 0.70),
    "candidate": (0.93, 0.88, 0.72, 1.0),
    "candidate_translucent": (0.93, 0.88, 0.72, 0.30),
    "outside": (1.0, 0.02, 0.01, 1.0),
    "vessels": (0.92, 0.06, 0.04, 0.92),
    "nerves": (0.98, 0.72, 0.02, 0.92),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask(asset: Any, tissues: set[str]) -> np.ndarray:
    selected = {name.strip().lower() for name in tissues}
    result = np.zeros(len(asset.vertices_rest), dtype=bool)
    for tissue, (start, stop) in zip(
        asset.source_tissues, np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    ):
        if str(tissue).strip().lower() in selected:
            result[int(start) : int(stop)] = True
    return result


def _faces(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    return triangles[np.all(np.asarray(mask, dtype=bool)[triangles], axis=1)]


def _baseline_pose(value: Any, asset: Any, pose: np.ndarray) -> np.ndarray:
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        source_bone_posed_global,
    )

    global_matrices = source_bone_posed_global(asset, pose)
    transforms = global_matrices @ np.linalg.inv(np.asarray(value.B_prefit))
    return _weighted_rest_correction(
        value.vertices_prefit, asset.driver_indices, asset.driver_weights, transforms
    ).astype(np.float32)


def _outside_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    bone_mask: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
) -> np.ndarray:
    import igl

    bone_ids = np.flatnonzero(bone_mask)
    winding = igl.winding_number(
        np.asarray(skin, dtype=np.float64),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(vertices, dtype=np.float64)[bone_ids],
    )
    outside = np.zeros(len(vertices), dtype=bool)
    outside[bone_ids] = np.abs(np.asarray(winding).reshape(-1)) < 0.5
    triangles = _faces(faces, bone_mask)
    return triangles[np.any(outside[triangles], axis=1)]


def _camera_manifest(skin: np.ndarray, frames: np.ndarray) -> dict[str, dict[str, Any]]:
    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    lower = np.min(skin, axis=0)
    upper = np.max(skin, axis=0)
    center = 0.5 * (lower + upper)
    cameras: dict[str, dict[str, Any]] = {
        "whole_ap": {
            "pos": tuple((center + (0.0, 0.0, 2.4)).tolist()),
            "lookat": tuple(center.tolist()), "up": (0.0, 1.0, 0.0), "fov": 34.0,
        }
    }
    hip_center = 0.5 * (
        frames[lookup["left_hip"], :3, 3] + frames[lookup["right_hip"], :3, 3]
    )
    cameras["pelvis_bilateral_ap"] = {
        "pos": tuple((hip_center + (0.0, 0.0, 0.62)).tolist()),
        "lookat": tuple(hip_center.tolist()), "up": (0.0, 1.0, 0.0), "fov": 32.0,
    }
    cameras["pelvis_axial"] = {
        "pos": tuple((hip_center + (0.0, -0.58, 0.0)).tolist()),
        "lookat": tuple(hip_center.tolist()), "up": (0.0, 0.0, 1.0), "fov": 32.0,
    }
    for side, sign in (("left", 1.0), ("right", -1.0)):
        wrist = frames[lookup[f"{side}_wrist"], :3, 3]
        elbow = frames[lookup[f"{side}_elbow"], :3, 3]
        distal = wrist - elbow
        distal /= np.linalg.norm(distal)
        hand_center = wrist + 0.09 * distal
        for view, offset, up in (
            ("dorsal", (0.0, 0.0, 0.34), (0.0, 1.0, 0.0)),
            ("palmar", (0.0, 0.0, -0.34), (0.0, 1.0, 0.0)),
            ("radial", (0.0, 0.30, 0.0), (0.0, 0.0, 1.0)),
            ("oblique", (0.18 * sign, 0.16, 0.25), (0.0, 1.0, 0.0)),
        ):
            cameras[f"{side}_hand_{view}"] = {
                "pos": tuple((hand_center + offset).tolist()),
                "lookat": tuple(hand_center.tolist()), "up": up, "fov": 31.0,
            }
        knee = frames[lookup[f"{side}_knee"], :3, 3]
        for view, offset, up in (
            ("ap", (0.0, 0.0, 0.34), (0.0, 1.0, 0.0)),
            ("lateral", (0.34 * sign, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ("condylar_axial", (0.0, -0.34, 0.0), (0.0, 0.0, 1.0)),
        ):
            cameras[f"{side}_knee_{view}"] = {
                "pos": tuple((knee + offset).tolist()),
                "lookat": tuple(knee.tolist()), "up": up, "fov": 31.0,
            }
    return cameras


def _render_layer(
    output: Path,
    *,
    entities: list[tuple[str, Path, tuple[float, float, float, float]]],
    cameras: dict[str, dict[str, Any]],
    backend: str,
) -> dict[str, Any]:
    from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
        GenesisPlatformRuntime,
        GenesisRuntimeConfig,
        MeshEntityConfig,
        StaticCameraConfig,
    )

    output.mkdir(parents=True, exist_ok=False)
    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=backend, show_viewer=False, show_fps=False,
            enable_collision=False, gravity=(0.0, 0.0, 0.0),
            plane_reflection=False, ambient_light=(0.44, 0.44, 0.44),
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
                    name=name, res=(720, 540), pos=spec["pos"],
                    lookat=spec["lookat"], up=spec["up"], fov=spec["fov"],
                    near=0.01, far=10.0, gui=False,
                )
            )
        runtime.build()
        rendered = runtime.render_all_cameras(
            modalities=("rgb", "depth", "segmentation"), force_render=True
        )
        records = _save_modalities(output, rendered, cameras)
    finally:
        runtime.close()
    sheet = _contact_sheet(
        [Path(record["rgb"]) for record in records],
        [record["camera"] for record in records],
        output / "contact_sheet.png",
        {},
    )
    return {
        "renders": records,
        "contact_sheet": str(sheet),
        "contact_sheet_sha256": _sha256(sheet),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--subject-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite truth render: {output}")
    output.mkdir(parents=True)
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(), operator=operator, required_scope="full_main_chain"
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    capture = args.capture.resolve()
    with np.load(capture, allow_pickle=False) as data:
        betas = np.asarray(data["shapes"]).reshape(-1)[:10]
        recorded_pose = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        )
    value = build_whole_chain_rest_fit_v1(
        operator,
        calibration,
        betas=betas,
        subject_label=args.subject_label,
        capture_sha256=_sha(capture),
        smplx_model=model,
        smplx_model_sha256=model_sha,
    )
    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=args.oracle.resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    bone_mask = _mask(asset, {"bone"})
    vessel_mask = _mask(asset, {"vessel"})
    nerve_mask = _mask(asset, {"nerve"})
    bone_faces = _faces(asset.faces, bone_mask)
    vessel_faces = _faces(asset.faces, vessel_mask)
    nerve_faces = _faces(asset.faces, nerve_mask)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "AlignmentTruthGenesisV1",
        "subject": args.subject_label,
        "poses": {},
        "camera_uses_candidate_geometry": False,
        "skin_frame_translation_applied": False,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
    }
    for pose_name, pose in (
        ("tpose", np.zeros((55, 3), dtype=np.float32)),
        (f"pose_{args.subject_label}", recorded_pose),
    ):
        baseline = _baseline_pose(value, asset, pose)
        candidate, _global = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=pose
        )
        skin, skin_faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
        frames, _widths, _details = _measure_frames(
            baseline, calibration.domains, calibration.joint_domain_bases,
            partition="validation",
        )
        cameras = _camera_manifest(skin, frames)
        assets = output / pose_name / "mesh_assets"
        skin_path = _export(assets / "skin.obj", skin, skin_faces)
        baseline_path = _export(assets / "baseline_bones.obj", baseline, bone_faces)
        candidate_path = _export(assets / "candidate_bones.obj", candidate, bone_faces)
        outside_path = _export(
            assets / "candidate_outside.obj",
            candidate,
            _outside_faces(candidate, asset.faces, bone_mask, skin, skin_faces),
        )
        vessel_path = _export(assets / "vessels.obj", candidate, vessel_faces)
        nerve_path = _export(assets / "nerves.obj", candidate, nerve_faces)
        layers = {
            "baseline_skin": [
                ("skin", skin_path, COLORS["skin"]),
                ("baseline", baseline_path, COLORS["candidate"]),
            ],
            "candidate_skin": [
                ("skin", skin_path, COLORS["skin"]),
                ("candidate", candidate_path, COLORS["candidate"]),
            ],
            "overlay": [
                ("skin", skin_path, (0.90, 0.58, 0.43, 0.10)),
                ("baseline", baseline_path, COLORS["baseline"]),
                ("candidate", candidate_path, COLORS["candidate_translucent"]),
            ],
            "outside_mask": [
                ("skin", skin_path, (0.90, 0.58, 0.43, 0.10)),
                ("candidate", candidate_path, COLORS["candidate_translucent"]),
                ("outside", outside_path, COLORS["outside"]),
            ],
            "tube_context": [
                ("skin", skin_path, (0.90, 0.58, 0.43, 0.08)),
                ("candidate", candidate_path, COLORS["candidate"]),
                ("vessels", vessel_path, COLORS["vessels"]),
                ("nerves", nerve_path, COLORS["nerves"]),
            ],
        }
        pose_report = {"camera_manifest": cameras, "layers": {}}
        for layer_name, entities in layers.items():
            pose_report["layers"][layer_name] = _render_layer(
                output / pose_name / layer_name,
                entities=entities,
                cameras=cameras,
                backend=args.backend,
            )
        manifest["poses"][pose_name] = pose_report
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"AlignmentTruthGenesisV1 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
