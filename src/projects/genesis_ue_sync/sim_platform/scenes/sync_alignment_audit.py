"""Genesis vs UE alignment diagnostics (canonical Genesis world vs UE cm bridge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bridge.adapters.ue import ue_camera_payload_from_spec, ue_world_point_from_genesis_m
from bridge.adapters.urdf import root_transform_from_pose
from bridge.core.camera import build_intrinsics_from_fov, opencv_camera_matrices_from_lookat
from bridge.core.transform import mat4_inv
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec
from projects.genesis_ue_sync.sim_platform.scenes.human_scene_placement import HumanScenePlacement, compute_human_scene_placement
from projects.genesis_ue_sync.urdf import compose_link_visual_world_transform, compute_link_world_transforms, parse_urdf_model


def _pose_matrix(pos: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float] | None) -> np.ndarray:
    return np.asarray(root_transform_from_pose(pos, quat_xyzw), dtype=np.float64).reshape(4, 4)


def _support_corners_genesis(surface) -> np.ndarray:
    half = 0.5 * np.asarray(surface.size, dtype=np.float64).reshape(3)
    local = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    pose = _pose_matrix(tuple(surface.pos), surface.quat_xyzw)
    out: list[np.ndarray] = []
    for pt in local:
        hom = np.concatenate([pt, [1.0]], axis=0)
        out.append((pose @ hom)[:3])
    return np.asarray(out, dtype=np.float64)


def _point_cm_ue(genesis_m: np.ndarray) -> list[float]:
    p = ue_world_point_from_genesis_m(genesis_m.reshape(3)).reshape(3)
    return [float(v * 100.0) for v in p.tolist()]


def _camera_roundtrip(camera_spec, ue_payload: dict[str, Any]) -> dict[str, Any]:
    w, h = int(camera_spec.res[0]), int(camera_spec.res[1])
    intrinsics = build_intrinsics_from_fov(width=w, height=h, fov_deg=float(camera_spec.fov))
    cam_from_world, world_from_cam = opencv_camera_matrices_from_lookat(
        camera_spec.pos,
        camera_spec.lookat,
        camera_spec.up,
        roll_deg=float(camera_spec.roll_deg),
    )
    probe_world = np.asarray(camera_spec.lookat, dtype=np.float64).reshape(3)
    hom = np.concatenate([probe_world, [1.0]], axis=0)
    proj = intrinsics @ cam_from_world[:3, :] @ hom
    z = float(proj[2])
    uv_scene = [float(proj[0] / z), float(proj[1] / z)] if z > 1e-9 else [float("nan"), float("nan")]

    loc_cm = np.asarray([ue_payload["x"], ue_payload["y"], ue_payload["z"]], dtype=np.float64).reshape(3)
    loc_genesis_m = ue_world_point_from_genesis_m(loc_cm / 100.0)
    look_cm = np.asarray(ue_payload["lookat_cm"], dtype=np.float64).reshape(3)
    look_genesis_m = ue_world_point_from_genesis_m(look_cm / 100.0)
    rot_deg_roll = float(ue_payload["roll"])
    rot_deg_pitch = float(ue_payload["pitch"])
    rot_deg_yaw = float(ue_payload["yaw"])
    _ = (rot_deg_roll, rot_deg_pitch, rot_deg_yaw)

    cam_from_world_ue_probe, world_from_cam_ue_probe = opencv_camera_matrices_from_lookat(
        tuple(loc_genesis_m.tolist()),
        tuple(look_genesis_m.tolist()),
        camera_spec.up,
        roll_deg=float(camera_spec.roll_deg),
    )
    proj_ue = intrinsics @ cam_from_world_ue_probe[:3, :] @ hom
    z2 = float(proj_ue[2])
    uv_ue_roundtrip = [float(proj_ue[0] / z2), float(proj_ue[1] / z2)] if z2 > 1e-9 else [float("nan"), float("nan")]
    delta = float(np.linalg.norm(np.asarray(uv_scene, dtype=np.float64) - np.asarray(uv_ue_roundtrip, dtype=np.float64)))

    return {
        "probe_world_genesis_m": probe_world.tolist(),
        "uv_direct_from_scene_spec": uv_scene,
        "uv_after_ue_location_cm_roundtrip": uv_ue_roundtrip,
        "pixel_l2_roundtrip": delta,
        "opencv_cam_from_world_det": float(np.linalg.det(cam_from_world[:3, :3])),
        "opencv_cam_from_world_ue_roundtrip_det": float(np.linalg.det(cam_from_world_ue_probe[:3, :3])),
        "world_from_camera_inv_matches_check": float(
            np.linalg.norm(mat4_inv(world_from_cam)[:3, :] - cam_from_world[:3, :])
        ),
    }


def _robot_link_report(scene_spec: SyncSceneSpec, *, visual_basis_rpy_deg: tuple[float, float, float]) -> dict[str, Any]:
    urdf_path = scene_spec.robot.resolved_urdf_path
    fk = compute_link_world_transforms(
        urdf_path=urdf_path,
        base_pos_m=tuple(scene_spec.robot.base_pos),
        base_quat_xyzw=scene_spec.robot.base_quat_xyzw,
        joint_positions=[float(v) for v in scene_spec.robot.joint_positions],
    )
    model = parse_urdf_model(urdf_path)
    links_out: dict[str, Any] = {}
    for name, lw in fk.items():
        t_gen = lw[:3, 3].copy()
        t_ue_cm = _point_cm_ue(t_gen)
        entry: dict[str, Any] = {
            "link_origin_genesis_m": t_gen.tolist(),
            "link_origin_ue_cm": t_ue_cm,
        }
        link_spec = model.links.get(name)
        if link_spec is not None and link_spec.visual_mesh:
            vw = compose_link_visual_world_transform(
                lw,
                visual_origin_xyz=link_spec.visual_origin_xyz,
                visual_origin_rpy=link_spec.visual_origin_rpy,
                visual_basis_rpy_deg=visual_basis_rpy_deg,
            )
            vg = vw[:3, 3]
            entry["visual_origin_genesis_m"] = vg.tolist()
            entry["visual_origin_ue_cm"] = _point_cm_ue(vg)
        links_out[name] = entry
    return {"urdf_path": str(urdf_path), "visual_basis_rpy_deg": list(visual_basis_rpy_deg), "links": links_out}


def build_genesis_ue_sync_audit_report(
    scene_spec: SyncSceneSpec,
    *,
    sequence: HumanMotionSequence | None = None,
    sequence_npz_path: str | None = None,
    device: str | None = "cpu",
    placement_sample_frames: int = 11,
    human_placement: HumanScenePlacement | None = None,
    robot_visual_basis_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Aggregate bed/camera/human/robot diagnostics for one export."""
    report: dict[str, Any] = {
        "scene_name": scene_spec.name,
        "conventions": {
            "genesis_world": "right_handed_z_up_meters",
            "ue_translation_from_genesis": "y_axis_mirror_then_scale_cm",
        },
    }

    if scene_spec.support_surface is not None:
        corners = _support_corners_genesis(scene_spec.support_surface)
        report["support_surface"] = {
            "semantic_role": scene_spec.support_surface.semantic_role,
            "genesis_center_m": list(scene_spec.support_surface.pos),
            "genesis_half_extents_m": [float(0.5 * x) for x in scene_spec.support_surface.size],
            "corner_world_genesis_m": corners.tolist(),
            "corner_world_ue_cm": [_point_cm_ue(p) for p in corners],
            "top_z_m": float(scene_spec.support_surface_top_z),
        }

    cameras_out: dict[str, Any] = {}
    for cam in scene_spec.cameras:
        ue_payload = ue_camera_payload_from_spec(cam)
        cameras_out[cam.name] = {
            "genesis_spec": {
                "pos_m": list(cam.pos),
                "lookat_m": list(cam.lookat),
                "up": list(cam.up),
                "fov_deg": float(cam.fov),
                "res": list(cam.res),
                "roll_deg": float(cam.roll_deg),
            },
            "ue_payload_cm_deg": ue_payload,
            "roundtrip_projection_check": _camera_roundtrip(cam, ue_payload),
        }
    report["cameras"] = cameras_out

    report["robot"] = _robot_link_report(scene_spec, visual_basis_rpy_deg=robot_visual_basis_rpy_deg)

    placement = human_placement
    if placement is None and sequence is not None:
        npz_ref = sequence_npz_path or sequence.source_path
        placement = compute_human_scene_placement(
            sequence,
            scene_spec=scene_spec,
            sequence_npz_path=str(npz_ref),
            device=device,
            placement_sample_frames=placement_sample_frames,
        )

    if placement is not None:
        report["human_scene_placement"] = placement.to_dict()
        human_ue: dict[str, Any] = {}
        for fr in placement.sampled_frames:
            human_ue[str(fr.frame_index)] = {
                "root_genesis_m": list(fr.root_translation_world_m),
                "root_ue_cm": _point_cm_ue(np.asarray(fr.root_translation_world_m, dtype=np.float64)),
                "pelvis_genesis_m": list(fr.pelvis_world_m),
                "pelvis_ue_cm": _point_cm_ue(np.asarray(fr.pelvis_world_m, dtype=np.float64)),
                "mesh_lowest_z_genesis_m": float(fr.mesh_lowest_z_m),
            }
        report["human_frames_genesis_vs_ue_cm"] = human_ue

    report["human_anchor_genesis_m"] = list(scene_spec.resolved_human_anchor())
    report["human_anchor_ue_cm"] = _point_cm_ue(np.asarray(scene_spec.resolved_human_anchor(), dtype=np.float64))

    return report


def write_sync_audit_json(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
