#!/usr/bin/env python3
"""Build RealUS scene YAML + tracking calibration from genesis_bundle.yaml."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLIDER_SPEC = REPO_ROOT / "rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _camera_scene_entry(name: str, cam: dict[str, Any]) -> dict[str, Any]:
    wfc = np.asarray(cam["world_from_camera"], dtype=np.float64).reshape(4, 4)
    pos = wfc[:3, 3].tolist()
    forward = wfc[:3, 2]
    up = (-wfc[:3, 1]).tolist()
    lookat = (wfc[:3, 3] + forward).tolist()
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    h = float(cam["image_size"][1])
    fy = float(K[1, 1])
    fov_v = float(2.0 * math.degrees(math.atan(h / (2.0 * fy)))) if fy > 1e-9 else 45.0
    return {
        "name": name,
        "res": [int(cam["image_size"][0]), int(cam["image_size"][1])],
        "pos": [float(v) for v in pos],
        "lookat": [float(v) for v in lookat],
        "up": [float(v) for v in up],
        "fov": fov_v,
        "fov_axis": "vertical",
        # Synthetic cameras share physical extrinsics/intrinsics, but do not
        # inherit Brown distortion from the physical lenses.
        "intrinsics": K.tolist(),
        "distortion": [0.0] * 5,
        "world_from_camera": wfc.tolist(),
        "metadata": {"projection_model": "opencv_pinhole", "source": "realus_calibration_bundle"},
    }


def _load_slider_rail(path: Path | None) -> dict[str, Any]:
    path = path or DEFAULT_SLIDER_SPEC
    payload = _slider_spec(path)
    travel_mm = float(payload["rail"]["effective_travel_mm"])
    return {
        "travel_m": travel_mm / 1000.0,
    }


def _slider_spec(path: Path) -> dict[str, Any]:
    # Same pure geometry/calibration code as RailGenesisScene. Never launch a
    # controller shell, and never silently replace failed calibration with an
    # obsolete default pose.
    control_root = str(REPO_ROOT / "rm75_control")
    if control_root not in sys.path:
        sys.path.insert(0, control_root)
    from rm75_control.control.joint_admittance_8dof.param_model.generator import load_spec
    spec = load_spec(path.resolve())
    if "base_pos_m" not in (spec.get("world_calib") or {}):
        raise ValueError(f"Missing calibrated base_link world pose in {path} and its calibrated overlay")
    return spec


def _rail_base_pose(slider_spec_path: Path | None) -> dict[str, list[float]]:
    spec = _slider_spec(slider_spec_path or DEFAULT_SLIDER_SPEC)
    from rm75_control.control.joint_admittance_8dof.param_model.generator import compute_layout
    from rm75_control.control.joint_admittance_8dof.param_model.placement import entity_pose_from_calib, resolve_world_calib
    pose = entity_pose_from_calib(resolve_world_calib(spec, compute_layout(spec)))
    return {
        "base_pos": [float(v) for v in pose["pos"]],
        "base_quat_wxyz": [float(v) for v in pose["quat_wxyz"]],
    }


def build_scene(
    *,
    bundle: dict[str, Any],
    slider: dict[str, Any],
    rail_base: dict[str, list[float]],
    body_name: str,
    skeletal_mesh_path: str,
) -> dict[str, Any]:
    control_root = str(REPO_ROOT / "rm75_control")
    if control_root not in sys.path:
        sys.path.insert(0, control_root)
    from rm75_control.control.joint_admittance_8dof.viewer.calib_scene import BedSurfaceCalib

    # Use the calibration viewer's actual parametric box, including its color,
    # rather than the legacy slab hint stored in the exported bundle.
    support = BedSurfaceCalib.from_bundle(bundle)
    if support is None:
        raise ValueError("Bundle must contain a calibrated bed with a positive top height")
    support_pos = list(support.center)
    support_size = list(support.size)
    cameras = [_camera_scene_entry(str(k), dict(v)) for k, v in (bundle.get("cameras") or {}).items()]
    support_yaw = support.rotation_deg
    # 8-DOF: rail prismatic + 7 arm joints
    joint_positions = [0.0] + [0.0] * 7
    return {
        "name": "realus_bed_rail_scene",
        "environment": {
            "ue_environment_preset": "soft_studio",
            "ue_ground_plane_color": [0.055, 0.060, 0.070, 1.0],
            "ue_map": "/Game/Bedlam/IBLMap",
            "ue_hdri_name": "abandoned_hopper_terminal_03",
            "ground_plane_color": [0.92, 0.92, 0.92, 1.0],
        },
        "support_surface": {
            "name": support.name,
            "pos": support_pos,
            "size": support_size,
            "rotation_deg": support_yaw,
            "quat_xyzw": [0.0, 0.0, math.sin(math.radians(support_yaw) / 2), math.cos(math.radians(support_yaw) / 2)],
            "color": list(support.color),
            "semantic_role": "bed",
            "spawn_in_genesis": True,
            "spawn_in_ue": True,
        },
        "robot_model_overrides": {
            "rm75_6f_8dof": {
                "joint_positions": joint_positions,
                "use_collision_geometry": False,
            },
            "rm75_6f": {
                "joint_positions": [0.0] * 7,
                "use_collision_geometry": False,
            },
        },
        "robot": {
            "model_id": "rm75_6f_8dof",
            "name": "robot_main",
            "base_pos": [float(v) for v in rail_base["base_pos"]],
            "base_quat_wxyz": [float(v) for v in rail_base["base_quat_wxyz"]],
            "joint_positions": joint_positions,
            "rail": {
                "enabled": True,
                "axis": "y",
                "travel_m": float(slider["travel_m"]),
                "joint_index": 0,
                "note": "RealUS 8-DOF: rail_y is URDF joint 0; UE spawn uses rail_base root pose",
            },
            "use_collision_geometry": False,
            "use_visual_mesh": True,
            "visual_mesh_format": "fbx",
            "ue_visual_asset_root": "/Game/Bedlam/Generated/RM758DofVisual",
            "color": [0.55, 0.55, 0.6, 1.0],
        },
        "human": {
            "anchor_pos": [0.0, 0.0, 0.0],
            "support_margin_m": 0.015,
            "support_reference": "support_surface_top",
            "align_floor": True,
            "ue_root_offset_genesis_m": [0.0, 0.0, 0.0],
        },
        "motion": {
            "source_id": "realus_easymocap_subject",
            "sequence_npz_path": "assets/human_reference/subject_shape_tpose.npz",
            "fps": 30.0,
            "frame_count": 1,
            "start_frame": 0,
            "frame_step": 1,
        },
        "render": {
            "fps": 30.0,
            "frame_limit": 1,
            "genesis_backend": "cuda",
            "ue_frame_count": 1,
            "ue_frame_step": 1,
            "ue_render_now": False,
        },
        "ue_avatar": {
            "body_mode": "official_retargeted_overlay",
            "body_name": body_name,
            "texture_body": "Male_Skin_Preset_0027",
            "texture_clothing": "gr_ben_004_M_texture_01",
            "texture_clothing_overlay": "T_gr_ben_004_M_texture_01_diffuse",
            "skeletal_mesh_path": skeletal_mesh_path,
            "animation_asset_root": "/Game/Bedlam/Generated/RetargetedAnimations",
            "imported_fbx_root": "/Game/Bedlam/Generated/ImportedSMPLMotion",
            "hidden_material_path": "/Engine/PS/Bedlam/Core/Materials/M_SMPLX_Hidden.M_SMPLX_Hidden",
            "fbx_global_scale": 100.0,
            "subject_betas_path": "outputs/offline_capture/latest/beta_calibration/betas.npy",
            "initial_pose": "tpose",
            "shape_authority": "ue_skeletal_mesh",
            "note": "BEDLAM skeletal mesh defines rendered shape; subject betas require an explicitly imported subject mesh.",
        },
        "cameras": cameras,
    }


def build_tracking_calibration(bundle: dict[str, Any], scene_rel: str) -> dict[str, Any]:
    return {
        "scene_spec": scene_rel,
        "metadata": {
            "rig_id": "realus_realsense",
            "source": "genesis_bundle",
            "note": "N-camera RealUS calibration; camera_ids follow bundle aliases",
        },
        "convention": (bundle.get("world_frame") or {}).get("convention")
        or {
            "units": "meters",
            "world_up_axis": "z",
            "world_handedness": "right",
            "image_origin": "top_left",
            "camera_forward_axis": "+z",
        },
        "cameras": bundle.get("cameras") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--slider-rail", type=Path, default=DEFAULT_SLIDER_SPEC)
    ap.add_argument("--out-scene", type=Path, default=Path("configs/scenes/realus_bed_rail_scene.yaml"))
    ap.add_argument("--out-calib", type=Path, default=Path("configs/calibration/realus_realsense/cameras.yaml"))
    ap.add_argument("--body-name", type=str, default="it_4375_M_2400")
    ap.add_argument(
        "--skeletal-mesh-path",
        type=str,
        default="/Engine/PS/Bedlam/SMPLX_LH_animations/it_4375_M/it_4375_M_2400",
    )
    args = ap.parse_args()

    bundle = _load(args.bundle)
    slider = _load_slider_rail(args.slider_rail)
    rail_base = _rail_base_pose(args.slider_rail)
    scene = build_scene(
        bundle=bundle,
        slider=slider,
        rail_base=rail_base,
        body_name=str(args.body_name),
        skeletal_mesh_path=str(args.skeletal_mesh_path),
    )
    _dump(args.out_scene, scene)
    calib = build_tracking_calibration(bundle, scene_rel=str(args.out_scene))
    _dump(args.out_calib, calib)
    print(f"wrote {args.out_scene}")
    print(f"wrote {args.out_calib} cameras={list((bundle.get('cameras') or {}).keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
