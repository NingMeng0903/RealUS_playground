#!/usr/bin/env python3
"""Validate robot URDF consistency between the shared scene spec, offline FK, and Genesis.

For DAE→OBJ baking and per-link mesh PCA vs FK, use audit_robot_visual_mesh.py in the same directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridge.adapters.genesis import xyzw_from_genesis_quat_wxyz
from bridge.core.rotation import quaternion_xyzw_to_matrix
from bridge.adapters.urdf import root_transform_from_pose
from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.embodiments import build_panda_ultrasound_preset
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime, GenesisRuntimeConfig
from projects.genesis_ue_sync.urdf import MESH_SOURCE_TO_UE_SCALE, compute_link_world_transforms, parse_urdf_model


def _matrix_from_pose_vector(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(f"Expected pose vector [x,y,z,qw,qx,qy,qz] or similar, got shape {values.shape}")
    pos = values[:3]
    quat_xyzw = xyzw_from_genesis_quat_wxyz(values[3:7])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quaternion_xyzw_to_matrix(quat_xyzw)
    mat[:3, 3] = pos
    return mat


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-spec", type=Path, default=paths.default_scene_spec_path)
    parser.add_argument("--output-json", type=Path, default=paths.tmp_root / "robot_urdf_consistency_report.json")
    parser.add_argument("--backend", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--offline-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_spec = load_sync_scene_spec(args.scene_spec.expanduser().resolve())
    model = parse_urdf_model(scene_spec.robot.resolved_urdf_path)
    offline_fk = compute_link_world_transforms(
        urdf_path=scene_spec.robot.resolved_urdf_path,
        base_pos_m=tuple(scene_spec.robot.base_pos),
        base_quat_xyzw=scene_spec.robot.base_quat_xyzw,
        joint_positions=[float(v) for v in scene_spec.robot.joint_positions],
    )

    report: dict[str, object] = {
        "scene_spec": str(args.scene_spec.expanduser().resolve()),
        "urdf_path": str(scene_spec.robot.resolved_urdf_path),
        "robot_name": scene_spec.robot.name,
        "joint_positions": [float(v) for v in scene_spec.robot.joint_positions],
        "root_link": model.root_link,
        "link_count": len(model.links),
        "base_pose_world_m": np.asarray(
            root_transform_from_pose(scene_spec.robot.base_pos, scene_spec.robot.base_quat_xyzw),
            dtype=np.float64,
        ).tolist(),
        "ue_self_parsed_fk": {name: mat.tolist() for name, mat in offline_fk.items()},
        "ue_visual_mesh_scale": {
            "scene_visual_mesh_scale": float(scene_spec.robot.visual_mesh_scale),
            "mesh_source_to_ue_scale": float(MESH_SOURCE_TO_UE_SCALE),
            "expected_applied_scale": float(scene_spec.robot.visual_mesh_scale) * float(MESH_SOURCE_TO_UE_SCALE),
        },
        "genesis_fk": None,
        "fk_delta": None,
        "issues": [],
        "warnings": [],
        "ok": True,
    }

    if not args.offline_only:
        runtime = GenesisPlatformRuntime(
            GenesisRuntimeConfig(
                backend=str(args.backend),
                show_viewer=False,
                show_fps=False,
                gravity=(0.0, 0.0, 0.0),
                enable_collision=False,
            )
        )
        try:
            runtime.initialize()
            embodiment = build_panda_ultrasound_preset(urdf_path=scene_spec.robot.resolved_urdf_path, camera_names=())
            runtime.add_articulated_entity(
                embodiment,
                name=scene_spec.robot.name,
                pos=scene_spec.robot.base_pos,
                quat_xyzw=scene_spec.robot.base_quat_xyzw,
            )
            runtime.build()
            runtime.reset()
            runtime.set_robot_joint_positions(
                scene_spec.robot.name,
                np.asarray(scene_spec.robot.joint_positions, dtype=np.float32),
            )

            genesis_fk: dict[str, list[list[float]]] = {}
            fk_delta: dict[str, dict[str, float]] = {}
            for link_name, ue_world in offline_fk.items():
                try:
                    genesis_pose = runtime.get_link_pose(scene_spec.robot.name, link_name)
                except Exception as exc:
                    report["issues"].append(f"Genesis missing link '{link_name}': {exc}")
                    continue
                genesis_world = _matrix_from_pose_vector(genesis_pose)
                genesis_fk[link_name] = genesis_world.tolist()
                trans_err = float(np.linalg.norm(genesis_world[:3, 3] - ue_world[:3, 3]))
                rot_err = float(np.linalg.norm(genesis_world[:3, :3] - ue_world[:3, :3], ord="fro"))
                fk_delta[link_name] = {
                    "translation_l2_m": trans_err,
                    "rotation_frobenius": rot_err,
                }
                if trans_err > 0.02:
                    report["issues"].append(f"{link_name}: translation delta {trans_err:.6f} m")
                if rot_err > 1e-2:
                    report["issues"].append(f"{link_name}: rotation delta {rot_err:.6f}")
            report["genesis_fk"] = genesis_fk
            report["fk_delta"] = fk_delta
        except ImportError as exc:
            report["warnings"].append(f"Genesis validation unavailable: {exc!r}")
        except Exception as exc:
            report["issues"].append(f"Genesis validation failed: {exc!r}")
        finally:
            runtime.close()

    report["ok"] = len(report["issues"]) == 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
