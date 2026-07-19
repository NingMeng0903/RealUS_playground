#!/usr/bin/env python3
"""Reframe a saved Stage-1 source rig onto its SMPL-X rest-joint anchors.

This is a one-time offline mapping refinement.  It preserves the imported
Blender mesh vertices, source-bone hierarchy, source weights, and all 55-joint
driver assignments.  Only the persisted target bind and controller coupling
are rebuilt, so a later pose is evaluated through the same source rig without
Blender or a pose-specific bake.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _anchor_errors(asset) -> np.ndarray:
    """Distance from each independently-driven source bind to its SMPL-X joint."""
    modes = list(asset.source_bone_driver_types or [])
    anchors = np.asarray(asset.target_bind_global, dtype=np.float64)[:, :3, 3]
    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    semantic = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    values = np.full(len(anchors), np.nan, dtype=np.float64)
    for index, mode in enumerate(modes):
        if mode != "bind_follow":
            values[index] = np.linalg.norm(anchors[index] - joints[semantic[index]])
    return values


def _hard_anchor_target_bind(asset, global_bind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put every controller root exactly at its declared SMPL-X rest joint."""
    original_local = np.asarray(asset.target_bind_local, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    semantic = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    modes = list(asset.source_bone_driver_types or [])
    anchored = np.asarray(global_bind, dtype=np.float64).copy()
    for bone, mode in enumerate(modes):
        parent = int(parents[bone])
        if mode == "bind_follow" and parent >= 0:
            anchored[bone] = anchored[parent] @ original_local[bone]
        else:
            anchored[bone, :3, 3] = joints[int(semantic[bone])]
    local = np.empty_like(anchored)
    for bone, parent in enumerate(parents):
        local[bone] = anchored[bone] if parent < 0 else np.linalg.inv(anchored[parent]) @ anchored[bone]
    return anchored, local


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hard-controller-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="snap independent source controller origins to their declared SMPL-X rest joints",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        skin_vertices,
        with_source_driver_coupling,
    )
    from projects.genesis_ue_sync.anatomy_retarget.material_fit import _fit_source_frames
    from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
        load_rigged_asset,
        save_rigged_asset,
    )

    asset = load_rigged_asset(args.asset, validate=True)
    before = _anchor_errors(asset)
    global_bind, local_bind, bone_delta = _fit_source_frames(asset)
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    if args.hard_controller_anchors:
        global_bind, local_bind = _hard_anchor_target_bind(asset, global_bind)
        bone_delta = global_bind @ np.linalg.inv(old_global)
    old_head = np.asarray(asset.target_bone_head, dtype=np.float64)
    old_tail = np.asarray(asset.target_bone_tail, dtype=np.float64)
    target_head = np.einsum("bij,bj->bi", bone_delta[:, :3, :3], old_head) + bone_delta[:, :3, 3]
    target_tail = np.einsum("bij,bj->bi", bone_delta[:, :3, :3], old_tail) + bone_delta[:, :3, 3]
    if not np.allclose(
        global_bind @ np.linalg.inv(global_bind), np.eye(4)[None], atol=1.0e-5, rtol=0.0
    ):
        raise RuntimeError("reframed target bind is not invertible")

    metadata = dict(asset.metadata or {})
    metadata["stage1_target_bind_reframe"] = {
        "base_asset": str(args.asset.resolve()),
        "method": "source_rig_target_rest_joint_reframe_v1",
        "hard_controller_anchors": bool(args.hard_controller_anchors),
        "preserves_source_weights": True,
        "preserves_source_hierarchy": True,
        "requires_blender_at_runtime": False,
        "requires_pose_rebake": False,
    }
    candidate = type(asset)(
        **{
            **asset.__dict__,
            "target_rest_global": np.asarray(global_bind, dtype=np.float32),
            "target_rest_local": np.asarray(local_bind, dtype=np.float32),
            "target_inverse_bind": np.linalg.inv(global_bind).astype(np.float32),
            "target_bone_head": target_head.astype(np.float32),
            "target_bone_tail": target_tail.astype(np.float32),
            "metadata": metadata,
        }
    )
    candidate = with_source_driver_coupling(candidate)
    candidate.validate()
    zero = skin_vertices(candidate, np.zeros((55, 3), dtype=np.float32))
    zero_error = float(np.max(np.linalg.norm(zero - candidate.vertices_rest, axis=1)))
    if zero_error > 1.0e-5:
        raise RuntimeError(f"reframed bind does not reproduce rest vertices: {zero_error:.8f} m")
    after = _anchor_errors(candidate)
    valid = np.isfinite(before) & np.isfinite(after)
    report = {
        "base_asset": str(args.asset.resolve()),
        "output": str(args.output.resolve()),
        "method": "source_rig_target_rest_joint_reframe_v1",
        "source_weights_unchanged": True,
        "source_hierarchy_unchanged": True,
        "runtime_requires_blender": False,
        "runtime_requires_pose_rebake": False,
        "zero_pose_vertex_error_m": zero_error,
        "controller_anchor_error_before_rms_m": float(np.sqrt(np.mean(before[valid] ** 2))),
        "controller_anchor_error_after_rms_m": float(np.sqrt(np.mean(after[valid] ** 2))),
        "controller_anchor_error_before_max_m": float(np.max(before[valid])),
        "controller_anchor_error_after_max_m": float(np.max(after[valid])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rigged_asset(args.output, candidate)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
