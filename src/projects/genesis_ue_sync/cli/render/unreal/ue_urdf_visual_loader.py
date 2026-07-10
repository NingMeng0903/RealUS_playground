"""Reusable URDF visual armature for UE Editor Python: one FK solve updates every StaticMesh link."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ue_common_scene_loader as scene_loader

_REGISTRY: dict[str, UrdfVisualArticulatedRobot] = {}


@dataclass
class UrdfVisualArticulatedRobot:
    """Drive Panda-like URDF visual meshes by joint angles (radians), matching Genesis FK semantics."""

    robot_id: str
    urdf_path: Path
    base_pos_m: tuple[float, float, float]
    base_quat_xyzw: tuple[float, float, float, float] | None
    use_collision_geometry: bool
    use_visual_mesh: bool
    allow_collision_fallback: bool
    visual_asset_root: str
    visual_mesh_scale: float
    folder: str
    color_rgba: tuple[float, float, float, float] | None
    actors_by_link: dict[str, object] = field(default_factory=dict)
    last_tip_actor_loc_cm: list[float] | None = None

    def apply_joint_positions(self, joint_positions: list[float]) -> dict[str, Any]:
        entries = scene_loader.compute_robot_visual_mesh_entries(
            urdf_path=self.urdf_path,
            base_pos_m=self.base_pos_m,
            joint_positions=[float(v) for v in joint_positions],
            base_quat_xyzw=self.base_quat_xyzw,
            use_collision_geometry=self.use_collision_geometry,
            use_visual_mesh=self.use_visual_mesh,
            allow_collision_fallback=self.allow_collision_fallback,
        )
        by_link = {str(item["link_name"]): item for item in entries}
        updated = 0
        missing: list[str] = []
        for link_name, editor_actor in self.actors_by_link.items():
            item = by_link.get(str(link_name))
            if item is None:
                missing.append(str(link_name))
                continue
            actor = scene_loader.resolve_active_world_actor(editor_actor)
            if actor is None:
                missing.append(str(link_name))
                continue
            scene_loader.apply_static_mesh_actor_pose_genesis(
                actor,
                pos_m=tuple(float(v) for v in item["pos_m"]),
                rot3=item["rot3"],
            )
            updated += 1
        result: dict[str, Any] = {
            "robot_id": self.robot_id,
            "updated_links": int(updated),
            "missing_pose_links": missing,
            "solve_links_n": len(by_link),
            "world_kind": scene_loader.world_kind_for_canonical_tick(),
        }
        link_names_sorted = sorted(str(k) for k in self.actors_by_link.keys())
        if link_names_sorted:
            link0 = link_names_sorted[0]
            link_tip = link_names_sorted[-1]
            actor0 = scene_loader.resolve_active_world_actor(self.actors_by_link.get(link0))
            actor_tip = scene_loader.resolve_active_world_actor(self.actors_by_link.get(link_tip))
            if actor0 is not None:
                try:
                    loc0 = actor0.get_actor_location()
                    result["link0_label"] = link0
                    result["link0_actor_loc_cm"] = [float(loc0.x), float(loc0.y), float(loc0.z)]
                except Exception as exc:
                    result["link0_actor_loc_error"] = repr(exc)
            if actor_tip is not None:
                try:
                    loc_t = actor_tip.get_actor_location()
                    tip_loc = [float(loc_t.x), float(loc_t.y), float(loc_t.z)]
                    result["link_tip_label"] = link_tip
                    result["link_tip_actor_loc_cm"] = tip_loc
                    if self.last_tip_actor_loc_cm is not None and len(self.last_tip_actor_loc_cm) == 3:
                        result["link_tip_delta_cm"] = [
                            float(tip_loc[i]) - float(self.last_tip_actor_loc_cm[i]) for i in range(3)
                        ]
                    self.last_tip_actor_loc_cm = list(tip_loc)
                except Exception as exc:
                    result["link_tip_actor_loc_error"] = repr(exc)
        return result


def register_articulated_robot(
    robot_id: str,
    *,
    actors_by_link: dict[str, object],
    urdf_path: Path | str,
    base_pos_m: tuple[float, float, float],
    base_quat_xyzw: tuple[float, float, float, float] | None,
    use_collision_geometry: bool,
    use_visual_mesh: bool,
    allow_collision_fallback: bool,
    visual_asset_root: str,
    visual_mesh_scale: float,
    folder: str,
    color_rgba: tuple[float, float, float, float] | None,
) -> UrdfVisualArticulatedRobot:
    rid = str(robot_id).strip()
    if not rid:
        raise ValueError("robot_id must be non-empty.")
    inst = UrdfVisualArticulatedRobot(
        robot_id=rid,
        urdf_path=Path(urdf_path).expanduser().resolve(),
        base_pos_m=tuple(float(v) for v in base_pos_m),
        base_quat_xyzw=(
            tuple(float(v) for v in base_quat_xyzw) if base_quat_xyzw is not None else None
        ),
        use_collision_geometry=bool(use_collision_geometry),
        use_visual_mesh=bool(use_visual_mesh),
        allow_collision_fallback=bool(allow_collision_fallback),
        visual_asset_root=str(visual_asset_root),
        visual_mesh_scale=float(visual_mesh_scale),
        folder=str(folder),
        color_rgba=tuple(float(v) for v in color_rgba) if color_rgba is not None else None,
        actors_by_link=dict(actors_by_link),
    )
    _REGISTRY[rid] = inst
    return inst


def get_articulated_robot(robot_id: str) -> UrdfVisualArticulatedRobot | None:
    return _REGISTRY.get(str(robot_id).strip())


def apply_articulated_robot_joints(robot_id: str, joint_positions: list[float]) -> dict[str, Any]:
    robot = get_articulated_robot(robot_id)
    if robot is None:
        raise KeyError(f"No articulated URDF robot registered as {robot_id!r}. Known: {sorted(_REGISTRY)!r}")
    return robot.apply_joint_positions(joint_positions)


def unregister_articulated_robot(robot_id: str) -> None:
    _REGISTRY.pop(str(robot_id).strip(), None)


def registered_robot_ids() -> list[str]:
    return sorted(_REGISTRY.keys())


def robot_articulation_id_from_env() -> str | None:
    raw = str(os.environ.get("AMONGUS_REGISTER_URDF_ARTICULATION_ID", "")).strip()
    return raw or None
