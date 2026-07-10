"""Versioned canonical dynamic scene snapshot (Genesis world frame).

Consumers include UE LiveSync, state logs, and external controllers.

Coordinates follow genesis_canonical_rh_z_up_m; UE-facing adapters flip Y only when bridging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = 1


def runtime_human_overlay(runtime: Any) -> dict[str, Any]:
    """Merge optional runtime-authored human dict/callback into canonical snapshots."""
    rh = getattr(runtime, "amongus_canonical_human", None)
    if callable(rh):
        rh = rh()
    return dict(rh) if isinstance(rh, dict) else {}


@dataclass
class CanonicalSceneStateV1:
    schema_version: int = SCHEMA_VERSION
    sim_step_index: int = 0
    wall_time_ns: int = 0
    sim_time_ns: int = 0
    source_time_ns: int = 0
    clock_domain: str = "genesis_sim"
    robot_entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    human: dict[str, Any] = field(default_factory=dict)
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    contacts: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def canonical_scene_state_to_dict(state: CanonicalSceneStateV1) -> dict[str, Any]:
    return {
        "schema_version": int(state.schema_version),
        "sim_step_index": int(state.sim_step_index),
        "frame_index": int(state.sim_step_index),
        "wall_time_ns": int(state.wall_time_ns),
        "sim_time_ns": int(state.sim_time_ns),
        "source_time_ns": int(state.source_time_ns),
        "clock_domain": str(state.clock_domain),
        "robot_entities": dict(state.robot_entities),
        "human": dict(state.human),
        "objects": dict(state.objects),
        "contacts": list(state.contacts),
        "extras": dict(state.extras),
    }


def snapshot_canonical_scene_state_v1(
    runtime: Any,
    *,
    robot_names: Iterable[str] | None = None,
    sim_step_index: int = 0,
    wall_time_ns: int = 0,
    sim_time_ns: int = 0,
    source_time_ns: int = 0,
    clock_domain: str = "genesis_sim",
    human: dict[str, Any] | None = None,
    objects: dict[str, dict[str, Any]] | None = None,
    contacts: list[dict[str, Any]] | None = None,
    extras: dict[str, Any] | None = None,
) -> CanonicalSceneStateV1:
    """Build a lightweight snapshot from GenesisPlatformRuntime (Genesis canonical frame)."""
    import os

    from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime

    if not isinstance(runtime, GenesisPlatformRuntime):
        raise TypeError(f"Expected GenesisPlatformRuntime, got {type(runtime).__name__}")

    merged_extras = dict(extras or {})
    sid = str(os.environ.get("AMONGUS_SESSION_ID", "") or "").strip()
    if sid:
        merged_extras.setdefault("session_id", sid)
    merged_extras.setdefault("geometry_authority", "genesis_world")

    names = list(robot_names) if robot_names is not None else list(runtime.embodiments.keys())
    robots: dict[str, dict[str, Any]] = {}
    for name in names:
        try:
            embodiment = runtime.embodiments[name]
            q = runtime.get_robot_joint_positions(name).reshape(-1).astype(np.float64)
            tcp = runtime.get_tcp_pose(name).reshape(-1).astype(np.float64)
            robots[name] = {
                "joint_positions": q.tolist(),
                "tcp_pose_pos_quat_wxyz": tcp.tolist(),
                "tcp_frame": str(embodiment.end_effector.tcp_frame),
            }
        except Exception as exc:
            robots[name] = {"error": repr(exc)}

    merged_human = runtime_human_overlay(runtime)
    if human:
        merged_human.update(human)
    runtime_objects = {}
    dyn = getattr(runtime, "dynamic_entities", None)
    if dyn is not None and hasattr(dyn, "snapshot"):
        try:
            runtime_objects = dict(dyn.snapshot())
        except Exception:
            runtime_objects = {}

    return CanonicalSceneStateV1(
        sim_step_index=int(sim_step_index),
        wall_time_ns=int(wall_time_ns),
        sim_time_ns=int(sim_time_ns),
        source_time_ns=int(source_time_ns or wall_time_ns),
        clock_domain=str(clock_domain or "genesis_sim"),
        robot_entities=robots,
        human=merged_human,
        objects={**runtime_objects, **dict(objects or {})},
        contacts=list(contacts or []),
        extras=dict(merged_extras),
    )
