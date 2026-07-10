"""Typed payloads for external controllers talking to Genesis (PEIRASTIC-compatible evolution)."""

from __future__ import annotations

from typing import Any, TypedDict


class ObservationEnvelopeV1(TypedDict, total=False):
    schema_version: int
    sim_step_index: int
    wall_time_ns: int
    canonical_scene: dict[str, Any]


def observation_envelope_v1_from_canonical(canonical_scene: dict[str, Any]) -> ObservationEnvelopeV1:
    return {
        "schema_version": 1,
        "sim_step_index": int(canonical_scene.get("sim_step_index", 0)),
        "wall_time_ns": int(canonical_scene.get("wall_time_ns", 0)),
        "canonical_scene": dict(canonical_scene),
    }


class ControlCommandV1(TypedDict, total=False):
    schema_version: int
    sim_step_index: int
    robot_joint_targets: dict[str, list[float]]
    robot_joint_efforts: dict[str, list[float]]
    extras: dict[str, Any]
