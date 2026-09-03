"""Typed mode payloads. ``to_json()`` matches the live SHM dict protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from peirastic.api.vel_filter import pack_vel_filter


def _as_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
    return value


def _dump(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in data.items():
        if val is None:
            continue
        out[key] = _as_list(val) if not isinstance(val, dict) else dict(val)
    return out


def _force_overlay(
    *,
    force: float | list[float] | None = None,
    force_axes: list[float] | None = None,
    track_axes: list[float] | None = None,
    selection: list[float] | None = None,
    mask_force_from_path: bool | None = None,
    filter: bool | list[float] | None = None,
    follow: bool | None = None,
    slew: bool | None = None,
    slew_axes: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = dict(extra or {})
    if force is not None:
        if isinstance(force, (int, float, np.floating)):
            raw["desired_z"] = float(force)
            raw["desired_force"] = float(force)
        else:
            arr = np.asarray(force, dtype=float).reshape(-1)
            raw["desired_force"] = arr.tolist()
            if arr.size >= 3:
                raw["desired_z"] = float(arr[2])
    if force_axes is not None:
        raw["force_axes"] = list(force_axes)
    if track_axes is not None:
        raw["track_axes"] = list(track_axes)
    if selection is not None:
        raw["selection"] = list(selection)
    if mask_force_from_path is not None:
        raw["mask_force_from_path"] = bool(mask_force_from_path)
    packed = pack_vel_filter(
        filter=filter, follow=follow, slew=slew, slew_axes=slew_axes
    )
    if packed is not None:
        raw["filter"] = packed
    return raw


@dataclass
class ServoTwistPayload:
    v_cmd: list[float] | None = None
    duration_s: float | None = None
    label: str | None = None
    filter: bool | list[float] | None = None
    follow: bool | None = None
    slew: bool | None = None
    slew_axes: list[float] | None = None

    def to_json(self) -> dict[str, Any]:
        return _dump(
            {
                "v_cmd": self.v_cmd,
                "duration_s": self.duration_s,
                "label": self.label,
                "filter": pack_vel_filter(
                    filter=self.filter,
                    follow=self.follow,
                    slew=self.slew,
                    slew_axes=self.slew_axes,
                ),
            }
        )


@dataclass
class TrackCartesianPayload:
    reference: str = "hold"
    poses: list[list[float]] | None = None
    points: list[list[float]] | None = None
    rpy: list[list[float]] | list[float] | None = None
    speed_m_s: float | None = None
    soft_start: bool | None = None
    ramp_s: float | None = None
    amplitude_x_m: float | None = None
    amplitude_y_m: float | None = None
    rot_amp_rad: list[float] | float | None = None
    rot_amp_deg: list[float] | float | None = None
    period_s: float | None = None
    max_vel_m_s: float | None = None
    duration_s: float | None = None
    max_lin_vel_m_s: float | None = None
    move_kp: float | None = None
    label: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _dump(
            {
                "reference": self.reference,
                "poses": self.poses,
                "points": self.points,
                "rpy": self.rpy,
                "speed_m_s": self.speed_m_s,
                "soft_start": self.soft_start,
                "ramp_s": self.ramp_s,
                "amplitude_x_m": self.amplitude_x_m,
                "amplitude_y_m": self.amplitude_y_m,
                "rot_amp_rad": self.rot_amp_rad,
                "rot_amp_deg": self.rot_amp_deg,
                "period_s": self.period_s,
                "max_vel_m_s": self.max_vel_m_s,
                "duration_s": self.duration_s,
                "max_lin_vel_m_s": self.max_lin_vel_m_s,
                "move_kp": self.move_kp,
                "label": self.label,
            }
        )


@dataclass
class HfpcPayload:
    """Position-force hybrid. Position axes track a pose reference."""

    reference: str = "polyline"
    poses: list[list[float]] | None = None
    speed_m_s: float | None = None
    law: str = "tff"
    duration_s: float | None = None
    label: str | None = None
    soft_start: bool | None = None
    ramp_s: float | None = None
    amplitude_x_m: float | None = None
    amplitude_y_m: float | None = None
    period_s: float | None = None
    max_vel_m_s: float | None = None
    force: float | list[float] | None = None
    force_axes: list[float] | None = None
    track_axes: list[float] | None = None
    selection: list[float] | None = None
    mask_force_from_path: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        law = str(self.law or "tff").lower()
        if law not in ("tff", "admittance"):
            raise ValueError(f"hfpc law must be 'tff' or 'admittance', got {self.law!r}")
        out = _dump(
            {
                "reference": self.reference,
                "poses": self.poses,
                "speed_m_s": self.speed_m_s,
                "use_tff_split": law == "tff",
                "duration_s": self.duration_s,
                "label": self.label,
                "soft_start": self.soft_start,
                "ramp_s": self.ramp_s,
                "amplitude_x_m": self.amplitude_x_m,
                "amplitude_y_m": self.amplitude_y_m,
                "period_s": self.period_s,
                "max_vel_m_s": self.max_vel_m_s,
            }
        )
        out.update(
            _force_overlay(
                force=self.force,
                force_axes=self.force_axes,
                track_axes=self.track_axes,
                selection=self.selection,
                mask_force_from_path=self.mask_force_from_path,
                extra=self.extra,
            )
        )
        return out


@dataclass
class HfvcPayload:
    """Velocity-force hybrid. Track axes follow a twist."""

    reference: str = "pad"
    v_cmd: list[float] | None = None
    duration_s: float | None = None
    label: str | None = None
    force: float | list[float] | None = None
    force_axes: list[float] | None = None
    track_axes: list[float] | None = None
    selection: list[float] | None = None
    mask_force_from_path: bool = True
    filter: bool | list[float] | None = None
    follow: bool | None = None
    slew: bool | None = None
    slew_axes: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        kind = str(self.reference or "pad")
        if kind not in ("pad", "twist", "servo"):
            raise ValueError(f"hfvc source must be pad|twist|servo, got {kind!r}")
        out = _dump(
            {
                "reference": kind,
                "v_cmd": self.v_cmd,
                "duration_s": self.duration_s,
                "label": self.label,
            }
        )
        out.update(
            _force_overlay(
                force=self.force,
                force_axes=self.force_axes,
                track_axes=self.track_axes,
                selection=self.selection,
                mask_force_from_path=self.mask_force_from_path,
                filter=self.filter,
                follow=self.follow,
                slew=self.slew,
                slew_axes=self.slew_axes,
                extra=self.extra,
            )
        )
        return out


@dataclass
class MoveJPayload:
    q_target: list[float] | None = None
    q_start: list[float] | None = None
    pose: list[float] | None = None
    rail_m: float | None = None
    duration_s: float | None = None
    v: float | None = None
    label: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _dump(
            {
                "q_target": self.q_target,
                "q_start": self.q_start,
                "pose": self.pose,
                "rail_m": self.rail_m,
                "duration_s": self.duration_s,
                "v": self.v,
                "label": self.label,
            }
        )


@dataclass
class MoveLPayload:
    pose: list[float] | None = None
    poses: list[list[float]] | None = None
    q_start: list[float] | None = None
    q_target: list[float] | None = None
    speed_m_s: float | None = None
    duration_s: float | None = None
    max_lin_vel_m_s: float | None = None
    v: float | None = None
    soft_start: bool | None = None
    ramp_s: float | None = None
    label: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _dump(
            {
                "pose": self.pose,
                "poses": self.poses,
                "q_start": self.q_start,
                "q_target": self.q_target,
                "speed_m_s": self.speed_m_s,
                "duration_s": self.duration_s,
                "max_lin_vel_m_s": self.max_lin_vel_m_s,
                "v": self.v,
                "soft_start": self.soft_start,
                "ramp_s": self.ramp_s,
                "label": self.label,
            }
        )
