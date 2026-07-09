"""Scan trajectory and waypoint types (world frame)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Waypoint:
    """One pose sample along a scan path (world coordinates)."""

    p_world: np.ndarray              # (3,) m
    tool_axis_world: np.ndarray      # (3,) unit — surface outward normal / TCP +Z desired
    axis_tol_deg: float = 10.0
    pos_tol_m: float = 0.015
    roll_range_deg: tuple[float, float] | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.p_world = np.asarray(self.p_world, dtype=np.float64).reshape(3)
        ax = np.asarray(self.tool_axis_world, dtype=np.float64).reshape(3)
        n = np.linalg.norm(ax)
        if n < 1e-9:
            raise ValueError("tool_axis_world must be non-zero")
        self.tool_axis_world = ax / n

    def with_relaxed_tolerances(self, factor: float = 2.0) -> "Waypoint":
        return Waypoint(
            p_world=self.p_world.copy(),
            tool_axis_world=self.tool_axis_world.copy(),
            axis_tol_deg=float(self.axis_tol_deg) * factor,
            pos_tol_m=float(self.pos_tol_m) * factor,
            roll_range_deg=self.roll_range_deg,
            weight=self.weight,
        )


@dataclass
class ScanTrajectory:
    waypoints: list[Waypoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def n(self) -> int:
        return len(self.waypoints)

    def arc_length_m(self, index: int) -> float:
        """Cumulative path length from waypoint 0 to ``index`` (inclusive)."""
        if index < 0:
            return 0.0
        index = min(index, self.n - 1)
        if index == 0:
            return 0.0
        pts = np.stack([wp.p_world for wp in self.waypoints[: index + 1]], axis=0)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(np.sum(seg))

    def segment_lengths(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros(0, dtype=np.float64)
        pts = np.stack([wp.p_world for wp in self.waypoints], axis=0)
        return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def load_trajectory_json(path: str | Path) -> ScanTrajectory:
    """Load trajectory from JSON.

    Format::

        {
          "waypoints": [
            {"p": [x,y,z], "tool_axis": [ux,uy,uz],
             "axis_tol_deg": 10, "pos_tol_m": 0.015, "weight": 1.0},
            ...
          ]
        }
    """
    raw = json.loads(Path(path).read_text())
    wps: list[Waypoint] = []
    for item in raw.get("waypoints", []):
        wps.append(
            Waypoint(
                p_world=np.asarray(item["p"], dtype=float),
                tool_axis_world=np.asarray(item.get("tool_axis", item.get("tool_axis_world", [0, 0, 1])), dtype=float),
                axis_tol_deg=float(item.get("axis_tol_deg", 10.0)),
                pos_tol_m=float(item.get("pos_tol_m", 0.015)),
                weight=float(item.get("weight", 1.0)),
            )
        )
    return ScanTrajectory(waypoints=wps)


def save_trajectory_json(traj: ScanTrajectory, path: str | Path) -> None:
    data = {
        "waypoints": [
            {
                "p": wp.p_world.tolist(),
                "tool_axis": wp.tool_axis_world.tolist(),
                "axis_tol_deg": wp.axis_tol_deg,
                "pos_tol_m": wp.pos_tol_m,
                "weight": wp.weight,
            }
            for wp in traj.waypoints
        ]
    }
    Path(path).write_text(json.dumps(data, indent=2))
