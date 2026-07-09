"""Pinocchio + HPP-FCL self-collision distance queries for CBF constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

DEFAULT_COLLISION_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.collision.urdf"
)
DEFAULT_PAIR_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "collision_pairs.yaml"
)


@dataclass
class CollisionPairInfo:
    pair_index: int
    geom_a: int
    geom_b: int
    name_a: str
    name_b: str
    distance: float
    normal: np.ndarray          # unit vector from B toward A (base frame)
    point_a: np.ndarray
    point_b: np.ndarray


@dataclass
class CollisionConfig:
    enabled: bool = True
    d_safe: float = 0.03
    d_activate: float = 0.08
    gamma: float = 5.0
    max_pairs: int = 8
    collision_urdf: Path = DEFAULT_COLLISION_URDF
    pair_config: Path = DEFAULT_PAIR_CONFIG


def _geom_name_map(geom_model: pin.GeometryModel) -> dict[str, int]:
    return {go.name: i for i, go in enumerate(geom_model.geometryObjects)}


def _disable_pairs(geom_model: pin.GeometryModel, disabled: list[list[str]]) -> None:
    name_to_id = _geom_name_map(geom_model)
    for pair in disabled:
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        if a not in name_to_id or b not in name_to_id:
            continue
        cp = pin.CollisionPair(name_to_id[a], name_to_id[b])
        if geom_model.existCollisionPair(cp):
            geom_model.removeCollisionPair(cp)


class CollisionModel:
    """Self-collision geometry loaded from a collision-capable URDF."""

    def __init__(
        self,
        kin_model: pin.Model,
        *,
        collision_urdf: str | Path | None = None,
        pair_config: str | Path | None = None,
    ) -> None:
        self.collision_urdf = Path(collision_urdf or DEFAULT_COLLISION_URDF)
        if not self.collision_urdf.exists():
            raise FileNotFoundError(f"collision URDF not found: {self.collision_urdf}")
        mesh_dir = self.collision_urdf.parent
        self.model = kin_model
        self.geom_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.collision_urdf),
            pin.COLLISION,
            package_dirs=[str(mesh_dir)],
        )
        self.geom_model.addAllCollisionPairs()
        pair_path = Path(pair_config or DEFAULT_PAIR_CONFIG)
        if pair_path.exists():
            raw = yaml.safe_load(pair_path.read_text()) or {}
            _disable_pairs(self.geom_model, raw.get("disabled_pairs", []))
        self.geom_data = self.geom_model.createData()
        self._kin_data = self.model.createData()
        self._q = np.zeros(self.model.nq, dtype=float)

    def update(self, q_rad: np.ndarray) -> None:
        self._q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self._kin_data, self._q)
        pin.updateGeometryPlacements(
            self.model, self._kin_data, self.geom_model, self.geom_data
        )
        pin.computeDistances(
            self.model, self._kin_data, self.geom_model, self.geom_data, self._q
        )

    def pair_info(self, pair_index: int) -> CollisionPairInfo | None:
        dr = self.geom_data.distanceResults[pair_index]
        d = float(dr.min_distance)
        if not np.isfinite(d):
            return None
        pa = np.asarray(dr.getNearestPoint1(), dtype=float)
        pb = np.asarray(dr.getNearestPoint2(), dtype=float)
        cp = self.geom_model.collisionPairs[pair_index]
        ga, gb = int(cp.first), int(cp.second)
        na = pa - pb
        n_norm = float(np.linalg.norm(na))
        if n_norm < 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = na / n_norm
        go_a = self.geom_model.geometryObjects[ga]
        go_b = self.geom_model.geometryObjects[gb]
        return CollisionPairInfo(
            pair_index=pair_index,
            geom_a=ga,
            geom_b=gb,
            name_a=go_a.name,
            name_b=go_b.name,
            distance=d,
            normal=normal,
            point_a=pa,
            point_b=pb,
        )

    def all_pairs(self) -> list[CollisionPairInfo]:
        out: list[CollisionPairInfo] = []
        for i in range(len(self.geom_model.collisionPairs)):
            info = self.pair_info(i)
            if info is not None:
                out.append(info)
        return out

    def active_pairs(self, d_activate: float) -> list[CollisionPairInfo]:
        pairs = [p for p in self.all_pairs() if p.distance < d_activate]
        pairs.sort(key=lambda p: p.distance)
        return pairs

    def min_distance(self) -> float:
        pairs = self.all_pairs()
        if not pairs:
            return float("inf")
        return min(p.distance for p in pairs)
