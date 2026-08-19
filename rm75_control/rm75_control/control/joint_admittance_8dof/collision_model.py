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


@dataclass(frozen=True)
class BoundingSphere:
    """Conservative sphere enclosing one collision geometry in local coords."""

    center: np.ndarray
    radius: float


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
        # The collision URDF currently contains triangle meshes.  Keeping a
        # local sphere for every geometry makes the broadphase independent of
        # HPP-FCL internals and, since it encloses every mesh vertex, strictly
        # conservative for triangle meshes as well.
        self._bounding_spheres = tuple(
            self._make_bounding_sphere(
                go.geometry,
                mesh_scale=getattr(go, "meshScale", np.ones(3)),
            )
            for go in self.geom_model.geometryObjects
        )
        n_geoms = len(self.geom_model.geometryObjects)
        n_pairs = len(self.geom_model.collisionPairs)
        self._sphere_centers = np.array(
            [s.center for s in self._bounding_spheres], dtype=float
        ).reshape(n_geoms, 3)
        self._sphere_radii = np.array(
            [s.radius for s in self._bounding_spheres], dtype=float
        )
        self._pair_first = np.array(
            [int(cp.first) for cp in self.geom_model.collisionPairs], dtype=np.intp
        )
        self._pair_second = np.array(
            [int(cp.second) for cp in self.geom_model.collisionPairs], dtype=np.intp
        )
        self._geom_translation = np.zeros((n_geoms, 3), dtype=float)
        self._geom_rotation = np.zeros((n_geoms, 3, 3), dtype=float)
        self._world_centers = np.zeros((n_geoms, 3), dtype=float)
        self._exact_pair_indices: tuple[int, ...] = ()
        self._last_lower_bounds = np.full(n_pairs, np.inf, dtype=float)
        self._last_distance_threshold: float | None = None
        self._last_skipped_pair_indices: tuple[int, ...] = ()

    @staticmethod
    def _make_bounding_sphere(
        geometry: object,
        *,
        mesh_scale: np.ndarray,
    ) -> BoundingSphere:
        """Build a conservative local sphere from the mesh vertices.

        ``coal`` exposes ``vertices`` as a method in some versions and as an
        array in others.  For an unsupported primitive, use an infinite
        radius; that preserves the old full narrow-phase behaviour rather
        than risking a false negative in collision checking.
        """

        try:
            scale = np.asarray(mesh_scale, dtype=float).reshape(-1)
            # Pinocchio/HPP-FCL versions differ on whether meshScale has
            # already been baked into ``vertices``.  A non-unit scale is
            # therefore ambiguous; disable broadphase for that geometry
            # instead of risking a sphere that is too small.
            if (
                scale.size < 3
                or not np.all(np.isfinite(scale[:3]))
                or not np.allclose(scale[:3], np.ones(3), atol=1.0e-12, rtol=0.0)
            ):
                raise ValueError("non-unit mesh scale is not safely inferable")
            vertices = getattr(geometry, "vertices")
            if callable(vertices):
                vertices = vertices()
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                raise ValueError("geometry has no Nx3 vertices")
            if not np.all(np.isfinite(vertices)):
                raise ValueError("geometry vertices are non-finite")
            center = np.mean(vertices, axis=0)
            radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
            if not np.isfinite(radius):
                raise ValueError("geometry radius is non-finite")
            return BoundingSphere(center=center, radius=radius)
        except Exception:
            return BoundingSphere(
                center=np.zeros(3, dtype=float), radius=float("inf")
            )

    @property
    def bounding_spheres(self) -> tuple[BoundingSphere, ...]:
        """Precomputed local-space spheres, exposed for diagnostics/tests."""

        return self._bounding_spheres

    @property
    def exact_pair_indices(self) -> tuple[int, ...]:
        """Pair indices whose narrow-phase distance was evaluated this tick."""

        return self._exact_pair_indices

    @property
    def broadphase_lower_bounds(self) -> np.ndarray:
        """Latest sphere lower bound for every collision pair."""

        return self._last_lower_bounds.copy()

    @property
    def skipped_pair_indices(self) -> tuple[int, ...]:
        """Pair indices skipped by the latest broadphase update."""

        return self._last_skipped_pair_indices

    @property
    def distance_query_count(self) -> int:
        """Number of exact narrow-phase pair queries in the latest update."""

        return len(self._exact_pair_indices)

    def _refresh_world_centers(self) -> None:
        n_geoms = self._geom_translation.shape[0]
        oMg = self.geom_data.oMg
        for i in range(n_geoms):
            T = oMg[i]
            self._geom_translation[i] = T.translation
            self._geom_rotation[i] = T.rotation
        np.einsum(
            "nij,nj->ni",
            self._geom_rotation,
            self._sphere_centers,
            out=self._world_centers,
        )
        self._world_centers += self._geom_translation

    def _pair_lower_bounds(self) -> np.ndarray:
        """Sphere-sphere separation for every collision pair.

        The sphere-sphere separation is a lower bound on the mesh distance.
        Do not clamp to zero: a negative value is still a valid conservative
        lower bound for overlapping spheres.  Infinite-radius spheres (unsafe
        mesh scale) force ``-inf`` so those pairs always take the narrow phase.
        """

        self._refresh_world_centers()
        ga = self._pair_first
        gb = self._pair_second
        delta = self._world_centers[ga] - self._world_centers[gb]
        dist = np.linalg.norm(delta, axis=1)
        dist -= self._sphere_radii[ga]
        dist -= self._sphere_radii[gb]
        bad = ~(
            np.isfinite(self._sphere_radii[ga]) & np.isfinite(self._sphere_radii[gb])
        )
        if np.any(bad):
            dist = dist.copy()
            dist[bad] = -np.inf
        return dist

    def _pair_lower_bound(self, pair_index: int) -> float:
        """Scalar wrapper kept for tests; uses the vectorized path."""

        return float(self._pair_lower_bounds()[int(pair_index)])

    def update(
        self,
        q_rad: np.ndarray,
        *,
        kinematic_data: pin.Data | None = None,
        kinematics_ready: bool = False,
        distance_threshold: float | None = None,
    ) -> None:
        """Update witness distances, optionally reusing this tick's FK data.

        ``RobotKinematics.jacobian`` has already computed joint Jacobians and
        frame placements for the immutable measured-state snapshot used by
        QPIK.  Reusing that data avoids a second forward-kinematics pass while
        preserving the exact same collision geometry and distance queries.
        Standalone callers retain the original self-contained behaviour.
        """

        self._q = np.asarray(q_rad, dtype=float)
        data = self._kin_data if kinematic_data is None else kinematic_data
        if not kinematics_ready:
            pin.forwardKinematics(self.model, data, self._q)
        pin.updateGeometryPlacements(
            self.model, data, self.geom_model, self.geom_data
        )
        # Placements are already current; the five-argument overload would
        # recompute them a second time.  A missing threshold preserves the
        # standalone/full narrow-phase API.  CBF callers provide the current
        # activation+hysteresis band and use the conservative sphere test.
        n_pairs = len(self.geom_model.collisionPairs)
        self._exact_pair_indices = ()
        self._last_distance_threshold = (
            None if distance_threshold is None else float(distance_threshold)
        )
        self._last_lower_bounds = np.full(n_pairs, np.inf, dtype=float)
        self._last_skipped_pair_indices = ()
        if distance_threshold is None:
            pin.computeDistances(self.geom_model, self.geom_data)
            self._exact_pair_indices = tuple(range(n_pairs))
            return

        threshold = float(distance_threshold)
        if not np.isfinite(threshold):
            # An infinite threshold is equivalent to the old full query and
            # avoids treating NaNs as an opportunity to skip safety checks.
            pin.computeDistances(self.geom_model, self.geom_data)
            self._exact_pair_indices = tuple(range(n_pairs))
            return

        self._last_lower_bounds = self._pair_lower_bounds()

        # Every pair whose true distance is <= threshold has a sphere lower
        # bound <= threshold, so this set cannot omit an active CBF pair.
        selected = np.flatnonzero(self._last_lower_bounds <= threshold).tolist()
        # Keep closest-pair telemetry meaningful even when every sphere is
        # outside the activation band.  The minimum lower-bound pair is the
        # only extra narrow-phase query needed for that telemetry.
        if n_pairs and not selected:
            selected = [int(np.argmin(self._last_lower_bounds))]
        selected = tuple(sorted(set(int(i) for i in selected)))
        for i in selected:
            pin.computeDistance(self.geom_model, self.geom_data, int(i))
        self._exact_pair_indices = selected
        selected_set = set(selected)
        self._last_skipped_pair_indices = tuple(
            i for i in range(n_pairs) if i not in selected_set
        )

    def pair_info(self, pair_index: int) -> CollisionPairInfo | None:
        if int(pair_index) not in self._exact_pair_indices:
            return None
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
        for i in self._exact_pair_indices:
            info = self.pair_info(i)
            if info is not None:
                out.append(info)
        return out

    def active_pairs(self, d_activate: float) -> list[CollisionPairInfo]:
        # Reading witness points/normals allocates several arrays per pair.
        # First filter on HPP-FCL's scalar distance result, then materialise
        # full information only for pairs that can enter/leave a CBF slot.
        threshold = float(d_activate)
        indices = [
            i
            for i in self._exact_pair_indices
            for result in (self.geom_data.distanceResults[i],)
            if np.isfinite(float(result.min_distance))
            and float(result.min_distance) < threshold
        ]
        pairs = [self.pair_info(i) for i in indices]
        pairs = [p for p in pairs if p is not None]
        pairs.sort(key=lambda p: p.distance)
        return pairs

    def min_distance(self) -> float:
        distances = [
            float(self.geom_data.distanceResults[i].min_distance)
            for i in self._exact_pair_indices
            for result in (self.geom_data.distanceResults[i],)
            if np.isfinite(float(result.min_distance))
        ]
        if not distances:
            return float("inf")
        return min(distances)

    def closest_pair(self) -> CollisionPairInfo | None:
        """Nearest pair after ``update``; not the CBF slot occupancy count."""
        best_i = -1
        best_d = float("inf")
        for i in self._exact_pair_indices:
            result = self.geom_data.distanceResults[i]
            d = float(result.min_distance)
            if np.isfinite(d) and d < best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            return None
        return self.pair_info(best_i)
