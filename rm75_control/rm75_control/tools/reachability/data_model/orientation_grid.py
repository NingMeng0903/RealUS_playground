"""Discretisations of the unit sphere for tool-axis sampling + roll grid.

Only the tool-axis (Zacharias 5-DOF) grid is required by default; the roll grid
is opt-in for 6-DOF maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from rm75_control.tools.reachability.data_model.schema import (
    OrientationGridConfig,
    RollGridConfig,
)

# ---------------------------------------------------------------------------
# Icosphere generation (subdivided icosahedron)
# ---------------------------------------------------------------------------
_PHI = (1.0 + 5.0**0.5) / 2.0


def _icosahedron_vertices() -> np.ndarray:
    v = np.array(
        [
            (-1, _PHI, 0),
            (1, _PHI, 0),
            (-1, -_PHI, 0),
            (1, -_PHI, 0),
            (0, -1, _PHI),
            (0, 1, _PHI),
            (0, -1, -_PHI),
            (0, 1, -_PHI),
            (_PHI, 0, -1),
            (_PHI, 0, 1),
            (-_PHI, 0, -1),
            (-_PHI, 0, 1),
        ],
        dtype=np.float64,
    )
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _icosahedron_faces() -> np.ndarray:
    return np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )


def subdivide_icosphere(subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices (V, 3) float64 unit-length, faces (F, 3) int64)``.

    ``subdiv=0`` → 12 verts / 20 faces (base icosahedron).
    ``subdiv=1`` → 42 verts / 80 faces (~31° avg spacing).
    ``subdiv=2`` → 162 verts / 320 faces (~19°).
    ``subdiv=3`` → 642 verts / 1280 faces (~11°).
    """
    if subdiv < 0:
        raise ValueError(f"subdiv must be >= 0, got {subdiv}")
    verts = _icosahedron_vertices()
    faces = _icosahedron_faces()
    for _ in range(subdiv):
        verts, faces = _subdivide_once(verts, faces)
    return verts, faces


def _subdivide_once(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint_cache: dict[tuple[int, int], int] = {}
    verts_list = verts.tolist()

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        idx = midpoint_cache.get(key)
        if idx is not None:
            return idx
        mp = (np.asarray(verts_list[a]) + np.asarray(verts_list[b])) * 0.5
        mp = mp / np.linalg.norm(mp)
        verts_list.append(mp.tolist())
        idx = len(verts_list) - 1
        midpoint_cache[key] = idx
        return idx

    new_faces: list[list[int]] = []
    for a, b, c in faces:
        ab = midpoint(int(a), int(b))
        bc = midpoint(int(b), int(c))
        ca = midpoint(int(c), int(a))
        new_faces.extend(
            [
                [int(a), ab, ca],
                [int(b), bc, ab],
                [int(c), ca, bc],
                [ab, bc, ca],
            ]
        )
    return np.asarray(verts_list, dtype=np.float64), np.asarray(new_faces, dtype=np.int64)


def fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) unit vectors placed by the Fibonacci lattice."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    idx = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0**0.5) * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# ToolAxisGrid (5-DOF)
# ---------------------------------------------------------------------------
@dataclass
class ToolAxisGrid:
    """Abstract base: exposes vectors + nearest-neighbour + neighbourhood query."""

    vectors: np.ndarray  # (N, 3) unit vectors
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vectors, dtype=np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-9):
            v = v / np.clip(norms, 1e-12, None)
        object.__setattr__(self, "vectors", v)
        object.__setattr__(self, "_tree", cKDTree(v))

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    def nearest(self, d: np.ndarray) -> int:
        """Nearest orientation index for one direction (3,)."""
        _, i = self._tree.query(np.asarray(d, dtype=np.float64))
        return int(i)

    def nearest_batch(self, d: np.ndarray) -> np.ndarray:
        """Nearest orientation index for a batch of directions (N,3)."""
        _, i = self._tree.query(np.asarray(d, dtype=np.float64))
        return i.astype(np.int64)

    def neighbors(self, seed_idx: int, half_angle_deg: float) -> np.ndarray:
        """Indices whose vector lies within ``half_angle_deg`` of vector ``seed_idx``.

        Uses chord-length: ``chord = 2 sin(angle/2)``.
        """
        chord = 2.0 * np.sin(0.5 * np.radians(float(half_angle_deg)))
        idx = self._tree.query_ball_point(self.vectors[seed_idx], r=chord)
        return np.asarray(sorted(int(i) for i in idx), dtype=np.int64)

    def neighbors_of_dir(self, d: np.ndarray, half_angle_deg: float) -> np.ndarray:
        chord = 2.0 * np.sin(0.5 * np.radians(float(half_angle_deg)))
        idx = self._tree.query_ball_point(np.asarray(d, dtype=np.float64), r=chord)
        return np.asarray(sorted(int(i) for i in idx), dtype=np.int64)


@dataclass
class IcosphereToolAxisGrid(ToolAxisGrid):
    """Icosphere-based grid; exposes faces for the direction-sphere glyph."""

    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    subdiv: int = 3

    @classmethod
    def build(cls, subdiv: int = 3) -> "IcosphereToolAxisGrid":
        verts, faces = subdivide_icosphere(subdiv)
        obj = cls(vectors=verts, faces=faces, subdiv=subdiv)
        return obj


def make_tool_axis_grid(cfg: OrientationGridConfig) -> ToolAxisGrid:
    if cfg.kind == "icosphere":
        return IcosphereToolAxisGrid.build(subdiv=cfg.subdiv)
    if cfg.kind == "fibonacci":
        return ToolAxisGrid(vectors=fibonacci_sphere(cfg.fibonacci_n))
    raise ValueError(f"unknown tool axis grid kind {cfg.kind!r}")


# ---------------------------------------------------------------------------
# RollGrid (6-DOF option)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RollGrid:
    """Uniform roll about tool axis in ``[0, 2π)``."""

    step_deg: float = 15.0

    @property
    def n(self) -> int:
        return int(round(360.0 / float(self.step_deg)))

    @property
    def angles_rad(self) -> np.ndarray:
        return np.deg2rad(np.arange(self.n, dtype=np.float64) * self.step_deg)

    def nearest(self, angle_rad: float) -> int:
        step = np.deg2rad(self.step_deg)
        a = float(angle_rad) % (2.0 * np.pi)
        return int(round(a / step)) % self.n

    @classmethod
    def from_config(cls, cfg: RollGridConfig | None) -> "RollGrid | None":
        if cfg is None or not cfg.enabled:
            return None
        return cls(step_deg=cfg.step_deg)
