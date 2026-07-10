from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeshDescriptor:
    vertex_count: int
    centroid_m: tuple[float, float, float]
    aabb_min_m: tuple[float, float, float]
    aabb_max_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    principal_axis_0_unit: tuple[float, float, float]
    principal_axis_1_unit: tuple[float, float, float]
    principal_axis_2_unit: tuple[float, float, float]
    principal_eigenvalues_m2: tuple[float, float, float]


def parse_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices parsed from {path}")
    return np.asarray(verts, dtype=np.float64)


def mesh_descriptor_from_vertices(vertices: np.ndarray) -> MeshDescriptor:
    pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    n = int(pts.shape[0])
    c = np.mean(pts, axis=0)
    centered = pts - c
    amin = np.min(pts, axis=0)
    amax = np.max(pts, axis=0)
    extent = tuple(float(x) for x in (amax - amin).tolist())
    if n < 4:
        u0 = (1.0, 0.0, 0.0)
        u1 = (0.0, 1.0, 0.0)
        u2 = (0.0, 0.0, 1.0)
        ev = (0.0, 0.0, 0.0)
    else:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        axes = vt
        u0 = tuple(float(x) for x in axes[0].tolist())
        u1 = tuple(float(x) for x in axes[1].tolist())
        u2 = tuple(float(x) for x in axes[2].tolist())
        ev = tuple(float(x * x) for x in s.tolist())
    return MeshDescriptor(
        vertex_count=n,
        centroid_m=tuple(float(x) for x in c.tolist()),
        aabb_min_m=tuple(float(x) for x in amin.tolist()),
        aabb_max_m=tuple(float(x) for x in amax.tolist()),
        extent_m=extent,
        principal_axis_0_unit=u0,
        principal_axis_1_unit=u1,
        principal_axis_2_unit=u2,
        principal_eigenvalues_m2=ev,
    )


def sorted_vertex_max_residual_m(a: np.ndarray, b: np.ndarray) -> float:
    """Max L2 distance after sorting rows lexicographically (multiset comparison)."""
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if aa.shape[0] != bb.shape[0]:
        raise ValueError(f"Vertex count mismatch: {aa.shape[0]} vs {bb.shape[0]}")
    sa = aa[np.lexsort((aa[:, 2], aa[:, 1], aa[:, 0]))]
    sb = bb[np.lexsort((bb[:, 2], bb[:, 1], bb[:, 0]))]
    return float(np.max(np.linalg.norm(sa - sb, axis=1)))
