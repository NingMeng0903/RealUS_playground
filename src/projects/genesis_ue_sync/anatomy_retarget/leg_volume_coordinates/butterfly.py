"""Interpolatory Butterfly-style subdivision for leg material surfaces.

Unlike Loop subdivision, this keeps every existing vertex fixed. Only edge
points are inserted, with a stencil back to the input atlas vertices so LBS
weights and material coordinates remain traceable after ICP/registration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atlas import LegVolumeAtlas, _compute_vertex_normals


@dataclass(frozen=True)
class ButterflySurface:
    vertices: np.ndarray
    faces: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    d: np.ndarray
    normals: np.ndarray
    full_vertex_indices: np.ndarray
    stencil_indptr: np.ndarray
    stencil_indices: np.ndarray
    stencil_weights: np.ndarray
    source_full_vertex_indices: np.ndarray
    level: int
    stencil_nnz: int


def _add_weight(row: dict[int, float], col: int, weight: float) -> None:
    if abs(float(weight)) <= 1.0e-14:
        return
    row[int(col)] = row.get(int(col), 0.0) + float(weight)


def _combine_rows(rows: list[dict[int, float]], terms: list[tuple[int, float]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for src, scale in terms:
        for col, weight in rows[int(src)].items():
            _add_weight(out, col, float(scale) * float(weight))
    return out


def _build_topology(vertex_count: int, faces: np.ndarray) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    edge_opp: dict[tuple[int, int], list[int]] = {}
    for face_idx, tri_raw in enumerate(np.asarray(faces, dtype=np.int64).reshape(-1, 3)):
        a, b, c = (int(v) for v in tri_raw.tolist())
        for u, v, opp in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(u, v), max(u, v))
            edge_faces.setdefault(key, []).append(int(face_idx))
            edge_opp.setdefault(key, []).append(int(opp))
    return edge_faces, edge_opp


def _opposite_across(edge_faces: dict[tuple[int, int], list[int]], edge_opp: dict[tuple[int, int], list[int]], a: int, b: int, avoid: set[int]) -> int | None:
    key = (min(int(a), int(b)), max(int(a), int(b)))
    for opp in edge_opp.get(key, []):
        if int(opp) not in avoid:
            return int(opp)
    return None


def _butterfly_edge_terms(
    edge_faces: dict[tuple[int, int], list[int]],
    edge_opp: dict[tuple[int, int], list[int]],
    a: int,
    b: int,
) -> list[tuple[int, float]]:
    """Return a classic interpolatory Butterfly stencil where topology allows it."""
    key = (min(a, b), max(a, b))
    opps = [int(v) for v in edge_opp.get(key, [])]
    if len(edge_faces.get(key, [])) < 2 or len(opps) < 2:
        return [(a, 0.5), (b, 0.5)]

    c, d = opps[:2]
    wings: list[int] = []
    for u, v in ((a, c), (b, c), (a, d), (b, d)):
        wing = _opposite_across(edge_faces, edge_opp, u, v, {a, b, c, d})
        if wing is not None:
            wings.append(wing)
    if len(wings) < 4:
        return [(a, 0.5), (b, 0.5), (c, 0.125), (d, 0.125), (a, -0.125), (b, -0.125)]
    return [(a, 0.5), (b, 0.5), (c, 0.125), (d, 0.125)] + [(w, -0.0625) for w in wings[:4]]


def _subdivide_once(rows_to_base: list[dict[int, float]], faces: np.ndarray) -> tuple[list[dict[int, float]], np.ndarray]:
    faces_i = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edge_faces, edge_opp = _build_topology(len(rows_to_base), faces_i)
    next_rows = [dict(row) for row in rows_to_base]
    edge_new_index: dict[tuple[int, int], int] = {}
    for a, b in sorted(edge_faces):
        edge_new_index[(a, b)] = len(next_rows)
        next_rows.append(_combine_rows(rows_to_base, _butterfly_edge_terms(edge_faces, edge_opp, a, b)))

    next_faces: list[list[int]] = []
    for tri_raw in faces_i:
        a, b, c = (int(v) for v in tri_raw.tolist())
        eab = edge_new_index[(min(a, b), max(a, b))]
        ebc = edge_new_index[(min(b, c), max(b, c))]
        eca = edge_new_index[(min(c, a), max(c, a))]
        next_faces.extend(([a, eab, eca], [b, ebc, eab], [c, eca, ebc], [eab, ebc, eca]))
    return next_rows, np.asarray(next_faces, dtype=np.int32)


def _apply_rows(rows: list[dict[int, float]], values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    out_shape = (len(rows),) + arr.shape[1:]
    out = np.zeros(out_shape, dtype=np.float64)
    values64 = arr.astype(np.float64, copy=False)
    for ridx, row in enumerate(rows):
        for col, weight in row.items():
            out[ridx] += float(weight) * values64[int(col)]
    return out


def _rows_to_csr(rows: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = np.zeros((len(rows) + 1,), dtype=np.int64)
    indices: list[int] = []
    weights: list[float] = []
    for ridx, row in enumerate(rows):
        for col, weight in sorted(row.items()):
            if abs(float(weight)) <= 1.0e-12:
                continue
            indices.append(int(col))
            weights.append(float(weight))
        indptr[ridx + 1] = len(indices)
    return indptr, np.asarray(indices, dtype=np.int32), np.asarray(weights, dtype=np.float32)


def _dominant_full_ids(rows: list[dict[int, float]], full_vertex_indices: np.ndarray) -> np.ndarray:
    full = np.asarray(full_vertex_indices, dtype=np.int64).reshape(-1)
    out = np.zeros((len(rows),), dtype=np.int32)
    for ridx, row in enumerate(rows):
        best_col = max(row.items(), key=lambda item: abs(float(item[1])))[0]
        out[ridx] = int(full[int(best_col)])
    return out


def make_butterfly_surface(atlas: LegVolumeAtlas, *, level: int = 2) -> ButterflySurface:
    """Return an interpolatory Butterfly-style surface with inherited chart data."""
    lvl = max(0, int(level))
    base_vertices = np.asarray(atlas.skin_vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(atlas.skin_faces, dtype=np.int32).reshape(-1, 3)
    rows: list[dict[int, float]] = [{i: 1.0} for i in range(base_vertices.shape[0])]
    for _ in range(lvl):
        rows, faces = _subdivide_once(rows, faces)

    vertices = _apply_rows(rows, base_vertices)
    h = _apply_rows(rows, np.asarray(atlas.skin_h, dtype=np.float64).reshape(-1)).reshape(-1)
    d = _apply_rows(rows, np.asarray(atlas.skin_d, dtype=np.float64).reshape(-1)).reshape(-1)
    theta = np.mod(np.asarray(atlas.skin_theta, dtype=np.float64).reshape(-1), 2.0 * np.pi)
    cos_t = _apply_rows(rows, np.cos(theta).reshape(-1)).reshape(-1)
    sin_t = _apply_rows(rows, np.sin(theta).reshape(-1)).reshape(-1)
    theta_out = np.mod(np.arctan2(sin_t, cos_t), 2.0 * np.pi)
    normals = _compute_vertex_normals(vertices.astype(np.float32), faces.astype(np.int32)).astype(np.float32)
    full_ids = _dominant_full_ids(rows, atlas.full_vertex_indices)
    indptr, indices, weights = _rows_to_csr(rows)
    return ButterflySurface(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        theta=theta_out.astype(np.float32),
        h=np.clip(h, 0.0, 1.0).astype(np.float32),
        d=np.clip(d, 0.0, 1.0).astype(np.float32),
        normals=normals,
        full_vertex_indices=full_ids,
        stencil_indptr=indptr,
        stencil_indices=indices,
        stencil_weights=weights,
        source_full_vertex_indices=np.asarray(atlas.full_vertex_indices, dtype=np.int32).reshape(-1),
        level=lvl,
        stencil_nnz=int(indices.shape[0]),
    )
