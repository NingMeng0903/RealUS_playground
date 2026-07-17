"""Extract native d=0 skin from structured layered Laplace volume grids."""

from __future__ import annotations

import numpy as np

from .atlas import _compute_vertex_normals


def layered_grid_vid(si: int, ri: int, ti: int, *, theta_count: int, radial_count: int) -> int:
    return (int(si) * int(radial_count) + int(ri)) * int(theta_count) + (int(ti) % int(theta_count))


def extract_native_layered_skin(
    vertices: np.ndarray,
    h: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    *,
    station_count: int,
    theta_count: int,
    radial_count: int,
    base_skin_vertices: np.ndarray,
    base_full_vertex_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build skin mesh from the outer structured shell (radial index = radial_count - 1)."""
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    h_flat = np.asarray(h, dtype=np.float64).reshape(-1)
    theta_flat = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), 2.0 * np.pi)
    d_flat = np.asarray(d, dtype=np.float64).reshape(-1)
    si_count = int(station_count)
    ti_count = int(theta_count)
    ri_outer = int(radial_count) - 1

    shell_ids = np.asarray(
        [layered_grid_vid(si, ri_outer, ti, theta_count=ti_count, radial_count=int(radial_count)) for si in range(si_count) for ti in range(ti_count)],
        dtype=np.int64,
    )
    skin_vertices = verts[shell_ids].astype(np.float32)
    skin_h = h_flat[shell_ids].astype(np.float32)
    skin_theta = theta_flat[shell_ids].astype(np.float32)
    skin_d = d_flat[shell_ids].astype(np.float32)

    id_map = {int(vid): int(local) for local, vid in enumerate(shell_ids.tolist())}
    faces: list[list[int]] = []
    for si in range(si_count - 1):
        for ti in range(ti_count):
            ti1 = (ti + 1) % ti_count
            corners = (
                layered_grid_vid(si, ri_outer, ti, theta_count=ti_count, radial_count=int(radial_count)),
                layered_grid_vid(si + 1, ri_outer, ti, theta_count=ti_count, radial_count=int(radial_count)),
                layered_grid_vid(si, ri_outer, ti1, theta_count=ti_count, radial_count=int(radial_count)),
                layered_grid_vid(si + 1, ri_outer, ti1, theta_count=ti_count, radial_count=int(radial_count)),
            )
            v00, v10, v01, v11 = (id_map[int(c)] for c in corners)
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    skin_faces = np.asarray(faces, dtype=np.int32)
    skin_normals = _compute_vertex_normals(skin_vertices, skin_faces)

    base_pts = np.asarray(base_skin_vertices, dtype=np.float64).reshape(-1, 3)
    base_ids = np.asarray(base_full_vertex_indices, dtype=np.int64).reshape(-1)
    full_vertex_indices = np.empty((skin_vertices.shape[0],), dtype=np.int32)
    chunk = 4096
    for start in range(0, skin_vertices.shape[0], chunk):
        stop = min(start + chunk, skin_vertices.shape[0])
        query = skin_vertices[start:stop].astype(np.float64)
        dist = np.sum(np.square(query[:, None, :] - base_pts[None, :, :]), axis=2)
        nn = np.argmin(dist, axis=1)
        full_vertex_indices[start:stop] = base_ids[nn].astype(np.int32)

    return {
        "skin_vertices": skin_vertices,
        "skin_faces": skin_faces,
        "skin_h": skin_h,
        "skin_theta": skin_theta.astype(np.float32),
        "skin_d": skin_d.astype(np.float32),
        "skin_normals": skin_normals.astype(np.float32),
        "full_vertex_indices": full_vertex_indices,
    }
