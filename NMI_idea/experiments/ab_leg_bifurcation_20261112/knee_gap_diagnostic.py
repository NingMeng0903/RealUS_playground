"""Independent geometry audit for the rows-104/105 repaired-atlas break.

The audit is deliberately local and read-only.  It checks the original vessel
XYZ rows against every repaired harmonic tetrahedron (not just a nearest
candidate cache), computes the nearest point on the actual repaired volume
boundary, and verifies that the boundary is a closed manifold.  A proposed
rebind is only the nearest physical boundary point; original XYZ and every
displacement are retained side-by-side and no asset is rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "LONG_LEG_ASSETS.npz"
ATLAS = HERE.parent / "ab_validation_20261110" / "repair_20261110" / "harmonic" / "shaft_roi_v2" / "atlas_right.npz"
SURFACE = HERE.parent / "ab_validation_20261110" / "repair_20261110" / "harmonic_full_domain_followup" / "contact_multichart_right.npz"
FOCUS_ROWS = np.arange(100, 108, dtype=np.int64)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _closest_point_triangle(point: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point and barycentric coordinates on one 3-D triangle."""

    a, b, c = tri
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0])
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0])
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, np.asarray([1.0 - v - w, v, w])


def _boundary_faces(tets: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    oriented = np.concatenate(
        [tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]], axis=0
    )
    keys = np.sort(oriented, axis=1)
    unique, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    boundary = oriented[first[counts == 1]]
    # A crack/nonmanifold seam would leave boundary edges with a count other
    # than two.  This check is independent of the point inverse below.
    edges = np.concatenate([boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]]], axis=0)
    edge_keys, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return boundary.astype(np.int64), {
        "tetra_count": int(len(tets)),
        "boundary_face_count": int(len(boundary)),
        "unique_face_count": int(len(unique)),
        "boundary_edge_count": int(len(edge_keys)),
        "boundary_edges_not_twice": int(np.sum(edge_counts != 2)),
        "boundary_edges_min_incidence": int(edge_counts.min()) if len(edge_counts) else 0,
        "boundary_edges_max_incidence": int(edge_counts.max()) if len(edge_counts) else 0,
    }


def _all_tet_inverse(point: np.ndarray, vertices: np.ndarray, tets: np.ndarray, theta: np.ndarray, h: np.ndarray, d: np.ndarray) -> dict[str, Any]:
    """Check every tetra; nearest-centroid candidate caches are not trusted."""

    matches: list[tuple[int, np.ndarray]] = []
    best_min = -np.inf
    best_tet = -1
    best_bary = np.full(4, np.nan)
    for start in range(0, len(tets), 100000):
        tet_rows = tets[start : start + 100000]
        corners = vertices[tet_rows]
        origin = corners[:, 0]
        matrix = np.transpose(corners[:, 1:] - origin[:, None, :], (0, 2, 1))
        rhs = (point[None, :] - origin)[..., None]
        local = np.linalg.solve(matrix, rhs)[..., 0]
        bary = np.concatenate([1.0 - local.sum(axis=1, keepdims=True), local], axis=1)
        score = np.min(bary, axis=1)
        slot = int(np.argmax(score))
        if float(score[slot]) > best_min:
            best_min = float(score[slot])
            best_tet = start + slot
            best_bary = bary[slot].copy()
        inside = (score >= -1.0e-8) & (np.max(bary, axis=1) <= 1.0 + 1.0e-8)
        for local_slot in np.flatnonzero(inside):
            matches.append((start + int(local_slot), np.clip(bary[local_slot], 0.0, 1.0)))
    def _xi(tet_index: int, bary: np.ndarray) -> list[float]:
        raw = np.asarray(theta[tets[tet_index]], dtype=float)
        unwrapped = raw[0] + (raw - raw[0] + np.pi) % (2.0 * np.pi) - np.pi
        return [float(np.mod(bary @ unwrapped, 2.0 * np.pi)), float(bary @ h[tets[tet_index]]), float(bary @ d[tets[tet_index]])]
    return {
        "containing_tet_count": int(len(matches)),
        "containing_tet_indices": [int(item[0]) for item in matches],
        "containing_xi": [_xi(tet, bary) for tet, bary in matches],
        "best_tet_index": int(best_tet),
        "best_barycentric": best_bary.tolist(),
        "best_min_barycentric": float(best_min),
        "best_xi_if_extrapolated": _xi(best_tet, best_bary) if best_tet >= 0 else None,
    }


def _nearest_boundary(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray, theta: np.ndarray, h: np.ndarray, d: np.ndarray) -> dict[str, Any]:
    best_dist = np.inf
    best_point = np.full(3, np.nan)
    best_bary = np.full(3, np.nan)
    best_face = -1
    for face_index, face in enumerate(faces):
        q, bary = _closest_point_triangle(point, vertices[face])
        distance = float(np.linalg.norm(q - point))
        if distance < best_dist:
            best_dist, best_point, best_bary, best_face = distance, q, bary, int(face_index)
    face = faces[best_face]
    raw = np.asarray(theta[face], dtype=float)
    unwrapped = raw[0] + (raw - raw[0] + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "boundary_face_index": best_face,
        "boundary_vertex_indices": face.tolist(),
        "point_m": best_point.tolist(),
        "distance_m": float(best_dist),
        "distance_mm": float(best_dist * 1000.0),
        "barycentric": best_bary.tolist(),
        "rebind_xi": [float(np.mod(best_bary @ unwrapped, 2.0 * np.pi)), float(best_bary @ h[face]), float(best_bary @ d[face])],
        "delta_m": (best_point - point).tolist(),
    }


def _nearest_vertex(point: np.ndarray, vertices: np.ndarray) -> dict[str, Any]:
    index = int(np.argmin(np.linalg.norm(vertices - point[None, :], axis=1)))
    return {"index": index, "distance_m": float(np.linalg.norm(vertices[index] - point)), "point_m": vertices[index].tolist()}


def build_diagnostic(*, output_path: Path, assets_path: Path = ASSETS, atlas_path: Path = ATLAS, surface_path: Path = SURFACE, rows: np.ndarray = FOCUS_ROWS) -> dict[str, Any]:
    with np.load(assets_path, allow_pickle=False) as source:
        assets = {key: np.asarray(source[key]) for key in source.files}
    with np.load(atlas_path, allow_pickle=False) as source:
        atlas = {key: np.asarray(source[key]) for key in source.files}
    rows = np.asarray(rows, dtype=np.int64)
    vessel = np.asarray(assets["vessel_m"], dtype=float)[rows]
    skin = np.asarray(assets["skin_m"], dtype=float)[rows]
    vertices = np.asarray(atlas["harmonic_vertices"], dtype=float)
    tets = np.asarray(atlas["harmonic_tets"], dtype=np.int64)
    theta = np.asarray(atlas["harmonic_theta"], dtype=float)
    h = np.asarray(atlas["harmonic_h"], dtype=float)
    d = np.asarray(atlas["harmonic_d"], dtype=float)
    boundary, topology = _boundary_faces(tets)
    # The contact atlas query is an independent nearest full-surface check.
    import sys

    source_root = Path("/media/camp/EXT_DRIVE/Among_US/src")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import SurfaceMaterialAtlas  # type: ignore

    contact = SurfaceMaterialAtlas.load(surface_path)
    records: list[dict[str, Any]] = []
    for row, point, old_skin in zip(rows.tolist(), vessel, skin, strict=True):
        inverse = _all_tet_inverse(point, vertices, tets, theta, h, d)
        nearest_boundary = _nearest_boundary(point, vertices, boundary, theta, h, d)
        full_surface = contact.query([point], project=True)[0]
        # The strict repaired skin is a separate, smaller mesh.  Its closest
        # vertex is reported as a lower-cost independent diagnostic; the
        # repaired tetra boundary distance above is the actual rebind measure.
        repaired_skin = np.asarray(atlas["skin_vertices"], dtype=float)
        rec = {
            "row_id": int(row),
            "label": str(assets["labels"][row]),
            "source_global_point_index": int(assets["source_global_point_indices"][row]),
            "original_vessel_m": point.tolist(),
            "old_full_surface_skin_m": old_skin.tolist(),
            "old_skin_to_vessel_depth_mm": float(np.linalg.norm(point - old_skin) * 1000.0),
            "asset_internal_valid": bool(assets["internal_valid"][row]),
            "asset_internal_xi": np.asarray(assets["internal_xi"][row], dtype=float).tolist(),
            "asset_legacy_xi_skin": np.asarray(assets["legacy_xi"][row], dtype=float).tolist(),
            "strict_all_tetra_inverse": inverse,
            "nearest_repaired_harmonic_vertex": _nearest_vertex(point, vertices),
            "nearest_repaired_skin_vertex": _nearest_vertex(point, repaired_skin),
            "nearest_repaired_volume_boundary": nearest_boundary,
            "full_contact_surface_query": {
                "chart_id": int(full_surface["chart_id"]),
                "source_face_id": int(full_surface["source_face_id"]),
                "projection_distance_m": float(full_surface["projection_distance_m"]),
                "projection_distance_mm": float(full_surface["projection_distance_m"] * 1000.0),
                "coordinate_uv_m": np.asarray(full_surface["coordinate_uv_m"], dtype=float).tolist(),
                "decoded_point_m": np.asarray(contact.decode(int(full_surface["chart_id"]), full_surface["coordinate_uv_m"])[0], dtype=float).tolist(),
            },
        }
        records.append(rec)
    diagnostics = {
        "schema": "knee_gap_diagnostic_v1",
        "status": "read_only_exact_local_geometry_audit",
        "focus_rows": rows.tolist(),
        "asset_paths": {
            "long_leg_assets": {"path": str(assets_path.resolve()), "sha256": _sha256(assets_path)},
            "repaired_atlas": {"path": str(atlas_path.resolve()), "sha256": _sha256(atlas_path)},
            "full_contact_surface": {"path": str(surface_path.resolve()), "sha256": _sha256(surface_path)},
        },
        "atlas_domain": {
            "h": [float(h.min()), float(h.max())],
            "theta_periodic": True,
            "d": [float(d.min()), float(d.max())],
            "harmonic_vertex_count": int(len(vertices)),
            "harmonic_tet_count": int(len(tets)),
        },
        "boundary_topology": topology,
        "interpretation": {
            "exact_tetra_inverse": "all tets are tested; containing_tet_count=0 is an actual outside query, not a nearest-cache miss",
            "rebind": "nearest point on closed repaired tetra boundary only; original vessel_m is never replaced",
            "surface_vs_volume": "old_full_surface_skin_m is full d=0 contact geometry; repaired volume boundary is the strict new atlas geometry",
            "no_dynamics": True,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "KNEE_GAP_DIAGNOSTIC.json")
    parser.add_argument("--rows", type=int, nargs="+", default=FOCUS_ROWS.tolist())
    args = parser.parse_args(argv)
    result = build_diagnostic(output_path=args.output, rows=np.asarray(args.rows, dtype=np.int64))
    print(json.dumps({"status": result["status"], "output": str(args.output), "rows": result["focus_rows"], "boundary_topology": result["boundary_topology"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

