"""Check transported material addresses on sampled native SMPL-X body shapes."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse.linalg import spsolve

from transport import ShapeTransport


STRESS_DELTAS = tuple(product((-2.0, 0.0, 2.0), repeat=2))
MODERATE_DELTAS = tuple(delta for delta in product((-1.0, 0.0, 1.0), repeat=2)
                        if delta != (0.0, 0.0))
POSES_DEG = (0, 45, 90)
SAMPLE_SEED = 113820
ADDRESS_TOL = np.array([1.e-6, 1.e-7, 1.e-7])  # rad, normalized h, normalized d
ROUNDTRIP_TOL_M = 1.e-6


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary(tets):
    sides = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    faces = np.concatenate([tets[:, face] for face in sides], axis=0)
    _, first, counts = np.unique(np.sort(faces, axis=1), return_index=True,
                                 return_counts=True, axis=0)
    if np.any(counts > 2):
        raise ValueError("nonmanifold volume topology")
    return faces[first[counts == 1]]


def samples(tets, theta, shell, rng):
    """Take interior points, periodic-seam points, and points on both shells."""
    tet_ids = rng.choice(len(tets), 192, replace=False)
    seam = np.flatnonzero(np.ptp(theta[tets], axis=1) > np.pi)
    tet_ids = np.r_[tet_ids, rng.choice(seam, 32, replace=False)]
    cells = [tets[tet_ids]]
    barycentric = [rng.dirichlet([2.] * 4, len(tet_ids))]
    for nodes in (shell[:, 0].ravel(), shell[:, -1].ravel()):
        shell_faces = boundary_faces_on(tets, nodes, len(theta))
        cells.append(rng.choice(shell_faces, 16, replace=False, axis=0))
        barycentric.append(rng.dirichlet([2.] * 3, 16))
    return tuple(zip(cells, barycentric, strict=True))


def boundary_faces_on(tets, nodes, count):
    mask = np.zeros(count, dtype=bool)
    mask[nodes] = True
    near = tets[np.sum(mask[tets], axis=1) >= 3]
    faces = boundary(near)
    return faces[np.all(mask[faces], axis=1)]


def points_from_samples(vertices, probes):
    return np.concatenate([np.einsum("ni,nij->nj", weights, vertices[ids])
                           for ids, weights in probes], axis=0)


def positive_overlap(kernel, vertices, tets):
    with TemporaryDirectory() as work:
        path = Path(work) / "tets.bin"
        with path.open("wb") as stream:
            stream.write(struct.pack("<QQ", len(vertices), len(tets)))
            stream.write(np.asarray(vertices, dtype="<f8").tobytes())
            stream.write(np.asarray(tets, dtype="<u4").tobytes())
            stream.write(np.zeros(len(tets), dtype="<i4").tobytes())
        result = subprocess.run([str(kernel), str(path)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    if not report["complete"] or report["tet_count"] != len(tets):
        raise RuntimeError("overlap checker did not finish all cells")
    return report["positive_interior_overlap_pairs"]


def depth_metrics(graph_laplacian, vertices, tets, shell, frozen):
    """Keep shape and cells fixed while re-solving only the depth field."""
    inner, outer = shell[:, 0].ravel(), shell[:, -1].ravel()
    fixed = np.r_[inner, outer]
    free = np.setdiff1d(np.arange(len(vertices)), fixed)
    matrix = graph_laplacian(vertices, tets)
    rhs = -(matrix[free][:, fixed] @ frozen[fixed])
    result = frozen.astype(float).copy()
    result[free] = spsolve(matrix[free][:, free], rhs)
    scale = max(np.linalg.norm(rhs), 1.e-30)
    delta = np.abs(result - frozen)
    metrics = {
        "transported_laplacian_relative_residual": float(np.linalg.norm((matrix @ frozen)[free]) / scale),
        "resolved_laplacian_relative_residual": float(np.linalg.norm((matrix @ result)[free]) / scale),
        "depth_change_p95": float(np.quantile(delta, .95)),
        "depth_change_max": float(delta.max()),
    }

    # At fixed h/theta each ray is an ordered chain of mesh edges.
    def half_depth_points(depth):
        values = np.moveaxis(depth[shell], 1, -1)
        xyz = np.moveaxis(vertices[shell], 1, -2)
        if np.any(np.diff(values, axis=-1) >= 0.):
            return None
        outer_idx = np.sum(values > 0.5, axis=-1)
        i, j = np.indices(outer_idx.shape)
        lo = values[i, j, outer_idx - 1]
        hi = values[i, j, outer_idx]
        weight = ((lo - .5) / (lo - hi))[..., None]
        return (1. - weight) * xyz[i, j, outer_idx - 1] + weight * xyz[i, j, outer_idx]

    old_mid = half_depth_points(frozen)
    new_mid = half_depth_points(result)
    metrics["resolved_radial_monotonicity"] = new_mid is not None
    if old_mid is not None and new_mid is not None:
        shift = 1000. * np.linalg.norm(new_mid - old_mid, axis=-1)
        metrics["half_depth_shift_p95_mm"] = float(np.quantile(shift, .95))
        metrics["half_depth_shift_max_mm"] = float(shift.max())
    return metrics


def run_side(model, repair, side, betas, poses):
    from dataclasses import replace

    from igl.copyleft.cgal import remesh_self_intersections
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import (
        graph_laplacian,
        tet_determinants,
    )
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
    from scipy.spatial.transform import Rotation
    import trimesh

    asset = model.sides[side]
    atlas_path = repair / "harmonic" / "shaft_roi_v2" / f"atlas_{side}.npz"
    atlas = load_leg_volume_atlas(atlas_path)
    kernel = repair / "harmonic_independent" / "tet_overlap"
    v0, tets = asset.vertices, asset.tets
    shell = np.arange(len(v0)).reshape(asset.grid_shape)
    outer = shell[:, -1].ravel()
    boundary_faces = boundary(tets).astype(np.int64)
    det0 = tet_determinants(v0, tets)
    if np.any(det0 <= 0.):
        raise ValueError("invalid source orientation")
    rng = np.random.default_rng(SAMPLE_SEED)
    probe_points = samples(tets, asset.theta, shell, rng)
    points0 = points_from_samples(v0, probe_points)
    address = VolumeTetRefiner.from_atlas(atlas).p_to_xi(points0)
    # Keep identical template material addresses and held-out triangle samples.
    skin_samples = rng.choice(len(asset.skin_faces), 256, replace=False)
    sampled_triangles = asset.skin_faces[skin_samples]
    skin_barycentric = rng.dirichlet([2.] * 3, len(skin_samples))
    h_indices = np.array([18, 36, 54])
    controls = []
    for scale in (1., 1.2):
        rot = Rotation.from_rotvec([.24, -.40, .31]).as_matrix()
        v = scale * np.asarray(v0, dtype=float) @ rot.T + [.12, -.08, .21]
        check = depth_metrics(graph_laplacian, v, tets, shell, asset.depth)
        controls.append(dict(uniform_scale=scale, **check))
        if check["depth_change_max"] > 1.e-6:
            raise RuntimeError("depth control exceeded numerical tolerance")

    rows = []
    for delta_beta, pose in product(betas, poses):
        vertices, skin, skin_faces, joints = model.deform(side, delta_beta, pose)
        # Match the stored atlas and public query precision for all metrics.
        vertices = np.asarray(vertices, dtype=np.float32)
        determinants = tet_determinants(vertices, tets)
        overlap = positive_overlap(kernel, vertices, tets)
        cgal = remesh_self_intersections(vertices.astype(float), boundary_faces,
                                         detect_only=True, first_only=False,
                                         cutoff=10_000_000)
        intersecting = len(cgal[2])
        native_cgal = remesh_self_intersections(skin.astype(float), skin_faces.astype(np.int64),
                                                detect_only=True, first_only=False,
                                                cutoff=10_000_000)
        native_intersections = len(native_cgal[2])
        ratios = determinants / det0
        rings = vertices[shell[h_indices, -1]]
        perimeter = np.linalg.norm(rings - np.roll(rings, 1, axis=1), axis=-1).sum(axis=1)
        hip, knee, ankle = (1, 4, 7) if side == "left" else (2, 5, 8)
        row = dict(side=side, beta_delta=list(delta_beta), knee_deg=pose, tetrahedra=len(tets),
                   nonpositive_tets=int(np.sum(determinants <= 0.)),
                   min_det_ratio=float(ratios.min()), positive_overlap_pairs=int(overlap),
                   volume_boundary_intersection_pairs=intersecting,
                   native_skin_intersection_pairs=native_intersections,
                   ring_perimeters_mm=(1000. * perimeter).tolist(),
                   segment_lengths_mm=(1000. * np.linalg.norm(
                       np.diff(joints[[hip, knee, ankle]], axis=0), axis=1)).tolist())
        row["geometry_pass"] = (
            row["nonpositive_tets"] == 0 and row["min_det_ratio"] > 1.e-3
            and overlap == 0 and intersecting == 0 and native_intersections == 0
        )
        if row["geometry_pass"]:
            posed_atlas = replace(atlas, harmonic_vertices=vertices, skin_vertices=vertices[outer])
            refiner = VolumeTetRefiner.from_atlas(posed_atlas)
            points = points_from_samples(vertices, probe_points)
            drift = refiner.p_to_xi(points).astype(float) - address
            drift[:, 0] = np.angle(np.exp(1j * drift[:, 0]))
            error = np.linalg.norm(refiner.xi_to_p(address) - points, axis=1)
            worst_drift = np.max(np.abs(drift), axis=0)
            row.update(material_points=len(points), max_label_drift_rad_h_d=worst_drift.tolist(),
                       max_material_roundtrip_m=float(error.max()),
                       material_check_pass=bool(np.all(worst_drift <= ADDRESS_TOL) and
                                                error.max() <= ROUNDTRIP_TOL_M))
            mesh = trimesh.Trimesh(vertices=skin, faces=skin_faces, process=False)
            skin_xyz = np.einsum("ni,nij->nj", skin_barycentric, vertices[outer][sampled_triangles])
            _, distances, _ = trimesh.proximity.closest_point(mesh, skin_xyz)
            row["outer_loft_distance_to_native_skin_mm"] = {
                "samples": len(distances), "p95": float(1000. * np.quantile(distances, .95)),
                "max": float(1000. * distances.max()),
            }
            if pose == 0:
                row["depth"] = depth_metrics(graph_laplacian, vertices, tets, shell, asset.depth)
        rows.append(row)
        print(side, delta_beta, pose, "pass" if row.get("material_check_pass") else "failed", flush=True)
    return {
        "h_circumference_samples": asset.h[shell[h_indices, -1, 0]].tolist(),
        "reference_lineage": asset.lineage,
        "similarity_controls": controls,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--grid", choices=("stress", "moderate", "smoke"), default="stress")
    args = parser.parse_args()
    among = args.among_us.resolve()
    repair = args.repair.resolve()
    sys.path.insert(0, str(among / "src"))
    model = ShapeTransport(among, repair)
    grid = {"stress": (STRESS_DELTAS, POSES_DEG),
            "moderate": (MODERATE_DELTAS, POSES_DEG),
            "smoke": (((0., 0.),), (0,))}
    betas, poses = grid[args.grid]
    sides = tuple(model.sides) if args.side == "both" else (args.side,)
    result = {
        "scope": "Sampled native SMPL-X male shape and knee pose, fixed shaft ROI addresses, model-defined interior extension",
        "grid": args.grid,
        "reference_beta": model.beta.tolist(),
        "beta_deltas": betas,
        "poses_deg": poses,
        "material_sample_seed": SAMPLE_SEED,
        "label_tolerance_rad_h_d": ADDRESS_TOL.tolist(),
        "roundtrip_tolerance_m": ROUNDTRIP_TOL_M,
        "input_sha256": {str(key): sha(Path(value)) for key, value in model.source_paths.items()
                         if Path(value).is_file()},
        "overlap_kernel_sha256": sha(repair / "harmonic_independent" / "tet_overlap"),
        "graph_laplacian_source_sha256": sha(
            among / "src" / "projects" / "genesis_ue_sync" / "anatomy_retarget"
            / "leg_volume_coordinates" / "shaft_rebuild.py"),
        "sides": {},
    }
    for side in sides:
        result["sides"][side] = run_side(model, repair, side, betas, poses)
    cases = [row for side in result["sides"].values() for row in side["cases"]]
    result["counts"] = dict(configurations=len(cases),
                            geometry_pass=sum(row["geometry_pass"] for row in cases),
                            address_pass=sum(row.get("material_check_pass", False) for row in cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["counts"], flush=True)


if __name__ == "__main__":
    main()
