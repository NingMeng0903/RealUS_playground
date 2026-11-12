"""Rebake a right-leg experiment copy with a smaller annular core."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from knee_gap_diagnostic import _boundary_faces, _sha256


HERE = Path(__file__).resolve().parent
AMONG = Path("/media/camp/EXT_DRIVE/Among_US")
HARMONIC = HERE.parent / "ab_validation_20261110/repair_20261110/harmonic"
SEALED = HARMONIC / "shaft_roi_v2"
OUT = HERE / "knee_rebake"
INNER_FRACTION = 0.01
TOLERANCE = 1.e-9


class TetCoverage:
    """Exact piecewise-linear segment coverage up to a barycentric tolerance."""

    def __init__(self, vertices, tets):
        self.corners = np.asarray(vertices, dtype=np.float64)[tets]
        self.origin = self.corners[:, 0]
        self.lower = self.corners.min(axis=1)
        self.upper = self.corners.max(axis=1)
        self.inverse = np.linalg.inv(np.swapaxes(self.corners[:, 1:]-self.corners[:, :1], 1, 2))

    def candidates(self, a, b):
        low = np.minimum(a, b) - 1.e-9
        high = np.maximum(a, b) + 1.e-9
        return np.flatnonzero(((self.lower <= high) & (self.upper >= low)).all(axis=1))

    def barycentric(self, point, ids):
        local = np.einsum("nij,nj->ni", self.inverse[ids], point-self.origin[ids])
        return np.column_stack((1.-local.sum(axis=1), local))

    def locate(self, point):
        ids = self.candidates(point, point)
        bary = self.barycentric(point, ids)
        inside = bary.min(axis=1) >= -TOLERANCE
        return ids[inside], bary[inside]

    def covered_intervals(self, a, b):
        ids = self.candidates(a, b)
        first = self.barycentric(a, ids)
        slope = self.barycentric(b, ids)-first
        root = np.divide(-TOLERANCE-first, slope, out=np.zeros_like(slope), where=slope != 0.)
        lower = np.maximum(0., np.where(slope > 0., root, -np.inf).max(axis=1))
        upper = np.minimum(1., np.where(slope < 0., root, np.inf).min(axis=1))
        valid = (lower <= upper) & ~((slope == 0.) & (first < -TOLERANCE)).any(axis=1)
        intervals = []
        for left, right in sorted(zip(lower[valid], upper[valid])):
            if intervals and left <= intervals[-1][1] + 1.e-10:
                intervals[-1][1] = max(intervals[-1][1], float(right))
            else:
                intervals.append([float(left), float(right)])
        return intervals


def longest_run(covered, points, steps):
    valid = np.array([len(covered.locate(p)[0]) > 0 for p in points])
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    spans = []
    start = 0
    for i, intervals in enumerate(steps):
        if intervals and intervals[0][0] <= 1.e-8 and intervals[-1][1] >= 1.-1.e-8 and len(intervals) == 1:
            continue
        if i > start:
            spans.append((start, i, float(1000.*distances[start:i].sum())))
        start = i+1
    if len(points)-1 > start:
        spans.append((start, len(points)-1, float(1000.*distances[start:].sum())))
    return {"valid_rows": int(valid.sum()), "continuous_runs": [s for s in spans if s[2] > 1.]}


def main():
    sys.path[:0] = [str(AMONG), str(AMONG/"src")]
    from igl.copyleft.cgal import remesh_self_intersections
    from DMP_playground.core_algorithms.patch_binding import load_canonical_bundle
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (
        load_leg_volume_atlas, save_leg_volume_atlas,
    )
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.lbs_bridge import blend_lbs_transforms
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import (
        build_shaft_roi, tet_determinants,
    )

    original = SEALED/"atlas_right.npz"
    old_hash = _sha256(original)
    old = load_leg_volume_atlas(original)
    base = load_leg_volume_atlas(HARMONIC/"rebuild_work/owned_base/atlas_right.npz")
    surface = load_leg_volume_atlas(HARMONIC/"ownership_smooth/owned_smooth_right_skin.npz")
    source = replace(surface, core_points=base.core_points, core_h=base.core_h,
                     metadata={**surface.metadata, "roi_source_surface": "ipsilateral_faces_of_original_production_loop_surface",
                               "centerline_source": "recomputed_ipsilateral_canonical_base"})
    with np.load(HARMONIC/"ownership_smooth/right_surface_lbs_weights.npz") as source_attachment:
        surface_weights = source_attachment["weights"]
        original_indices = source_attachment["source_skin_vertex_indices"]
    new, attachment, bake_metrics = build_shaft_roi(
        source, surface_weights, station_count=72, theta_count=108, inner_frac=INNER_FRACTION,
    )
    attachment["source_owned_surface_weights"] = surface_weights
    attachment["source_original_skin_vertex_indices"] = original_indices
    OUT.mkdir(exist_ok=False)
    new_path = OUT/"atlas_right.npz"
    save_leg_volume_atlas(new_path, new)
    np.savez_compressed(OUT/"attachments_right.npz", **attachment)
    new = load_leg_volume_atlas(new_path)
    assert np.array_equal(old.skin_vertices, new.skin_vertices)
    assert np.array_equal(old.skin_faces, new.skin_faces)
    assert np.array_equal(old.core_points, new.core_points)
    assert np.array_equal(old.harmonic_tets, new.harmonic_tets)
    assert _sha256(original) == old_hash

    with np.load(HERE/"LONG_LEG_ASSETS.npz") as assets:
        points = assets["vessel_m"]
    segments = list(zip(points[:-1], points[1:]))
    old_cover = TetCoverage(old.harmonic_vertices, old.harmonic_tets)
    old_steps = [old_cover.covered_intervals(a, b) for a, b in segments]
    old_summary = longest_run(old_cover, points, old_steps)
    del old_cover
    new_cover = TetCoverage(new.harmonic_vertices, new.harmonic_tets)
    new_steps = [new_cover.covered_intervals(a, b) for a, b in segments]
    new_summary = longest_run(new_cover, points, new_steps)
    gap = {}
    for row in (104, 105):
        ids, bary = new_cover.locate(points[row])
        assert len(ids) == 1
        verts = new.harmonic_tets[ids[0]]
        xyz = bary[0]@new.harmonic_vertices[verts]
        theta = new.harmonic_theta[verts].astype(float)
        theta = theta[0]+(theta-theta[0]+np.pi)%(2.*np.pi)-np.pi
        gap[str(row)] = {
            "original_xyz_m": points[row].tolist(),
            "tet_index": int(ids[0]),
            "barycentric": bary[0].tolist(),
            "rebound_theta_h_d": [float((bary[0]@theta)%(2.*np.pi)),
                                  float(bary[0]@new.harmonic_h[verts]),
                                  float(bary[0]@new.harmonic_d[verts])],
            "roundtrip_m": float(np.linalg.norm(xyz-points[row])),
        }
    assert any(a <= 103 and b >= 106 for a, b, _ in new_summary["continuous_runs"])
    del new_cover

    faces, topology = _boundary_faces(new.harmonic_tets)
    assert topology["boundary_edges_not_twice"] == 0
    canonical = load_canonical_bundle(AMONG/"outputs/anatomy_retarget/recovered_smplx_20261108")
    knee = list(map(str, canonical["joint_names"])).index("right_knee")
    vertex_h = np.column_stack([new.harmonic_vertices, np.ones(len(new.harmonic_vertices), np.float32)])
    rest_det = tet_determinants(new.harmonic_vertices, new.harmonic_tets)
    poses = []
    for angle in range(0, 91, 15):
        pose = np.zeros((len(canonical["joint_names"]), 3), np.float32)
        pose[knee, 0] = np.deg2rad(angle)
        transforms = joint_global_transforms(pose_axis_angle=pose.ravel(), rest_joints=canonical["rest_joints"],
                                             parents=canonical["parents"]) @ canonical["inverse_bind"]
        blend = blend_lbs_transforms(attachment["volume_lbs_weights"], transforms)
        posed = np.einsum("nij,nj->ni", blend, vertex_h)[:, :3].astype(np.float32)
        det = tet_determinants(posed, new.harmonic_tets)
        pairs = remesh_self_intersections(posed.astype(float), faces, detect_only=True,
                                          first_only=False, cutoff=10000000)[2]
        inverse = TetCoverage(posed, new.harmonic_tets)
        counts, errors = [], []
        for record in gap.values():
            ids, bary = record["tet_index"], np.array(record["barycentric"])
            tracked = bary@posed[new.harmonic_tets[ids]]
            matches, decoded = inverse.locate(tracked)
            counts.append(len(matches))
            assert len(matches) == 1 and matches[0] == ids
            errors.append(float(np.max(abs(decoded[0]-bary))))
        assert np.all(det > 0.) and len(pairs) == 0
        poses.append({"knee_deg": angle, "min_det_ratio": float((det/rest_det).min()),
                      "nonpositive_tets": int((det <= 0.).sum()),
                      "boundary_intersection_pairs": len(pairs),
                      "recovered_knee_row_tet_counts": counts,
                      "max_barycentric_roundtrip_error": max(errors)})
        print(f"Verified pose {angle} deg", flush=True)
    report = {
        "scope": "Experimental right shaft rebake. Same skin and centerline, same original vessel XYZ.",
        "inner_fraction_before": .04, "inner_fraction_after": INNER_FRACTION,
        "old": old_summary, "new": new_summary,
        "old_knee_intervals": old_steps[103:106], "new_knee_intervals": new_steps[103:106],
        "rebound_rows": gap, "bake_metrics": bake_metrics, "boundary_topology": topology,
        "poses": poses, "old_sha256": old_hash, "new_sha256": _sha256(new_path),
        "new_attachments_sha256": _sha256(OUT/"attachments_right.npz"),
    }
    (OUT/"SUMMARY.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("old","new","rebound_rows")}, indent=2))


if __name__ == "__main__":
    main()
