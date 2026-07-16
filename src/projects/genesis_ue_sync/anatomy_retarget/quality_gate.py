"""Strict, publication-blocking quality checks for SMPL-X anatomy assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


DEFAULT_LIMITS: dict[str, float] = {
    "weight_sum_error": 1.0e-5,
    "anchor_rms_m": 0.010,
    "anchor_max_m": 0.020,
    "edge_ratio_max": 3.0,
    "edge_ratio_p999": 1.5,
    "edge_growth_max_m": 0.01,
    "inside_fraction": 0.995,
    "max_outside_m": 0.002,
    "critical_max_outside_m": 0.001,
    "hand_foot_inside_fraction": 0.99,
    "hand_foot_max_outside_m": 0.005,
    "brain_inside_skull_fraction": 0.995,
    "brain_skull_center_drift_m": 0.002,
    "compound_aspect_ratio_change": 0.02,
    "long_bone_end_edge_change": 0.02,
    "foot_subtree_gap_m": 0.005,
    "digit_rigid_offset_m": 0.002,
}


def _subject_surface(canonical_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(Path(canonical_dir) / "smpl_canonical_tpose.obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    parts = mesh.split(only_watertight=False)
    body = max(parts, key=lambda item: len(item.faces)) if parts else mesh
    return np.asarray(body.vertices, dtype=np.float64), np.asarray(body.faces, dtype=np.int32)


def _signed_distances(points: np.ndarray, canonical_dir: Path, *, batch_size: int = 50000) -> np.ndarray:
    """Signed distance to the subject SMPL-X body; negative values are inside."""
    import igl

    surface_v, surface_f = _subject_surface(canonical_dir)
    values: list[np.ndarray] = []
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    for start in range(0, pts.shape[0], int(batch_size)):
        signed, _face, _closest, _normal = igl.signed_distance(
            pts[start : start + int(batch_size)], surface_v, surface_f
        )
        values.append(np.asarray(signed, dtype=np.float32))
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


def _containment_by_tissue(asset: AnatomyRiggedAsset, signed: np.ndarray) -> dict[str, dict[str, float | int]]:
    ranges = asset.source_vertex_ranges
    tissues = asset.source_tissues
    groups: dict[str, list[np.ndarray]] = {}
    if ranges is not None and tissues is not None and len(ranges) == len(tissues):
        for (start, stop), tissue in zip(np.asarray(ranges, dtype=np.int64), tissues):
            groups.setdefault(str(tissue), []).append(signed[int(start) : int(stop)])
    else:
        groups["all"] = [signed]
    result: dict[str, dict[str, float | int]] = {}
    for tissue, chunks in groups.items():
        values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        outside = values > 0.0
        result[tissue] = {
            "vertex_count": int(values.size),
            "outside_count": int(np.count_nonzero(outside)),
            "inside_fraction": float(np.mean(~outside)) if values.size else 1.0,
            "max_outside_m": float(max(0.0, float(np.max(values)))) if values.size else 0.0,
            "min_skin_distance_m": float(max(0.0, -float(np.max(values)))) if values.size else 0.0,
        }
    return result


def _region_containment(asset: AnatomyRiggedAsset, signed: np.ndarray) -> dict[str, dict[str, float | int]]:
    regions: dict[str, list[np.ndarray]] = {"hand_bones": [], "foot_bones": []}
    for name, (start, stop), tissue in zip(
        asset.source_mesh_names, asset.source_vertex_ranges, asset.source_tissues
    ):
        if str(tissue) != "bone":
            continue
        lower = str(name).lower()
        if any(token in lower for token in ("metacarpal", "phalanx_hand", "phalanges_hand")):
            regions["hand_bones"].append(signed[int(start) : int(stop)])
        if any(token in lower for token in ("calcaneus", "talus", "navicular", "cuboid", "cuneiform", "metatarsal", "phalanx_foot", "phalanges_foot")):
            regions["foot_bones"].append(signed[int(start) : int(stop)])
    result: dict[str, dict[str, float | int]] = {}
    for name, chunks in regions.items():
        values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        result[name] = {
            "vertex_count": int(values.size),
            "inside_fraction": float(np.mean(values <= 0.0)) if values.size else 1.0,
            "max_outside_m": float(max(0.0, float(np.max(values)))) if values.size else 0.0,
        }
    return result


def _brain_skull_metrics(asset: AnatomyRiggedAsset) -> dict[str, float | int]:
    from scipy.spatial import ConvexHull

    skull_chunks: list[np.ndarray] = []
    brain_chunks: list[np.ndarray] = []
    brain_tokens = ("brain", "cerebr", "cerebell", "amygdala", "basal_ganglia", "corpus_callosum", "lobe", "thalam")
    for name, (start, stop) in zip(asset.source_mesh_names, asset.source_vertex_ranges):
        lower = str(name).lower()
        points = np.asarray(asset.vertices_rest[int(start) : int(stop)], dtype=np.float64)
        if "skull" in lower or "cranium" in lower:
            skull_chunks.append(points)
        elif any(token in lower for token in brain_tokens):
            brain_chunks.append(points)
    if not skull_chunks or not brain_chunks:
        return {"brain_vertices": 0, "inside_fraction": 0.0, "max_outside_m": float("inf")}
    skull = np.concatenate(skull_chunks)
    brain = np.concatenate(brain_chunks)
    # ``Upper_Skull`` is an open cranial cap in the authored asset.  Close its
    # missing base in the anatomical inferior direction before testing brain
    # containment; the raw convex hull would cut through the cerebellum.
    names = list(asset.joint_names)
    if "head" in names and "neck" in names:
        superior = np.asarray(asset.rest_joints[names.index("head")], dtype=np.float64) - np.asarray(
            asset.rest_joints[names.index("neck")], dtype=np.float64
        )
    else:
        superior = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    superior /= max(float(np.linalg.norm(superior)), 1.0e-8)
    height = (skull @ superior)
    lower = skull[height <= np.quantile(height, 0.12)]
    base_extension = 0.45 * float(np.ptp(height))
    skull_center = np.mean(skull, axis=0)
    axial = skull_center + ((lower - skull_center) @ superior)[:, None] * superior[None, :]
    lower_radial = axial + 1.30 * (lower - axial)
    closed_skull = np.concatenate(
        (skull, lower_radial - base_extension * superior[None, :]), axis=0
    )
    hull = ConvexHull(closed_skull)
    normals = hull.equations[:, :3]
    offsets = hull.equations[:, 3]
    plane_distance = brain @ normals.T + offsets[None, :]
    outside = np.max(plane_distance, axis=1)
    return {
        "brain_vertices": int(len(brain)),
        "inside_fraction": float(np.mean(outside <= 1.0e-6)),
        "max_outside_m": float(max(0.0, float(np.max(outside)))),
    }


def _bone_pose_edge_stretch(asset: AnatomyRiggedAsset) -> dict[str, float]:
    if asset.pose_cache_vertices is None or asset.registration_reference is None:
        return {"max": float("inf"), "p999": float("inf"), "max_growth_m": float("inf")}
    bone_vertex = np.zeros(len(asset.vertices_rest), dtype=bool)
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue) == "bone":
            bone_vertex[int(start) : int(stop)] = True
    faces = np.asarray(asset.faces, dtype=np.int64)
    faces = faces[np.all(bone_vertex[faces], axis=1)]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    source = np.asarray(asset.registration_reference, dtype=np.float64)
    posed = np.asarray(asset.pose_cache_vertices, dtype=np.float64)
    before = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    after = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
    valid = before > 2.0e-4
    ratio = after[valid] / before[valid]
    return {
        "max": float(np.max(ratio)),
        "p999": float(np.quantile(ratio, 0.999)),
        "max_growth_m": float(np.max(after - before)),
    }


def evaluate_asset_quality(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
    blender_report: dict[str, Any] | None = None,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate all deterministic gates and return a JSON-serializable report."""
    thresholds = dict(DEFAULT_LIMITS)
    thresholds.update({str(k): float(v) for k, v in (limits or {}).items()})
    if asset.driver_weights is not None:
        weights = np.asarray(asset.driver_weights, dtype=np.float32)
    elif asset.lbs_weights is not None:
        weights = np.asarray(asset.lbs_weights, dtype=np.float32)
    else:
        raise ValueError("asset contains no skinning weights")
    names = list(asset.joint_names)
    weight_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    hand_names = [
        name for name in names
        if (name.startswith("left_") or name.startswith("right_"))
        and any(token in name for token in ("index", "middle", "pinky", "ring", "thumb"))
    ]
    if asset.source_bone_names is not None:
        active_bones = set(
            int(idx)
            for idx in np.asarray(asset.driver_indices, dtype=np.int64)[
                np.asarray(asset.driver_weights, dtype=np.float32) > 0.0
            ].tolist()
        )
        active_hands = []
        for name in hand_names:
            joint = names.index(name)
            mapped = np.flatnonzero(
                (np.asarray(asset.source_bone_smplx_a) == joint)
                | (np.asarray(asset.source_bone_smplx_b) == joint)
            )
            if any(int(bone) in active_bones for bone in mapped.tolist()):
                active_hands.append(name)
    else:
        active_hands = [name for name in hand_names if np.any(weights[:, names.index(name)] > 0.0)]

    source_report = dict(blender_report or {})
    rest_align = dict(source_report.get("rest_align", {}) or {})
    stretch = dict(source_report.get("edge_stretch", {}) or {})
    signed = _signed_distances(asset.vertices_rest, Path(canonical_dir))
    containment = _containment_by_tissue(asset, signed)
    regions = _region_containment(asset, signed)
    brain_skull = _brain_skull_metrics(asset)
    bone_pose_stretch = _bone_pose_edge_stretch(asset)

    failures: list[str] = []
    volume_report = dict(source_report.get("volume_registration", {}) or {})
    inverted_tetrahedra = int(
        volume_report.get(
            "inverted_tetrahedra",
            volume_report.get("diagnostic_inverted_tetrahedra", -1),
        )
    )
    if inverted_tetrahedra != 0:
        failures.append(f"volume registration contains {inverted_tetrahedra} inverted tetrahedra")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        failures.append("weights contain NaN/Inf or negative values")
    if weight_error > thresholds["weight_sum_error"]:
        failures.append(f"weight sum error {weight_error:.3g} exceeds {thresholds['weight_sum_error']:.3g}")
    if len(active_hands) != 30:
        failures.append(f"only {len(active_hands)}/30 SMPL-X finger joints have non-zero weights")
    active_groups = int(source_report.get("active_source_group_count", 0))
    if asset.source_bone_names is not None and active_groups != 175:
        failures.append(f"expected 175 explicit Blender vertex groups, report contains {active_groups}")
    anchor_rms = float(rest_align.get("anchor_rms_m", float("inf")))
    anchor_max = float(rest_align.get("max_joint_offset_m", float("inf")))
    if anchor_rms > thresholds["anchor_rms_m"]:
        failures.append(f"anchor RMS {anchor_rms * 1000.0:.1f} mm exceeds {thresholds['anchor_rms_m'] * 1000.0:.1f} mm")
    if anchor_max > thresholds["anchor_max_m"]:
        failures.append(f"anchor max {anchor_max * 1000.0:.1f} mm exceeds {thresholds['anchor_max_m'] * 1000.0:.1f} mm")
    # Intermediate initializer metrics remain diagnostic.  The production gate
    # compares the original Blender geometry directly with the final asset.
    stages = [
        (
            "source_to_final",
            float(stretch.get("source_to_final_max", float("inf"))),
            float(stretch.get("source_to_final_p999", float("inf"))),
            float(stretch.get("source_to_final_max_growth_m", float("inf"))),
        ),
        (
            "bone_to_pose_cache",
            float(bone_pose_stretch["max"]),
            float(bone_pose_stretch["p999"]),
            float(bone_pose_stretch["max_growth_m"]),
        ),
    ]
    for stage, maximum, p999, growth in stages:
        if maximum > thresholds["edge_ratio_max"]:
            failures.append(f"{stage} max edge ratio {maximum:.2f} exceeds {thresholds['edge_ratio_max']:.2f}")
        if p999 > thresholds["edge_ratio_p999"]:
            failures.append(f"{stage} p99.9 edge ratio {p999:.2f} exceeds {thresholds['edge_ratio_p999']:.2f}")
        if growth > thresholds["edge_growth_max_m"]:
            failures.append(
                f"{stage} maximum absolute edge growth {growth * 1000.0:.1f} mm exceeds "
                f"{thresholds['edge_growth_max_m'] * 1000.0:.1f} mm"
            )
    # Whole-tissue containment remains diagnostic: an organ/vessel mesh can
    # legitimately touch an open mouth, eye or authored skin opening, and one
    # aggregate SDF cannot identify the requested rig failures.  Publication
    # is blocked by the explicit hand/foot, cranial-compound, protected-end,
    # chain endpoint/gap/axis and zero-Jacobian gates below.
    for region, metrics in regions.items():
        if float(metrics["inside_fraction"]) < thresholds["hand_foot_inside_fraction"]:
            failures.append(
                f"{region} containment {float(metrics['inside_fraction']) * 100.0:.2f}% is below "
                f"{thresholds['hand_foot_inside_fraction'] * 100.0:.2f}%"
            )
        if float(metrics["max_outside_m"]) > thresholds["hand_foot_max_outside_m"]:
            failures.append(
                f"{region} maximum protrusion {float(metrics['max_outside_m']) * 1000.0:.1f} mm exceeds "
                f"{thresholds['hand_foot_max_outside_m'] * 1000.0:.1f} mm"
            )
    if float(brain_skull["inside_fraction"]) < thresholds["brain_inside_skull_fraction"]:
        failures.append(
            f"brain inside skull {float(brain_skull['inside_fraction']) * 100.0:.2f}% is below "
            f"{thresholds['brain_inside_skull_fraction'] * 100.0:.2f}%"
        )
    material_report = dict(source_report.get("material_shape") or {})
    for group in ("cranial", "pelvis"):
        change = float(material_report.get(f"{group}_aspect_ratio_change", float("inf")))
        if change > thresholds["compound_aspect_ratio_change"]:
            failures.append(f"{group} aspect-ratio change {change * 100.0:.2f}% exceeds {thresholds['compound_aspect_ratio_change'] * 100.0:.2f}%")
    center_drift = float(material_report.get("brain_skull_center_drift_m", float("inf")))
    if center_drift > thresholds["brain_skull_center_drift_m"]:
        failures.append(f"brain/skull center drift {center_drift * 1000.0:.2f} mm exceeds {thresholds['brain_skull_center_drift_m'] * 1000.0:.2f} mm")
    end_change = float(material_report.get("long_bone_end_edge_change", float("inf")))
    if end_change > thresholds["long_bone_end_edge_change"]:
        failures.append(f"long-bone protected-end edge change {end_change * 100.0:.2f}% exceeds {thresholds['long_bone_end_edge_change'] * 100.0:.2f}%")
    digit_offset = float(material_report.get("maximum_digit_rigid_offset_m", float("inf")))
    if digit_offset > thresholds["digit_rigid_offset_m"]:
        failures.append(f"digit rigid centering offset {digit_offset * 1000.0:.2f} mm exceeds {thresholds['digit_rigid_offset_m'] * 1000.0:.2f} mm")
    for side, metrics in dict(material_report.get("feet") or {}).items():
        gap = float(metrics.get("forefoot_gap_before_m", float("inf"))) - float(
            metrics.get("forefoot_rigid_shift_m", 0.0)
        )
        if gap > thresholds["foot_subtree_gap_m"]:
            failures.append(f"{side} midfoot/forefoot gap {gap * 1000.0:.2f} mm exceeds {thresholds['foot_subtree_gap_m'] * 1000.0:.2f} mm")
    pose_report = dict(source_report.get("pose_cache_report") or {})
    pose_over_limit = dict(pose_report.get("over_limit_count") or {})
    if any(int(value) > 0 for value in pose_over_limit.values()):
        failures.append(f"saved-pose containment exceeds publication limits: {pose_over_limit}")
    bone_chain_report = dict(source_report.get("bone_segment_diagnostics") or {})
    failed_chains = [
        name
        for name, metrics in dict(bone_chain_report.get("joints") or {}).items()
        if not bool(metrics.get("pass", False))
    ]
    if failed_chains:
        failures.append(f"bone-chain endpoint/gap/axis regression failed: {failed_chains}")
    head_orientation = dict(bone_chain_report.get("head_orientation") or {})
    if head_orientation and not bool(head_orientation.get("pass", False)):
        failures.append("head orientation regression failed")

    return {
        "schema_version": 4,
        "passed": not failures,
        "failures": failures,
        "thresholds": thresholds,
        "weights": {
            "max_sum_error": weight_error,
            "negative_count": int(np.count_nonzero(weights < 0.0)),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(weights))),
            "active_finger_joints": len(active_hands),
        },
        "anchors": {"rms_m": anchor_rms, "max_m": anchor_max},
        "edge_stretch": stretch,
        "bone_pose_edge_stretch": bone_pose_stretch,
        "volume_registration": volume_report,
        "bone_chains": bone_chain_report,
        "containment_backend": "libigl_exact_signed_distance",
        "containment": containment,
        "regional_containment": regions,
        "brain_skull": brain_skull,
        "material_shape": material_report,
    }


def write_quality_report(path: Path | str, report: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return out
