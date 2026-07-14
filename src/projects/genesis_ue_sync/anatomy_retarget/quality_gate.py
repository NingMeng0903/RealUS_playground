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
    "inside_fraction": 0.995,
    "max_outside_m": 0.002,
    "critical_max_outside_m": 0.001,
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

    failures: list[str] = []
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
    stages = ["source_to_final"]
    if asset.pose_cache_vertices is not None:
        stages.append("source_to_pose_cache")
    for stage in stages:
        maximum = float(stretch.get(f"{stage}_max", float("inf")))
        p999 = float(stretch.get(f"{stage}_p999", float("inf")))
        if maximum > thresholds["edge_ratio_max"]:
            failures.append(f"{stage} max edge ratio {maximum:.2f} exceeds {thresholds['edge_ratio_max']:.2f}")
        if p999 > thresholds["edge_ratio_p999"]:
            failures.append(f"{stage} p99.9 edge ratio {p999:.2f} exceeds {thresholds['edge_ratio_p999']:.2f}")
    for tissue, metrics in containment.items():
        inside = float(metrics["inside_fraction"])
        outside_m = float(metrics["max_outside_m"])
        max_allowed = thresholds["critical_max_outside_m"] if tissue == "bone" else thresholds["max_outside_m"]
        if inside < thresholds["inside_fraction"]:
            failures.append(f"{tissue} containment {inside * 100.0:.2f}% is below {thresholds['inside_fraction'] * 100.0:.2f}%")
        if outside_m > max_allowed:
            failures.append(f"{tissue} maximum protrusion {outside_m * 1000.0:.1f} mm exceeds {max_allowed * 1000.0:.1f} mm")
    pose_report = dict((asset.metadata or {}).get("pose_cache_report") or {})
    pose_over_limit = dict(pose_report.get("over_limit_count") or {})
    if any(int(value) > 0 for value in pose_over_limit.values()):
        failures.append(f"saved-pose containment exceeds publication limits: {pose_over_limit}")

    return {
        "schema_version": 2,
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
        "containment_backend": "libigl_exact_signed_distance",
        "containment": containment,
    }


def write_quality_report(path: Path | str, report: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return out
