"""Bone segment and ligament classification diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms, source_bone_skinning_transforms, skin_vertices
from .rigged_asset import AnatomyRiggedAsset

ENDPOINT_LIMIT_M = 0.005
AXIS_LIMIT_DEG = 3.0

SEGMENT_MESHES = {
    "forearm_left": ("Radius_L", "Ulna_L", "Forearm_Bone_L", "Forearm_Twist_L"),
    "forearm_right": ("Radius_R", "Ulna_R", "Forearm_Bone_R", "Forearm_Twist_R"),
    "shin_left": ("Tibia_L", "Fibula_L", "Tibia_Bone_L", "Tibia_Twist_L", "Patella_L"),
    "shin_right": ("Tibia_R", "Fibula_R", "Tibia_Bone_R", "Tibia_Twist_R", "Patella_R"),
    "shoulder_left": ("Scapula_L", "Humerus_L"),
    "shoulder_right": ("Scapula_R", "Humerus_R"),
    "head": ("Upper_Skull",),
}

LIGAMENT_HELPERS = frozenset(
    {
        "Interspinous_Ligament",
        "Supraspinous_Ligament",
        "Nuchal_Ligament",
        "Intertransverse_Ligament",
        "Ligamentum_Flavum",
    }
)


def _mesh_slice(asset: AnatomyRiggedAsset, name: str) -> slice | None:
    if name not in asset.source_mesh_names:
        return None
    idx = asset.source_mesh_names.index(name)
    start, stop = map(int, asset.source_vertex_ranges[idx])
    return slice(start, stop)


def _bone_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis /= max(float(np.linalg.norm(axis)), 1.0e-10)
    return axis


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1.0e-10)
    b = b / max(float(np.linalg.norm(b)), 1.0e-10)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _endpoint_error(rest: np.ndarray, posed: np.ndarray) -> dict[str, float]:
    rest_axis = _bone_axis(rest)
    t = rest @ rest_axis
    p = posed @ rest_axis
    rest_span = float(t.max() - t.min())
    if rest_span < 1.0e-6:
        return {"endpoint_error_m": 0.0, "axis_error_deg": 0.0, "length_error_m": 0.0}
    rest_a, rest_b = float(t.min()), float(t.max())
    posed_a = float(p[np.argmin(t)])
    posed_b = float(p[np.argmax(t)])
    endpoint = max(abs(posed_a - rest_a), abs(posed_b - rest_b))
    posed_axis = _bone_axis(posed)
    length_err = abs(float(np.ptp(p)) - rest_span)
    return {
        "endpoint_error_m": float(endpoint),
        "axis_error_deg": _angle_deg(rest_axis, posed_axis),
        "length_error_m": float(length_err),
    }


def classify_ligament_meshes(asset: AnatomyRiggedAsset, mesh_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rigid_meshes = set((asset.metadata or {}).get("rigid_meshes") or [])
    entries: list[dict[str, Any]] = []
    for item in mesh_diagnostics.get("meshes", []):
        name = str(item.get("mesh", ""))
        ratio = float(item.get("extent_aspect_ratio", 0.0))
        driver = str(item.get("driver_bone", ""))
        tissue = str(item.get("tissue", ""))
        flags: list[str] = []
        if name in LIGAMENT_HELPERS or ratio >= 8.0:
            if name in rigid_meshes and "Hip_bone" in driver:
                flags.append("mis_rigid_collapse")
            if ratio >= 8.0 and tissue == "bone":
                flags.append("high_aspect_ligament")
        if "Spine_C" in driver and ratio >= 8.0:
            flags.append("single_spine_driver")
        if flags:
            entries.append({"mesh": name, "flags": flags, "driver_bone": driver, "extent_aspect_ratio": ratio})
    return entries


def write_bone_segment_diagnostics(
    asset: AnatomyRiggedAsset,
    *,
    pose_axis_angle: np.ndarray,
    transl: np.ndarray | None,
    output_path: Path | str,
    mesh_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    zero = skin_vertices(asset, np.zeros((55, 3), dtype=np.float32))
    posed = skin_vertices(asset, pose_axis_angle, transl=transl)
    segments: dict[str, Any] = {}
    failures: list[str] = []
    for label, mesh_names in SEGMENT_MESHES.items():
        items: list[dict[str, Any]] = []
        for name in mesh_names:
            sl = _mesh_slice(asset, name)
            if sl is None:
                continue
            rest = np.asarray(asset.vertices_rest, dtype=np.float64)[sl]
            err = _endpoint_error(rest, posed[sl])
            err["mesh"] = name
            err["pass"] = bool(
                err["endpoint_error_m"] <= ENDPOINT_LIMIT_M and err["axis_error_deg"] <= AXIS_LIMIT_DEG
            )
            if not err["pass"]:
                failures.append(f"{label}/{name}")
            items.append(err)
        segments[label] = items

    ligaments = classify_ligament_meshes(asset, mesh_diagnostics or {})
    report = {
        "endpoint_limit_m": ENDPOINT_LIMIT_M,
        "axis_limit_deg": AXIS_LIMIT_DEG,
        "segments": segments,
        "ligament_flags": ligaments,
        "passed": len(failures) == 0,
        "failures": failures,
    }
    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report
