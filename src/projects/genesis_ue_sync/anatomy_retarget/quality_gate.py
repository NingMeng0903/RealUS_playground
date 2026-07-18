"""Strict, publication-blocking quality checks for SMPL-X anatomy assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import ANATOMY_ASSET_SCHEMA_VERSION, AnatomyRiggedAsset


DEFAULT_LIMITS: dict[str, float] = {
    "weight_sum_error": 1.0e-5,
    "bind_roundtrip_error_m": 1.0e-5,
    "blender_parity_error_m": 5.0e-4,
    "anchor_rms_m": 0.002,
    "anchor_max_m": 0.002,
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
    # Soft-tissue meshes have to be judged independently.  A handful of
    # badly sheared vessel triangles used to disappear in the global mesh
    # percentile, even though they are conspicuous in the arm preview.
    "soft_edge_ratio_q99": 1.25,
    "soft_edge_ratio_max": 1.50,
    "tube_graph_ratio_p99": 1.25,
    "tube_graph_ratio_max": 1.50,
    "cranial_shared_transform_rms_m": 1.0e-3,
    "upper_teeth_skull_distance_drift_m": 0.001,
}

SOFT_TISSUES = frozenset({"vessel", "nerve", "organ", "connective_tissue", "heart"})
CRITICAL_SOFT_TISSUES = frozenset({"vessel", "nerve"})
REQUIRED_CONTAINMENT_STAGES = ("neutral_canonical", "subject_rest", "final_pose")


def _required_mapping(
    parent: dict[str, Any],
    key: str,
    *,
    failures: list[str],
    label: str,
) -> dict[str, Any] | None:
    value = parent.get(key)
    if not isinstance(value, dict) or not value:
        failures.append(f"{label} report is missing")
        return None
    return value


def _required_number(
    report: dict[str, Any],
    key: str,
    *,
    failures: list[str],
    label: str,
) -> float | None:
    if key not in report:
        failures.append(f"{label} field {key!r} is missing")
        return None
    try:
        value = float(report[key])
    except (TypeError, ValueError):
        failures.append(f"{label} field {key!r} is not numeric")
        return None
    if not np.isfinite(value):
        failures.append(f"{label} field {key!r} is not finite")
        return None
    return value


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


def _containment_by_tissue(asset: AnatomyRiggedAsset, signed: np.ndarray) -> dict[str, dict[str, Any]]:
    ranges = asset.source_vertex_ranges
    tissues = asset.source_tissues
    groups: dict[str, list[np.ndarray]] = {}
    if ranges is not None and tissues is not None and len(ranges) == len(tissues):
        for (start, stop), tissue in zip(np.asarray(ranges, dtype=np.int64), tissues):
            groups.setdefault(str(tissue), []).append(signed[int(start) : int(stop)])
    result: dict[str, dict[str, Any]] = {}
    for tissue, chunks in groups.items():
        values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        outside = values > 0.0
        result[tissue] = {
            "available": bool(values.size),
            "vertex_count": int(values.size),
            "outside_count": int(np.count_nonzero(outside)) if values.size else None,
            "inside_fraction": float(np.mean(~outside)) if values.size else None,
            "max_outside_m": float(max(0.0, float(np.max(values)))) if values.size else None,
            "min_skin_distance_m": float(max(0.0, -float(np.max(values)))) if values.size else None,
        }
    return result


def _region_containment(asset: AnatomyRiggedAsset, signed: np.ndarray) -> dict[str, dict[str, Any]]:
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
    result: dict[str, dict[str, Any]] = {}
    for name, chunks in regions.items():
        values = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        result[name] = {
            "available": bool(values.size),
            "vertex_count": int(values.size),
            "inside_fraction": float(np.mean(values <= 0.0)) if values.size else None,
            "max_outside_m": float(max(0.0, float(np.max(values)))) if values.size else None,
        }
    return result


def _brain_skull_metrics(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    from scipy.spatial import ConvexHull

    skull_chunks: list[np.ndarray] = []
    brain_chunks: list[np.ndarray] = []
    # Keep this deliberately broader than the visible lobe meshes.  The
    # previous list missed deep-brain meshes (fornix/hippocampus/ventricles),
    # so a visibly displaced brain could still pass the publication gate.
    brain_tokens = (
        "brain", "cerebr", "cerebell", "amygdala", "basal_ganglia",
        "corpus_callosum", "lobe", "thalam", "hypothalam", "midbrain",
        "pons", "medulla", "fornix", "hippocamp", "ventric", "pituitar",
        "pineal", "olfactory", "optic_chiasm", "chiasm",
    )
    for name, (start, stop) in zip(asset.source_mesh_names, asset.source_vertex_ranges):
        lower = str(name).lower()
        points = np.asarray(asset.vertices_rest[int(start) : int(stop)], dtype=np.float64)
        if "skull" in lower or "cranium" in lower:
            skull_chunks.append(points)
        elif any(token in lower for token in brain_tokens):
            brain_chunks.append(points)
    if not skull_chunks or not brain_chunks:
        return {
            "available": False,
            "brain_vertices": 0,
            "inside_fraction": None,
            "max_outside_m": None,
        }
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
        "available": True,
        "brain_vertices": int(len(brain)),
        "inside_fraction": float(np.mean(outside <= 1.0e-6)),
        "max_outside_m": float(max(0.0, float(np.max(outside)))),
    }


def _mesh_edges(asset: AnatomyRiggedAsset, start: int, stop: int) -> np.ndarray:
    """Return mesh-local triangles as global unique-undirected edges."""
    faces = np.asarray(asset.faces, dtype=np.int64)
    selected = faces[np.all((faces >= int(start)) & (faces < int(stop)), axis=1)]
    if not len(selected):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate((selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _soft_mesh_pose_stretch(asset: AnatomyRiggedAsset) -> dict[str, dict[str, float | int]]:
    """Per-mesh pose deformation diagnostics for every soft-tissue mesh."""
    if asset.pose_cache_vertices is None:
        return {}
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    posed = np.asarray(asset.pose_cache_vertices, dtype=np.float64)
    result: dict[str, dict[str, float | int]] = {}
    for name, (start, stop), tissue in zip(asset.source_mesh_names, asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue) not in SOFT_TISSUES:
            continue
        edges = _mesh_edges(asset, int(start), int(stop))
        if not len(edges):
            continue
        before = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
        after = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
        valid = before > 2.0e-4
        if not np.any(valid):
            continue
        ratio = after[valid] / before[valid]
        result[str(name)] = {
            "tissue": str(tissue),
            "edge_count": int(np.count_nonzero(valid)),
            "ratio_q99": float(np.quantile(ratio, 0.99)),
            "ratio_p999": float(np.quantile(ratio, 0.999)),
            "ratio_max": float(np.max(ratio)),
            "max_growth_m": float(np.max(after - before)),
        }
    return result


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the least-squares proper similarity mapping ``source`` to target."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src_center, dst_center = np.mean(src, axis=0), np.mean(dst, axis=0)
    a, b = src - src_center, dst - dst_center
    u, singular, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / max(float(np.sum(a * a)), 1.0e-12))
    translation = dst_center - scale * (src_center @ rotation)
    return rotation, scale, translation


def _cranial_compound_metrics(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    """Verify head compound membership and the upper-teeth/skull transform.

    Mesh names are retained for reporting, while source-rig hierarchy is used
    when available: any mesh fully controlled by Head_Bone descendants but not
    Jaw_Bone descendants belongs to the cranial compound.  This catches names
    such as Fornix and Upper_Teeth without maintaining a fragile whitelist.
    """
    # The cranial material fit starts from the all-harmonic subject reference,
    # not the authored/source registration mesh.  Measuring against the latter
    # incorrectly assigns beta-field deformation to the compound fit.
    source = getattr(asset, "harmonic_reference_vertices", None)
    if source is None:
        source = asset.registration_reference
    if source is None:
        return {"available": False, "member_meshes": [], "upper_teeth_meshes": []}
    source = np.asarray(source, dtype=np.float64)
    final = np.asarray(asset.vertices_rest, dtype=np.float64)
    from .material_fit import cranial_material_mask

    compound_mask = cranial_material_mask(asset)
    bone_names = [str(name).lower() for name in (asset.source_bone_names or [])]
    head_index = next((i for i, name in enumerate(bone_names) if name in {"head_bone", "head"}), None)
    jaw_index = next((i for i, name in enumerate(bone_names) if name in {"jaw_bone", "jaw"}), None)
    descendants: set[int] = set()
    jaw_descendants: set[int] = set()
    if head_index is not None and asset.source_bone_parents is not None:
        parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
        for index in range(len(parents)):
            ancestry: set[int] = set()
            cursor = index
            while cursor >= 0 and cursor not in ancestry:
                ancestry.add(cursor)
                cursor = int(parents[cursor])
            if head_index in ancestry:
                descendants.add(index)
            if jaw_index is not None and jaw_index in ancestry:
                jaw_descendants.add(index)
    skull_ranges: list[tuple[int, int]] = []
    member_ranges: list[tuple[str, int, int]] = []
    teeth_ranges: list[tuple[str, int, int]] = []
    for mesh_index, (name, vertex_range) in enumerate(zip(asset.source_mesh_names, asset.source_vertex_ranges)):
        start, stop = map(int, vertex_range)
        lower = str(name).lower()
        is_head_member = bool(np.mean(compound_mask[start:stop]) >= 0.5)
        is_upper_teeth = (
            ("upper" in lower and ("tooth" in lower or "teeth" in lower))
            or any(token in lower for token in ("molar", "premolar", "incisor", "canine"))
        ) and is_head_member
        is_skull = "skull" in lower or "cranium" in lower
        is_member = is_head_member and (is_skull or any(token in lower for token in (
            "brain", "cerebr", "cerebell", "amygdala", "fornix", "hippocamp", "ventric",
            "thalam", "hypothalam", "midbrain", "pons", "medulla", "pituitar", "pineal",
            "olfactory", "optic_chiasm", "chiasm",
        )) or is_upper_teeth)
        if is_skull:
            skull_ranges.append((start, stop))
        if is_member:
            member_ranges.append((str(name), start, stop))
        if is_upper_teeth:
            teeth_ranges.append((str(name), start, stop))
    if not skull_ranges:
        return {"available": False, "member_meshes": [name for name, *_ in member_ranges], "upper_teeth_meshes": [name for name, *_ in teeth_ranges]}
    skull_idx = np.concatenate([np.arange(start, stop) for start, stop in skull_ranges])
    rotation, scale, translation = _similarity(source[skull_idx], final[skull_idx])
    def residual(start: int, stop: int) -> np.ndarray:
        predicted = scale * (source[start:stop] @ rotation) + translation
        return np.linalg.norm(final[start:stop] - predicted, axis=1)
    member_errors = [residual(start, stop) for _name, start, stop in member_ranges]
    teeth_errors = [residual(start, stop) for _name, start, stop in teeth_ranges]
    errors = np.concatenate(member_errors) if member_errors else np.zeros(0)
    teeth = np.concatenate(teeth_errors) if teeth_errors else np.zeros(0)
    return {
        "available": True,
        "member_meshes": [name for name, *_ in member_ranges],
        "upper_teeth_meshes": [name for name, *_ in teeth_ranges],
        "member_count": len(member_ranges),
        "shared_transform_rms_m": float(np.sqrt(np.mean(errors * errors))) if len(errors) else None,
        "upper_teeth_skull_distance_drift_m": float(np.sqrt(np.mean(teeth * teeth))) if len(teeth) else None,
    }


def _foot_reach_metrics(asset: AnatomyRiggedAsset, canonical_dir: Path) -> dict[str, dict[str, Any]]:
    """Measure complete foot-bone reach against the final SMPL-X foot surface."""
    surface, _faces = _subject_surface(canonical_dir)
    names = list(asset.joint_names)
    result: dict[str, dict[str, Any]] = {}
    foot_tokens = ("calcaneus", "talus", "navicular", "cuboid", "cuneiform", "metatarsal", "phalanx_foot", "phalanges_foot")
    for side in ("left", "right"):
        ankle_name, foot_name = f"{side}_ankle", f"{side}_foot"
        if ankle_name not in names or foot_name not in names:
            continue
        ankle = np.asarray(asset.rest_joints[names.index(ankle_name)], dtype=np.float64)
        forward = np.asarray(asset.rest_joints[names.index(foot_name)], dtype=np.float64) - ankle
        forward /= max(float(np.linalg.norm(forward)), 1.0e-8)
        chunks: list[np.ndarray] = []
        suffix = "_l" if side == "left" else "_r"
        for name, (start, stop), tissue in zip(asset.source_mesh_names, asset.source_vertex_ranges, asset.source_tissues):
            lower = str(name).lower()
            if str(tissue) == "bone" and any(token in lower for token in foot_tokens) and (lower.endswith(suffix) or f"{suffix}_" in lower):
                chunks.append(np.asarray(asset.vertices_rest[int(start):int(stop)], dtype=np.float64))
        if not chunks:
            continue
        # A local cylinder around ankle->foot avoids taking a leg or opposite
        # foot point when the person is lying down.
        local = surface[np.linalg.norm(np.cross(surface - ankle, forward), axis=1) < 0.16]
        if not len(local):
            result[side] = {"available": False, "reason": "subject foot surface is unavailable"}
            continue
        target = float(np.quantile((local - ankle) @ forward, 0.995))
        if target <= 1.0e-8:
            result[side] = {"available": False, "reason": "subject foot reach is degenerate"}
            continue
        reach = float(np.quantile((np.concatenate(chunks) - ankle) @ forward, 0.995))
        result[side] = {
            "available": True,
            "bone_reach_m": reach,
            "skin_reach_m": target,
            "reach_ratio": reach / target,
        }
    return result


def _bone_pose_edge_stretch(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    """Pose stretch on bone meshes only, measured from fitted rest → pose cache."""
    if asset.pose_cache_vertices is None:
        return {
            "available": False,
            "max": None,
            "p999": None,
            "max_growth_m": None,
        }
    bone_vertex = np.zeros(len(asset.vertices_rest), dtype=bool)
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue) == "bone":
            bone_vertex[int(start) : int(stop)] = True
    faces = np.asarray(asset.faces, dtype=np.int64)
    faces = faces[np.all(bone_vertex[faces], axis=1)]
    if len(faces) == 0:
        return {
            "available": False,
            "max": None,
            "p999": None,
            "max_growth_m": None,
        }
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    posed = np.asarray(asset.pose_cache_vertices, dtype=np.float64)
    before = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    after = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
    # Ignore sub-5 mm edges: finger CAD seams dominate ratio noise while the
    # absolute-growth gate still catches real long-bone explosions (tibia).
    valid = before > 5.0e-3
    if not np.any(valid):
        return {
            "available": False,
            "max": None,
            "p999": None,
            "max_growth_m": None,
        }
    ratio = after[valid] / before[valid]
    return {
        "available": True,
        "max": float(np.max(ratio)),
        "p999": float(np.quantile(ratio, 0.999)),
        "max_growth_m": float(np.max(after[valid] - before[valid])),
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
    failures: list[str] = []
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
    schema_report = _required_mapping(
        source_report, "schema", failures=failures, label="asset schema"
    )
    if schema_report is not None:
        actual_schema = _required_number(
            schema_report,
            "asset_schema_version",
            failures=failures,
            label="asset schema",
        )
        expected_schema = _required_number(
            schema_report,
            "expected_schema_version",
            failures=failures,
            label="asset schema",
        )
        if actual_schema is not None and int(actual_schema) != ANATOMY_ASSET_SCHEMA_VERSION:
            failures.append(
                f"asset schema {int(actual_schema)} does not match required schema "
                f"{ANATOMY_ASSET_SCHEMA_VERSION}"
            )
        if expected_schema is not None and int(expected_schema) != ANATOMY_ASSET_SCHEMA_VERSION:
            failures.append("schema report was generated for a different asset schema")
        if schema_report.get("passed") is not True:
            failures.append("asset schema report did not pass")

    manifest_report = _required_mapping(
        source_report, "manifest", failures=failures, label="canonical manifest"
    )
    if manifest_report is not None:
        for key in ("source", "gender", "betas", "sha256"):
            if key not in manifest_report or manifest_report[key] in (None, "", []):
                failures.append(f"canonical manifest field {key!r} is missing")
    run_manifest = _required_mapping(
        source_report,
        "run_manifest",
        failures=failures,
        label="content-addressed run manifest",
    )
    if run_manifest is not None:
        for key in (
            "schema_version",
            "content_hash",
            "git",
            "inputs",
            "code_files",
            "solver_versions",
            "coordinate_contract",
            "environment",
        ):
            if key not in run_manifest or run_manifest[key] in (None, "", [], {}):
                failures.append(f"run manifest field {key!r} is missing")
        coordinate_contract = run_manifest.get("coordinate_contract")
        if isinstance(coordinate_contract, dict):
            if coordinate_contract.get("asset") != "smplx_y_up_m":
                failures.append("run manifest asset coordinate must be smplx_y_up_m")
            if int(coordinate_contract.get("viewer_transform_count", -1)) != 1:
                failures.append("run manifest must specify exactly one viewer axis transform")
    bind_report = _required_mapping(
        source_report,
        "source_bind_roundtrip",
        failures=failures,
        label="source/target bind roundtrip",
    )
    if bind_report is not None:
        if bind_report.get("pass") is not True:
            failures.append("source/target bind roundtrip did not pass")
        for key in (
            "max_matrix_error",
            "zero_pose_vertex_error_m",
            "target_bind_max_matrix_error",
            "target_zero_pose_vertex_error_m",
        ):
            error = _required_number(
                bind_report,
                key,
                failures=failures,
                label="source/target bind roundtrip",
            )
            if error is not None and error > thresholds["bind_roundtrip_error_m"]:
                failures.append(
                    f"source/target bind roundtrip {key}={error:.6g} exceeds "
                    f"{thresholds['bind_roundtrip_error_m']:.6g}"
                )

    rest_align = _required_mapping(
        source_report, "rest_align", failures=failures, label="source rest-align"
    )
    rest_anchor_rms = (
        _required_number(
            rest_align,
            "anchor_rms_m",
            failures=failures,
            label="source rest-align",
        )
        if rest_align is not None
        else None
    )
    rest_anchor_max = (
        _required_number(
            rest_align,
            "max_joint_offset_m",
            failures=failures,
            label="source rest-align",
        )
        if rest_align is not None
        else None
    )
    landmark_report = _required_mapping(
        source_report, "landmark_report", failures=failures, label="geometry landmark"
    )
    if landmark_report is not None and landmark_report.get("passed") is not True:
        failures.append("geometry landmark report did not pass")

    stretch = _required_mapping(
        source_report, "edge_stretch", failures=failures, label="edge stretch"
    ) or {}
    signed = _signed_distances(asset.vertices_rest, Path(canonical_dir))
    containment = _containment_by_tissue(asset, signed)
    regions = _region_containment(asset, signed)
    brain_skull = _brain_skull_metrics(asset)
    bone_pose_stretch = _bone_pose_edge_stretch(asset)
    soft_pose_stretch = _soft_mesh_pose_stretch(asset)
    cranial_compound = _cranial_compound_metrics(asset)
    foot_reach = _foot_reach_metrics(asset, Path(canonical_dir))

    volume_report = _required_mapping(
        source_report,
        "volume_registration",
        failures=failures,
        label="source volume registration",
    )
    subject_volume_report = _required_mapping(
        source_report,
        "shape",
        failures=failures,
        label="subject beta volume registration",
    )
    for stage, stage_report in (
        ("source volume registration", volume_report),
        ("subject beta volume registration", subject_volume_report),
    ):
        if stage_report is None:
            continue
        inverted_key = (
            "inverted_tetrahedra"
            if "inverted_tetrahedra" in stage_report
            else "diagnostic_inverted_tetrahedra"
            if "diagnostic_inverted_tetrahedra" in stage_report
            else ""
        )
        if not inverted_key:
            failures.append(f"{stage} field 'inverted_tetrahedra' is missing")
            inverted_tetrahedra = None
        else:
            inverted_tetrahedra = _required_number(
                stage_report,
                inverted_key,
                failures=failures,
                label=stage,
            )
        outside_soft = _required_number(
            stage_report,
            "outside_soft_material_count",
            failures=failures,
            label=stage,
        )
        minimum_jacobian = _required_number(
            stage_report,
            "minimum_jacobian_ratio",
            failures=failures,
            label=stage,
        )
        if inverted_tetrahedra is not None and int(inverted_tetrahedra) != 0:
            failures.append(
                f"{stage} contains {int(inverted_tetrahedra)} inverted tetrahedra"
            )
        if outside_soft is not None and int(outside_soft) != 0:
            failures.append(f"{stage} excludes {int(outside_soft)} soft material vertices")
        if minimum_jacobian is not None and minimum_jacobian < 0.05:
            failures.append(
                f"{stage} minimum Jacobian ratio {minimum_jacobian:.6f} is below 0.050000"
            )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        failures.append("weights contain NaN/Inf or negative values")
    if weight_error > thresholds["weight_sum_error"]:
        failures.append(f"weight sum error {weight_error:.3g} exceeds {thresholds['weight_sum_error']:.3g}")
    if len(active_hands) != 30:
        failures.append(f"only {len(active_hands)}/30 SMPL-X finger joints have non-zero weights")
    active_groups = _required_number(
        source_report,
        "active_source_group_count",
        failures=failures,
        label="Blender source groups",
    )
    if active_groups is not None and int(active_groups) <= 0:
        failures.append("Blender source contains no active vertex groups")
    source_audit = _required_mapping(
        source_report,
        "source_audit",
        failures=failures,
        label="Blender source audit",
    )
    if source_audit is not None:
        coordinate = source_audit.get("coordinate_metadata")
        if not isinstance(coordinate, dict):
            failures.append("Blender source audit coordinate metadata is missing")
        elif coordinate.get("output_coordinate_system") != "smplx_y_up_m":
            failures.append("Blender source audit output coordinate must be smplx_y_up_m")
        armature = source_audit.get("armature")
        if not isinstance(armature, dict):
            failures.append("Blender source audit armature record is missing")
        elif int(armature.get("bone_count", -1)) != len(asset.source_bone_names or []):
            failures.append("Blender source audit bone count does not match the asset")
        selection = source_audit.get("selection")
        if not isinstance(selection, dict):
            failures.append("Blender source audit selection record is missing")
        elif int(selection.get("selected_mesh_count", -1)) != len(asset.source_mesh_names):
            failures.append("Blender source audit mesh count does not match the asset")
        semantic_audit = source_audit.get("semantic_manifest")
        if not isinstance(semantic_audit, dict):
            failures.append("resolved source semantic manifest is missing")
        elif int(semantic_audit.get("resolved_mesh_count", -1)) != len(
            asset.source_mesh_names
        ):
            failures.append("semantic manifest did not resolve every selected mesh")
        weight_audit = source_audit.get("weight_influences")
        if not isinstance(weight_audit, dict):
            failures.append("full source influence audit is missing")
        parity = source_audit.get("blender_lbs_parity")
        if not isinstance(parity, dict):
            failures.append("Blender depsgraph LBS parity report is missing")
        else:
            parity_mesh_count = _required_number(
                parity,
                "mesh_count",
                failures=failures,
                label="Blender depsgraph LBS parity",
            )
            parity_max = _required_number(
                parity,
                "max_error_m",
                failures=failures,
                label="Blender depsgraph LBS parity",
            )
            if (
                parity_mesh_count is not None
                and int(parity_mesh_count) != len(asset.source_mesh_names)
            ):
                failures.append("Blender parity did not cover every source mesh")
            if (
                parity_max is not None
                and parity_max > thresholds["blender_parity_error_m"]
            ):
                failures.append(
                    f"Blender LBS parity max {parity_max * 1000.0:.3f} mm exceeds "
                    f"{thresholds['blender_parity_error_m'] * 1000.0:.3f} mm"
                )

    material_report = _required_mapping(
        source_report, "material_shape", failures=failures, label="material post-fit"
    )
    anchor_rms = (
        _required_number(
            material_report,
            "anchor_rms_m",
            failures=failures,
            label="material post-fit",
        )
        if material_report is not None
        else None
    )
    anchor_max = (
        _required_number(
            material_report,
            "anchor_max_m",
            failures=failures,
            label="material post-fit",
        )
        if material_report is not None
        else None
    )
    if anchor_rms is not None and anchor_rms > thresholds["anchor_rms_m"]:
        failures.append(
            f"post-fit anchor RMS {anchor_rms * 1000.0:.1f} mm exceeds "
            f"{thresholds['anchor_rms_m'] * 1000.0:.1f} mm"
        )
    if anchor_max is not None and anchor_max > thresholds["anchor_max_m"]:
        failures.append(
            f"post-fit anchor max {anchor_max * 1000.0:.1f} mm exceeds "
            f"{thresholds['anchor_max_m'] * 1000.0:.1f} mm"
        )

    stages: list[tuple[str, float | None, float | None, float | None]] = [
        (
            "source_to_final",
            _required_number(
                stretch,
                "source_to_final_max",
                failures=failures,
                label="source-to-final edge stretch",
            ),
            _required_number(
                stretch,
                "source_to_final_p999",
                failures=failures,
                label="source-to-final edge stretch",
            ),
            _required_number(
                stretch,
                "source_to_final_max_growth_m",
                failures=failures,
                label="source-to-final edge stretch",
            ),
        ),
        (
            "bone_to_pose_cache",
            _required_number(
                bone_pose_stretch,
                "max",
                failures=failures,
                label="bone pose edge stretch",
            ),
            _required_number(
                bone_pose_stretch,
                "p999",
                failures=failures,
                label="bone pose edge stretch",
            ),
            _required_number(
                bone_pose_stretch,
                "max_growth_m",
                failures=failures,
                label="bone pose edge stretch",
            ),
        ),
    ]
    for stage, maximum, p999, growth in stages:
        if maximum is not None and maximum > thresholds["edge_ratio_max"]:
            failures.append(f"{stage} max edge ratio {maximum:.2f} exceeds {thresholds['edge_ratio_max']:.2f}")
        if p999 is not None and p999 > thresholds["edge_ratio_p999"]:
            failures.append(f"{stage} p99.9 edge ratio {p999:.2f} exceeds {thresholds['edge_ratio_p999']:.2f}")
        if growth is not None and growth > thresholds["edge_growth_max_m"]:
            failures.append(
                f"{stage} maximum absolute edge growth {growth * 1000.0:.1f} mm exceeds "
                f"{thresholds['edge_growth_max_m'] * 1000.0:.1f} mm"
            )

    present_soft_tissues = sorted(set(asset.source_tissues or []) & SOFT_TISSUES)
    for tissue in present_soft_tissues:
        metrics = containment.get(tissue)
        if metrics is None or metrics.get("available") is not True:
            failures.append(f"{tissue} subject-rest signed-distance containment is unavailable")
            continue
        inside = _required_number(
            metrics,
            "inside_fraction",
            failures=failures,
            label=f"{tissue} subject-rest containment",
        )
        outside = _required_number(
            metrics,
            "max_outside_m",
            failures=failures,
            label=f"{tissue} subject-rest containment",
        )
        if inside is not None and inside < thresholds["inside_fraction"]:
            failures.append(
                f"{tissue} containment {inside * 100.0:.2f}% is below "
                f"{thresholds['inside_fraction'] * 100.0:.2f}%"
            )
        outside_limit = (
            thresholds["critical_max_outside_m"]
            if tissue in CRITICAL_SOFT_TISSUES
            else thresholds["max_outside_m"]
        )
        if outside is not None and outside > outside_limit:
            failures.append(
                f"{tissue} maximum protrusion {outside * 1000.0:.1f} mm exceeds "
                f"{outside_limit * 1000.0:.1f} mm"
            )

    for region, metrics in regions.items():
        if metrics.get("available") is not True:
            failures.append(f"{region} containment is unavailable")
            continue
        inside = _required_number(
            metrics,
            "inside_fraction",
            failures=failures,
            label=f"{region} containment",
        )
        outside = _required_number(
            metrics,
            "max_outside_m",
            failures=failures,
            label=f"{region} containment",
        )
        if inside is not None and inside < thresholds["hand_foot_inside_fraction"]:
            failures.append(
                f"{region} containment {inside * 100.0:.2f}% is below "
                f"{thresholds['hand_foot_inside_fraction'] * 100.0:.2f}%"
            )
        if outside is not None and outside > thresholds["hand_foot_max_outside_m"]:
            failures.append(
                f"{region} maximum protrusion {outside * 1000.0:.1f} mm exceeds "
                f"{thresholds['hand_foot_max_outside_m'] * 1000.0:.1f} mm"
            )
    if brain_skull.get("available") is not True:
        failures.append("brain/skull containment is unavailable")
    else:
        brain_inside = _required_number(
            brain_skull,
            "inside_fraction",
            failures=failures,
            label="brain/skull containment",
        )
        if brain_inside is not None and brain_inside < thresholds["brain_inside_skull_fraction"]:
            failures.append(
                f"brain inside skull {brain_inside * 100.0:.2f}% is below "
                f"{thresholds['brain_inside_skull_fraction'] * 100.0:.2f}%"
            )
    if not bool(cranial_compound.get("available", False)):
        failures.append("cranial compound membership/skull reference is unavailable")
    else:
        shared_rms = _required_number(
            cranial_compound,
            "shared_transform_rms_m",
            failures=failures,
            label="cranial compound",
        )
        if shared_rms is not None and shared_rms > thresholds["cranial_shared_transform_rms_m"]:
            failures.append(
                f"cranial compound shared-transform RMS {shared_rms * 1000.0:.3f} mm exceeds "
                f"{thresholds['cranial_shared_transform_rms_m'] * 1000.0:.3f} mm"
            )
        upper_teeth = list(cranial_compound.get("upper_teeth_meshes", []))
        if not upper_teeth:
            failures.append("cranial compound is missing an Upper_Teeth mesh")
        else:
            drift = _required_number(
                cranial_compound,
                "upper_teeth_skull_distance_drift_m",
                failures=failures,
                label="cranial compound",
            )
            if drift is not None and drift > thresholds["upper_teeth_skull_distance_drift_m"]:
                failures.append(
                    f"upper-teeth/skull transform drift {drift * 1000.0:.3f} mm exceeds "
                    f"{thresholds['upper_teeth_skull_distance_drift_m'] * 1000.0:.3f} mm"
                )

    if material_report is not None:
        for group in ("cranial", "pelvis"):
            change = _required_number(
                material_report,
                f"{group}_aspect_ratio_change",
                failures=failures,
                label="material post-fit",
            )
            if change is not None and change > thresholds["compound_aspect_ratio_change"]:
                failures.append(
                    f"{group} aspect-ratio change {change * 100.0:.2f}% exceeds "
                    f"{thresholds['compound_aspect_ratio_change'] * 100.0:.2f}%"
                )
            scale_report = _required_mapping(
                material_report,
                f"{group}_scale_report",
                failures=failures,
                label=f"{group} scale",
            )
            if scale_report is not None:
                if "saturated" not in scale_report:
                    failures.append(f"{group} scale field 'saturated' is missing")
                elif bool(scale_report["saturated"]):
                    failures.append(f"{group} uniform scale saturated")

        hip_geometry = _required_mapping(
            material_report,
            "hip_geometry",
            failures=failures,
            label="material hip geometry",
        )
        for side in ("left", "right"):
            metrics = (
                hip_geometry.get(side)
                if isinstance(hip_geometry, dict)
                else None
            )
            if not isinstance(metrics, dict) or not metrics:
                failures.append(f"{side} femoral-head/acetabulum landmark is missing")
                continue
            gap = _required_number(
                metrics,
                "femoral_head_to_acetabulum_m",
                failures=failures,
                label=f"{side} hip landmark",
            )
            if gap is not None and gap > 0.002:
                failures.append(
                    f"{side} femoral-head/acetabulum gap {gap * 1000.0:.2f} mm exceeds 2.00 mm"
                )
        center_drift = _required_number(
            material_report,
            "brain_skull_center_drift_m",
            failures=failures,
            label="material post-fit",
        )
        if center_drift is not None and center_drift > thresholds["brain_skull_center_drift_m"]:
            failures.append(
                f"brain/skull center drift {center_drift * 1000.0:.2f} mm exceeds "
                f"{thresholds['brain_skull_center_drift_m'] * 1000.0:.2f} mm"
            )
        end_change = _required_number(
            material_report,
            "long_bone_end_edge_change",
            failures=failures,
            label="material post-fit",
        )
        if end_change is not None and end_change > thresholds["long_bone_end_edge_change"]:
            failures.append(
                f"long-bone protected-end edge change {end_change * 100.0:.2f}% exceeds "
                f"{thresholds['long_bone_end_edge_change'] * 100.0:.2f}%"
            )
        feet = _required_mapping(
            material_report,
            "feet",
            failures=failures,
            label="material foot landmarks",
        )
        for side in ("left", "right"):
            metrics = feet.get(side) if isinstance(feet, dict) else None
            if not isinstance(metrics, dict) or not metrics:
                failures.append(f"{side} material foot landmark is missing")
                continue
            if metrics.get("fit_policy") not in {
                "ankle_foot_similarity_compound",
                "ef58024_ankle_axial_material_fit",
            }:
                failures.append(f"{side} foot used an unknown fit policy")
            if metrics.get("post_projection_applied") is not False:
                failures.append(f"{side} foot used a forbidden post-fit projection")

    soft_meshes = [
        str(name)
        for name, tissue in zip(asset.source_mesh_names, asset.source_tissues or [])
        if str(tissue) in SOFT_TISSUES
    ]
    for mesh in soft_meshes:
        metrics = soft_pose_stretch.get(mesh)
        if metrics is None:
            failures.append(f"{mesh} posed soft-tissue stretch report is missing")
            continue
        q99 = _required_number(
            metrics,
            "ratio_q99",
            failures=failures,
            label=f"{mesh} posed soft-tissue stretch",
        )
        maximum = _required_number(
            metrics,
            "ratio_max",
            failures=failures,
            label=f"{mesh} posed soft-tissue stretch",
        )
        if q99 is not None and q99 > thresholds["soft_edge_ratio_q99"]:
            failures.append(
                f"{mesh} soft edge q99 ratio {q99:.3f} exceeds "
                f"{thresholds['soft_edge_ratio_q99']:.3f}"
            )
        if maximum is not None and maximum > thresholds["soft_edge_ratio_max"]:
            failures.append(
                f"{mesh} soft edge max ratio {maximum:.3f} exceeds "
                f"{thresholds['soft_edge_ratio_max']:.3f}"
            )
    tube_reports = _required_mapping(
        source_report,
        "tube_graphs",
        failures=failures,
        label="vessel/nerve material graphs",
    )
    critical_graph_meshes = [
        str(name)
        for name, tissue in zip(
            asset.source_mesh_names,
            asset.source_tissues or [],
        )
        if str(tissue) in CRITICAL_SOFT_TISSUES
    ]
    if tube_reports is not None:
        for mesh in critical_graph_meshes:
            graph_report = tube_reports.get(mesh)
            if not isinstance(graph_report, dict):
                failures.append(f"tube material graph {mesh!r} is missing")
                continue
            if graph_report.get("topology_preserved") is not True:
                failures.append(f"tube material graph {mesh!r} changed topology")
            for stage in ("neutral_to_subject", "subject_to_pose"):
                metrics = graph_report.get(stage)
                if not isinstance(metrics, dict):
                    failures.append(f"tube material graph {mesh!r} {stage} metrics are missing")
                    continue
                degenerate = _required_number(
                    metrics,
                    "degenerate_edge_count",
                    failures=failures,
                    label=f"tube graph {mesh} {stage}",
                )
                p99 = _required_number(
                    metrics,
                    "length_ratio_p99",
                    failures=failures,
                    label=f"tube graph {mesh} {stage}",
                )
                maximum = _required_number(
                    metrics,
                    "length_ratio_max",
                    failures=failures,
                    label=f"tube graph {mesh} {stage}",
                )
                minimum = _required_number(
                    metrics,
                    "length_ratio_min",
                    failures=failures,
                    label=f"tube graph {mesh} {stage}",
                )
                if degenerate is not None and int(degenerate) != 0:
                    failures.append(
                        f"tube graph {mesh} {stage} has {int(degenerate)} collapsed edges"
                    )
                if p99 is not None and p99 > thresholds["tube_graph_ratio_p99"]:
                    failures.append(
                        f"tube graph {mesh} {stage} p99 ratio {p99:.3f} exceeds "
                        f"{thresholds['tube_graph_ratio_p99']:.3f}"
                    )
                if maximum is not None and maximum > thresholds["tube_graph_ratio_max"]:
                    failures.append(
                        f"tube graph {mesh} {stage} max ratio {maximum:.3f} exceeds "
                        f"{thresholds['tube_graph_ratio_max']:.3f}"
                    )
                if minimum is not None and minimum < 1.0 / thresholds["tube_graph_ratio_max"]:
                    failures.append(
                        f"tube graph {mesh} {stage} min ratio {minimum:.3f} is below "
                        f"{1.0 / thresholds['tube_graph_ratio_max']:.3f}"
                    )
    tube_bone_intersections = _required_mapping(
        source_report,
        "tube_bone_intersections",
        failures=failures,
        label="tube/bone triangle intersections",
    )
    if tube_bone_intersections is not None:
        total_net_new = _required_number(
            tube_bone_intersections,
            "positive_per_mesh_total_net_new_count",
            failures=failures,
            label="tube/bone triangle intersections",
        )
        station_net_new = _required_number(
            tube_bone_intersections,
            "positive_per_mesh_station_follow_net_new_count",
            failures=failures,
            label="tube/bone triangle intersections",
        )
        if total_net_new is not None and total_net_new > 0:
            failures.append(
                f"final anatomy has {int(total_net_new)} per-mesh net-new tube/bone "
                "triangle intersections relative to the all-harmonic reference"
            )
        if station_net_new is not None and station_net_new > 0:
            failures.append(
                f"station soft follow has {int(station_net_new)} per-mesh net-new "
                "tube/bone triangle intersections relative to final bones with harmonic tubes"
            )
    runtime_matrix = _required_mapping(
        source_report,
        "runtime_pose_matrix",
        failures=failures,
        label="runtime pose validation matrix",
    )
    if runtime_matrix is not None:
        cases = runtime_matrix.get("cases")
        if not isinstance(cases, dict) or not cases:
            failures.append("runtime pose validation cases are missing")
        else:
            required_cases = {
                "pose_zero",
                "pose_upper_limb_flex",
                "pose_lower_limb_flex",
                "pose_finger_flex",
                "pose_axial_twist",
            }
            missing_cases = sorted(required_cases - set(cases))
            if missing_cases:
                failures.append(f"runtime pose validation cases are missing: {missing_cases}")
            for case_name, case in cases.items():
                if not isinstance(case, dict) or case.get("finite") is not True:
                    failures.append(f"runtime pose case {case_name!r} is non-finite")
                    continue
                soft = case.get("soft_meshes")
                if not isinstance(soft, dict):
                    failures.append(f"runtime pose case {case_name!r} soft metrics are missing")
                    continue
                for mesh in critical_graph_meshes:
                    metrics = soft.get(mesh)
                    if not isinstance(metrics, dict):
                        failures.append(
                            f"runtime pose case {case_name!r} omits critical mesh {mesh!r}"
                        )
                        continue
                    q99 = _required_number(
                        metrics,
                        "ratio_q99",
                        failures=failures,
                        label=f"runtime pose {case_name} {mesh}",
                    )
                    maximum = _required_number(
                        metrics,
                        "ratio_max",
                        failures=failures,
                        label=f"runtime pose {case_name} {mesh}",
                    )
                    if q99 is not None and q99 > thresholds["soft_edge_ratio_q99"]:
                        failures.append(
                            f"runtime pose {case_name} {mesh} q99 {q99:.3f} exceeds "
                            f"{thresholds['soft_edge_ratio_q99']:.3f}"
                        )
                    if (
                        maximum is not None
                        and maximum > thresholds["soft_edge_ratio_max"]
                    ):
                        failures.append(
                            f"runtime pose {case_name} {mesh} max {maximum:.3f} exceeds "
                            f"{thresholds['soft_edge_ratio_max']:.3f}"
                        )

    containment_stages_raw = source_report.get("containment_stages")
    containment_stages: list[dict[str, Any]] = []
    if not isinstance(containment_stages_raw, list) or not containment_stages_raw:
        failures.append("signed-distance containment stage reports are missing")
    else:
        containment_stages = [
            dict(item) for item in containment_stages_raw if isinstance(item, dict)
        ]
    containment_by_stage = {
        str(report.get("stage")): report
        for report in containment_stages
        if report.get("stage")
    }
    critical_meshes = [
        (str(name), str(tissue))
        for name, tissue in zip(asset.source_mesh_names, asset.source_tissues or [])
        if str(tissue) in CRITICAL_SOFT_TISSUES
    ]
    for stage in REQUIRED_CONTAINMENT_STAGES:
        report = containment_by_stage.get(stage)
        if report is None:
            failures.append(f"{stage} signed-distance containment report is missing")
            continue
        backend = str(report.get("backend", ""))
        if "signed_distance" not in backend:
            failures.append(f"{stage} containment backend is missing signed-distance evidence")
        over_limit = report.get("over_limit_count")
        if not isinstance(over_limit, dict):
            failures.append(f"{stage} containment field 'over_limit_count' is missing")
        else:
            for tissue in present_soft_tissues:
                if tissue not in over_limit:
                    failures.append(f"{stage} containment omits {tissue}")
                    continue
                try:
                    count = int(over_limit[tissue])
                except (TypeError, ValueError):
                    failures.append(f"{stage} containment count for {tissue} is invalid")
                    continue
                if count > 0:
                    failures.append(
                        f"{stage} {tissue} containment exceeds publication limits: {count}"
                    )
        per_mesh = report.get("per_mesh")
        if not isinstance(per_mesh, dict):
            failures.append(f"{stage} per-region vessel/nerve containment is missing")
        else:
            for mesh, tissue in critical_meshes:
                metrics = per_mesh.get(mesh)
                if not isinstance(metrics, dict):
                    failures.append(f"{stage} containment region {mesh!r} is missing")
                    continue
                count = _required_number(
                    metrics,
                    "over_limit_count",
                    failures=failures,
                    label=f"{stage} containment region {mesh}",
                )
                if count is not None and int(count) > 0:
                    failures.append(
                        f"{stage} {tissue} region {mesh} exceeds containment limits: {int(count)}"
                    )

    pose_report = _required_mapping(
        source_report,
        "pose_cache_report",
        failures=failures,
        label="posed signed-distance containment",
    )
    if pose_report is not None and not isinstance(pose_report.get("over_limit_count"), dict):
        failures.append("posed containment field 'over_limit_count' is missing")

    bone_chain_report = _required_mapping(
        source_report,
        "bone_segment_diagnostics",
        failures=failures,
        label="bone segment diagnostics",
    )
    if bone_chain_report is not None:
        if bone_chain_report.get("passed") is not True:
            failures.append("bone segment diagnostics top-level passed is false")
        report_failures = bone_chain_report.get("failures")
        if not isinstance(report_failures, list):
            failures.append("bone segment diagnostics failure list is missing")
        elif report_failures:
            failures.append(f"bone segment diagnostics failures: {report_failures}")
        joints = bone_chain_report.get("joints")
        if not isinstance(joints, dict) or not joints:
            failures.append("bone joint landmark diagnostics are missing")
        else:
            failed_chains = [
                name
                for name, metrics in joints.items()
                if not isinstance(metrics, dict) or metrics.get("pass") is not True
            ]
            if failed_chains:
                failures.append(
                    f"bone-chain controller/geometry regression failed: {failed_chains}"
                )
            for side in ("left", "right"):
                label = f"hip_{side}"
                metrics = joints.get(label)
                if not isinstance(metrics, dict):
                    failures.append(f"{side} hip bone landmark diagnostic is missing")
                    continue
                geometry = metrics.get("geometry_landmarks")
                if not isinstance(geometry, dict) or geometry.get("available") is not True:
                    failures.append(f"{side} hip geometry landmark is unavailable")
        rigidity = bone_chain_report.get("rigidity_segments")
        if not isinstance(rigidity, dict) or not rigidity:
            failures.append("bone rigidity diagnostics are missing")
        else:
            rigidity_failures = [
                f"{segment}/{item.get('mesh', '<unknown>')}"
                for segment, items in rigidity.items()
                if isinstance(items, list)
                for item in items
                if not isinstance(item, dict) or item.get("pass") is not True
            ]
            unavailable_segments = [
                segment
                for segment, items in rigidity.items()
                if not isinstance(items, list) or not items
            ]
            if unavailable_segments:
                failures.append(f"bone rigidity segments are unavailable: {unavailable_segments}")
            if rigidity_failures:
                failures.append(f"bone rigidity regressions failed: {rigidity_failures}")
        head_orientation = bone_chain_report.get("head_orientation")
        if not isinstance(head_orientation, dict) or head_orientation.get("pass") is not True:
            failures.append("head orientation regression failed or is unavailable")

    critical_tokens = (
        "schema",
        "non-finite",
        "nonfinite",
        "missing",
        "unavailable",
        "inverted",
        "zero-pose",
        "zero pose",
        "topology",
        "degenerate",
    )
    issues = [
        {
            "severity": (
                "critical"
                if any(token in str(message).lower() for token in critical_tokens)
                else "warning"
            ),
            "message": str(message),
        }
        for message in failures
    ]
    baseline = dict((asset.metadata or {}).get("ef58024_quality_baseline") or {})
    regression_comparison = {
        "baseline_commit": "ef58024a15765e9b5701323259687a97141024b",
        "available": bool(baseline),
        "baseline_metrics": baseline if baseline else None,
    }
    return {
        "schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
        "passed": not failures,
        "failures": failures,
        "issues": issues,
        "critical_failures": [
            issue["message"] for issue in issues if issue["severity"] == "critical"
        ],
        "warnings": [
            issue["message"] for issue in issues if issue["severity"] == "warning"
        ],
        "regression_comparison": regression_comparison,
        "thresholds": thresholds,
        "asset_schema": schema_report,
        "canonical_manifest": manifest_report,
        "run_manifest": run_manifest,
        "bind_roundtrip": bind_report,
        "source_audit": source_audit,
        "weights": {
            "max_sum_error": weight_error,
            "negative_count": int(np.count_nonzero(weights < 0.0)),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(weights))),
            "active_finger_joints": len(active_hands),
        },
        "anchors": {
            "source_rest_align_rms_m": rest_anchor_rms,
            "source_rest_align_max_m": rest_anchor_max,
            "post_fit_rms_m": anchor_rms,
            "post_fit_max_m": anchor_max,
        },
        "landmarks": landmark_report,
        "edge_stretch": stretch,
        "bone_pose_edge_stretch": bone_pose_stretch,
        "soft_mesh_pose_stretch": soft_pose_stretch,
        "tube_graphs": tube_reports or {},
        "tube_bone_intersections": tube_bone_intersections or {},
        "runtime_pose_matrix": runtime_matrix or {},
        "volume_registration": volume_report or {},
        "subject_volume_registration": subject_volume_report or {},
        "bone_chains": bone_chain_report or {},
        "containment_backend": "libigl_exact_signed_distance",
        "containment": containment,
        "containment_stages": containment_stages,
        "regional_containment": regions,
        "brain_skull": brain_skull,
        "cranial_compound": cranial_compound,
        "foot_reach": foot_reach,
        "material_shape": material_report or {},
    }


def write_quality_report(path: Path | str, report: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    return out
