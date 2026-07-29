"""Length-incompatible V8.10 leg retargeting.

SMPL-X joints provide pose directions, not mandatory anatomical endpoints.
The V71 bind hierarchy keeps its authored lengths and articular pivots.  A
reviewed BA9 subject supplies only the radial femur direction, while axial
length residuals are measured and retained.  The expensive work is baked at
L0/L1; pose-time evaluation remains the existing 235-bone parent-local FK.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .acceptance_v8 import fit_sphere
from .rigged_asset import AnatomyRiggedAsset


LEG_CENTERLINE_SCHEMA_VERSION_V810 = 810
_PREFIX = "leg_centerline_v810."
_FEMUR_STATIONS = (0.15, 0.30, 0.50, 0.70, 0.85)
_SHANK_STATIONS = (0.25, 0.50, 0.75)
_FOOT_TOKENS = (
    "talus",
    "calcaneus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
    "phalanx_foot",
)


def _mesh_vertex_ids(asset: AnatomyRiggedAsset, mesh_name: str) -> np.ndarray:
    try:
        index = list(asset.source_mesh_names or ()).index(mesh_name)
    except ValueError as exc:
        raise ValueError(f"required V8.10 mesh {mesh_name!r} is missing") from exc
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[index]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _domain_ids(domains: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    missing = [name for name in names if name not in domains]
    if missing:
        raise ValueError(f"required V8.10 material domains are missing: {missing}")
    return np.unique(
        np.concatenate(
            [np.asarray(domains[name], dtype=np.int64).reshape(-1) for name in names]
        )
    )


def _global_to_local(global_frames: np.ndarray, parents: np.ndarray) -> np.ndarray:
    frames = np.asarray(global_frames, dtype=np.float64)
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    if frames.shape != (len(parent_ids), 4, 4):
        raise ValueError("global frames and parents have incompatible shapes")
    local = frames.copy()
    for bone, parent in enumerate(parent_ids.tolist()):
        if parent >= 0:
            local[bone] = np.linalg.inv(frames[parent]) @ frames[bone]
    return local


def _descendant_mask(
    names: Sequence[str],
    parents: np.ndarray,
    ancestor: str,
) -> np.ndarray:
    if ancestor not in names:
        raise ValueError(f"source rig is missing required bone {ancestor!r}")
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    root = list(names).index(ancestor)
    result = np.zeros(len(names), dtype=bool)
    for bone in range(len(names)):
        current = bone
        for _ in range(len(names) + 1):
            if current == root:
                result[bone] = True
                break
            if current < 0:
                break
            current = int(parent_ids[current])
        else:
            raise ValueError("source bone hierarchy contains a cycle")
    return result


def _foot_bone_vertex_ids(
    asset: AnatomyRiggedAsset,
    *,
    suffix: str,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    names = list(asset.source_mesh_names or ())
    tissues = list(asset.source_tissues or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    for name, tissue, (start, stop) in zip(names, tissues, ranges):
        lower = str(name).lower()
        if (
            str(tissue).strip().lower() == "bone"
            and str(name).endswith(f"_{suffix}")
            and any(token in lower for token in _FOOT_TOKENS)
        ):
            selected.append(np.arange(int(start), int(stop), dtype=np.int64))
    if not selected:
        raise ValueError(f"V8.10 found no {suffix} foot bone meshes")
    return np.unique(np.concatenate(selected))


def _smootherstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _axis_parameter(
    vertices: np.ndarray,
    *,
    proximal: np.ndarray,
    distal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    axis = np.asarray(distal, dtype=np.float64) - np.asarray(
        proximal, dtype=np.float64
    )
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-8:
        raise ValueError("long-bone endpoints are degenerate")
    direction = axis / length
    parameter = (
        np.asarray(vertices, dtype=np.float64)
        - np.asarray(proximal, dtype=np.float64)
    ) @ direction / length
    return parameter, direction, length


def _proper_direction_rotation(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_vectors, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vectors, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 2:
        raise ValueError("direction fit requires matching vector sets")
    source_norm = np.linalg.norm(source, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    if np.any(source_norm <= 1.0e-8) or np.any(target_norm <= 1.0e-8):
        raise ValueError("direction fit contains a degenerate vector")
    source = source / source_norm[:, None]
    target = target / target_norm[:, None]
    u, _singular, vt = np.linalg.svd(source.T @ target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10)
    ):
        raise ValueError("direction fit did not produce a proper rotation")
    return rotation


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.asarray(Rotation.from_matrix(matrix).as_rotvec(), dtype=np.float64)


def _station_centers(
    *,
    vertices: np.ndarray,
    sample_ids: np.ndarray,
    parameter: np.ndarray,
    fractions: Sequence[float],
    half_width: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    values: list[np.ndarray] = []
    selected_ids: list[np.ndarray] = []
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    s = np.asarray(parameter, dtype=np.float64).reshape(-1)
    for fraction in fractions:
        distance = np.abs(s - float(fraction))
        local = np.flatnonzero(distance <= float(half_width))
        if len(local) < 12:
            local = np.argsort(distance)[: min(48, len(distance))]
        chosen = ids[local]
        values.append(
            np.mean(np.asarray(vertices, dtype=np.float64)[chosen], axis=0)
        )
        selected_ids.append(chosen)
    return np.asarray(values, dtype=np.float64), selected_ids


def _direction_fit_report(
    *,
    source: AnatomyRiggedAsset,
    reference: AnatomyRiggedAsset,
    sample_ids: np.ndarray,
    proximal_ids: np.ndarray,
    distal_ids: np.ndarray,
    fractions: Sequence[float],
    half_width: float,
    pivot: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_vertices = np.asarray(source.vertices_rest, dtype=np.float64)
    reference_vertices = np.asarray(reference.vertices_rest, dtype=np.float64)
    proximal_ids = np.asarray(proximal_ids, dtype=np.int64)
    distal_ids = np.asarray(distal_ids, dtype=np.int64)
    source_proximal = np.mean(source_vertices[proximal_ids], axis=0)
    source_distal = np.mean(source_vertices[distal_ids], axis=0)
    reference_proximal = np.mean(reference_vertices[proximal_ids], axis=0)
    reference_distal = np.mean(reference_vertices[distal_ids], axis=0)
    parameter, _direction, source_length = _axis_parameter(
        source_vertices[np.asarray(sample_ids, dtype=np.int64)],
        proximal=source_proximal,
        distal=source_distal,
    )
    source_centers, station_ids = _station_centers(
        vertices=source_vertices,
        sample_ids=sample_ids,
        parameter=parameter,
        fractions=fractions,
        half_width=half_width,
    )
    reference_centers = np.asarray(
        [np.mean(reference_vertices[ids], axis=0) for ids in station_ids],
        dtype=np.float64,
    )
    if pivot == "distal":
        source_pivot = source_distal
        reference_pivot = reference_distal
    elif pivot == "proximal":
        source_pivot = source_proximal
        reference_pivot = reference_proximal
    else:
        raise ValueError(f"unsupported V8.10 direction pivot {pivot!r}")
    rotation = _proper_direction_rotation(
        source_centers - source_pivot,
        reference_centers - reference_pivot,
    )
    mapped = (source_centers - source_pivot) @ rotation.T + source_pivot
    target = reference_centers + (source_pivot - reference_pivot)
    reference_axis = reference_distal - reference_proximal
    reference_length = float(np.linalg.norm(reference_axis))
    if reference_length <= 1.0e-8:
        raise ValueError("reference long-bone endpoints are degenerate")
    reference_direction = reference_axis / reference_length
    error = mapped - target
    axial_error = error @ reference_direction
    radial_error = error - axial_error[:, None] * reference_direction
    radial_norm = np.linalg.norm(radial_error, axis=1)
    angle = float(np.linalg.norm(_rotation_vector(rotation)))
    report = {
        "method": "unit_ray_direction_fit_with_axial_residual_v810",
        "pivot": pivot,
        "station_parameter": [float(value) for value in fractions],
        "source_station_centers_m": source_centers.tolist(),
        "reference_station_centers_m": reference_centers.tolist(),
        "mapped_station_centers_m": mapped.tolist(),
        "radial_errors_m": radial_norm.tolist(),
        "maximum_radial_error_m": float(np.max(radial_norm)),
        "axial_residuals_m": axial_error.tolist(),
        "maximum_abs_axial_residual_m": float(np.max(np.abs(axial_error))),
        "source_anatomical_length_m": source_length,
        "reference_anatomical_length_m": reference_length,
        "reference_length_residual_m": reference_length - source_length,
        "rotation_vector_rad": _rotation_vector(rotation).tolist(),
        "rotation_angle_deg": float(np.degrees(angle)),
        "det_rotation": float(np.linalg.det(rotation)),
        "scale": 1.0,
    }
    return rotation, report


def _mesh_edges(
    asset: AnatomyRiggedAsset,
    vertex_ids: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    lookup = np.full(len(asset.vertices_rest), -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids), dtype=np.int64)
    faces = np.asarray(asset.faces, dtype=np.int64)
    local_mask = np.all(lookup[faces] >= 0, axis=1)
    local_faces = lookup[faces[local_mask]]
    edges = np.concatenate(
        (
            local_faces[:, (0, 1)],
            local_faces[:, (1, 2)],
            local_faces[:, (2, 0)],
        ),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _edge_strain_report(
    *,
    before: np.ndarray,
    after: np.ndarray,
    edges: np.ndarray,
    activation: np.ndarray,
) -> dict[str, Any]:
    original = np.linalg.norm(
        before[edges[:, 0]] - before[edges[:, 1]], axis=1
    )
    final = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    valid = original > 1.0e-10
    relative = np.abs(final[valid] / original[valid] - 1.0)
    valid_edges = edges[valid]
    core = (activation[valid_edges[:, 0]] >= 1.0 - 1.0e-8) & (
        activation[valid_edges[:, 1]] >= 1.0 - 1.0e-8
    )
    transition = (
        (activation[valid_edges[:, 0]] > 1.0e-8)
        | (activation[valid_edges[:, 1]] > 1.0e-8)
    ) & ~core

    def metrics(mask: np.ndarray) -> dict[str, float | int]:
        values = relative[mask]
        if not len(values):
            return {"edge_count": 0, "q99": 0.0, "maximum": 0.0}
        return {
            "edge_count": int(len(values)),
            "q99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }

    return {
        "all": metrics(np.ones(len(relative), dtype=bool)),
        "rigid_shaft_core": metrics(core),
        "anatomical_adapter_zones": metrics(transition),
    }


def _rigid_rotation_about_pivot(
    rotation: np.ndarray,
    pivot: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    if (
        not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-10, rtol=0.0)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-10, rtol=0.0)
    ):
        raise ValueError("V8.10 segment rotation must be a proper rotation")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = matrix
    affine[:3, 3] = center - matrix @ center
    return affine


def _apply_rigid_segment_rotation_v810(
    asset: AnatomyRiggedAsset,
    *,
    side: str,
    rotvec: np.ndarray,
    vertex_ids: np.ndarray,
    pivot_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.spatial.transform import Rotation

    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    pivot_vertex_ids = np.asarray(pivot_ids, dtype=np.int64).reshape(-1)
    if (
        not len(ids)
        or not len(pivot_vertex_ids)
        or np.any(ids < 0)
        or np.any(ids >= len(vertices))
        or np.any(pivot_vertex_ids < 0)
        or np.any(pivot_vertex_ids >= len(vertices))
    ):
        raise ValueError("V8.10 rigid segment references invalid vertices")
    before = vertices[ids]
    pivot = np.mean(vertices[pivot_vertex_ids], axis=0)
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle > np.radians(20.0):
        raise ValueError(f"{side} segment direction correction exceeds 20 degrees")
    rotation = Rotation.from_rotvec(vector).as_matrix()
    affine = _rigid_rotation_about_pivot(rotation, pivot)
    after = before @ rotation.T + affine[:3, 3]
    edges = _mesh_edges(asset, ids)
    activation = np.ones(len(ids), dtype=np.float64)
    strain = _edge_strain_report(
        before=before,
        after=after,
        edges=edges,
        activation=activation,
    )
    all_edges = strain["all"]
    if float(all_edges["q99"]) > 1.0e-9 or float(all_edges["maximum"]) > 1.0e-8:
        raise ValueError("V8.10 rigid segment changed mesh edge lengths")
    delta = after - before
    return delta, {
        "side": side,
        "method": "whole_segment_unit_scale_so3_v810",
        "rotation_vector_rad": vector.tolist(),
        "rotation_angle_deg": float(np.degrees(angle)),
        "pivot_m": pivot.tolist(),
        "maximum_translation_m": float(
            np.max(np.linalg.norm(delta, axis=1))
        ),
        "pivot_translation_m": float(
            np.linalg.norm(pivot @ rotation.T + affine[:3, 3] - pivot)
        ),
        "vertex_count": int(len(ids)),
        "frame_determinant": float(np.linalg.det(rotation)),
        "frame_scale": 1.0,
        "cross_section_scale": 1.0,
        "affine": affine.tolist(),
        "edge_strain": strain,
    }


def _fit_beta_linear_vectors(
    betas: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    origin_beta = np.asarray(betas[0], dtype=np.float64).reshape(10)
    origin_value = np.asarray(values[0], dtype=np.float64).reshape(3)
    design = np.stack(
        [np.asarray(beta, dtype=np.float64).reshape(10) - origin_beta for beta in betas[1:]],
        axis=0,
    )
    observations = np.stack(
        [np.asarray(value, dtype=np.float64).reshape(3) - origin_value for value in values[1:]],
        axis=0,
    )
    basis = np.linalg.pinv(design) @ observations
    reconstructed = design @ basis
    error = reconstructed - observations
    return origin_value, basis, {
        "rank": int(np.linalg.matrix_rank(design)),
        "fit_rms": float(np.sqrt(np.mean(error * error))),
        "fit_max": float(np.max(np.abs(error))),
    }


def has_leg_centerline_v810(coefficients: Mapping[str, np.ndarray]) -> bool:
    key = f"{_PREFIX}schema_version"
    if key not in coefficients:
        return False
    schema = int(np.asarray(coefficients[key]).reshape(-1)[0])
    if schema != LEG_CENTERLINE_SCHEMA_VERSION_V810:
        raise ValueError(f"unsupported leg centerline schema {schema}")
    return True


def leg_centerline_delta_v810(
    source: AnatomyRiggedAsset,
    reference: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the reviewed femur rest warp and length-residual diagnostics."""

    source.validate()
    reference.validate()
    if (
        len(source.vertices_rest) != len(reference.vertices_rest)
        or not np.array_equal(source.faces, reference.faces)
        or source.source_mesh_names != reference.source_mesh_names
        or not np.array_equal(source.source_vertex_ranges, reference.source_vertex_ranges)
    ):
        raise ValueError("V8.10 centerline reference must have identical topology")
    ids_out: list[np.ndarray] = []
    delta_out: list[np.ndarray] = []
    report: dict[str, Any] = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "direction_only_leg_centerline_v810",
        "pelvis_correction": "identity",
        "changes_vessel_route": False,
        "segments": {},
    }
    for side, suffix in (("left", "L"), ("right", "R")):
        femur_ids = _mesh_vertex_ids(source, f"Femur_{suffix}")
        femur_head = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
            f"{side}/femoral_head.validation",
        )
        femur_distal = _domain_ids(
            domains,
            f"{side}/femoral_condyle_medial.fit",
            f"{side}/femoral_condyle_medial.validation",
            f"{side}/femoral_condyle_lateral.fit",
            f"{side}/femoral_condyle_lateral.validation",
            f"{side}/trochlea.fit",
            f"{side}/trochlea.validation",
        )
        femur_rotation, femur_fit = _direction_fit_report(
            source=source,
            reference=reference,
            sample_ids=femur_ids,
            proximal_ids=femur_head,
            distal_ids=femur_distal,
            fractions=_FEMUR_STATIONS,
            half_width=0.045,
            pivot="distal",
        )
        if float(femur_fit["maximum_radial_error_m"]) > 0.003:
            raise ValueError(
                f"{side} femur radial direction fit exceeds 3 mm"
            )
        femur_delta, femur_warp = _apply_rigid_segment_rotation_v810(
            source,
            side=side,
            rotvec=_rotation_vector(femur_rotation),
            vertex_ids=femur_ids,
            pivot_ids=femur_distal,
        )
        ids_out.append(femur_ids)
        delta_out.append(femur_delta)

        tibia_ids = _mesh_vertex_ids(source, f"Tibia_{suffix}")
        fibula_ids = _mesh_vertex_ids(source, f"Fibula_{suffix}")
        shank_ids = np.unique(np.concatenate((tibia_ids, fibula_ids)))
        shank_proximal = _domain_ids(
            domains,
            f"{side}/tibial_plateau_medial.fit",
            f"{side}/tibial_plateau_medial.validation",
            f"{side}/tibial_plateau_lateral.fit",
            f"{side}/tibial_plateau_lateral.validation",
        )
        shank_distal = _domain_ids(
            domains,
            f"ankle/{side}/tibia.fit",
            f"ankle/{side}/tibia.validation",
            f"ankle/{side}/fibula.fit",
            f"ankle/{side}/fibula.validation",
        )
        shank_rotation, shank_fit = _direction_fit_report(
            source=source,
            reference=reference,
            sample_ids=tibia_ids,
            proximal_ids=shank_proximal,
            distal_ids=shank_distal,
            fractions=_SHANK_STATIONS,
            half_width=0.055,
            pivot="proximal",
        )
        shank_delta, shank_warp = _apply_rigid_segment_rotation_v810(
            source,
            side=side,
            rotvec=_rotation_vector(shank_rotation),
            vertex_ids=shank_ids,
            pivot_ids=shank_proximal,
        )
        ids_out.append(shank_ids)
        delta_out.append(shank_delta)
        report["segments"][side] = {
            "femur": {**femur_fit, "warp": femur_warp},
            "shank": {**shank_fit, "warp": shank_warp},
        }
    vertex_ids = np.concatenate(ids_out).astype(np.int32)
    delta = np.concatenate(delta_out).astype(np.float32)
    order = np.argsort(vertex_ids)
    return vertex_ids[order], delta[order], report


def build_leg_centerline_coefficients_v810(
    *,
    samples: Sequence[
        tuple[np.ndarray, AnatomyRiggedAsset, AnatomyRiggedAsset]
    ],
    domains: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Bake beta-linear femur direction adapters from reviewed BA9 subjects."""

    if len(samples) < 2:
        raise ValueError("V8.10 leg calibration requires at least two beta samples")
    betas: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    rotvecs: dict[str, dict[str, list[np.ndarray]]] = {
        side: {"femur": [], "shank": []}
        for side in ("left", "right")
    }
    for beta_raw, source, reference in samples:
        beta = np.asarray(beta_raw, dtype=np.float64).reshape(-1)
        if beta.shape != (10,) or not np.all(np.isfinite(beta)):
            raise ValueError("each V8.10 calibration beta must contain ten values")
        _ids, _delta, report = leg_centerline_delta_v810(
            source,
            reference,
            domains=domains,
        )
        betas.append(beta)
        reports.append(report)
        for side in ("left", "right"):
            for segment in ("femur", "shank"):
                rotvecs[side][segment].append(
                    np.asarray(
                        report["segments"][side][segment][
                            "rotation_vector_rad"
                        ],
                        dtype=np.float64,
                    )
                )
    coefficients: dict[str, np.ndarray] = {
        f"{_PREFIX}schema_version": np.asarray(
            [LEG_CENTERLINE_SCHEMA_VERSION_V810], dtype=np.int32
        ),
        f"{_PREFIX}beta_origin": betas[0].astype(np.float32),
    }
    fit_reports: dict[str, Any] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        side_fit: dict[str, Any] = {}
        for segment in ("femur", "shank"):
            origin, basis, fit_report = _fit_beta_linear_vectors(
                betas,
                rotvecs[side][segment],
            )
            coefficients[
                f"{_PREFIX}{side}.{segment}_rotvec_origin_rad"
            ] = origin.astype(np.float32)
            coefficients[
                f"{_PREFIX}{side}.{segment}_rotvec_beta_basis_rad"
            ] = basis.astype(np.float32)
            side_fit[segment] = fit_report
        coefficients[f"{_PREFIX}{side}.femur_vertex_ids"] = _mesh_vertex_ids(
            samples[0][1], f"Femur_{suffix}"
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.femur_head_ids"] = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
            f"{side}/femoral_head.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.femur_distal_ids"] = _domain_ids(
            domains,
            f"{side}/femoral_condyle_medial.fit",
            f"{side}/femoral_condyle_medial.validation",
            f"{side}/femoral_condyle_lateral.fit",
            f"{side}/femoral_condyle_lateral.validation",
            f"{side}/trochlea.fit",
            f"{side}/trochlea.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.femur_head_ids"] = _domain_ids(
            domains,
            f"{side}/femoral_head.fit",
            f"{side}/femoral_head.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.acetabulum_ids"] = _domain_ids(
            domains,
            f"{side}/acetabulum.fit",
            f"{side}/acetabulum.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.shank_vertex_ids"] = np.unique(
            np.concatenate(
                (
                    _mesh_vertex_ids(samples[0][1], f"Tibia_{suffix}"),
                    _mesh_vertex_ids(samples[0][1], f"Fibula_{suffix}"),
                )
            )
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.shank_proximal_ids"] = _domain_ids(
            domains,
            f"{side}/tibial_plateau_medial.fit",
            f"{side}/tibial_plateau_medial.validation",
            f"{side}/tibial_plateau_lateral.fit",
            f"{side}/tibial_plateau_lateral.validation",
        ).astype(np.int32)
        coefficients[f"{_PREFIX}{side}.shank_distal_ids"] = _domain_ids(
            domains,
            f"ankle/{side}/tibia.fit",
            f"ankle/{side}/tibia.validation",
            f"ankle/{side}/fibula.fit",
            f"ankle/{side}/fibula.validation",
        ).astype(np.int32)
        fit_reports[side] = side_fit
    calibration_report = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "sample_count": len(samples),
        "vertex_count": int(
            sum(
                len(coefficients[f"{_PREFIX}{side}.femur_vertex_ids"])
                for side in ("left", "right")
            )
        ),
        "method": "beta_linear_whole_segment_direction_v810",
        "pelvis_correction": "identity",
        "fit": fit_reports,
        "samples": reports,
    }
    return coefficients, calibration_report


def transport_coupled_rbf_parent_frames_v810(
    metadata: Mapping[str, Any],
    *,
    old_global: np.ndarray,
    new_global: np.ndarray,
    parents: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Express existing RBF translations in corrected parent-local frames."""

    result = dict(metadata)
    responses = dict(result.get("source_coupled_joint_response_v8", {}))
    if not responses:
        return result, {
            "available": False,
            "reason": "asset has no coupled RBF responses",
        }
    old_frames = np.asarray(old_global, dtype=np.float64)
    new_frames = np.asarray(new_global, dtype=np.float64)
    parent_ids = np.asarray(parents, dtype=np.int64).reshape(-1)
    new_local = _global_to_local(new_frames, parent_ids)
    updated: dict[str, Any] = {}
    maximum_change = 0.0
    transported_count = 0
    for key, raw in responses.items():
        bone = int(key)
        if bone < 0 or bone >= len(parent_ids):
            raise ValueError("coupled RBF response has an invalid bone index")
        parent = int(parent_ids[bone])
        old_parent_rotation = (
            np.eye(3, dtype=np.float64)
            if parent < 0
            else old_frames[parent, :3, :3]
        )
        new_parent_rotation = (
            np.eye(3, dtype=np.float64)
            if parent < 0
            else new_frames[parent, :3, :3]
        )
        row_transport = old_parent_rotation.T @ new_parent_rotation
        response = dict(raw)
        for field in (
            "rbf_values_parent_local_m",
            "rbf_zero_parent_local_m",
            "rbf_weights_parent_local_m",
        ):
            if field not in response:
                continue
            values = np.asarray(response[field], dtype=np.float64)
            if values.shape[-1:] != (3,):
                raise ValueError(f"{field} must end with a 3-vector")
            transported = values @ row_transport
            maximum_change = max(
                maximum_change,
                float(np.max(np.abs(transported - values))),
            )
            response[field] = transported.tolist()
            transported_count += int(values.reshape(-1, 3).shape[0])
        response["anatomical_pivot_target_bind_m"] = new_frames[
            bone, :3, 3
        ].tolist()
        response["anatomical_pivot_parent_local_m"] = new_local[
            bone, :3, 3
        ].tolist()
        updated[str(bone)] = response
    result["source_coupled_joint_response_v8"] = updated
    return result, {
        "available": True,
        "response_count": int(len(updated)),
        "transported_vector_count": transported_count,
        "maximum_coefficient_change_m": maximum_change,
        "method": "old_parent_world_to_new_parent_local_v810",
    }


def reconstruct_leg_centerline_compounds_v810(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    target_surface_gap_m: float = 0.0015,
    maximum_platform_shift_m: float = 0.025,
    search_steps: int = 81,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Rebuild hip/shank rest compounds using only unit-scale corrections."""

    from scipy.spatial import cKDTree

    asset.validate()
    if (
        asset.target_bind_global is None
        or asset.target_bone_head is None
        or asset.target_bone_tail is None
        or asset.source_bone_names is None
        or asset.source_bone_parents is None
    ):
        raise ValueError("V8.10 leg reconstruction requires complete target FK")
    if search_steps < 3:
        raise ValueError("V8.10 platform search needs at least three steps")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    target_global = old_global.copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    bone_names = list(asset.source_bone_names)
    report: dict[str, Any] = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "unit_scale_hip_shank_compounds_v810",
        "pelvis_correction": "identity",
        "sides": {},
    }
    for side, suffix, hip_joint, knee_joint, ankle_joint in (
        ("left", "L", 1, 4, 7),
        ("right", "R", 2, 5, 8),
    ):
        femur_ids = _mesh_vertex_ids(asset, f"Femur_{suffix}")
        femur_head_ids = np.asarray(
            domains[f"{side}/femoral_head.fit"], dtype=np.int64
        )
        head_fit = fit_sphere(vertices[femur_head_ids])
        if not head_fit.get("available", False):
            raise ValueError(f"{side} femoral head fit is unavailable")
        femur_bone = bone_names.index(f"Femur_Rot_{suffix}")
        bind_socket = target_head[femur_bone]
        head_translation = bind_socket - np.asarray(
            head_fit["center"], dtype=np.float64
        )
        if float(np.linalg.norm(head_translation)) > 0.002:
            raise ValueError(f"{side} rigid femur socket correction exceeds 2 mm")
        vertices[femur_ids] += head_translation

        tibia_ids = _mesh_vertex_ids(asset, f"Tibia_{suffix}")
        fibula_ids = _mesh_vertex_ids(asset, f"Fibula_{suffix}")
        medial_condyle = np.asarray(
            domains[f"{side}/femoral_condyle_medial.fit"], dtype=np.int64
        )
        lateral_condyle = np.asarray(
            domains[f"{side}/femoral_condyle_lateral.fit"], dtype=np.int64
        )
        medial_platform = np.asarray(
            domains[f"{side}/tibial_plateau_medial.fit"], dtype=np.int64
        )
        lateral_platform = np.asarray(
            domains[f"{side}/tibial_plateau_lateral.fit"], dtype=np.int64
        )
        distal_tibia = np.asarray(
            domains[f"ankle/{side}/tibia.fit"], dtype=np.int64
        )
        source_platform = np.mean(
            vertices[np.concatenate((medial_platform, lateral_platform))],
            axis=0,
        )
        condyle_center = np.mean(
            vertices[np.concatenate((medial_condyle, lateral_condyle))],
            axis=0,
        )
        source_distal = np.mean(vertices[distal_tibia], axis=0)
        direction = condyle_center - source_platform
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-8:
            raise ValueError(f"{side} knee contact direction is degenerate")
        direction /= direction_norm
        medial_tree = cKDTree(vertices[medial_condyle])
        lateral_tree = cKDTree(vertices[lateral_condyle])
        best: tuple[float, float, float, float] | None = None
        for shift in np.linspace(
            0.0,
            float(maximum_platform_shift_m),
            int(search_steps),
        ):
            translation = float(shift) * direction
            medial_gap = float(
                medial_tree.query(vertices[medial_platform] + translation)[0].min()
            )
            lateral_gap = float(
                lateral_tree.query(vertices[lateral_platform] + translation)[0].min()
            )
            objective = (
                (medial_gap - float(target_surface_gap_m)) ** 2
                + (lateral_gap - float(target_surface_gap_m)) ** 2
            )
            candidate = (objective, float(shift), medial_gap, lateral_gap)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise AssertionError("V8.10 knee platform search produced no candidate")
        _objective, shift, medial_gap, lateral_gap = best
        shank_translation = shift * direction
        foot_ids = _foot_bone_vertex_ids(asset, suffix=suffix)
        moved_ids = np.unique(
            np.concatenate((tibia_ids, fibula_ids, foot_ids))
        )
        vertices[moved_ids] += shank_translation
        shank_root = f"Tibia_Bone_{suffix}"
        subtree = _descendant_mask(bone_names, parents, shank_root)
        target_global[subtree, :3, 3] += shank_translation
        target_head[subtree] += shank_translation
        target_tail[subtree] += shank_translation
        anatomical_length = float(np.linalg.norm(source_distal - source_platform))
        smplx_length = float(
            np.linalg.norm(
                np.asarray(asset.rest_joints[ankle_joint], dtype=np.float64)
                - np.asarray(asset.rest_joints[knee_joint], dtype=np.float64)
            )
        )
        report["sides"][side] = {
            "femur": {
                "bone_index": femur_bone,
                "socket_translation_m": head_translation.tolist(),
                "socket_translation_norm_m": float(
                    np.linalg.norm(head_translation)
                ),
                "scale": 1.0,
            },
            "shank": {
                "platform_shift_m": shift,
                "translation_m": shank_translation.tolist(),
                "fit_medial_gap_m": medial_gap,
                "fit_lateral_gap_m": lateral_gap,
                "anatomical_length_m": anatomical_length,
                "smplx_knee_ankle_length_m": smplx_length,
                "axial_length_residual_m": smplx_length - anatomical_length,
                "moved_bone_vertex_count": int(len(moved_ids)),
                "moved_bind_bone_count": int(np.count_nonzero(subtree)),
                "scale": 1.0,
            },
            "hip_joint": hip_joint,
            "knee_joint": knee_joint,
            "ankle_joint": ankle_joint,
        }
    metadata, rbf_report = transport_coupled_rbf_parent_frames_v810(
        dict(asset.metadata or {}),
        old_global=old_global,
        new_global=target_global,
        parents=parents,
    )
    target_local = _global_to_local(target_global, parents)
    metadata["leg_compounds_v810"] = report
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result.validate()
    report["rbf_frame_transport"] = rbf_report
    return result, report


def apply_leg_centerline_v810(
    asset: AnatomyRiggedAsset,
    *,
    betas: Any,
    coefficients: Mapping[str, np.ndarray],
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Apply beta-specific rigid leg directions and rebuild one local FK chain."""

    if not has_leg_centerline_v810(coefficients):
        return asset, {
            "available": False,
            "reason": "operator has no V8.10 leg centerline coefficients",
        }
    required = ["beta_origin"]
    for side in ("left", "right"):
        required.extend(
            (
                f"{side}.femur_rotvec_origin_rad",
                f"{side}.femur_rotvec_beta_basis_rad",
                f"{side}.shank_rotvec_origin_rad",
                f"{side}.shank_rotvec_beta_basis_rad",
                f"{side}.femur_vertex_ids",
                f"{side}.femur_head_ids",
                f"{side}.femur_distal_ids",
                f"{side}.acetabulum_ids",
                f"{side}.shank_vertex_ids",
                f"{side}.shank_proximal_ids",
                f"{side}.shank_distal_ids",
            )
        )
    missing = [
        name for name in required if f"{_PREFIX}{name}" not in coefficients
    ]
    if missing:
        return asset, {
            "available": False,
            "reason": f"operator has incomplete V8.10 coefficients: {missing}",
        }
    beta = np.asarray(betas, dtype=np.float64).reshape(10)
    beta_origin = np.asarray(
        coefficients[f"{_PREFIX}beta_origin"], dtype=np.float64
    ).reshape(10)
    beta_delta = beta - beta_origin
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    if (
        asset.target_bind_global is None
        or asset.target_bone_head is None
        or asset.target_bone_tail is None
        or asset.source_bone_names is None
        or asset.source_bone_parents is None
    ):
        raise ValueError("V8.10 centerline apply requires complete target FK")
    target_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    old_global = target_global.copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    bone_names = list(asset.source_bone_names)
    bone_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    guide_joints = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else asset.rest_joints,
        dtype=np.float64,
    ).copy()
    smplx_parents = np.asarray(asset.parents, dtype=np.int64)
    side_reports: dict[str, Any] = {}
    for side, suffix, hip_joint, knee_joint, ankle_joint, foot_joint in (
        ("left", "L", 1, 4, 7, 10),
        ("right", "R", 2, 5, 8, 11),
    ):
        segment_rotvec: dict[str, np.ndarray] = {}
        for segment in ("femur", "shank"):
            origin = np.asarray(
                coefficients[
                    f"{_PREFIX}{side}.{segment}_rotvec_origin_rad"
                ],
                dtype=np.float64,
            ).reshape(3)
            basis = np.asarray(
                coefficients[
                    f"{_PREFIX}{side}.{segment}_rotvec_beta_basis_rad"
                ],
                dtype=np.float64,
            ).reshape(10, 3)
            segment_rotvec[segment] = origin + beta_delta @ basis
        femur_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.femur_vertex_ids"],
            dtype=np.int64,
        ).reshape(-1)
        head_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.femur_head_ids"],
            dtype=np.int64,
        ).reshape(-1)
        distal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.femur_distal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        acetabulum_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.acetabulum_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_vertex_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_proximal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_proximal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        shank_distal_ids = np.asarray(
            coefficients[f"{_PREFIX}{side}.shank_distal_ids"],
            dtype=np.int64,
        ).reshape(-1)
        coefficient_ids = (
            femur_ids,
            head_ids,
            distal_ids,
            acetabulum_ids,
            shank_ids,
            shank_proximal_ids,
            shank_distal_ids,
        )
        if any(
            not len(ids)
            or np.any(ids < 0)
            or np.any(ids >= len(vertices))
            for ids in coefficient_ids
        ):
            raise ValueError("V8.10 leg coefficients reference an invalid vertex")

        working = replace(asset, vertices_rest=vertices.astype(np.float32))
        femur_delta, femur_report = _apply_rigid_segment_rotation_v810(
            working,
            side=side,
            rotvec=segment_rotvec["femur"],
            vertex_ids=femur_ids,
            pivot_ids=distal_ids,
        )
        vertices[femur_ids] += femur_delta
        femur_affine = np.asarray(femur_report["affine"], dtype=np.float64)
        femur_bone = bone_names.index(f"Femur_Rot_{suffix}")
        target_global[femur_bone] = femur_affine @ target_global[femur_bone]
        target_head[femur_bone] = (
            target_head[femur_bone] @ femur_affine[:3, :3].T
            + femur_affine[:3, 3]
        )
        target_tail[femur_bone] = (
            target_tail[femur_bone] @ femur_affine[:3, :3].T
            + femur_affine[:3, 3]
        )

        foot_ids = _foot_bone_vertex_ids(asset, suffix=suffix)
        patella_ids = _mesh_vertex_ids(asset, f"Patella_{suffix}")
        moved_shank_ids = np.unique(
            np.concatenate((shank_ids, foot_ids, patella_ids))
        )
        working = replace(asset, vertices_rest=vertices.astype(np.float32))
        shank_delta, shank_report = _apply_rigid_segment_rotation_v810(
            working,
            side=side,
            rotvec=segment_rotvec["shank"],
            vertex_ids=moved_shank_ids,
            pivot_ids=shank_proximal_ids,
        )
        vertices[moved_shank_ids] += shank_delta
        shank_affine = np.asarray(shank_report["affine"], dtype=np.float64)
        shank_subtree = _descendant_mask(
            bone_names,
            bone_parents,
            f"Tibia_Bone_{suffix}",
        )
        target_global[shank_subtree] = (
            shank_affine[None] @ target_global[shank_subtree]
        )
        target_head[shank_subtree] = (
            target_head[shank_subtree] @ shank_affine[:3, :3].T
            + shank_affine[:3, 3]
        )
        target_tail[shank_subtree] = (
            target_tail[shank_subtree] @ shank_affine[:3, :3].T
            + shank_affine[:3, 3]
        )

        head_fit = fit_sphere(vertices[head_ids])
        socket_fit = fit_sphere(vertices[acetabulum_ids])
        if not head_fit.get("available", False) or not socket_fit.get(
            "available", False
        ):
            raise ValueError(f"{side} V8.10 hip sphere fit is unavailable")
        moved_head = np.asarray(head_fit["center"], dtype=np.float64)
        socket = np.asarray(socket_fit["center"], dtype=np.float64)
        knee_station = np.mean(vertices[shank_proximal_ids], axis=0)
        ankle_station = np.mean(vertices[shank_distal_ids], axis=0)
        guide_joints[hip_joint] = moved_head
        guide_joints[knee_joint] = knee_station

        joint_descendants = np.zeros(len(guide_joints), dtype=bool)
        for joint in range(len(guide_joints)):
            current = joint
            for _ in range(len(guide_joints) + 1):
                if current == ankle_joint:
                    joint_descendants[joint] = True
                    break
                if current < 0:
                    break
                current = int(smplx_parents[current])
            else:
                raise ValueError("SMPL-X guide hierarchy contains a cycle")
        transformed_guide = (
            guide_joints[joint_descendants] @ shank_affine[:3, :3].T
            + shank_affine[:3, 3]
        )
        guide_joints[joint_descendants] = transformed_guide
        guide_offset = ankle_station - guide_joints[ankle_joint]
        guide_joints[joint_descendants] += guide_offset
        guide_joints[ankle_joint] = ankle_station
        if foot_joint >= len(guide_joints):
            raise ValueError("V8.10 foot guide joint is unavailable")

        side_reports[side] = {
            "femur": {
                **femur_report,
                "bone_index": int(femur_bone),
                "head_center_m": moved_head.tolist(),
                "acetabulum_center_m": socket.tolist(),
                "head_socket_residual_m": float(
                    np.linalg.norm(moved_head - socket)
                ),
                "hip_station_unreachable_with_fixed_socket": True,
            },
            "shank": {
                **shank_report,
                "subtree_bone_count": int(np.count_nonzero(shank_subtree)),
                "knee_station_m": knee_station.tolist(),
                "ankle_station_m": ankle_station.tolist(),
            },
            "guide_joints": {
                "hip": int(hip_joint),
                "knee": int(knee_joint),
                "ankle": int(ankle_joint),
                "foot": int(foot_joint),
            },
        }

    metadata = dict(asset.metadata or {})
    metadata, rbf_report = transport_coupled_rbf_parent_frames_v810(
        metadata,
        old_global=old_global,
        new_global=target_global,
        parents=bone_parents,
    )
    target_local = _global_to_local(target_global, bone_parents)
    report = {
        "available": True,
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "beta_linear_whole_segment_so3_guide_fk_v810",
        "pelvis_correction": "identity",
        "changes_bind_frames": True,
        "changes_vessel_route": False,
        "guide_fk": "source_driver_rest_joints_v810",
        "rbf_frame_transport": rbf_report,
        "sides": side_reports,
        "maximum_translation_m": float(
            max(
                max(
                    side_reports[side]["femur"]["maximum_translation_m"],
                    side_reports[side]["shank"]["maximum_translation_m"],
                )
                for side in ("left", "right")
            )
        ),
    }
    metadata["source_anatomical_guide_fk_v810"] = True
    metadata["leg_centerline_v810"] = report
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        source_driver_rest_joints=guide_joints.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result.validate()
    return result, report


__all__ = [
    "LEG_CENTERLINE_SCHEMA_VERSION_V810",
    "apply_leg_centerline_v810",
    "build_leg_centerline_coefficients_v810",
    "has_leg_centerline_v810",
    "leg_centerline_delta_v810",
    "reconstruct_leg_centerline_compounds_v810",
    "transport_coupled_rbf_parent_frames_v810",
]
