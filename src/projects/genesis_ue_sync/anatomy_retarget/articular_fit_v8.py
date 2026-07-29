"""Pure whole-bone articular rest fitting for schema V8.

The operations in this module are deliberately smaller than subject
materialization.  They move an entire femur or tibia/fibula compound with one
constant affine transform and update its target bind authority consistently.
There is no endpoint profile, containment shrink, pose hinge, or patella
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .acceptance_v8 import fit_sphere, fit_sphere_center_fixed_radius
from .mechanism_v8 import WholeBoneRestFitV8, fit_whole_bone_rest_v8
from .rigged_asset import AnatomyRiggedAsset


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _points(value: Any, *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must be [N, 3]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must be finite")
    return points


def _point(value: Any, *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{label} must be finite")
    return point


def whole_bone_affine_matrix_v8(fit: WholeBoneRestFitV8) -> np.ndarray:
    """Return the one homogeneous affine represented by ``fit``."""

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.asarray(fit.linear, dtype=np.float64)
    affine[:3, 3] = (
        np.asarray(fit.target_head, dtype=np.float64)
        - affine[:3, :3] @ np.asarray(fit.source_head, dtype=np.float64)
    )
    return affine


def apply_whole_bone_affine_v8(
    points: Any,
    fit: WholeBoneRestFitV8,
) -> np.ndarray:
    """Apply exactly one affine to every point without spatial blending."""

    vertices = _points(points, label="whole-bone points")
    affine = whole_bone_affine_matrix_v8(fit)
    homogeneous = np.concatenate(
        (vertices, np.ones((len(vertices), 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ affine.T)[:, :3]


@dataclass(frozen=True)
class ArticularWholeBoneFitV8:
    """One fitted mesh and the explicit endpoint/affine authority behind it."""

    role: str
    source_proximal: np.ndarray
    source_distal: np.ndarray
    target_proximal: np.ndarray
    target_distal: np.ndarray
    fit: WholeBoneRestFitV8
    vertices: np.ndarray
    affine: np.ndarray

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        if not role:
            raise ValueError("whole-bone fit role is required")
        for name in (
            "source_proximal",
            "source_distal",
            "target_proximal",
            "target_distal",
        ):
            object.__setattr__(
                self, name, _readonly(_point(getattr(self, name), label=name))
            )
        vertices = _points(self.vertices, label=f"{role} vertices")
        affine = np.asarray(self.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError("whole-bone affine must be finite [4, 4]")
        if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
            raise ValueError("whole-bone affine must have an affine bottom row")
        expected = whole_bone_affine_matrix_v8(self.fit)
        if not np.allclose(affine, expected, atol=1.0e-12, rtol=0.0):
            raise ValueError("stored whole-bone affine disagrees with fit")
        mapped = apply_whole_bone_affine_v8(
            np.stack((self.source_proximal, self.source_distal)), self.fit
        )
        targets = np.stack((self.target_proximal, self.target_distal))
        if not np.allclose(mapped, targets, atol=1.0e-9, rtol=0.0):
            raise ValueError("whole-bone fit does not map its articular endpoints")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "vertices", _readonly(vertices))
        object.__setattr__(self, "affine", _readonly(affine))

    @property
    def target_length(self) -> float:
        return float(np.linalg.norm(self.target_distal - self.target_proximal))


@dataclass(frozen=True)
class TibiaFibulaWholeBoneFitV8:
    """Tibia and fibula transformed by the exact same whole-shank affine."""

    fit: WholeBoneRestFitV8
    source_platform: np.ndarray
    source_distal: np.ndarray
    target_platform: np.ndarray
    target_distal: np.ndarray
    tibia_vertices: np.ndarray
    fibula_vertices: np.ndarray
    affine: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "source_platform",
            "source_distal",
            "target_platform",
            "target_distal",
        ):
            object.__setattr__(
                self, name, _readonly(_point(getattr(self, name), label=name))
            )
        tibia = _points(self.tibia_vertices, label="tibia_vertices")
        fibula = _points(self.fibula_vertices, label="fibula_vertices")
        affine = np.asarray(self.affine, dtype=np.float64)
        expected = whole_bone_affine_matrix_v8(self.fit)
        if affine.shape != (4, 4) or not np.allclose(
            affine, expected, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("tibia/fibula must store the shared whole-bone affine")
        mapped = apply_whole_bone_affine_v8(
            np.stack((self.source_platform, self.source_distal)), self.fit
        )
        if not np.allclose(
            mapped,
            np.stack((self.target_platform, self.target_distal)),
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError("shank fit does not map platform/distal endpoints")
        object.__setattr__(self, "tibia_vertices", _readonly(tibia))
        object.__setattr__(self, "fibula_vertices", _readonly(fibula))
        object.__setattr__(self, "affine", _readonly(affine))


def fit_femur_to_acetabulum_v8(
    *,
    femur_vertices: Any,
    current_femoral_head_center: Any,
    current_condyle_endpoint: Any,
    target_acetabulum_center: Any,
    radial_scales: tuple[float, float] = (1.0, 1.0),
) -> ArticularWholeBoneFitV8:
    """Move the complete femur head into its socket while preserving its knee.

    The current condyle endpoint is intentionally both the source and target
    distal endpoint.  Any change in head-to-condyle length is distributed by
    the one whole-bone axial scale, never by deforming the head, shaft, or
    condyles separately.
    """

    vertices = _points(femur_vertices, label="femur_vertices")
    source_head = _point(
        current_femoral_head_center, label="current_femoral_head_center"
    )
    condyle = _point(current_condyle_endpoint, label="current_condyle_endpoint")
    socket = _point(target_acetabulum_center, label="target_acetabulum_center")
    fit = fit_whole_bone_rest_v8(
        source_head=source_head,
        source_tail=condyle,
        target_head=socket,
        target_tail=condyle,
        radial_scales=radial_scales,
    )
    return ArticularWholeBoneFitV8(
        role="femur",
        source_proximal=source_head,
        source_distal=condyle,
        target_proximal=socket,
        target_distal=condyle,
        fit=fit,
        vertices=apply_whole_bone_affine_v8(vertices, fit),
        affine=whole_bone_affine_matrix_v8(fit),
    )


def fit_tibia_fibula_to_platform_v8(
    *,
    tibia_vertices: Any,
    fibula_vertices: Any,
    current_platform_center: Any,
    current_distal_endpoint: Any,
    target_platform_center: Any,
    target_distal_endpoint: Any | None = None,
    radial_scales: tuple[float, float] = (1.0, 1.0),
) -> TibiaFibulaWholeBoneFitV8:
    """Fit the complete tibia/fibula compound with one shared affine.

    By default the ankle-side endpoint stays fixed.  A caller may provide a
    different distal target, but tibia and fibula always receive the identical
    transform.
    """

    tibia = _points(tibia_vertices, label="tibia_vertices")
    fibula = _points(fibula_vertices, label="fibula_vertices")
    source_platform = _point(
        current_platform_center, label="current_platform_center"
    )
    source_distal = _point(
        current_distal_endpoint, label="current_distal_endpoint"
    )
    target_platform = _point(target_platform_center, label="target_platform_center")
    target_distal = (
        source_distal.copy()
        if target_distal_endpoint is None
        else _point(target_distal_endpoint, label="target_distal_endpoint")
    )
    fit = fit_whole_bone_rest_v8(
        source_head=source_platform,
        source_tail=source_distal,
        target_head=target_platform,
        target_tail=target_distal,
        radial_scales=radial_scales,
    )
    affine = whole_bone_affine_matrix_v8(fit)
    return TibiaFibulaWholeBoneFitV8(
        fit=fit,
        source_platform=source_platform,
        source_distal=source_distal,
        target_platform=target_platform,
        target_distal=target_distal,
        tibia_vertices=apply_whole_bone_affine_v8(tibia, fit),
        fibula_vertices=apply_whole_bone_affine_v8(fibula, fit),
        affine=affine,
    )


def apply_fit_to_meshes_v8(
    meshes: Mapping[str, Any],
    fit: WholeBoneRestFitV8,
) -> dict[str, np.ndarray]:
    """Apply one fit to an arbitrary set of named compound meshes."""

    if not meshes:
        raise ValueError("at least one compound mesh is required")
    return {
        str(name): apply_whole_bone_affine_v8(vertices, fit)
        for name, vertices in meshes.items()
    }


def _closest_proper_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _singular, right = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    )
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _validate_parent_order(parents: np.ndarray, count: int) -> None:
    if parents.shape != (count,):
        raise ValueError("parents must have one entry per target bone")
    for bone, parent in enumerate(parents.tolist()):
        if parent < -1 or parent >= bone:
            raise ValueError("target bind parents must be parent-before-child")


def _global_to_local(global_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.empty_like(global_bind)
    for bone, parent in enumerate(parents.tolist()):
        local[bone] = (
            global_bind[bone]
            if parent < 0
            else np.linalg.inv(global_bind[parent]) @ global_bind[bone]
        )
    return local


@dataclass(frozen=True)
class TargetBindUpdateV8:
    """Detached target bind arrays after one offline whole-bone correction."""

    target_bone_head: np.ndarray
    target_bone_tail: np.ndarray
    target_rest_global: np.ndarray
    target_rest_local: np.ndarray
    target_inverse_bind: np.ndarray
    transformed_bone_indices: np.ndarray
    affine: np.ndarray

    def __post_init__(self) -> None:
        head = np.asarray(self.target_bone_head, dtype=np.float64)
        tail = np.asarray(self.target_bone_tail, dtype=np.float64)
        global_bind = np.asarray(self.target_rest_global, dtype=np.float64)
        local_bind = np.asarray(self.target_rest_local, dtype=np.float64)
        inverse = np.asarray(self.target_inverse_bind, dtype=np.float64)
        count = len(head)
        if head.shape != (count, 3) or tail.shape != (count, 3):
            raise ValueError("target bone endpoints must be [B, 3]")
        expected = (count, 4, 4)
        if (
            global_bind.shape != expected
            or local_bind.shape != expected
            or inverse.shape != expected
        ):
            raise ValueError("target bind matrices must be [B, 4, 4]")
        if not all(
            np.all(np.isfinite(value))
            for value in (head, tail, global_bind, local_bind, inverse)
        ):
            raise ValueError("target bind update contains non-finite values")
        indices = np.asarray(self.transformed_bone_indices, dtype=np.int64).reshape(-1)
        if (
            indices.size == 0
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= count)
        ):
            raise ValueError("transformed_bone_indices is invalid")
        affine = np.asarray(self.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError("target bind affine must be finite [4,4]")
        identity = np.broadcast_to(np.eye(4), expected)
        if not np.allclose(inverse @ global_bind, identity, atol=1.0e-8, rtol=0.0):
            raise ValueError("target_inverse_bind disagrees with target_rest_global")
        object.__setattr__(self, "target_bone_head", _readonly(head))
        object.__setattr__(self, "target_bone_tail", _readonly(tail))
        object.__setattr__(self, "target_rest_global", _readonly(global_bind))
        object.__setattr__(self, "target_rest_local", _readonly(local_bind))
        object.__setattr__(self, "target_inverse_bind", _readonly(inverse))
        object.__setattr__(
            self, "transformed_bone_indices", _readonly(indices, np.int32)
        )
        object.__setattr__(self, "affine", _readonly(affine))


def update_target_bind_with_whole_bone_fit_v8(
    *,
    target_bone_head: Any,
    target_bone_tail: Any,
    target_rest_global: Any,
    parents: Any,
    transformed_bone_indices: Any,
    fit: WholeBoneRestFitV8,
) -> TargetBindUpdateV8:
    """Purely update endpoints and bind frames with the same whole-bone fit.

    Only the explicitly selected bone frames are transported.  Unselected
    global frames remain fixed and all parent-local matrices are then
    recomputed, so runtime FK remains the sole authority and no global child
    anchor is stored.
    """

    head = np.asarray(target_bone_head, dtype=np.float64)
    tail = np.asarray(target_bone_tail, dtype=np.float64)
    global_bind = np.asarray(target_rest_global, dtype=np.float64)
    if head.ndim != 2 or head.shape[1] != 3:
        raise ValueError("target_bone_head must be [B,3]")
    count = len(head)
    if tail.shape != (count, 3) or global_bind.shape != (count, 4, 4):
        raise ValueError("target bone endpoints/global bind have inconsistent shapes")
    if not all(np.all(np.isfinite(value)) for value in (head, tail, global_bind)):
        raise ValueError("target bind inputs must be finite")
    parent_array = np.asarray(parents, dtype=np.int64).reshape(-1)
    _validate_parent_order(parent_array, count)
    indices = np.asarray(transformed_bone_indices, dtype=np.int64).reshape(-1)
    if (
        indices.size == 0
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= count)
    ):
        raise ValueError("transformed_bone_indices is invalid")

    affine = whole_bone_affine_matrix_v8(fit)
    new_head = head.copy()
    new_tail = tail.copy()
    new_global = global_bind.copy()
    new_head[indices] = apply_whole_bone_affine_v8(head[indices], fit)
    new_tail[indices] = apply_whole_bone_affine_v8(tail[indices], fit)
    for bone in indices.tolist():
        old_rotation = global_bind[bone, :3, :3]
        new_global[bone, :3, :3] = _closest_proper_rotation(
            affine[:3, :3] @ old_rotation
        )
        new_global[bone, :3, 3] = apply_whole_bone_affine_v8(
            global_bind[bone : bone + 1, :3, 3], fit
        )[0]
        new_global[bone, 3] = (0.0, 0.0, 0.0, 1.0)
    new_local = _global_to_local(new_global, parent_array)
    inverse = np.linalg.inv(new_global)
    return TargetBindUpdateV8(
        target_bone_head=new_head,
        target_bone_tail=new_tail,
        target_rest_global=new_global,
        target_rest_local=new_local,
        target_inverse_bind=inverse,
        transformed_bone_indices=indices,
        affine=affine,
    )


def _combined_domain_ids(
    domains: Mapping[str, np.ndarray],
    *base_names: str,
) -> np.ndarray:
    names = [
        f"{base}.{partition}"
        for base in base_names
        for partition in ("fit", "validation")
    ]
    missing = [name for name in names if name not in domains]
    if missing:
        raise ValueError(f"V8 articular domains are missing: {missing}")
    return np.unique(
        np.concatenate(
            [np.asarray(domains[name], dtype=np.int64).reshape(-1) for name in names]
        )
    )


def _mesh_vertex_ids(asset: AnatomyRiggedAsset, mesh_name: str) -> np.ndarray:
    try:
        mesh_index = list(asset.source_mesh_names).index(mesh_name)
    except ValueError as exc:
        raise ValueError(f"required V8 articular mesh {mesh_name!r} is missing") from exc
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[mesh_index]
    return np.arange(int(start), int(stop), dtype=np.int64)


def reconstruct_hip_compounds_v8(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    maximum_bind_socket_error_m: float = 0.002,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Align each complete femur mesh to its already beta-fitted FK endpoints.

    The fitted target bind is the motion authority and already places the
    femur pivot at the beta-specific socket.  This pass repairs the historical
    mismatch where the *mesh* head remained several centimetres away while the
    bind pivot was correct.  Socket material is fitted from the frozen ``fit``
    subset only; the disjoint validation subset is intentionally untouched and
    reserved for acceptance.
    """

    asset.validate()
    if (
        asset.target_bone_head is None
        or asset.source_bone_names is None
        or asset.source_vertex_ranges is None
    ):
        raise ValueError("V8 hip reconstruction requires target FK and mesh topology")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64)
    report: dict[str, Any] = {
        "schema_version": 8,
        "method": "whole_femur_affine_to_beta_fk_socket_v8",
        "uses_validation_for_fit": False,
        "sides": {},
    }
    for side, suffix in (("left", "L"), ("right", "R")):
        femur_ids = _mesh_vertex_ids(asset, f"Femur_{suffix}")
        domain_femur = _combined_domain_ids(domains, f"{side}/femur")
        if not np.array_equal(femur_ids, domain_femur):
            raise ValueError(
                f"{side} frozen femur domain does not cover the complete mesh"
            )
        head_fit = fit_sphere(
            vertices[
                np.asarray(
                    domains[f"{side}/femoral_head.fit"], dtype=np.int64
                )
            ]
        )
        if not head_fit.get("available", False):
            raise ValueError(f"{side} femoral-head fit is unavailable")
        bone_index = list(asset.source_bone_names).index(f"Femur_Rot_{suffix}")
        bind_socket = target_head[bone_index]
        socket_fit = fit_sphere_center_fixed_radius(
            vertices[
                np.asarray(domains[f"{side}/acetabulum.fit"], dtype=np.int64)
            ],
            radius_m=float(head_fit["radius_m"]),
            initial_center=bind_socket,
            multistart=False,
        )
        if not socket_fit.get("available", False):
            raise ValueError(f"{side} acetabulum fit is unavailable")
        bind_socket_error = float(
            np.linalg.norm(
                bind_socket - np.asarray(socket_fit["center"], dtype=np.float64)
            )
        )
        if bind_socket_error > float(maximum_bind_socket_error_m):
            raise ValueError(
                f"{side} beta FK socket disagrees with frozen socket fit by "
                f"{bind_socket_error * 1000.0:.3f} mm"
            )
        condyle_ids = _combined_domain_ids(
            domains,
            f"{side}/femoral_condyle_medial",
            f"{side}/femoral_condyle_lateral",
        )
        condyle_center = np.mean(vertices[condyle_ids], axis=0)
        fitted = fit_femur_to_acetabulum_v8(
            femur_vertices=vertices[femur_ids],
            current_femoral_head_center=head_fit["center"],
            current_condyle_endpoint=condyle_center,
            target_acetabulum_center=bind_socket,
        )
        if not 0.85 <= float(fitted.fit.axial_scale) <= 1.15:
            raise ValueError(
                f"{side} whole-femur axial scale "
                f"{fitted.fit.axial_scale:.6f} is outside [0.85, 1.15]"
            )
        vertices[femur_ids] = fitted.vertices
        report["sides"][side] = {
            "bone_index": int(bone_index),
            "whole_mesh_vertex_count": int(len(femur_ids)),
            "source_head_center_m": np.asarray(
                head_fit["center"], dtype=np.float64
            ).tolist(),
            "target_bind_socket_m": bind_socket.tolist(),
            "surface_socket_center_m": np.asarray(
                socket_fit["center"], dtype=np.float64
            ).tolist(),
            "bind_socket_error_m": bind_socket_error,
            "source_condyle_center_m": condyle_center.tolist(),
            "axial_scale": float(fitted.fit.axial_scale),
            "radial_scales": [1.0, 1.0],
            "affine": fitted.affine.tolist(),
        }
    metadata = dict(asset.metadata or {})
    metadata["v8_whole_bone_hip_fit"] = report
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result.validate()
    return result, report


def reconstruct_knee_ankle_compounds_v8(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    target_surface_gap_m: float = 0.0015,
    maximum_platform_shift_m: float = 0.025,
    search_steps: int = 101,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Fit each complete shank between the knee surfaces and fixed ankle.

    The proximal platform is advanced toward the frozen femoral-condyle
    surfaces while the distal tibia/talus interface remains fixed.  Tibia and
    fibula receive one shared whole-bone affine; the same affine transports the
    shank target-bind frames.  The ankle subtree keeps its global bind and is
    reconnected by recomputing the single parent-local chain.
    """

    from scipy.spatial import cKDTree

    asset.validate()
    if (
        asset.target_bone_head is None
        or asset.target_bone_tail is None
        or asset.target_bind_global is None
        or asset.source_bone_names is None
        or asset.source_bone_parents is None
    ):
        raise ValueError("V8 knee/ankle reconstruction requires target FK authority")
    if search_steps < 3:
        raise ValueError("search_steps must be at least three")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    target_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    bone_names = list(asset.source_bone_names)
    report: dict[str, Any] = {
        "schema_version": 8,
        "method": "whole_shank_surface_gap_to_fixed_ankle_v8",
        "target_surface_gap_m": float(target_surface_gap_m),
        "uses_validation_for_fit": False,
        "sides": {},
    }

    for side, suffix, effector in (
        ("left", "L", "eff16"),
        ("right", "R", "eff74"),
    ):
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
        required = (
            medial_condyle,
            lateral_condyle,
            medial_platform,
            lateral_platform,
            distal_tibia,
        )
        if any(len(ids) < 4 for ids in required):
            raise ValueError(f"{side} knee/ankle frozen fit domains are incomplete")

        source_platform = np.mean(
            vertices[np.concatenate((medial_platform, lateral_platform))], axis=0
        )
        condyle_center = np.mean(
            vertices[np.concatenate((medial_condyle, lateral_condyle))], axis=0
        )
        source_distal = np.mean(vertices[distal_tibia], axis=0)
        proximal_direction = condyle_center - source_platform
        direction_norm = float(np.linalg.norm(proximal_direction))
        if direction_norm <= 1.0e-8:
            raise ValueError(f"{side} knee platform/condyle direction is degenerate")
        proximal_direction /= direction_norm
        medial_tree = cKDTree(vertices[medial_condyle])
        lateral_tree = cKDTree(vertices[lateral_condyle])

        best: tuple[
            float, float, float, float, TibiaFibulaWholeBoneFitV8
        ] | None = None
        for shift in np.linspace(
            0.0, float(maximum_platform_shift_m), int(search_steps)
        ):
            fitted = fit_tibia_fibula_to_platform_v8(
                tibia_vertices=vertices[tibia_ids],
                fibula_vertices=vertices[fibula_ids],
                current_platform_center=source_platform,
                current_distal_endpoint=source_distal,
                target_platform_center=source_platform
                + float(shift) * proximal_direction,
            )
            mapped_medial = apply_whole_bone_affine_v8(
                vertices[medial_platform], fitted.fit
            )
            mapped_lateral = apply_whole_bone_affine_v8(
                vertices[lateral_platform], fitted.fit
            )
            medial_gap = float(medial_tree.query(mapped_medial)[0].min())
            lateral_gap = float(lateral_tree.query(mapped_lateral)[0].min())
            objective = (
                (medial_gap - float(target_surface_gap_m)) ** 2
                + (lateral_gap - float(target_surface_gap_m)) ** 2
            )
            candidate = (
                objective,
                float(shift),
                medial_gap,
                lateral_gap,
                fitted,
            )
            if best is None or candidate[:4] < best[:4]:
                best = candidate
        if best is None:
            raise AssertionError("knee platform search produced no candidate")
        _objective, shift, medial_gap, lateral_gap, fitted = best
        axial_scale = float(fitted.fit.axial_scale)
        if not 0.90 <= axial_scale <= 1.10:
            raise ValueError(
                f"{side} whole-shank axial scale {axial_scale:.6f} "
                "is outside [0.90, 1.10]"
            )
        vertices[tibia_ids] = fitted.tibia_vertices
        vertices[fibula_ids] = fitted.fibula_vertices

        transformed_bones = np.asarray(
            [
                bone_names.index(f"Tibia_Bone_{suffix}"),
                bone_names.index(f"Tibia_Twist_{suffix}"),
                bone_names.index(effector),
            ],
            dtype=np.int64,
        )
        bind = update_target_bind_with_whole_bone_fit_v8(
            target_bone_head=target_head,
            target_bone_tail=target_tail,
            target_rest_global=target_global,
            parents=parents,
            transformed_bone_indices=transformed_bones,
            fit=fitted.fit,
        )
        target_head = np.asarray(bind.target_bone_head, dtype=np.float64)
        target_tail = np.asarray(bind.target_bone_tail, dtype=np.float64)
        target_global = np.asarray(bind.target_rest_global, dtype=np.float64)
        report["sides"][side] = {
            "platform_shift_m": shift,
            "fit_medial_gap_m": medial_gap,
            "fit_lateral_gap_m": lateral_gap,
            "axial_scale": axial_scale,
            "radial_scales": [1.0, 1.0],
            "distal_anchor_m": source_distal.tolist(),
            "transformed_bones": [
                bone_names[index] for index in transformed_bones.tolist()
            ],
            "affine": fitted.affine.tolist(),
        }

    target_local = _global_to_local(target_global, parents)
    metadata = dict(asset.metadata or {})
    metadata["v8_whole_shank_knee_ankle_fit"] = report
    result = replace(
        asset,
        vertices_rest=vertices.astype(np.float32),
        target_bone_head=target_head.astype(np.float32),
        target_bone_tail=target_tail.astype(np.float32),
        target_rest_global=target_global.astype(np.float32),
        target_rest_local=target_local.astype(np.float32),
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        source_driver_coupling=None,
        metadata=metadata,
    )
    result.validate()
    return result, report


def calibrate_ankle_roll_glide_v8(
    asset: AnatomyRiggedAsset,
    *,
    domains: Mapping[str, np.ndarray],
    target_surface_gap_m: float = 0.0015,
    maximum_translation_m: float = 0.012,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Bake a small whole-foot translation law from frozen contact surfaces.

    The law moves the complete ankle/foot subtree in parent-local coordinates;
    it never projects talus vertices or changes foot topology.  Fit domains
    determine the coefficients and disjoint validation domains measure them.
    """

    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    from .anatomy_lbs import source_bone_posed_global, skin_vertices

    names = list(asset.source_bone_names or ())
    metadata = dict(asset.metadata or {})
    metadata.pop("source_ankle_roll_glide_v8", None)
    base = replace(asset, metadata=metadata)
    knot_angles_deg = np.asarray(
        (-60.0, -30.0, 0.0, 30.0, 60.0), dtype=np.float64
    )
    validation_rotvecs: list[np.ndarray] = []
    for axis in range(3):
        for angle in (-45.0, -15.0, 15.0, 45.0):
            value = np.zeros(3, dtype=np.float64)
            value[axis] = np.radians(angle)
            validation_rotvecs.append(value)
    report: dict[str, Any] = {
        "available": True,
        "fit_uses_validation_vertices": False,
        "whole_foot_translation_only": True,
        "sides": {},
    }
    responses: dict[str, Any] = {}
    for side, suffix, joint in (("left", "L", 7), ("right", "R", 8)):
        ankle_bone = names.index(f"Ankle_Rot_{suffix}")
        parent = int(np.asarray(asset.source_bone_parents)[ankle_bone])
        fit_ids = {
            part: np.asarray(
                domains[f"ankle/{side}/{part}.fit"], dtype=np.int64
            )
            for part in ("tibia", "fibula", "talus")
        }
        validation_ids = {
            part: np.asarray(
                domains[f"ankle/{side}/{part}.validation"], dtype=np.int64
            )
            for part in ("tibia", "fibula", "talus")
        }
        axis_translations = np.zeros(
            (3, len(knot_angles_deg), 3), dtype=np.float64
        )
        fit_gaps: list[list[float]] = []
        for axis in range(3):
            for knot_index, angle_deg in enumerate(knot_angles_deg):
                if angle_deg == 0.0:
                    continue
                rotvec = np.zeros(3, dtype=np.float64)
                rotvec[axis] = np.radians(angle_deg)
                pose = np.zeros((55, 3), dtype=np.float32)
                pose[joint] = rotvec.astype(np.float32)
                posed = skin_vertices(base, pose, validate=False)
                tibia = np.asarray(
                    posed[fit_ids["tibia"]], dtype=np.float64
                )
                fibula = np.asarray(
                    posed[fit_ids["fibula"]], dtype=np.float64
                )
                talus = np.asarray(
                    posed[fit_ids["talus"]], dtype=np.float64
                )

                def objective(raw_translation: Any) -> float:
                    translation = np.asarray(raw_translation, dtype=np.float64)
                    shifted = talus + translation
                    tree = cKDTree(shifted)
                    gaps = (
                        float(tree.query(tibia)[0].min()),
                        float(tree.query(fibula)[0].min()),
                    )
                    return float(
                        sum(
                            (gap - float(target_surface_gap_m)) ** 2
                            for gap in gaps
                        )
                        + 0.02 * float(np.dot(translation, translation))
                    )

                solved = minimize(
                    objective,
                    np.zeros(3, dtype=np.float64),
                    method="Powell",
                    bounds=[
                        (-maximum_translation_m, maximum_translation_m)
                    ] * 3,
                    options={
                        "maxiter": 80,
                        "xtol": 1.0e-7,
                        "ftol": 1.0e-12,
                    },
                )
                translation_world = np.asarray(solved.x, dtype=np.float64)
                if (
                    not solved.success
                    or not np.all(np.isfinite(translation_world))
                    or float(np.linalg.norm(translation_world))
                    > float(maximum_translation_m) + 1.0e-7
                ):
                    raise ValueError(
                        f"{side} ankle roll-glide calibration failed"
                    )
                posed_bones = source_bone_posed_global(base, pose)
                axis_translations[axis, knot_index] = (
                    posed_bones[parent, :3, :3].T @ translation_world
                )
                shifted = talus + translation_world
                tree = cKDTree(shifted)
                fit_gaps.append(
                    [
                        float(tree.query(tibia)[0].min()),
                        float(tree.query(fibula)[0].min()),
                    ]
                )
        response = {
            "smplx_joint": joint,
            "axis_knots_rad": np.tile(
                np.radians(knot_angles_deg), (3, 1)
            ).tolist(),
            "axis_translation_parent_local_m": axis_translations.tolist(),
            "maximum_translation_m": float(maximum_translation_m * 2.0),
        }
        responses[str(ankle_bone)] = response
        candidate_metadata = {
            **metadata,
            "source_ankle_roll_glide_v8": {str(ankle_bone): response},
        }
        candidate = replace(base, metadata=candidate_metadata)
        validation_gaps: list[list[float]] = []
        for rotvec in validation_rotvecs:
            pose = np.zeros((55, 3), dtype=np.float32)
            pose[joint] = rotvec.astype(np.float32)
            posed = skin_vertices(candidate, pose, validate=False)
            talus = np.asarray(
                posed[validation_ids["talus"]], dtype=np.float64
            )
            tree = cKDTree(talus)
            validation_gaps.append(
                [
                    float(
                        tree.query(posed[validation_ids["tibia"]])[0].min()
                    ),
                    float(
                        tree.query(posed[validation_ids["fibula"]])[0].min()
                    ),
                ]
            )
        report["sides"][side] = {
            "ankle_bone": ankle_bone,
            "smplx_joint": joint,
            "training_sample_count": int(3 * (len(knot_angles_deg) - 1)),
            "validation_sample_count": len(validation_rotvecs),
            "fit_max_gap_m": float(np.max(fit_gaps)),
            "validation_max_gap_m": float(np.max(validation_gaps)),
            "coefficient_norm": float(np.linalg.norm(axis_translations)),
        }
    metadata["source_ankle_roll_glide_v8"] = responses
    result = replace(base, metadata=metadata)
    result.validate()
    return result, report


__all__ = [
    "ArticularWholeBoneFitV8",
    "TargetBindUpdateV8",
    "TibiaFibulaWholeBoneFitV8",
    "apply_fit_to_meshes_v8",
    "apply_whole_bone_affine_v8",
    "calibrate_ankle_roll_glide_v8",
    "fit_femur_to_acetabulum_v8",
    "fit_tibia_fibula_to_platform_v8",
    "reconstruct_hip_compounds_v8",
    "reconstruct_knee_ankle_compounds_v8",
    "update_target_bind_with_whole_bone_fit_v8",
    "whole_bone_affine_matrix_v8",
]
