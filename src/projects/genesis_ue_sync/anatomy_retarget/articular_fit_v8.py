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


__all__ = [
    "ArticularWholeBoneFitV8",
    "TargetBindUpdateV8",
    "TibiaFibulaWholeBoneFitV8",
    "apply_fit_to_meshes_v8",
    "apply_whole_bone_affine_v8",
    "fit_femur_to_acetabulum_v8",
    "fit_tibia_fibula_to_platform_v8",
    "reconstruct_hip_compounds_v8",
    "update_target_bind_with_whole_bone_fit_v8",
    "whole_bone_affine_matrix_v8",
]
