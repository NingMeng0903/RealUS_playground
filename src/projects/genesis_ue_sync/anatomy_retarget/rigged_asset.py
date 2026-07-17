"""Schema helpers for anatomy meshes driven by SMPL-X through a source rig."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POSE_FORMAT = "smplx_body55_axis_angle"
DEFAULT_COORDINATE_SYSTEM = "smplx_y_up_m"
ANATOMY_ASSET_SCHEMA_VERSION = 6
SOURCE_DRIVER_MODES: tuple[str, ...] = (
    "joint_local",
    "segment_root",
    "twist",
    "bind_follow",
    "rigid_group",
)

# Compact serialization for Blender's Bone.inherit_scale enum.  Keep this
# stable: source templates are intended to outlive the Blender process that
# produced them.
def _string_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "U", "O"}:
        return np.asarray([str(v.decode("utf-8") if isinstance(v, bytes) else v) for v in arr.reshape(-1)], dtype=object)
    return np.asarray([str(v) for v in arr.reshape(-1)], dtype=object)


def source_global_from_local(rest_local: Any, parents: Any) -> np.ndarray:
    """Reconstruct source-rig global bind frames from the only persisted frames."""
    local = np.asarray(rest_local, dtype=np.float64).reshape(-1, 4, 4)
    pa = np.asarray(parents, dtype=np.int64).reshape(-1)
    if len(local) != len(pa):
        raise ValueError("source rest_local/parents length mismatch")
    result = np.empty_like(local)
    for bone, parent in enumerate(pa.tolist()):
        if int(parent) < 0:
            result[bone] = local[bone]
        else:
            if int(parent) >= bone:
                raise ValueError("source parents must be parent-before-child")
            result[bone] = result[int(parent)] @ local[bone]
    return result.astype(np.float32)


def _points_to_bone_local(points: Any, rest_global: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inverse = np.linalg.inv(np.asarray(rest_global, dtype=np.float64).reshape(-1, 4, 4))
    return np.einsum("bij,bj->bi", inverse[:, :3, :3], pts) + inverse[:, :3, 3]


def _points_from_bone_local(points: Any, rest_global: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    global_bind = np.asarray(rest_global, dtype=np.float64).reshape(-1, 4, 4)
    return np.einsum("bij,bj->bi", global_bind[:, :3, :3], pts) + global_bind[:, :3, 3]


@dataclass(frozen=True)
class AnatomyRiggedAsset:
    vertices_rest: np.ndarray
    faces: np.ndarray
    lbs_weights: np.ndarray | None
    joint_names: list[str]
    parents: np.ndarray
    rest_joints: np.ndarray
    inverse_bind: np.ndarray
    source_mesh_names: list[str]
    source_vertex_ranges: np.ndarray | None = None
    source_tissues: list[str] | None = None
    # V5 makes mesh material semantics explicit.  These are per
    # ``source_mesh_names`` entry, not per vertex: consumers must never infer a
    # controller from an arbitrary vertex or from an object name at runtime.
    source_mesh_controller_bones: np.ndarray | None = None
    source_mesh_material_groups: list[str] | None = None
    source_mesh_roles: list[str] | None = None
    source_fit_policies: list[str] | None = None
    source_driver_policies: list[str] | None = None
    source_compound_ids: list[str] | None = None
    source_sides: list[str] | None = None
    source_landmarks: list[tuple[str, ...]] | None = None
    target_landmark_recipes: list[str] | None = None
    source_quality_profiles: list[str] | None = None
    driver_indices: np.ndarray | None = None
    driver_weights: np.ndarray | None = None
    source_bone_names: list[str] | None = None
    source_bone_parents: np.ndarray | None = None
    # These compatibility field names hold the immutable authored source bind,
    # never a target bind or a runtime controller frame.  The schema-v6 names
    # are exposed by source_bind_global/source_bind_local below.
    source_rest_global: np.ndarray | None = None
    source_rest_local: np.ndarray | None = None
    source_inverse_bind: np.ndarray | None = None
    # Authored Blender bone endpoints are independent of SMPL-X driver probes.
    source_bone_head: np.ndarray | None = None
    source_bone_tail: np.ndarray | None = None
    source_bone_smplx_a: np.ndarray | None = None
    source_bone_smplx_b: np.ndarray | None = None
    source_bone_blend: np.ndarray | None = None
    source_bone_driver_types: list[str] | None = None
    # Up to three explicit SMPL-X joints used to construct a source-bone
    # driver frame.  ``-1`` is padding; column zero is always the primary
    # driver and agrees with source_bone_smplx_a.
    source_bone_frame_joints: np.ndarray | None = None
    # For every independently driven source bone, C = inv(F_rest) @ B_bind.
    # bind_follow entries are identity because their authored local bind is the
    # authority.  Persisting this matrix prevents runtime from inventing a
    # second rest frame from warped mesh vertices.
    source_driver_coupling: np.ndarray | None = None
    # Subject-fitted source-rig bind used by runtime skinning.  The authored
    # source bind above remains immutable and available for Blender parity.
    target_rest_global: np.ndarray | None = None
    target_rest_local: np.ndarray | None = None
    target_inverse_bind: np.ndarray | None = None
    target_bone_head: np.ndarray | None = None
    target_bone_tail: np.ndarray | None = None
    rigid_component_ids: np.ndarray | None = None
    registration_reference: np.ndarray | None = None
    source_skin_vertices: np.ndarray | None = None
    source_skin_faces: np.ndarray | None = None
    pose_cache_vertices: np.ndarray | None = None
    pose_cache_hash: str = ""
    pose_format: str = DEFAULT_POSE_FORMAT
    coordinate_system: str = DEFAULT_COORDINATE_SYSTEM
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Detach and freeze authored source-rig authority arrays.

        A frozen dataclass alone does not protect mutable NumPy buffers.  Copying
        these arrays also prevents a caller's target-fit scratch buffer from
        aliasing and later overwriting the persisted source bind or endpoints.
        """
        immutable_source_fields = (
            "source_rest_global",
            "source_rest_local",
            "source_inverse_bind",
            "source_bone_head",
            "source_bone_tail",
            "source_driver_coupling",
            "target_rest_global",
            "target_rest_local",
            "target_inverse_bind",
            "target_bone_head",
            "target_bone_tail",
        )
        for name in immutable_source_fields:
            value = getattr(self, name)
            if value is None:
                continue
            detached = np.array(value, copy=True)
            detached.setflags(write=False)
            object.__setattr__(self, name, detached)

    @property
    def source_bind_global(self) -> np.ndarray | None:
        """Immutable authored source-bone global bind matrices."""
        return self.source_rest_global

    @property
    def source_bind_local(self) -> np.ndarray | None:
        """Immutable authored source-bone local bind matrices."""
        return self.source_rest_local

    @property
    def target_bind_global(self) -> np.ndarray | None:
        """Subject-fitted bind, or the authored bind before regional fitting."""
        return (
            self.target_rest_global
            if self.target_rest_global is not None
            else self.source_rest_global
        )

    @property
    def target_bind_local(self) -> np.ndarray | None:
        return (
            self.target_rest_local
            if self.target_rest_local is not None
            else self.source_rest_local
        )

    @property
    def runtime_inverse_bind(self) -> np.ndarray | None:
        return (
            self.target_inverse_bind
            if self.target_inverse_bind is not None
            else self.source_inverse_bind
        )

    def validate(self) -> None:
        vertices = np.asarray(self.vertices_rest, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        weights = None if self.lbs_weights is None else np.asarray(self.lbs_weights, dtype=np.float32)
        parents = np.asarray(self.parents, dtype=np.int32).reshape(-1)
        rest_joints = np.asarray(self.rest_joints, dtype=np.float32)
        inverse_bind = np.asarray(self.inverse_bind, dtype=np.float32)
        joint_count = len(self.joint_names)

        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError(f"vertices_rest must be [N, 3], got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be [F, 3], got {faces.shape}")
        if weights is not None:
            if weights.shape != (vertices.shape[0], joint_count):
                raise ValueError(f"legacy lbs_weights must be {(vertices.shape[0], joint_count)}, got {weights.shape}")
            if np.any(weights < 0.0):
                raise ValueError("lbs_weights contains negative values")
        if self.driver_indices is not None or self.driver_weights is not None:
            if self.driver_indices is None or self.driver_weights is None:
                raise ValueError("driver_indices and driver_weights must be stored together")
            sparse_i = np.asarray(self.driver_indices, dtype=np.int32)
            sparse_w = np.asarray(self.driver_weights, dtype=np.float32)
            if sparse_i.shape != sparse_w.shape or sparse_i.ndim != 2 or sparse_i.shape[0] != vertices.shape[0]:
                raise ValueError("sparse drivers must both be [N, K]")
            source_count = len(self.source_bone_names or [])
            driver_count = source_count if source_count else joint_count
            if sparse_i.size and (int(sparse_i.min()) < 0 or int(sparse_i.max()) >= driver_count):
                raise ValueError("driver_indices contains an invalid source bone/joint")
            if not np.allclose(sparse_w.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0):
                raise ValueError("driver_weights rows must sum to one")
        if parents.shape != (joint_count,):
            raise ValueError(f"parents must be [{joint_count}], got {parents.shape}")
        if rest_joints.shape != (joint_count, 3):
            raise ValueError(f"rest_joints must be [{joint_count}, 3], got {rest_joints.shape}")
        if inverse_bind.shape != (joint_count, 4, 4):
            raise ValueError(f"inverse_bind must be [{joint_count}, 4, 4], got {inverse_bind.shape}")
        if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
            raise ValueError("faces contain vertex indices outside vertices_rest")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_rest contains non-finite values")
        if self.registration_reference is not None and np.asarray(self.registration_reference).shape != vertices.shape:
            raise ValueError("registration_reference must match vertices_rest")
        mesh_count = len(self.source_mesh_names)
        if self.source_vertex_ranges is None or np.asarray(self.source_vertex_ranges).shape != (mesh_count, 2):
            raise ValueError("source_vertex_ranges must have one [start, stop] range per source mesh")
        ranges = np.asarray(self.source_vertex_ranges, dtype=np.int64)
        if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] < ranges[:, 0]) or (mesh_count and int(ranges[-1, 1]) != len(vertices)):
            raise ValueError("source_vertex_ranges must be ordered and cover vertices_rest")
        mesh_semantics = {
            "source_mesh_controller_bones": self.source_mesh_controller_bones,
            "source_mesh_material_groups": self.source_mesh_material_groups,
            "source_mesh_roles": self.source_mesh_roles,
        }
        for name, value in mesh_semantics.items():
            if value is None or len(value) != mesh_count:
                raise ValueError(f"{name} must have one entry per source mesh")
        extended_semantics = {
            "source_fit_policies": self.source_fit_policies,
            "source_driver_policies": self.source_driver_policies,
            "source_compound_ids": self.source_compound_ids,
            "source_sides": self.source_sides,
            "source_landmarks": self.source_landmarks,
            "target_landmark_recipes": self.target_landmark_recipes,
            "source_quality_profiles": self.source_quality_profiles,
        }
        if any(value is not None for value in extended_semantics.values()):
            for name, value in extended_semantics.items():
                if value is None or len(value) != mesh_count:
                    raise ValueError(f"{name} must have one entry per source mesh")
        controllers = np.asarray(self.source_mesh_controller_bones, dtype=np.int32).reshape(-1)
        source_count_for_mesh = len(self.source_bone_names or [])
        if controllers.size and (int(controllers.min()) < 0 or int(controllers.max()) >= source_count_for_mesh):
            raise ValueError("source_mesh_controller_bones contains an invalid source bone")
        if any(not str(value) for value in (self.source_mesh_material_groups or [])):
            raise ValueError("source_mesh_material_groups may not contain empty values")
        if any(not str(value) for value in (self.source_mesh_roles or [])):
            raise ValueError("source_mesh_roles may not contain empty values")
        if self.source_skin_vertices is not None:
            skin_v = np.asarray(self.source_skin_vertices)
            skin_f = np.asarray(self.source_skin_faces)
            if skin_v.ndim != 2 or skin_v.shape[1] != 3 or skin_f.ndim != 2 or skin_f.shape[1] != 3:
                raise ValueError("source skin must be [N,3] vertices and [F,3] faces")
        if self.pose_cache_vertices is not None:
            cached = np.asarray(self.pose_cache_vertices)
            if cached.shape != vertices.shape or not np.all(np.isfinite(cached)):
                raise ValueError("pose_cache_vertices must be finite and match vertices_rest")
            if not str(self.pose_cache_hash):
                raise ValueError("pose_cache_hash is required with pose_cache_vertices")
        if weights is not None:
            if not np.all(np.isfinite(weights)):
                raise ValueError("lbs_weights contains non-finite values")
            row_sums = weights.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1.0e-5, rtol=0.0):
                raise ValueError(f"lbs_weights rows must sum to 1; max error={float(np.max(np.abs(row_sums - 1.0))):.6g}")
        if joint_count and int(parents[0]) not in (-1, 0):
            raise ValueError("parents[0] must be -1 or 0 for root")
        for idx, parent in enumerate(parents.tolist()):
            if idx == 0:
                continue
            if parent < 0 or parent >= idx:
                raise ValueError(f"parents[{idx}]={parent} must point to an earlier joint")

        if self.source_bone_names is not None:
            bone_count = len(self.source_bone_names)
            source_arrays = {
                "source_bone_parents": (self.source_bone_parents, (bone_count,)),
                "source_rest_global": (self.source_rest_global, (bone_count, 4, 4)),
                "source_rest_local": (self.source_rest_local, (bone_count, 4, 4)),
                "source_inverse_bind": (self.source_inverse_bind, (bone_count, 4, 4)),
                "source_bone_head": (self.source_bone_head, (bone_count, 3)),
                "source_bone_tail": (self.source_bone_tail, (bone_count, 3)),
                "source_bone_smplx_a": (self.source_bone_smplx_a, (bone_count,)),
                "source_bone_smplx_b": (self.source_bone_smplx_b, (bone_count,)),
                "source_bone_blend": (self.source_bone_blend, (bone_count,)),
            }
            for name, (value, shape) in source_arrays.items():
                if value is None or np.asarray(value).shape != shape:
                    raise ValueError(f"{name} must be {shape} for source-rig v2")
                if not np.all(np.isfinite(np.asarray(value))):
                    raise ValueError(f"{name} contains non-finite values")
            if self.source_bone_driver_types is None or len(self.source_bone_driver_types) != bone_count:
                raise ValueError("source_bone_driver_types must have one entry per source bone")
            unknown_modes = sorted(set(self.source_bone_driver_types) - set(SOURCE_DRIVER_MODES))
            if unknown_modes:
                raise ValueError(f"unknown source driver mode(s): {unknown_modes}")
            if self.source_bone_frame_joints is None:
                raise ValueError("schema v6 requires explicit source_bone_frame_joints")
            frame_joints = np.asarray(self.source_bone_frame_joints, dtype=np.int32)
            if frame_joints.shape != (bone_count, 3):
                raise ValueError("source_bone_frame_joints must be [source_bone_count, 3]")
            if np.any(frame_joints < -1) or np.any(frame_joints >= joint_count):
                raise ValueError("source_bone_frame_joints contains an invalid SMPL-X joint")
            if np.any(frame_joints[:, 0] < 0):
                raise ValueError("source_bone_frame_joints requires a primary joint in column zero")
            if not np.array_equal(frame_joints[:, 0], np.asarray(self.source_bone_smplx_a, dtype=np.int32)):
                raise ValueError("source_bone_frame_joints[:, 0] must match source_bone_smplx_a")
            source_a = np.asarray(self.source_bone_smplx_a, dtype=np.int32)
            source_b = np.asarray(self.source_bone_smplx_b, dtype=np.int32)
            if (
                np.any(source_a < 0)
                or np.any(source_a >= joint_count)
                or np.any(source_b < 0)
                or np.any(source_b >= joint_count)
            ):
                raise ValueError("source bone driver contains an unmapped or invalid SMPL-X joint")
            blends = np.asarray(self.source_bone_blend, dtype=np.float64)
            if np.any(blends < 0.0) or np.any(blends > 1.0):
                raise ValueError("source_bone_blend must be within [0, 1]")
            if self.source_driver_coupling is not None:
                coupling = np.asarray(self.source_driver_coupling)
                if coupling.shape != (bone_count, 4, 4) or not np.all(np.isfinite(coupling)):
                    raise ValueError("source_driver_coupling must be finite [source_bone_count, 4, 4]")
                if (
                    not np.allclose(
                        coupling[:, 3, :],
                        np.asarray((0.0, 0.0, 0.0, 1.0)),
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                    or np.any(
                        np.abs(np.linalg.det(coupling[:, :3, :3])) <= 1.0e-10
                    )
                ):
                    raise ValueError("source_driver_coupling must be invertible affine")
            source_parents = np.asarray(self.source_bone_parents, dtype=np.int32)
            for idx, parent in enumerate(source_parents.tolist()):
                if parent >= idx or parent < -1:
                    raise ValueError(f"source_bone_parents[{idx}]={parent} is not topological")
                if (
                    self.source_driver_coupling is not None
                    and self.source_bone_driver_types[idx] == "bind_follow"
                    and parent >= 0
                    and not np.allclose(
                        coupling[idx],
                        np.eye(4),
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                ):
                    raise ValueError("bind_follow source_driver_coupling must be identity")
            if self.driver_indices is None or self.driver_weights is None:
                raise ValueError("source-rig v2 requires sparse driver indices and weights")

            source_global = np.asarray(self.source_bind_global, dtype=np.float64)
            source_local = np.asarray(self.source_bind_local, dtype=np.float64)
            inverse_source = np.asarray(self.source_inverse_bind, dtype=np.float64)
            for name, matrices in (
                ("source_bind_global", source_global),
                ("source_bind_local", source_local),
                ("source_inverse_bind", inverse_source),
            ):
                if not np.allclose(
                    matrices[:, 3, :],
                    np.asarray((0.0, 0.0, 0.0, 1.0)),
                    atol=1.0e-6,
                    rtol=0.0,
                ):
                    raise ValueError(f"{name} must contain affine transforms")
                determinants = np.linalg.det(matrices[:, :3, :3])
                if np.any(np.abs(determinants) <= 1.0e-10):
                    raise ValueError(f"{name} contains a singular transform")
            for idx, parent in enumerate(source_parents.tolist()):
                reconstructed = (
                    source_local[idx]
                    if int(parent) < 0
                    else source_global[int(parent)] @ source_local[idx]
                )
                if not np.allclose(reconstructed, source_global[idx], atol=1.0e-4, rtol=0.0):
                    raise ValueError(
                        f"source_bind_local[{idx}] does not reconstruct source_bind_global"
                    )
            if not np.allclose(
                source_global @ inverse_source,
                np.eye(4, dtype=np.float64),
                atol=1.0e-4,
                rtol=0.0,
            ):
                raise ValueError("source_inverse_bind does not invert source_bind_global")
            heads = np.asarray(self.source_bone_head, dtype=np.float64)
            tails = np.asarray(self.source_bone_tail, dtype=np.float64)
            if np.any(np.linalg.norm(tails - heads, axis=1) <= 1.0e-8):
                raise ValueError("source bone head/tail contains a zero-length bone")
            target_global = np.asarray(self.target_bind_global, dtype=np.float64)
            target_local = np.asarray(self.target_bind_local, dtype=np.float64)
            target_inverse = np.asarray(self.runtime_inverse_bind, dtype=np.float64)
            if (
                target_global.shape != source_global.shape
                or target_local.shape != source_local.shape
                or target_inverse.shape != inverse_source.shape
            ):
                raise ValueError("target bind matrices must match the source rig")
            for idx, parent in enumerate(source_parents.tolist()):
                reconstructed = (
                    target_local[idx]
                    if int(parent) < 0
                    else target_global[int(parent)] @ target_local[idx]
                )
                if not np.allclose(
                    reconstructed,
                    target_global[idx],
                    atol=1.0e-4,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"target_bind_local[{idx}] does not reconstruct target_bind_global"
                    )
            if not np.allclose(
                target_global @ target_inverse,
                np.eye(4, dtype=np.float64),
                atol=1.0e-4,
                rtol=0.0,
            ):
                raise ValueError("target_inverse_bind does not invert target_bind_global")
            target_heads = np.asarray(
                self.target_bone_head if self.target_bone_head is not None else heads,
                dtype=np.float64,
            )
            target_tails = np.asarray(
                self.target_bone_tail if self.target_bone_tail is not None else tails,
                dtype=np.float64,
            )
            if target_heads.shape != heads.shape or target_tails.shape != tails.shape:
                raise ValueError("target bone probes must match the source bone count")
            if np.any(np.linalg.norm(target_tails - target_heads, axis=1) <= 1.0e-8):
                raise ValueError("target bone head/tail contains a zero-length bone")


def _semantic_defaults(asset: AnatomyRiggedAsset) -> AnatomyRiggedAsset:
    """Materialize explicit v6 semantics for programmatically built assets.

    Blender extraction always supplies these fields.  This narrow fallback
    keeps small unit fixtures usable while ensuring the serialized V5 asset
    never relies on runtime inference.
    """
    mesh_count = len(asset.source_mesh_names)
    if asset.source_bone_names is None:
        return asset
    frame = asset.source_bone_frame_joints
    if frame is None and asset.source_bone_smplx_a is not None and asset.source_bone_smplx_b is not None:
        frame = np.full((len(asset.source_bone_names), 3), -1, dtype=np.int16)
        frame[:, 0] = np.asarray(asset.source_bone_smplx_a, dtype=np.int16)
        frame[:, 1] = np.asarray(asset.source_bone_smplx_b, dtype=np.int16)
    controllers = asset.source_mesh_controller_bones
    if controllers is None and asset.source_vertex_ranges is not None and asset.driver_indices is not None and asset.driver_weights is not None:
        controllers = np.empty(mesh_count, dtype=np.int16)
        for mi, (start, stop) in enumerate(np.asarray(asset.source_vertex_ranges, dtype=np.int64)):
            indices = np.asarray(asset.driver_indices[start:stop], dtype=np.int64)
            weights = np.asarray(asset.driver_weights[start:stop], dtype=np.float64)
            mass = np.zeros(len(asset.source_bone_names), dtype=np.float64)
            if indices.size:
                np.add.at(mass, indices.reshape(-1), weights.reshape(-1))
            controllers[mi] = int(np.argmax(mass))
    tissues = list(asset.source_tissues or [])
    groups = asset.source_mesh_material_groups
    if groups is None and len(tissues) == mesh_count:
        groups = ["soft_tissue" if tissue in {"vessel", "nerve", "organ", "connective_tissue"} else "skeletal" for tissue in tissues]
    roles = asset.source_mesh_roles
    if roles is None:
        roles = ["authored_mesh"] * mesh_count
    fit_policies = asset.source_fit_policies
    if fit_policies is None:
        fit_policies = ["volume_field"] * mesh_count
    driver_policies = asset.source_driver_policies
    if driver_policies is None:
        driver_policies = ["bind_follow"] * mesh_count
    compound_ids = asset.source_compound_ids
    if compound_ids is None:
        compound_ids = [""] * mesh_count
    sides = asset.source_sides
    if sides is None:
        sides = ["center"] * mesh_count
    landmarks = asset.source_landmarks
    if landmarks is None:
        landmarks = [("auto_geometry_landmarks",)] * mesh_count
    target_recipes = asset.target_landmark_recipes
    if target_recipes is None:
        target_recipes = ["auto_geometry"] * mesh_count
    quality_profiles = asset.source_quality_profiles
    if quality_profiles is None:
        quality_profiles = ["default"] * mesh_count
    result = replace(
        asset,
        source_bone_frame_joints=frame,
        source_mesh_controller_bones=controllers,
        source_mesh_material_groups=groups,
        source_mesh_roles=roles,
        source_fit_policies=fit_policies,
        source_driver_policies=driver_policies,
        source_compound_ids=compound_ids,
        source_sides=sides,
        source_landmarks=landmarks,
        target_landmark_recipes=target_recipes,
        source_quality_profiles=quality_profiles,
    )
    if result.source_driver_coupling is None:
        # Local import avoids a module cycle while allowing small programmatic
        # fixtures and Blender extraction to use the same coupling builder.
        from .anatomy_lbs import with_source_driver_coupling

        result = with_source_driver_coupling(result)
    return result


def save_rigged_asset(path: Path | str, asset: AnatomyRiggedAsset) -> Path:
    asset = _semantic_defaults(asset)
    asset.validate()
    if asset.source_bone_names is None:
        raise ValueError("schema v6 requires a complete source rig")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if asset.driver_indices is None or asset.driver_weights is None:
        if asset.lbs_weights is None:
            raise ValueError("asset requires either sparse drivers or legacy dense weights")
        driver_indices, driver_weights = sparse_driver_weights(asset.lbs_weights)
    else:
        driver_indices, driver_weights = asset.driver_indices, asset.driver_weights
    payload: dict[str, Any] = dict(
        schema_version=np.asarray(ANATOMY_ASSET_SCHEMA_VERSION, dtype=np.int32),
        vertices_rest=np.asarray(asset.vertices_rest, dtype=np.float32),
        faces=np.asarray(asset.faces, dtype=np.int32),
        joint_names=np.asarray(asset.joint_names, dtype=object),
        parents=np.asarray(asset.parents, dtype=np.int32),
        rest_joints=np.asarray(asset.rest_joints, dtype=np.float32),
        inverse_bind=np.asarray(asset.inverse_bind, dtype=np.float32),
        source_mesh_names=np.asarray(asset.source_mesh_names, dtype=object),
        source_vertex_ranges=np.asarray(
            asset.source_vertex_ranges if asset.source_vertex_ranges is not None else [], dtype=np.int32
        ).reshape(-1, 2),
        source_tissues=np.asarray(asset.source_tissues or [], dtype=object),
        source_mesh_controller_bones=np.asarray(asset.source_mesh_controller_bones, dtype=np.int16),
        source_mesh_material_groups=np.asarray(asset.source_mesh_material_groups, dtype=object),
        source_mesh_roles=np.asarray(asset.source_mesh_roles, dtype=object),
        source_fit_policies=np.asarray(asset.source_fit_policies, dtype=object),
        source_driver_policies=np.asarray(asset.source_driver_policies, dtype=object),
        source_compound_ids=np.asarray(asset.source_compound_ids, dtype=object),
        source_sides=np.asarray(asset.source_sides, dtype=object),
        source_landmarks_json=np.asarray(
            [
                json.dumps(list(values), ensure_ascii=True)
                for values in (asset.source_landmarks or [])
            ],
            dtype=object,
        ),
        target_landmark_recipes=np.asarray(asset.target_landmark_recipes, dtype=object),
        source_quality_profiles=np.asarray(asset.source_quality_profiles, dtype=object),
        driver_indices=np.asarray(driver_indices, dtype=np.int16),
        driver_weights=np.asarray(driver_weights, dtype=np.float32),
        pose_format=np.asarray(str(asset.pose_format), dtype=object),
        coordinate_system=np.asarray(str(asset.coordinate_system), dtype=object),
        metadata=np.asarray(asset.metadata or {}, dtype=object),
    )
    if asset.source_bone_names is not None:
        if asset.source_rest_local is None:
            raise ValueError("schema v6 requires source_rest_local")
        source_global = source_global_from_local(asset.source_rest_local, asset.source_bone_parents)
        target_local = np.asarray(asset.target_bind_local, dtype=np.float32)
        target_global = source_global_from_local(target_local, asset.source_bone_parents)
        head_local = _points_to_bone_local(asset.source_bone_head, source_global)
        tail_local = _points_to_bone_local(asset.source_bone_tail, source_global)
        target_head = (
            asset.target_bone_head
            if asset.target_bone_head is not None
            else asset.source_bone_head
        )
        target_tail = (
            asset.target_bone_tail
            if asset.target_bone_tail is not None
            else asset.source_bone_tail
        )
        target_head_local = _points_to_bone_local(target_head, target_global)
        target_tail_local = _points_to_bone_local(target_tail, target_global)
        payload.update(
            source_bone_names=np.asarray(asset.source_bone_names, dtype=object),
            source_bone_parents=np.asarray(asset.source_bone_parents, dtype=np.int16),
            source_bind_global=np.asarray(source_global, dtype=np.float32),
            source_bind_local=np.asarray(asset.source_rest_local, dtype=np.float32),
            source_bone_head_local=np.asarray(head_local, dtype=np.float32),
            source_bone_tail_local=np.asarray(tail_local, dtype=np.float32),
            target_bind_global=np.asarray(target_global, dtype=np.float32),
            target_bind_local=np.asarray(target_local, dtype=np.float32),
            target_bone_head_local=np.asarray(target_head_local, dtype=np.float32),
            target_bone_tail_local=np.asarray(target_tail_local, dtype=np.float32),
            source_bone_smplx_a=np.asarray(asset.source_bone_smplx_a, dtype=np.int16),
            source_bone_smplx_b=np.asarray(asset.source_bone_smplx_b, dtype=np.int16),
            source_bone_blend=np.asarray(asset.source_bone_blend, dtype=np.float32),
            source_bone_driver_types=np.asarray(asset.source_bone_driver_types, dtype=object),
            source_bone_frame_joints=np.asarray(asset.source_bone_frame_joints, dtype=np.int16),
            source_driver_coupling=np.asarray(asset.source_driver_coupling, dtype=np.float32),
            rigid_component_ids=np.asarray(
                asset.rigid_component_ids if asset.rigid_component_ids is not None else [], dtype=np.int32
            ),
            registration_reference=np.asarray(
                asset.registration_reference if asset.registration_reference is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            source_skin_vertices=np.asarray(
                asset.source_skin_vertices if asset.source_skin_vertices is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            source_skin_faces=np.asarray(
                asset.source_skin_faces if asset.source_skin_faces is not None else [], dtype=np.int32
            ).reshape(-1, 3),
            posed_vertices=np.asarray(
                asset.pose_cache_vertices if asset.pose_cache_vertices is not None else [], dtype=np.float32
            ).reshape(-1, 3),
            pose_hash=np.asarray(str(asset.pose_cache_hash), dtype=object),
        )
    np.savez_compressed(
        out,
        **payload,
    )
    return out


def sparse_driver_weights(
    weights: Any,
    *,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense weights without truncating authoritative influences."""
    dense = np.asarray(weights, dtype=np.float32)
    if dense.ndim != 2 or dense.shape[1] == 0:
        raise ValueError(f"weights must be [N, J], got {dense.shape}")
    if top_k is None:
        k = max(1, int(np.max(np.count_nonzero(dense > 0.0, axis=1))))
    else:
        k = max(1, min(int(top_k), int(dense.shape[1])))
    indices = np.argpartition(dense, -k, axis=1)[:, -k:]
    values = np.take_along_axis(dense, indices, axis=1)
    order = np.argsort(-values, axis=1)
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int16)
    values = np.take_along_axis(values, order, axis=1).astype(np.float32)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1.0e-8)
    return indices, values


def load_rigged_asset(path: Path | str, *, validate: bool = True) -> AnatomyRiggedAsset:
    data = np.load(Path(path), allow_pickle=True)
    schema = int(np.asarray(data["schema_version"]).reshape(-1)[0]) if "schema_version" in data.files else 0
    if schema != ANATOMY_ASSET_SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses anatomy schema {schema}; schema {ANATOMY_ASSET_SCHEMA_VERSION} "
            "is required, rebuild from the source blend"
        )
    metadata: dict[str, Any] | None = None
    if "metadata" in data.files:
        raw_meta = data["metadata"]
        try:
            metadata = dict(raw_meta.item())
        except Exception:
            metadata = {}
    required = {
        "driver_indices", "driver_weights", "source_bone_names", "source_bone_parents",
        "source_bind_global", "source_bind_local", "source_bone_head_local", "source_bone_tail_local",
        "target_bind_global", "target_bind_local", "target_bone_head_local", "target_bone_tail_local",
        "source_bone_smplx_a", "source_bone_smplx_b", "source_bone_blend",
        "source_bone_driver_types", "source_bone_frame_joints",
        "source_driver_coupling",
        "source_mesh_controller_bones", "source_mesh_material_groups", "source_mesh_roles",
        "source_fit_policies", "source_driver_policies", "source_compound_ids",
        "source_sides", "source_landmarks_json", "target_landmark_recipes",
        "source_quality_profiles",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{path} is missing schema-v6 fields: {missing}")
    driver_indices = np.asarray(data["driver_indices"], dtype=np.int16)
    driver_weights = np.asarray(data["driver_weights"], dtype=np.float32)
    source_parents = (
        np.asarray(data["source_bone_parents"], dtype=np.int32)
        if "source_bone_parents" in data.files else None
    )
    source_local = (
        np.asarray(data["source_bind_local"], dtype=np.float32)
        if "source_bind_local" in data.files else None
    )
    source_global = np.asarray(data["source_bind_global"], dtype=np.float32)
    reconstructed_global = source_global_from_local(source_local, source_parents)
    if not np.allclose(source_global, reconstructed_global, atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{path} source bind global/local matrices are inconsistent")
    source_head = (
        _points_from_bone_local(data["source_bone_head_local"], source_global).astype(np.float32)
        if "source_bone_head_local" in data.files and source_global is not None else None
    )
    source_tail = (
        _points_from_bone_local(data["source_bone_tail_local"], source_global).astype(np.float32)
        if "source_bone_tail_local" in data.files and source_global is not None else None
    )
    target_local = np.asarray(data["target_bind_local"], dtype=np.float32)
    target_global = np.asarray(data["target_bind_global"], dtype=np.float32)
    reconstructed_target = source_global_from_local(target_local, source_parents)
    if not np.allclose(target_global, reconstructed_target, atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{path} target bind global/local matrices are inconsistent")
    target_head = _points_from_bone_local(
        data["target_bone_head_local"], target_global
    ).astype(np.float32)
    target_tail = _points_from_bone_local(
        data["target_bone_tail_local"], target_global
    ).astype(np.float32)
    asset = AnatomyRiggedAsset(
        vertices_rest=np.asarray(data["vertices_rest"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int32),
        lbs_weights=None,
        joint_names=[str(v) for v in _string_array(data["joint_names"]).tolist()],
        parents=np.asarray(data["parents"], dtype=np.int32),
        rest_joints=np.asarray(data["rest_joints"], dtype=np.float32),
        inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float32),
        source_mesh_names=[str(v) for v in _string_array(data["source_mesh_names"]).tolist()],
        source_vertex_ranges=(
            np.asarray(data["source_vertex_ranges"], dtype=np.int32).reshape(-1, 2)
            if "source_vertex_ranges" in data.files
            else None
        ),
        source_tissues=(
            [str(v) for v in _string_array(data["source_tissues"]).tolist()]
            if "source_tissues" in data.files and data["source_tissues"].size
            else None
        ),
        source_mesh_controller_bones=np.asarray(data["source_mesh_controller_bones"], dtype=np.int32),
        source_mesh_material_groups=[str(v) for v in _string_array(data["source_mesh_material_groups"]).tolist()],
        source_mesh_roles=[str(v) for v in _string_array(data["source_mesh_roles"]).tolist()],
        source_fit_policies=[str(v) for v in _string_array(data["source_fit_policies"]).tolist()],
        source_driver_policies=[
            str(v) for v in _string_array(data["source_driver_policies"]).tolist()
        ],
        source_compound_ids=[
            str(v) for v in _string_array(data["source_compound_ids"]).tolist()
        ],
        source_sides=[str(v) for v in _string_array(data["source_sides"]).tolist()],
        source_landmarks=[
            tuple(
                str(value)
                for value in json.loads(str(serialized))
            )
            for serialized in _string_array(data["source_landmarks_json"]).tolist()
        ],
        target_landmark_recipes=[
            str(v) for v in _string_array(data["target_landmark_recipes"]).tolist()
        ],
        source_quality_profiles=[
            str(v) for v in _string_array(data["source_quality_profiles"]).tolist()
        ],
        driver_indices=driver_indices,
        driver_weights=driver_weights,
        source_bone_names=(
            [str(v) for v in _string_array(data["source_bone_names"]).tolist()]
            if "source_bone_names" in data.files
            else None
        ),
        source_bone_parents=source_parents,
        source_rest_global=source_global,
        source_rest_local=source_local,
        source_inverse_bind=(np.linalg.inv(source_global).astype(np.float32) if source_global is not None else None),
        source_bone_head=source_head,
        source_bone_tail=source_tail,
        source_bone_smplx_a=np.asarray(data["source_bone_smplx_a"], dtype=np.int32) if "source_bone_smplx_a" in data.files else None,
        source_bone_smplx_b=np.asarray(data["source_bone_smplx_b"], dtype=np.int32) if "source_bone_smplx_b" in data.files else None,
        source_bone_blend=np.asarray(data["source_bone_blend"], dtype=np.float32) if "source_bone_blend" in data.files else None,
        source_bone_driver_types=(
            [str(v) for v in _string_array(data["source_bone_driver_types"]).tolist()]
            if "source_bone_driver_types" in data.files
            else None
        ),
        source_bone_frame_joints=np.asarray(data["source_bone_frame_joints"], dtype=np.int32),
        source_driver_coupling=np.asarray(data["source_driver_coupling"], dtype=np.float32),
        target_rest_global=target_global,
        target_rest_local=target_local,
        target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
        target_bone_head=target_head,
        target_bone_tail=target_tail,
        rigid_component_ids=np.asarray(data["rigid_component_ids"], dtype=np.int32) if "rigid_component_ids" in data.files else None,
        registration_reference=np.asarray(data["registration_reference"], dtype=np.float32).reshape(-1, 3) if "registration_reference" in data.files and data["registration_reference"].size else None,
        source_skin_vertices=np.asarray(data["source_skin_vertices"], dtype=np.float32).reshape(-1, 3) if "source_skin_vertices" in data.files and data["source_skin_vertices"].size else None,
        source_skin_faces=np.asarray(data["source_skin_faces"], dtype=np.int32).reshape(-1, 3) if "source_skin_faces" in data.files and data["source_skin_faces"].size else None,
        pose_cache_vertices=np.asarray(data["posed_vertices"], dtype=np.float32).reshape(-1, 3) if "posed_vertices" in data.files and data["posed_vertices"].size else None,
        pose_cache_hash=str(data["pose_hash"].item()) if "pose_hash" in data.files else "",
        pose_format=str(data["pose_format"].item()) if "pose_format" in data.files else DEFAULT_POSE_FORMAT,
        coordinate_system=str(data["coordinate_system"].item()) if "coordinate_system" in data.files else DEFAULT_COORDINATE_SYSTEM,
        metadata=metadata,
    )
    if validate:
        asset.validate()
    return asset
