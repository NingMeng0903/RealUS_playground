"""Schema-v8 mechanism primitives for selective anatomy retargeting.

This module intentionally has no Blender, SMPL-X, SciPy, or schema-v7 runtime
dependency.  It defines the small, fail-closed contracts shared by the V8
offline baker and resident evaluator:

* material domains are immutable and tied to one topology digest;
* the ba9 head compound is selected from authored Armature weights;
* V71 remains a complete parent-local FK authority;
* a whole bone receives one rest transform, never an endpoint shrink profile;
* limb children are composed from their parent and an authored joint, rather
  than independently anchored to global SMPL-X joints.

The functions operate on NumPy arrays so the exact same math can be exercised
by the offline baker, runtime, and independent acceptance harness.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .fk_policy_v8 import validate_source_fk_policy_v8


ANATOMY_V8_SCHEMA_VERSION = 8
V71_SOURCE_BONE_COUNT = 235
_DIGEST_HEX = frozenset("0123456789abcdef")
_LEGACY_V7_KEYS = frozenset(
    {
        "source_leg_hinge_solve_v1",
        "leg_hinge_solve_v1",
        "patella_oracle",
        "patella_oracle_v7",
        "patella_oracle_digest",
        "frozen_v71_patella_oracle",
        "source_knee_hinge_splines_v7",
        "source_tibia_glide_splines_v7",
        "source_patella_splines_v7",
        "source_patella_v71_response_v8",
        "source_patella_response_v7",
    }
)
_GLOBAL_CHILD_ANCHOR_KEYS = frozenset(
    {
        "child_global_anchor",
        "child_global_anchors",
        "global_child_anchor",
        "global_child_anchors",
        "independent_global_anchor",
        "independent_global_anchors",
    }
)
_LEGACY_SHRINK_KEYS = frozenset(
    {
        "fit_subject_bone_containment",
        "bone_containment_shrink",
        "containment_shrink",
        "endpoint_compression",
        "endpoint_scale_profile",
        "joint_lobe_scale",
        "hand_radial_scale",
        "hand_bone_radial_scale",
        "femur_joint_lobe_scale",
    }
)
_AXIAL_STRAIN_LIMITS_V810 = MappingProxyType(
    {
        "femur": 0.12,
        "shank": 0.08,
    }
)


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _unit(vector: Any, *, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    length = float(np.linalg.norm(value))
    if length <= 1.0e-10:
        raise ValueError(f"{label} may not be zero")
    return value / length


def _validate_digest(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in _DIGEST_HEX for char in digest):
        raise ValueError(f"{label} must be a full lowercase SHA-256 digest")
    return digest


def _proper_rotation(value: Any, *, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7):
        raise ValueError(f"{label} must be a proper rotation")
    return rotation


def axis_rotation_v8(axis: Any, angle_rad: float) -> np.ndarray:
    """Return a proper rotation around ``axis`` without a SciPy dependency."""

    direction = _unit(axis, label="rotation axis")
    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError("rotation angle must be finite")
    x, y, z = direction
    cross = np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64
    )
    return (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def _rotation_from_to(source: Any, target: Any) -> np.ndarray:
    first = _unit(source, label="source axis")
    second = _unit(target, label="target axis")
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    if sine > 1.0e-10:
        return axis_rotation_v8(cross / sine, math.atan2(sine, cosine))
    if cosine > 0.0:
        return np.eye(3)
    seed = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(seed, first))) > 0.8:
        seed = np.asarray((0.0, 0.0, 1.0))
    return axis_rotation_v8(np.cross(first, seed), math.pi)


def _frame_from_axis(axis: Any) -> np.ndarray:
    longitudinal = _unit(axis, label="bone axis")
    seed = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(seed, longitudinal))) > 0.85:
        seed = np.asarray((1.0, 0.0, 0.0))
    radial_a = _unit(np.cross(seed, longitudinal), label="radial axis")
    radial_b = _unit(
        np.cross(longitudinal, radial_a), label="secondary radial axis"
    )
    return np.stack((radial_a, longitudinal, radial_b), axis=1)


def _walk_mapping(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).strip().lower(), item
            yield from _walk_mapping(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_mapping(item)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def strip_v7_leg_oracle_metadata_v8(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached mapping without V7 hinge/oracle payloads.

    This helper is for an explicit migration step.  V8 validation itself never
    silently strips the payload: a runtime pack containing one of these keys is
    rejected.
    """

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): clean(item)
                for key, item in value.items()
                if str(key).strip().lower() not in _LEGACY_V7_KEYS
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clean(item) for item in value)
        return copy.deepcopy(value)

    return clean(metadata)


def reject_obsolete_mechanism_config_v8(metadata: Mapping[str, Any]) -> None:
    """Reject V7 self-oracles, global child anchors, and local bone shrink."""

    for key, value in _walk_mapping(metadata):
        if key in _LEGACY_V7_KEYS:
            raise ValueError(f"V8 forbids legacy V7 hinge/oracle field {key!r}")
        if key in _GLOBAL_CHILD_ANCHOR_KEYS:
            raise ValueError(f"V8 forbids independent global child anchor {key!r}")
        if key in _LEGACY_SHRINK_KEYS:
            raise ValueError(f"V8 forbids legacy local bone shrink field {key!r}")
        if isinstance(value, str) and value.strip().lower() in _LEGACY_V7_KEYS:
            raise ValueError(
                f"V8 forbids legacy V7 hinge/oracle value {value!r}"
            )
        if key in {"head_scale", "fast_head_scale"}:
            try:
                scale = float(value)
            except (TypeError, ValueError):
                continue
            if math.isclose(scale, 0.70, abs_tol=1.0e-12, rel_tol=0.0):
                raise ValueError("V8 rejects the legacy head_scale=0.70 fast patch")


@dataclass(frozen=True)
class FrozenMaterialDomainV8:
    """One topology-bound domain with independent fit and validation probes."""

    name: str
    topology_digest: str
    fit_vertex_ids: np.ndarray
    validation_vertex_ids: np.ndarray

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("material domain name is required")
        topology = _validate_digest(self.topology_digest, label=f"{name}.topology_digest")
        fit = np.asarray(self.fit_vertex_ids, dtype=np.int64).reshape(-1)
        validation = np.asarray(self.validation_vertex_ids, dtype=np.int64).reshape(-1)
        if fit.size == 0 or validation.size == 0:
            raise ValueError(f"{name} requires non-empty fit and validation vertex IDs")
        if np.any(fit < 0) or np.any(validation < 0):
            raise ValueError(f"{name} vertex IDs must be non-negative")
        if len(np.unique(fit)) != len(fit) or len(np.unique(validation)) != len(validation):
            raise ValueError(f"{name} contains duplicate vertex IDs")
        if np.intersect1d(fit, validation).size:
            raise ValueError(f"{name} fit and validation domains must be disjoint")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "topology_digest", topology)
        object.__setattr__(self, "fit_vertex_ids", _readonly(fit, np.int64))
        object.__setattr__(
            self, "validation_vertex_ids", _readonly(validation, np.int64)
        )

    def validate_vertex_count(self, vertex_count: int) -> None:
        count = int(vertex_count)
        if count <= 0:
            raise ValueError("vertex_count must be positive")
        maximum = max(
            int(np.max(self.fit_vertex_ids)),
            int(np.max(self.validation_vertex_ids)),
        )
        if maximum >= count:
            raise ValueError(
                f"{self.name} vertex ID {maximum} is outside topology size {count}"
            )


@dataclass(frozen=True)
class FrozenMaterialDomainsV8:
    """Immutable collection whose members all address one source topology."""

    topology_digest: str
    domains: tuple[FrozenMaterialDomainV8, ...]

    def __post_init__(self) -> None:
        topology = _validate_digest(self.topology_digest, label="topology_digest")
        domains = tuple(self.domains)
        if not domains:
            raise ValueError("at least one frozen material domain is required")
        names = [domain.name for domain in domains]
        if len(set(names)) != len(names):
            raise ValueError("frozen material domain names must be unique")
        for domain in domains:
            if domain.topology_digest != topology:
                raise ValueError(
                    f"{domain.name} addresses a different topology digest"
                )
        object.__setattr__(self, "topology_digest", topology)
        object.__setattr__(self, "domains", domains)

    def require(self, name: str) -> FrozenMaterialDomainV8:
        for domain in self.domains:
            if domain.name == str(name):
                return domain
        raise KeyError(f"missing frozen V8 material domain {name!r}")

    def validate_vertex_count(self, vertex_count: int) -> None:
        for domain in self.domains:
            domain.validate_vertex_count(vertex_count)


@dataclass(frozen=True)
class TongueProvenanceV8:
    """Provenance required before the V8 head compound may be published."""

    source: str
    license_id: str
    content_digest: str
    topology_digest: str

    def validate(self) -> None:
        if not str(self.source).strip():
            raise ValueError("tongue provenance requires a source")
        if not str(self.license_id).strip():
            raise ValueError("tongue provenance requires a license identifier")
        _validate_digest(self.content_digest, label="tongue.content_digest")
        _validate_digest(self.topology_digest, label="tongue.topology_digest")


def require_publishable_tongue_v8(
    provenance: TongueProvenanceV8 | None,
) -> TongueProvenanceV8:
    if provenance is None:
        raise ValueError(
            "V8 publish is blocked until a legally sourced tongue with "
            "content/topology provenance is supplied"
        )
    provenance.validate()
    return provenance


def _descendant_mask(names: Sequence[str], parents: Any, ancestor: str) -> np.ndarray:
    if ancestor not in names:
        raise ValueError(f"source rig is missing required bone {ancestor!r}")
    parent_array = np.asarray(parents, dtype=np.int64).reshape(-1)
    if parent_array.shape != (len(names),):
        raise ValueError("source_bone_parents must match source_bone_names")
    if np.any(parent_array < -1) or np.any(parent_array >= len(names)):
        raise ValueError("source_bone_parents contains an invalid index")
    root = list(names).index(ancestor)
    result = np.zeros(len(names), dtype=bool)
    for bone in range(len(names)):
        current = bone
        visited = 0
        while current >= 0:
            if current == root:
                result[bone] = True
                break
            current = int(parent_array[current])
            visited += 1
            if visited > len(names):
                raise ValueError("source bone hierarchy contains a cycle")
    return result


@dataclass(frozen=True)
class HeadCompoundSelectionV8:
    """Weight-derived ba9 head selection; hyoid is REST-only."""

    topology_digest: str
    cranial_mask: np.ndarray
    jaw_mask: np.ndarray
    rigid_attachment_mask: np.ndarray
    hyoid_rest_mask: np.ndarray
    tongue_provenance: TongueProvenanceV8 | None = None

    def __post_init__(self) -> None:
        topology = _validate_digest(self.topology_digest, label="head.topology_digest")
        masks = {}
        size: int | None = None
        for name in (
            "cranial_mask",
            "jaw_mask",
            "rigid_attachment_mask",
            "hyoid_rest_mask",
        ):
            mask = np.asarray(getattr(self, name), dtype=bool).reshape(-1)
            if size is None:
                size = len(mask)
            if len(mask) != size:
                raise ValueError("all head compound masks must have the same length")
            masks[name] = _readonly(mask, bool)
        if np.any(masks["cranial_mask"] & masks["jaw_mask"]):
            raise ValueError("cranial and jaw material masks must be disjoint")
        object.__setattr__(self, "topology_digest", topology)
        for name, mask in masks.items():
            object.__setattr__(self, name, mask)
        if self.tongue_provenance is not None:
            self.tongue_provenance.validate()

    @property
    def rest_transform_mask(self) -> np.ndarray:
        result = self.rigid_attachment_mask | self.hyoid_rest_mask
        result.setflags(write=False)
        return result

    @property
    def publishable(self) -> bool:
        try:
            require_publishable_tongue_v8(self.tongue_provenance)
        except ValueError:
            return False
        return True


def build_ba9_head_selection_v8(
    asset: Any,
    *,
    topology_digest: str,
    tongue_provenance: TongueProvenanceV8 | None = None,
    minimum_subtree_weight: float = 1.0 - 1.0e-6,
) -> HeadCompoundSelectionV8:
    """Reproduce ba9's weight-driven head/jaw attachment semantics.

    Whole rigid meshes must be authored exclusively to the Head_Bone subtree.
    Mixed vessels, nerves, and connective tissue remain on their Blender
    weights.  The hyoid receives the shared REST placement but its driver
    arrays are neither returned nor modified.
    """

    vertices = np.asarray(asset.vertices_rest)
    count = len(vertices)
    names = list(asset.source_bone_names or [])
    head_bones = _descendant_mask(names, asset.source_bone_parents, "Head_Bone")
    jaw_bones = (
        _descendant_mask(names, asset.source_bone_parents, "Jaw_Bone_tip")
        if "Jaw_Bone_tip" in names
        else np.zeros(len(names), dtype=bool)
    )
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    if (
        indices.ndim != 2
        or weights.shape != indices.shape
        or indices.shape[0] != count
        or np.any(indices < 0)
        or np.any(indices >= len(names))
    ):
        raise ValueError("head selection requires valid per-vertex source drivers")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError("source driver weights must sum to one")
    head_weight = np.sum(weights * head_bones[indices], axis=1)
    jaw_weight = np.sum(weights * jaw_bones[indices], axis=1)
    cranial = (head_weight - jaw_weight) >= 0.5
    jaw = jaw_weight >= 0.5

    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    mesh_names = list(asset.source_mesh_names)
    tissues = list(getattr(asset, "source_tissues", None) or [""] * len(mesh_names))
    if len(ranges) != len(mesh_names) or len(tissues) != len(mesh_names):
        raise ValueError("source mesh ranges/names/tissues are inconsistent")
    rigid = np.zeros(count, dtype=bool)
    hyoid = np.zeros(count, dtype=bool)
    excluded = {"vessel", "nerve", "connective_tissue"}
    for (start, stop), mesh_name, tissue in zip(ranges, mesh_names, tissues):
        lo, hi = int(start), int(stop)
        if lo < 0 or hi > count or hi <= lo:
            raise ValueError("source mesh vertex range is invalid")
        if "hyoid" in str(mesh_name).lower():
            hyoid[lo:hi] = True
        if str(tissue).strip().lower() in excluded:
            continue
        if bool(np.all(head_weight[lo:hi] >= float(minimum_subtree_weight))):
            rigid[lo:hi] = True
    return HeadCompoundSelectionV8(
        topology_digest=topology_digest,
        cranial_mask=cranial,
        jaw_mask=jaw,
        rigid_attachment_mask=rigid,
        hyoid_rest_mask=hyoid,
        tongue_provenance=tongue_provenance,
    )


@dataclass(frozen=True)
class UniformHeadTransformV8:
    """The sole similarity transform applied to the ba9 head compound."""

    source_origin: np.ndarray
    target_origin: np.ndarray
    rotation: np.ndarray
    uniform_scale: float

    def __post_init__(self) -> None:
        source = np.asarray(self.source_origin, dtype=np.float64).reshape(3)
        target = np.asarray(self.target_origin, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("head transform origins must be finite")
        rotation = _proper_rotation(self.rotation, label="head rotation")
        scale = float(self.uniform_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("head uniform_scale must be finite and positive")
        if math.isclose(scale, 0.70, abs_tol=1.0e-12, rel_tol=0.0):
            raise ValueError("V8 rejects the legacy head_scale=0.70 fast patch")
        object.__setattr__(self, "source_origin", _readonly(source, np.float64))
        object.__setattr__(self, "target_origin", _readonly(target, np.float64))
        object.__setattr__(self, "rotation", _readonly(rotation, np.float64))
        object.__setattr__(self, "uniform_scale", scale)

    def apply(self, points: Any) -> np.ndarray:
        vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return self.target_origin + self.uniform_scale * (
            (vertices - self.source_origin) @ self.rotation.T
        )


def apply_head_compound_rest_v8(
    vertices: Any,
    selection: HeadCompoundSelectionV8,
    transform: UniformHeadTransformV8,
) -> np.ndarray:
    """Apply one transform to rigid head attachments and hyoid REST placement."""

    result = np.asarray(vertices, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("vertices must be [N, 3]")
    mask = selection.rest_transform_mask
    if len(mask) != len(result):
        raise ValueError("head selection does not match vertex count")
    result[mask] = transform.apply(result[mask])
    return result


@dataclass(frozen=True)
class V71ParentLocalFKV8:
    """Complete V71 parent-local rig authority embedded in a V8 operator."""

    bone_names: tuple[str, ...]
    parents: np.ndarray
    rest_local: np.ndarray
    bone_head: np.ndarray
    bone_tail: np.ndarray
    bone_roll: np.ndarray
    bone_use_connect: np.ndarray
    bone_inherit_scale: np.ndarray
    driver_types: tuple[str, ...]
    driver_coupling: np.ndarray
    driver_indices: np.ndarray
    driver_weights: np.ndarray
    metadata: Mapping[str, Any]
    expected_bone_count: int = V71_SOURCE_BONE_COUNT

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.bone_names)
        count = len(names)
        if count != int(self.expected_bone_count):
            raise ValueError(
                f"V71 parent-local authority requires {self.expected_bone_count} "
                f"bones, found {count}"
            )
        if len(set(names)) != count or any(not name for name in names):
            raise ValueError("V71 bone names must be non-empty and unique")
        parents = np.asarray(self.parents, dtype=np.int64).reshape(-1)
        if parents.shape != (count,):
            raise ValueError("V71 parents must have one entry per bone")
        for bone, parent in enumerate(parents.tolist()):
            if parent >= bone:
                raise ValueError("V71 hierarchy must be parent-before-child")
            if parent < -1:
                raise ValueError("V71 parent index is invalid")
        local = np.asarray(self.rest_local, dtype=np.float64)
        coupling = np.asarray(self.driver_coupling, dtype=np.float64)
        if local.shape != (count, 4, 4) or coupling.shape != (count, 4, 4):
            raise ValueError("V71 local bind and driver coupling must be [B,4,4]")
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(coupling)):
            raise ValueError("V71 bind/coupling matrices must be finite")
        expected_bottom = np.broadcast_to((0.0, 0.0, 0.0, 1.0), (count, 4))
        if not np.allclose(local[:, 3], expected_bottom, atol=1.0e-7):
            raise ValueError("V71 local bind matrices must be affine SE(3)")
        if not np.allclose(coupling[:, 3], expected_bottom, atol=1.0e-7):
            raise ValueError("V71 driver coupling matrices must be affine SE(3)")
        shapes = {
            "bone_head": (self.bone_head, (count, 3)),
            "bone_tail": (self.bone_tail, (count, 3)),
            "bone_roll": (self.bone_roll, (count,)),
            "bone_use_connect": (self.bone_use_connect, (count,)),
            "bone_inherit_scale": (self.bone_inherit_scale, (count,)),
        }
        frozen = {}
        for field, (value, shape) in shapes.items():
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"V71 {field} must have shape {shape}")
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"V71 {field} must be finite")
            frozen[field] = _readonly(array)
        types = tuple(str(value) for value in self.driver_types)
        if len(types) != count or any(not value for value in types):
            raise ValueError("V71 driver_types must have one non-empty entry per bone")
        indices = np.asarray(self.driver_indices, dtype=np.int64)
        weights = np.asarray(self.driver_weights, dtype=np.float64)
        if (
            indices.ndim != 2
            or weights.shape != indices.shape
            or indices.shape[0] == 0
            or np.any(indices < 0)
            or np.any(indices >= count)
        ):
            raise ValueError("V71 sparse Armature drivers must be [N,K]")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("V71 Armature weights must be finite and non-negative")
        if indices.shape[1] > 14:
            raise ValueError("V71 runtime drivers may contain at most 14 slots")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5, rtol=0.0):
            raise ValueError("V71 Armature weights must sum to one")
        metadata = dict(self.metadata)
        validate_source_fk_policy_v8(
            metadata,
            bone_count=count,
            bone_names=names,
        )
        reject_obsolete_mechanism_config_v8(metadata)
        object.__setattr__(self, "bone_names", names)
        object.__setattr__(self, "parents", _readonly(parents, np.int32))
        object.__setattr__(self, "rest_local", _readonly(local, np.float64))
        object.__setattr__(self, "driver_coupling", _readonly(coupling, np.float64))
        object.__setattr__(self, "driver_indices", _readonly(indices, np.int32))
        object.__setattr__(self, "driver_weights", _readonly(weights, np.float32))
        object.__setattr__(self, "driver_types", types)
        object.__setattr__(self, "metadata", _deep_freeze_json(metadata))
        for field, value in frozen.items():
            object.__setattr__(self, field, value)

    def rest_global(self) -> np.ndarray:
        result = np.empty_like(self.rest_local)
        for bone, parent in enumerate(self.parents.tolist()):
            result[bone] = (
                self.rest_local[bone]
                if parent < 0
                else result[parent] @ self.rest_local[bone]
            )
        return result


@dataclass(frozen=True)
class WholeBoneRestFitV8:
    """One affine rest fit shared by every vertex of a bone compound."""

    source_head: np.ndarray
    target_head: np.ndarray
    linear: np.ndarray
    axial_scale: float
    radial_scales: tuple[float, float]

    def __post_init__(self) -> None:
        source = np.asarray(self.source_head, dtype=np.float64).reshape(3)
        target = np.asarray(self.target_head, dtype=np.float64).reshape(3)
        linear = np.asarray(self.linear, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("bone fit heads must be finite")
        if not np.all(np.isfinite(linear)) or abs(float(np.linalg.det(linear))) <= 1.0e-12:
            raise ValueError("bone fit linear transform must be finite and invertible")
        axial = float(self.axial_scale)
        radial = tuple(float(value) for value in self.radial_scales)
        if (
            not math.isfinite(axial)
            or axial <= 0.0
            or len(radial) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in radial)
        ):
            raise ValueError("bone scales must be finite and positive")
        object.__setattr__(self, "source_head", _readonly(source, np.float64))
        object.__setattr__(self, "target_head", _readonly(target, np.float64))
        object.__setattr__(self, "linear", _readonly(linear, np.float64))
        object.__setattr__(self, "axial_scale", axial)
        object.__setattr__(self, "radial_scales", radial)

    def apply(self, points: Any) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return self.target_head + (value - self.source_head) @ self.linear.T


@dataclass(frozen=True)
class CapPreservingAxialRestResultV810:
    """Result and analytic strain report for one axial rest adaptation."""

    vertices: np.ndarray
    displacement: np.ndarray
    axial_parameter: np.ndarray
    phi: np.ndarray
    profile_derivative: np.ndarray
    axial_jacobian: np.ndarray
    axial_strain: np.ndarray
    axis_direction: np.ndarray
    source_length_m: float
    requested_delta_m: float
    applied_delta_m: float
    remaining_residual_m: float
    segment: str
    proximal_cap_fraction: float
    distal_cap_fraction: float
    max_abs_axial_strain: float
    profile_peak_slope: float
    minimum_axial_jacobian: float
    maximum_axial_jacobian: float
    maximum_abs_applied_strain: float
    cross_section_scale: float = 1.0

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        displacement = np.asarray(
            self.displacement,
            dtype=np.float64,
        ).reshape(-1, 3)
        if vertices.shape != displacement.shape or not len(vertices):
            raise ValueError("axial rest result requires matching non-empty vertices")
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "axial_parameter",
            "phi",
            "profile_derivative",
            "axial_jacobian",
            "axial_strain",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if len(value) != len(vertices):
                raise ValueError(f"{name} must have one value per vertex")
            arrays[name] = value
        if (
            not np.all(np.isfinite(vertices))
            or not np.all(np.isfinite(displacement))
            or any(not np.all(np.isfinite(value)) for value in arrays.values())
        ):
            raise ValueError("axial rest result must be finite")
        if np.any(arrays["axial_parameter"] < 0.0) or np.any(
            arrays["axial_parameter"] > 1.0
        ):
            raise ValueError("axial parameters must lie in [0, 1]")
        if np.any(arrays["phi"] < 0.0) or np.any(arrays["phi"] > 1.0):
            raise ValueError("axial profile values must lie in [0, 1]")
        if np.any(arrays["profile_derivative"] < 0.0):
            raise ValueError("axial profile must be monotone")
        if not np.allclose(
            arrays["axial_jacobian"],
            1.0 + arrays["axial_strain"],
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("axial Jacobian and strain are inconsistent")

        direction = _unit(self.axis_direction, label="axial rest target axis")
        scalar_names = (
            "source_length_m",
            "requested_delta_m",
            "applied_delta_m",
            "remaining_residual_m",
            "proximal_cap_fraction",
            "distal_cap_fraction",
            "max_abs_axial_strain",
            "profile_peak_slope",
            "minimum_axial_jacobian",
            "maximum_axial_jacobian",
            "maximum_abs_applied_strain",
            "cross_section_scale",
        )
        scalars = {name: float(getattr(self, name)) for name in scalar_names}
        if any(not math.isfinite(value) for value in scalars.values()):
            raise ValueError("axial rest scalar report must be finite")
        if scalars["source_length_m"] <= 1.0e-8:
            raise ValueError("axial rest source length must be positive")
        if not math.isclose(
            scalars["requested_delta_m"],
            scalars["applied_delta_m"] + scalars["remaining_residual_m"],
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError("axial rest applied delta and residual are inconsistent")
        if scalars["minimum_axial_jacobian"] <= 0.0:
            raise ValueError("axial rest mapping must have a positive Jacobian")
        if (
            scalars["maximum_axial_jacobian"]
            < scalars["minimum_axial_jacobian"]
            or scalars["maximum_abs_applied_strain"] < 0.0
        ):
            raise ValueError("axial rest analytic strain report is invalid")
        if not math.isclose(
            scalars["cross_section_scale"],
            1.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError("axial rest cross-section scale must be one")

        segment = str(self.segment).strip().lower()
        if segment not in _AXIAL_STRAIN_LIMITS_V810:
            raise ValueError("axial rest segment must be 'femur' or 'shank'")
        object.__setattr__(self, "vertices", _readonly(vertices, np.float64))
        object.__setattr__(
            self,
            "displacement",
            _readonly(displacement, np.float64),
        )
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly(value, np.float64))
        object.__setattr__(
            self,
            "axis_direction",
            _readonly(direction, np.float64),
        )
        object.__setattr__(self, "segment", segment)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)


def _low_peak_axial_profile_v810(
    axial_parameter: np.ndarray,
    *,
    proximal_cap_fraction: float,
    distal_cap_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate a C2 profile with two truly rigid end caps.

    Each cap is followed by an equally wide smootherstep derivative ramp.  The
    remaining shaft uses the lowest constant derivative that integrates to one.
    Consequently every point in the proximal cap has ``phi=0`` and every point
    in the distal cap has ``phi=1``; only the shaft absorbs axial strain.
    """

    parameter = np.asarray(axial_parameter, dtype=np.float64).reshape(-1)
    proximal_fraction = float(proximal_cap_fraction)
    distal_fraction = float(distal_cap_fraction)
    if (
        not math.isfinite(proximal_fraction)
        or not math.isfinite(distal_fraction)
        or proximal_fraction <= 0.0
        or distal_fraction <= 0.0
        or 2.0 * (proximal_fraction + distal_fraction) >= 1.0
    ):
        raise ValueError(
            "axial cap fractions must be finite, positive, and leave a shaft "
            "between their C2 transition zones"
        )

    # Integral(smootherstep(t), t=0..1) = 1/2.  The two half ramps and
    # constant-slope shaft therefore have this effective unit length.
    effective_length = 1.0 - 1.5 * (
        proximal_fraction + distal_fraction
    )
    peak = 1.0 / effective_length
    phi = np.zeros_like(parameter)
    derivative = np.zeros_like(parameter)

    left_ramp_start = proximal_fraction
    left_ramp_stop = 2.0 * proximal_fraction
    right_ramp_start = 1.0 - 2.0 * distal_fraction
    right_ramp_stop = 1.0 - distal_fraction

    def smootherstep(value: np.ndarray) -> np.ndarray:
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)

    def smootherstep_integral(value: np.ndarray) -> np.ndarray:
        return value**6 - 3.0 * value**5 + 2.5 * value**4

    left_ramp = (parameter > left_ramp_start) & (
        parameter < left_ramp_stop
    )
    left_t = (
        parameter[left_ramp] - left_ramp_start
    ) / proximal_fraction
    phi[left_ramp] = (
        peak
        * proximal_fraction
        * smootherstep_integral(left_t)
    )
    derivative[left_ramp] = peak * smootherstep(left_t)

    core = (parameter >= left_ramp_stop) & (
        parameter <= right_ramp_start
    )
    left_ramp_area = 0.5 * proximal_fraction
    phi[core] = peak * (
        left_ramp_area + parameter[core] - left_ramp_stop
    )
    derivative[core] = peak

    right_ramp = (parameter > right_ramp_start) & (
        parameter < right_ramp_stop
    )
    right_t = (
        parameter[right_ramp] - right_ramp_start
    ) / distal_fraction
    core_length = right_ramp_start - left_ramp_stop
    right_integral = right_t - smootherstep_integral(right_t)
    phi[right_ramp] = peak * (
        left_ramp_area
        + core_length
        + distal_fraction * right_integral
    )
    derivative[right_ramp] = peak * (
        1.0 - smootherstep(right_t)
    )

    distal_cap = parameter >= right_ramp_stop
    phi[distal_cap] = 1.0
    phi = np.clip(phi, 0.0, 1.0)
    return phi, derivative, peak


def apply_cap_preserving_axial_rest_v810(
    vertices: Any,
    *,
    proximal: Any,
    distal: Any,
    target_length_delta_m: float,
    axial_parameter: Any,
    segment: str = "femur",
    proximal_cap_fraction: float = 0.10,
    distal_cap_fraction: float = 0.10,
    max_abs_axial_strain: float | None = None,
) -> CapPreservingAxialRestResultV810:
    """Apply a bounded axial delta without changing any cross-section scale.

    The default maximum absolute axial strain is 0.12 for ``femur`` and 0.08
    for ``shank``.  The requested length delta is clipped analytically against
    that limit, and the unapplied part is returned as ``remaining_residual_m``.
    """

    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    parameter = np.asarray(axial_parameter, dtype=np.float64).reshape(-1)
    if not len(points) or len(parameter) != len(points):
        raise ValueError("axial rest adapter requires one lambda per vertex")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(parameter)):
        raise ValueError("axial rest vertices and lambda must be finite")
    if np.any(parameter < 0.0) or np.any(parameter > 1.0):
        raise ValueError("axial rest lambda must lie in [0, 1]")

    proximal_point = np.asarray(proximal, dtype=np.float64).reshape(3)
    distal_point = np.asarray(distal, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(proximal_point)) or not np.all(
        np.isfinite(distal_point)
    ):
        raise ValueError("axial rest endpoints must be finite")
    axis = distal_point - proximal_point
    source_length = float(np.linalg.norm(axis))
    if source_length <= 1.0e-8:
        raise ValueError("axial rest endpoints must be non-degenerate")
    direction = axis / source_length

    requested_delta = float(target_length_delta_m)
    if not math.isfinite(requested_delta):
        raise ValueError("axial rest target length delta must be finite")
    selected_segment = str(segment).strip().lower()
    if selected_segment not in _AXIAL_STRAIN_LIMITS_V810:
        raise ValueError("axial rest segment must be 'femur' or 'shank'")
    phi, profile_derivative, peak_slope = _low_peak_axial_profile_v810(
        parameter,
        proximal_cap_fraction=proximal_cap_fraction,
        distal_cap_fraction=distal_cap_fraction,
    )

    strain_limit = (
        _AXIAL_STRAIN_LIMITS_V810[selected_segment]
        if max_abs_axial_strain is None
        else float(max_abs_axial_strain)
    )
    if (
        not math.isfinite(strain_limit)
        or strain_limit < 0.0
        or strain_limit >= 1.0
    ):
        raise ValueError(
            "max_abs_axial_strain must be finite and lie in [0, 1)"
        )
    delta_limit = strain_limit * source_length / peak_slope
    applied_delta = float(np.clip(requested_delta, -delta_limit, delta_limit))

    peak_signed_strain = applied_delta * peak_slope / source_length
    minimum_axial_jacobian = 1.0 + min(0.0, peak_signed_strain)
    maximum_axial_jacobian = 1.0 + max(0.0, peak_signed_strain)

    axial_strain = applied_delta * profile_derivative / source_length
    axial_jacobian = 1.0 + axial_strain
    displacement = applied_delta * phi[:, None] * direction[None, :]
    return CapPreservingAxialRestResultV810(
        vertices=points + displacement,
        displacement=displacement,
        axial_parameter=parameter,
        phi=phi,
        profile_derivative=profile_derivative,
        axial_jacobian=axial_jacobian,
        axial_strain=axial_strain,
        axis_direction=direction,
        source_length_m=source_length,
        requested_delta_m=requested_delta,
        applied_delta_m=applied_delta,
        remaining_residual_m=requested_delta - applied_delta,
        segment=selected_segment,
        proximal_cap_fraction=float(proximal_cap_fraction),
        distal_cap_fraction=float(distal_cap_fraction),
        max_abs_axial_strain=strain_limit,
        profile_peak_slope=peak_slope,
        minimum_axial_jacobian=minimum_axial_jacobian,
        maximum_axial_jacobian=maximum_axial_jacobian,
        maximum_abs_applied_strain=abs(peak_signed_strain),
        cross_section_scale=1.0,
    )


@dataclass(frozen=True)
class ProjectedStationRestFitV810:
    """One unit-scale rigid segment projected onto a driver direction."""

    anchor: str
    source_a: np.ndarray
    source_b: np.ndarray
    driver_a: np.ndarray
    driver_b: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    affine: np.ndarray
    source_length_m: float
    driver_length_m: float
    driver_length_residual_m: float
    free_endpoint_residual_m: np.ndarray
    free_endpoint_residual_norm_m: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        anchor = str(self.anchor).strip().lower()
        if anchor not in {"proximal", "distal"}:
            raise ValueError("projected station anchor must be 'proximal' or 'distal'")

        points: dict[str, np.ndarray] = {}
        for name in (
            "source_a",
            "source_b",
            "driver_a",
            "driver_b",
            "target_a",
            "target_b",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            points[name] = value

        rotation = _proper_rotation(
            self.rotation,
            label="projected station rotation",
        )
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(translation)):
            raise ValueError("projected station translation must be finite")
        affine = np.asarray(self.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError("projected station affine must be finite [4, 4]")
        if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
            raise ValueError("projected station affine has an invalid bottom row")
        expected_affine = np.eye(4, dtype=np.float64)
        expected_affine[:3, :3] = rotation
        expected_affine[:3, 3] = translation
        if not np.allclose(
            affine,
            expected_affine,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("projected station affine disagrees with R/t")

        source_length = float(
            np.linalg.norm(points["source_b"] - points["source_a"])
        )
        driver_length = float(
            np.linalg.norm(points["driver_b"] - points["driver_a"])
        )
        if source_length <= 1.0e-8 or driver_length <= 1.0e-8:
            raise ValueError("projected station endpoints must be non-degenerate")
        if not math.isclose(
            float(self.source_length_m),
            source_length,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        ):
            raise ValueError("projected station source length is inconsistent")
        if not math.isclose(
            float(self.driver_length_m),
            driver_length,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        ):
            raise ValueError("projected station driver length is inconsistent")
        length_residual = driver_length - source_length
        if not math.isclose(
            float(self.driver_length_residual_m),
            length_residual,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        ):
            raise ValueError("projected station length residual is inconsistent")
        if not math.isclose(
            float(self.scale),
            1.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError("projected station scale must be exactly one")

        source = np.stack((points["source_a"], points["source_b"]))
        mapped = source @ rotation.T + translation
        target = np.stack((points["target_a"], points["target_b"]))
        if not np.allclose(mapped, target, atol=1.0e-9, rtol=0.0):
            raise ValueError("projected station R/t does not map its endpoints")
        target_length = float(
            np.linalg.norm(points["target_b"] - points["target_a"])
        )
        if not math.isclose(
            target_length,
            source_length,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        ):
            raise ValueError("projected station changed anatomical length")

        if anchor == "proximal":
            anchor_error = points["target_a"] - points["driver_a"]
            expected_residual = points["target_b"] - points["driver_b"]
        else:
            anchor_error = points["target_b"] - points["driver_b"]
            expected_residual = points["target_a"] - points["driver_a"]
        if float(np.linalg.norm(anchor_error)) > 1.0e-10:
            raise ValueError("projected station did not preserve its anchor")
        residual = np.asarray(
            self.free_endpoint_residual_m,
            dtype=np.float64,
        ).reshape(3)
        if not np.allclose(
            residual,
            expected_residual,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("projected station free-endpoint residual is inconsistent")
        residual_norm = float(np.linalg.norm(residual))
        if not math.isclose(
            float(self.free_endpoint_residual_norm_m),
            residual_norm,
            abs_tol=1.0e-10,
            rel_tol=0.0,
        ):
            raise ValueError("projected station residual norm is inconsistent")

        object.__setattr__(self, "anchor", anchor)
        for name, value in points.items():
            object.__setattr__(self, name, _readonly(value, np.float64))
        object.__setattr__(self, "rotation", _readonly(rotation, np.float64))
        object.__setattr__(
            self,
            "translation",
            _readonly(translation, np.float64),
        )
        object.__setattr__(self, "affine", _readonly(affine, np.float64))
        object.__setattr__(
            self,
            "free_endpoint_residual_m",
            _readonly(residual, np.float64),
        )
        object.__setattr__(self, "source_length_m", source_length)
        object.__setattr__(self, "driver_length_m", driver_length)
        object.__setattr__(
            self,
            "driver_length_residual_m",
            length_residual,
        )
        object.__setattr__(
            self,
            "free_endpoint_residual_norm_m",
            residual_norm,
        )
        object.__setattr__(self, "scale", 1.0)

    def apply(self, points: Any) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(value)):
            raise ValueError("projected station points must be finite")
        return value @ self.rotation.T + self.translation


def fit_projected_station_rest_v810(
    source_a: Any,
    source_b: Any,
    driver_a: Any,
    driver_b: Any,
    *,
    anchor: str,
) -> ProjectedStationRestFitV810:
    """Project one anatomical segment onto a driver ray without scaling it."""

    source_start = np.asarray(source_a, dtype=np.float64).reshape(3)
    source_end = np.asarray(source_b, dtype=np.float64).reshape(3)
    driver_start = np.asarray(driver_a, dtype=np.float64).reshape(3)
    driver_end = np.asarray(driver_b, dtype=np.float64).reshape(3)
    for name, value in (
        ("source_a", source_start),
        ("source_b", source_end),
        ("driver_a", driver_start),
        ("driver_b", driver_end),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    source_axis = source_end - source_start
    driver_axis = driver_end - driver_start
    source_length = float(np.linalg.norm(source_axis))
    driver_length = float(np.linalg.norm(driver_axis))
    if source_length <= 1.0e-8 or driver_length <= 1.0e-8:
        raise ValueError("projected station endpoints must be non-degenerate")
    driver_direction = driver_axis / driver_length
    rotation = _rotation_from_to(source_axis, driver_axis)

    selected_anchor = str(anchor).strip().lower()
    if selected_anchor == "proximal":
        target_start = driver_start.copy()
        target_end = target_start + source_length * driver_direction
        translation = target_start - rotation @ source_start
        free_residual = target_end - driver_end
    elif selected_anchor == "distal":
        target_end = driver_end.copy()
        target_start = target_end - source_length * driver_direction
        translation = target_end - rotation @ source_end
        free_residual = target_start - driver_start
    else:
        raise ValueError("projected station anchor must be 'proximal' or 'distal'")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation
    affine[:3, 3] = translation
    return ProjectedStationRestFitV810(
        anchor=selected_anchor,
        source_a=source_start,
        source_b=source_end,
        driver_a=driver_start,
        driver_b=driver_end,
        target_a=target_start,
        target_b=target_end,
        rotation=rotation,
        translation=translation,
        affine=affine,
        source_length_m=source_length,
        driver_length_m=driver_length,
        driver_length_residual_m=driver_length - source_length,
        free_endpoint_residual_m=free_residual,
        free_endpoint_residual_norm_m=float(np.linalg.norm(free_residual)),
        scale=1.0,
    )


def fit_whole_bone_rest_v8(
    *,
    source_head: Any,
    source_tail: Any,
    target_head: Any,
    target_tail: Any,
    radial_scales: tuple[float, float] = (1.0, 1.0),
) -> WholeBoneRestFitV8:
    """Fit an entire bone with one axial/bi-radial transform.

    There is deliberately no station profile or endpoint parameter.  Articular
    ends, shaft, and any attached validation domain therefore receive exactly
    the same longitudinal/radial scaling law.
    """

    source_a = np.asarray(source_head, dtype=np.float64).reshape(3)
    source_b = np.asarray(source_tail, dtype=np.float64).reshape(3)
    target_a = np.asarray(target_head, dtype=np.float64).reshape(3)
    target_b = np.asarray(target_tail, dtype=np.float64).reshape(3)
    source_axis = source_b - source_a
    target_axis = target_b - target_a
    source_length = float(np.linalg.norm(source_axis))
    target_length = float(np.linalg.norm(target_axis))
    if source_length <= 1.0e-8 or target_length <= 1.0e-8:
        raise ValueError("whole-bone fit requires non-zero source and target lengths")
    radial = tuple(float(value) for value in radial_scales)
    if len(radial) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in radial
    ):
        raise ValueError("radial_scales must contain two finite positive values")
    source_frame = _frame_from_axis(source_axis)
    alignment = _rotation_from_to(source_axis, target_axis)
    target_frame = alignment @ source_frame
    axial = target_length / source_length
    local_scale = np.diag((radial[0], axial, radial[1]))
    linear = target_frame @ local_scale @ source_frame.T
    return WholeBoneRestFitV8(
        source_head=source_a,
        target_head=target_a,
        linear=linear,
        axial_scale=axial,
        radial_scales=radial,
    )


@dataclass(frozen=True)
class StationThicknessMetricsV8:
    station_centers: np.ndarray
    reference_thickness: np.ndarray
    candidate_thickness: np.ndarray
    thickness_ratios: np.ndarray
    rms_relative_error: float
    max_relative_error: float
    max_adjacent_ratio_change: float

    def __post_init__(self) -> None:
        for name in (
            "station_centers",
            "reference_thickness",
            "candidate_thickness",
            "thickness_ratios",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float64))


def station_thickness_metrics_v8(
    *,
    reference_vertices: Any,
    candidate_vertices: Any,
    reference_head: Any,
    reference_tail: Any,
    candidate_head: Any,
    candidate_tail: Any,
    station_count: int = 5,
    quantile: float = 0.95,
) -> StationThicknessMetricsV8:
    """Measure paired bi-radial thickness and taper at fixed material stations."""

    reference = np.asarray(reference_vertices, dtype=np.float64).reshape(-1, 3)
    candidate = np.asarray(candidate_vertices, dtype=np.float64).reshape(-1, 3)
    if reference.shape != candidate.shape or len(reference) < int(station_count) * 4:
        raise ValueError("paired station metrics require enough corresponding vertices")
    count = int(station_count)
    if count < 5:
        raise ValueError("V8 bone thickness requires at least five axial stations")
    q = float(quantile)
    if not 0.5 < q <= 1.0:
        raise ValueError("quantile must be in (0.5, 1.0]")
    ref_head = np.asarray(reference_head, dtype=np.float64).reshape(3)
    ref_tail = np.asarray(reference_tail, dtype=np.float64).reshape(3)
    cand_head = np.asarray(candidate_head, dtype=np.float64).reshape(3)
    cand_tail = np.asarray(candidate_tail, dtype=np.float64).reshape(3)
    ref_axis = ref_tail - ref_head
    cand_axis = cand_tail - cand_head
    ref_length = float(np.linalg.norm(ref_axis))
    cand_length = float(np.linalg.norm(cand_axis))
    if ref_length <= 1.0e-8 or cand_length <= 1.0e-8:
        raise ValueError("station axes must be non-zero")
    ref_frame = _frame_from_axis(ref_axis)
    # Parallel-transport the reference radial axes onto the candidate axis.
    # Constructing an unrelated frame from the candidate's world direction can
    # introduce a false 90-degree radial swap when it crosses a seed threshold.
    cand_frame = _rotation_from_to(ref_axis, cand_axis) @ ref_frame
    parameter = np.clip(
        ((reference - ref_head) @ ref_frame[:, 1]) / ref_length, 0.0, 1.0
    )
    boundaries = np.linspace(0.0, 1.0, count + 1)
    reference_size = np.empty((count, 2), dtype=np.float64)
    candidate_size = np.empty((count, 2), dtype=np.float64)
    for station in range(count):
        if station + 1 == count:
            selected = (parameter >= boundaries[station]) & (parameter <= 1.0)
        else:
            selected = (parameter >= boundaries[station]) & (
                parameter < boundaries[station + 1]
            )
        if int(np.count_nonzero(selected)) < 4:
            raise ValueError(f"station {station} has fewer than four vertices")
        ref_local = reference[selected] @ ref_frame[:, (0, 2)]
        cand_local = candidate[selected] @ cand_frame[:, (0, 2)]
        ref_local -= np.median(ref_local, axis=0)
        cand_local -= np.median(cand_local, axis=0)
        reference_size[station] = 2.0 * np.quantile(np.abs(ref_local), q, axis=0)
        candidate_size[station] = 2.0 * np.quantile(np.abs(cand_local), q, axis=0)
    if np.any(reference_size <= 1.0e-10):
        raise ValueError("reference station thickness is degenerate")
    ratios = candidate_size / reference_size
    relative = ratios - 1.0
    adjacent = (
        float(np.max(np.abs(np.diff(ratios, axis=0)))) if count > 1 else 0.0
    )
    return StationThicknessMetricsV8(
        station_centers=0.5 * (boundaries[:-1] + boundaries[1:]),
        reference_thickness=reference_size,
        candidate_thickness=candidate_size,
        thickness_ratios=ratios,
        rms_relative_error=float(np.sqrt(np.mean(relative**2))),
        max_relative_error=float(np.max(np.abs(relative))),
        max_adjacent_ratio_change=adjacent,
    )


def rotation_about_pivot_v8(rotation: Any, pivot: Any) -> np.ndarray:
    """Homogeneous transform that leaves ``pivot`` exactly invariant."""

    matrix = _proper_rotation(rotation, label="pivot rotation")
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = matrix
    transform[:3, 3] = center - matrix @ center
    return transform


@dataclass(frozen=True)
class CoupledLimbPoseV8:
    """Result of one parent-local, fixed-length two-segment solve."""

    proximal_pivot: np.ndarray
    joint_pivot: np.ndarray
    distal_endpoint: np.ndarray
    hinge_axis_world: np.ndarray
    proximal_rotation: np.ndarray
    distal_rotation: np.ndarray
    proximal_transform: np.ndarray
    distal_transform: np.ndarray
    bind_flexion_deg: float
    relative_flexion_deg: float
    axial_twist_deg: float
    proximal_length: float
    distal_length: float

    def __post_init__(self) -> None:
        for name in (
            "proximal_pivot",
            "joint_pivot",
            "distal_endpoint",
            "hinge_axis_world",
            "proximal_rotation",
            "distal_rotation",
            "proximal_transform",
            "distal_transform",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float64))


def coupled_fixed_length_limb_v8(
    *,
    bind_proximal_pivot: Any,
    bind_joint_pivot: Any,
    bind_distal_endpoint: Any,
    posed_proximal_pivot: Any,
    proximal_rotation: Any,
    hinge_axis_proximal_local: Any,
    flexion_deg: float,
    bind_flexion_deg: float = 0.0,
    axial_twist_deg: float = 0.0,
) -> CoupledLimbPoseV8:
    """Compose a child from its parent while preserving both segment lengths.

    The API deliberately accepts no posed/global child or distal target.
    Consequently a caller cannot reintroduce the V7 double-drive by pulling the
    knee/elbow or ankle/wrist to an independent global SMPL-X anchor.
    """

    bind_a = np.asarray(bind_proximal_pivot, dtype=np.float64).reshape(3)
    bind_b = np.asarray(bind_joint_pivot, dtype=np.float64).reshape(3)
    bind_c = np.asarray(bind_distal_endpoint, dtype=np.float64).reshape(3)
    posed_a = np.asarray(posed_proximal_pivot, dtype=np.float64).reshape(3)
    rotation = _proper_rotation(proximal_rotation, label="proximal rotation")
    hinge_local = _unit(
        hinge_axis_proximal_local, label="hinge_axis_proximal_local"
    )
    values = np.concatenate((bind_a, bind_b, bind_c, posed_a))
    if not np.all(np.isfinite(values)):
        raise ValueError("limb pivots must be finite")
    proximal_vector = bind_b - bind_a
    distal_vector = bind_c - bind_b
    proximal_length = float(np.linalg.norm(proximal_vector))
    distal_length = float(np.linalg.norm(distal_vector))
    if proximal_length <= 1.0e-8 or distal_length <= 1.0e-8:
        raise ValueError("limb bind segments must have non-zero length")
    flexion = float(flexion_deg)
    bind_flexion = float(bind_flexion_deg)
    twist = float(axial_twist_deg)
    if not all(math.isfinite(value) for value in (flexion, bind_flexion, twist)):
        raise ValueError("limb angles must be finite")
    relative_deg = flexion - bind_flexion
    hinge_world = rotation @ hinge_local
    hinge_rotation = axis_rotation_v8(hinge_world, math.radians(relative_deg))
    joint = posed_a + rotation @ proximal_vector
    posed_distal_vector = hinge_rotation @ (rotation @ distal_vector)
    distal = joint + posed_distal_vector
    shank_axis = _unit(posed_distal_vector, label="posed distal segment")
    twist_rotation = axis_rotation_v8(shank_axis, math.radians(twist))
    distal_rotation = twist_rotation @ hinge_rotation @ rotation

    proximal_transform = np.eye(4, dtype=np.float64)
    proximal_transform[:3, :3] = rotation
    proximal_transform[:3, 3] = posed_a - rotation @ bind_a
    distal_transform = np.eye(4, dtype=np.float64)
    distal_transform[:3, :3] = distal_rotation
    distal_transform[:3, 3] = joint - distal_rotation @ bind_b
    # Twisting around the posed segment must not move its endpoint.
    mapped_distal = distal_transform[:3, :3] @ bind_c + distal_transform[:3, 3]
    if not np.allclose(mapped_distal, distal, atol=1.0e-9, rtol=0.0):
        raise RuntimeError("internal V8 limb composition failed fixed-endpoint invariant")
    return CoupledLimbPoseV8(
        proximal_pivot=posed_a,
        joint_pivot=joint,
        distal_endpoint=distal,
        hinge_axis_world=hinge_world,
        proximal_rotation=rotation,
        distal_rotation=distal_rotation,
        proximal_transform=proximal_transform,
        distal_transform=distal_transform,
        bind_flexion_deg=bind_flexion,
        relative_flexion_deg=relative_deg,
        axial_twist_deg=twist,
        proximal_length=proximal_length,
        distal_length=distal_length,
    )


def _smooth_bounded_twist(
    relative_flexion_deg: float,
    *,
    gain: float,
    maximum_abs_deg: float,
) -> float:
    gain_value = float(gain)
    maximum = float(maximum_abs_deg)
    if not math.isfinite(gain_value) or not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("twist gain/cap must be finite and cap non-negative")
    raw = gain_value * float(relative_flexion_deg)
    if maximum == 0.0:
        return 0.0
    return maximum * math.tanh(raw / maximum)


def pose_hip_common_pivot_v8(
    points: Any,
    *,
    common_pivot: Any,
    femur_rotation: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the complete femur/head compound around one hip/socket pivot."""

    transform = rotation_about_pivot_v8(femur_rotation, common_pivot)
    vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    posed = vertices @ transform[:3, :3].T + transform[:3, 3]
    return posed, transform


def pose_knee_v8(
    *,
    bind_hip: Any,
    bind_knee: Any,
    bind_ankle: Any,
    posed_hip: Any,
    femur_rotation: Any,
    hinge_axis_femur_local: Any,
    flexion_deg: float,
    bind_flexion_deg: float,
    screw_home_gain: float = 0.0,
    maximum_screw_home_deg: float = 10.0,
) -> CoupledLimbPoseV8:
    """Fixed-length hip→knee→ankle chain with continuous relative-bind twist."""

    relative = float(flexion_deg) - float(bind_flexion_deg)
    twist = _smooth_bounded_twist(
        relative,
        gain=screw_home_gain,
        maximum_abs_deg=maximum_screw_home_deg,
    )
    return coupled_fixed_length_limb_v8(
        bind_proximal_pivot=bind_hip,
        bind_joint_pivot=bind_knee,
        bind_distal_endpoint=bind_ankle,
        posed_proximal_pivot=posed_hip,
        proximal_rotation=femur_rotation,
        hinge_axis_proximal_local=hinge_axis_femur_local,
        flexion_deg=flexion_deg,
        bind_flexion_deg=bind_flexion_deg,
        axial_twist_deg=twist,
    )


def pose_elbow_v8(
    *,
    bind_shoulder: Any,
    bind_elbow: Any,
    bind_wrist: Any,
    posed_shoulder: Any,
    humerus_rotation: Any,
    hinge_axis_humerus_local: Any,
    flexion_deg: float,
    bind_flexion_deg: float = 0.0,
    forearm_twist_deg: float = 0.0,
    bind_forearm_twist_deg: float = 0.0,
) -> CoupledLimbPoseV8:
    """Parent-local elbow flexion with an independent authored forearm twist."""

    return coupled_fixed_length_limb_v8(
        bind_proximal_pivot=bind_shoulder,
        bind_joint_pivot=bind_elbow,
        bind_distal_endpoint=bind_wrist,
        posed_proximal_pivot=posed_shoulder,
        proximal_rotation=humerus_rotation,
        hinge_axis_proximal_local=hinge_axis_humerus_local,
        flexion_deg=flexion_deg,
        bind_flexion_deg=bind_flexion_deg,
        axial_twist_deg=float(forearm_twist_deg) - float(bind_forearm_twist_deg),
    )


def topology_digest_v8(vertices: Any, faces: Any) -> str:
    """Digest topology identity without making posed/rest coordinates an oracle."""

    vertex_count = int(np.asarray(vertices).reshape(-1, 3).shape[0])
    triangles = np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3))
    digest = hashlib.sha256(b"anatomy-v8-topology-v1\0")
    digest.update(np.asarray((vertex_count,), dtype=np.int64).tobytes())
    digest.update(np.asarray(triangles.shape, dtype=np.int64).tobytes())
    digest.update(triangles.tobytes())
    return digest.hexdigest()


__all__ = [
    "ANATOMY_V8_SCHEMA_VERSION",
    "V71_SOURCE_BONE_COUNT",
    "CapPreservingAxialRestResultV810",
    "CoupledLimbPoseV8",
    "FrozenMaterialDomainV8",
    "FrozenMaterialDomainsV8",
    "HeadCompoundSelectionV8",
    "ProjectedStationRestFitV810",
    "StationThicknessMetricsV8",
    "TongueProvenanceV8",
    "UniformHeadTransformV8",
    "V71ParentLocalFKV8",
    "WholeBoneRestFitV8",
    "apply_cap_preserving_axial_rest_v810",
    "apply_head_compound_rest_v8",
    "axis_rotation_v8",
    "build_ba9_head_selection_v8",
    "coupled_fixed_length_limb_v8",
    "fit_projected_station_rest_v810",
    "fit_whole_bone_rest_v8",
    "pose_elbow_v8",
    "pose_hip_common_pivot_v8",
    "pose_knee_v8",
    "reject_obsolete_mechanism_config_v8",
    "require_publishable_tongue_v8",
    "rotation_about_pivot_v8",
    "station_thickness_metrics_v8",
    "strip_v7_leg_oracle_metadata_v8",
    "topology_digest_v8",
]
