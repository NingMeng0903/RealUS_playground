"""Frozen pose-space tube corrections evaluated after authoritative LBS.

The V8 tube coupling path deliberately owns the vessel/nerve vertices through
the original Blender fourteen-slot matrix LBS.  This module adds a small,
fully-baked local displacement after that result.  It never asks a surface,
spatial index, graph, or solver for pose-time information: runtime consists of
Gaussian RBF evaluation, dense small-matrix products, and one sparse add.

``local_displacement_basis`` has shape ``[V, 3, K]``.  For a pose, the RBF
evaluates ``K`` component coefficients and the local correction is
``basis[v] @ coefficients``.  Each correction vector is carried through the
same *linear* fourteen-slot LBS blend as its tube vertex before it is added to
the already posed tube result.  Translation is intentionally excluded from
that vector transform.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TUBE_POSE_CORRECTIVE_SCHEMA_V1 = 1
TUBE_POSE_CORRECTIVE_ARTIFACT_KIND_V1 = "TubePoseCorrectivePackV1"
TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1 = "tube_pose_corrective_v1."
TUBE_POSE_CORRECTIVE_INFLUENCE_SLOTS_V1 = 14
SMPLX_JOINT_COUNT_V1 = 55

_RUNTIME_FIELD_NAMES_V1 = frozenset(
    (
        "schema_version",
        "artifact_kind",
        "vertex_ids",
        "local_displacement_basis",
        "driver_joint_ids",
        "rbf_centers",
        "rbf_widths",
        "center_coefficients",
        "maximum_displacement_m",
        "content_digest",
    )
)


def _readonly_copy(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _sha256_arrays_v1(namespace: bytes, *values: np.ndarray) -> str:
    digest = hashlib.sha256(namespace + b"\0")
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _ascii_u8(value: str) -> np.ndarray:
    try:
        encoded = str(value).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("tube pose corrective runtime metadata must be ASCII") from exc
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def _decode_ascii_u8(
    fields: Mapping[str, Any], name: str, *, length: int | None = None
) -> str:
    key = f"{TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1}{name}"
    array = np.asarray(fields[key])
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional uint8 array")
    if length is not None and len(array) != int(length):
        raise ValueError(f"{key} has an invalid encoded length")
    try:
        return array.tobytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{key} is not valid ASCII") from exc


def _runtime_array_v1(
    fields: Mapping[str, Any], name: str, *, dtype: Any, ndim: int
) -> np.ndarray:
    key = f"{TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1}{name}"
    array = np.asarray(fields[key])
    expected = np.dtype(dtype)
    if array.dtype != expected or array.ndim != int(ndim):
        raise ValueError(f"{key} must have dtype {expected} and ndim {int(ndim)}")
    return array


def _is_lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class TubePoseCorrectivePackV1:
    """Mmap-friendly numeric data for one sparse tube pose corrective.

    Arrays use stable dtypes so they can be placed directly in an NPZ/NPY
    subject artifact and memory-mapped after materialization.  ``vertex_ids``
    reference the global posed vertex buffer; every other per-vertex array is
    in that same ordered sparse domain.
    """

    vertex_ids: np.ndarray
    local_displacement_basis: np.ndarray
    driver_joint_ids: np.ndarray
    rbf_centers: np.ndarray
    rbf_widths: np.ndarray
    center_coefficients: np.ndarray
    maximum_displacement_m: float

    def __post_init__(self) -> None:
        for name, dtype in (
            ("vertex_ids", np.int32),
            ("local_displacement_basis", np.float32),
            ("driver_joint_ids", np.int16),
            ("rbf_centers", np.float32),
            ("rbf_widths", np.float32),
            ("center_coefficients", np.float32),
        ):
            object.__setattr__(self, name, _readonly_copy(getattr(self, name), dtype))
        maximum = np.asarray(self.maximum_displacement_m, dtype=np.float32)
        if maximum.ndim != 0:
            raise ValueError("tube pose corrective maximum_displacement_m must be scalar")
        object.__setattr__(
            self, "maximum_displacement_m", float(maximum)
        )
        self.validate()

    @property
    def vertex_count(self) -> int:
        """Number of sparse tube vertices affected by this corrective."""
        return int(len(self.vertex_ids))

    @property
    def component_count(self) -> int:
        """Number of PCA components in ``local_displacement_basis``."""
        return int(self.local_displacement_basis.shape[2])

    @property
    def center_count(self) -> int:
        """Number of Gaussian RBF centers."""
        return int(len(self.rbf_widths))

    def validate(self) -> None:
        """Check the runtime-safe array schema and numerical constraints."""
        vertex_ids = np.asarray(self.vertex_ids, dtype=np.int64).reshape(-1)
        basis = np.asarray(self.local_displacement_basis)
        joint_ids = np.asarray(self.driver_joint_ids, dtype=np.int64).reshape(-1)
        centers = np.asarray(self.rbf_centers)
        widths = np.asarray(self.rbf_widths).reshape(-1)
        coefficients = np.asarray(self.center_coefficients)

        if (
            not len(vertex_ids)
            or np.any(vertex_ids < 0)
            or not np.array_equal(vertex_ids, np.unique(vertex_ids))
        ):
            raise ValueError("tube pose corrective vertex_ids must be unique and ordered")
        if (
            basis.ndim != 3
            or basis.shape[0] != len(vertex_ids)
            or basis.shape[1] != 3
            or basis.shape[2] <= 0
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError(
                "tube pose corrective local_displacement_basis must be finite [V,3,K]"
            )
        if (
            not len(joint_ids)
            or np.any(joint_ids < 0)
            or np.any(joint_ids >= SMPLX_JOINT_COUNT_V1)
            or not np.array_equal(joint_ids, np.unique(joint_ids))
        ):
            raise ValueError(
                "tube pose corrective driver_joint_ids must be unique SMPL-X joint ids"
            )
        if (
            centers.ndim != 3
            or centers.shape[0] <= 0
            or centers.shape[1:] != (len(joint_ids), 3)
            or not np.all(np.isfinite(centers))
        ):
            raise ValueError("tube pose corrective rbf_centers must be finite [C,D,3]")
        if (
            widths.shape != (centers.shape[0],)
            or np.any(widths <= 0.0)
            or not np.all(np.isfinite(widths))
        ):
            raise ValueError("tube pose corrective rbf_widths must be finite positive [C]")
        if (
            coefficients.shape != (centers.shape[0], basis.shape[2])
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("tube pose corrective center_coefficients must be finite [C,K]")
        if not np.isfinite(self.maximum_displacement_m) or self.maximum_displacement_m < 0.0:
            raise ValueError(
                "tube pose corrective maximum_displacement_m must be finite and non-negative"
            )

    def content_digest(self) -> str:
        """Return the authenticated digest for every behavior-affecting field."""
        return _sha256_arrays_v1(
            b"anatomy-tube-pose-corrective-v1",
            np.asarray((TUBE_POSE_CORRECTIVE_SCHEMA_V1,), dtype=np.int32),
            np.asarray((self.maximum_displacement_m,), dtype=np.float32),
            self.vertex_ids,
            self.local_displacement_basis,
            self.driver_joint_ids,
            self.rbf_centers,
            self.rbf_widths,
            self.center_coefficients,
        )


def tube_pose_corrective_pack_to_runtime_fields_v1(
    pack: TubePoseCorrectivePackV1,
) -> dict[str, np.ndarray]:
    """Flatten a corrective into typed, digest-authenticated runtime fields."""
    pack.validate()
    prefix = TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1
    return {
        f"{prefix}schema_version": np.asarray(
            TUBE_POSE_CORRECTIVE_SCHEMA_V1, dtype=np.int32
        ),
        f"{prefix}artifact_kind": _ascii_u8(TUBE_POSE_CORRECTIVE_ARTIFACT_KIND_V1),
        f"{prefix}vertex_ids": np.asarray(pack.vertex_ids, dtype=np.int32).copy(),
        f"{prefix}local_displacement_basis": np.asarray(
            pack.local_displacement_basis, dtype=np.float32
        ).copy(),
        f"{prefix}driver_joint_ids": np.asarray(
            pack.driver_joint_ids, dtype=np.int16
        ).copy(),
        f"{prefix}rbf_centers": np.asarray(pack.rbf_centers, dtype=np.float32).copy(),
        f"{prefix}rbf_widths": np.asarray(pack.rbf_widths, dtype=np.float32).copy(),
        f"{prefix}center_coefficients": np.asarray(
            pack.center_coefficients, dtype=np.float32
        ).copy(),
        f"{prefix}maximum_displacement_m": np.asarray(
            pack.maximum_displacement_m, dtype=np.float32
        ),
        f"{prefix}content_digest": _ascii_u8(pack.content_digest()),
    }


def has_tube_pose_corrective_v1(runtime_fields: Mapping[str, Any] | None) -> bool:
    """Return whether a mapping contains any V1 pose-corrective field."""
    return bool(
        runtime_fields
        and any(
            str(name).startswith(TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1)
            for name in runtime_fields
        )
    )


def tube_pose_corrective_pack_from_runtime_fields_v1(
    fields: Mapping[str, Any],
) -> TubePoseCorrectivePackV1:
    """Restore a pack and fail closed for missing, unknown, or altered fields."""
    if not isinstance(fields, Mapping):
        raise ValueError("tube pose corrective runtime fields must be a mapping")
    prefix = TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1
    present = {
        str(key)[len(prefix) :] for key in fields if str(key).startswith(prefix)
    }
    missing = sorted(_RUNTIME_FIELD_NAMES_V1 - present)
    unknown = sorted(present - _RUNTIME_FIELD_NAMES_V1)
    if missing:
        raise ValueError(
            f"tube pose corrective runtime fields missing required fields: {missing}"
        )
    if unknown:
        raise ValueError(
            f"tube pose corrective runtime fields contain unknown fields: {unknown}"
        )
    schema = _runtime_array_v1(fields, "schema_version", dtype=np.int32, ndim=0)
    if int(schema) != TUBE_POSE_CORRECTIVE_SCHEMA_V1:
        raise ValueError("tube pose corrective runtime fields require schema_version 1")
    if (
        _decode_ascii_u8(fields, "artifact_kind")
        != TUBE_POSE_CORRECTIVE_ARTIFACT_KIND_V1
    ):
        raise ValueError("invalid tube pose corrective runtime artifact kind")
    pack = TubePoseCorrectivePackV1(
        vertex_ids=_runtime_array_v1(fields, "vertex_ids", dtype=np.int32, ndim=1),
        local_displacement_basis=_runtime_array_v1(
            fields, "local_displacement_basis", dtype=np.float32, ndim=3
        ),
        driver_joint_ids=_runtime_array_v1(
            fields, "driver_joint_ids", dtype=np.int16, ndim=1
        ),
        rbf_centers=_runtime_array_v1(fields, "rbf_centers", dtype=np.float32, ndim=3),
        rbf_widths=_runtime_array_v1(fields, "rbf_widths", dtype=np.float32, ndim=1),
        center_coefficients=_runtime_array_v1(
            fields, "center_coefficients", dtype=np.float32, ndim=2
        ),
        maximum_displacement_m=float(
            _runtime_array_v1(
                fields, "maximum_displacement_m", dtype=np.float32, ndim=0
            )
        ),
    )
    expected = _decode_ascii_u8(fields, "content_digest", length=64)
    if not _is_lower_sha256(expected):
        raise ValueError("tube pose corrective runtime content_digest is invalid")
    if pack.content_digest() != expected:
        raise ValueError("tube pose corrective runtime content digest mismatch")
    return pack


def _pose55_v1(pose_axis_angle: Any) -> np.ndarray:
    pose = np.asarray(pose_axis_angle, dtype=np.float64)
    if pose.size != SMPLX_JOINT_COUNT_V1 * 3:
        raise ValueError("tube pose corrective pose must contain exactly 55 axis-angle joints")
    pose = pose.reshape(SMPLX_JOINT_COUNT_V1, 3)
    if not np.all(np.isfinite(pose)):
        raise ValueError("tube pose corrective pose contains non-finite values")
    return pose


def _gaussian_rbf_values_v1(
    features: np.ndarray, centers: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    """Evaluate unnormalized Gaussian RBFs with one width per center."""
    delta = np.asarray(centers, dtype=np.float64) - np.asarray(
        features, dtype=np.float64
    )[None, ...]
    squared_distance = np.einsum("c...,c...->c", delta, delta, optimize=True)
    return np.exp(
        -0.5 * squared_distance / np.square(np.asarray(widths, dtype=np.float64))
    )


def _gaussian_rbf_matrix_v1(
    features: np.ndarray, centers: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    """Evaluate a sample-by-center Gaussian RBF design matrix."""
    sample = np.asarray(features, dtype=np.float64)
    center = np.asarray(centers, dtype=np.float64)
    delta = center[None, ...] - sample[:, None, ...]
    squared_distance = np.sum(delta * delta, axis=tuple(range(2, delta.ndim)))
    return np.exp(
        -0.5
        * squared_distance
        / np.square(np.asarray(widths, dtype=np.float64))[None, :]
    )


def evaluate_tube_pose_corrective_coefficients_v1(
    pack: TubePoseCorrectivePackV1,
    pose_axis_angle: Any,
    *,
    validate_pack: bool = True,
) -> np.ndarray:
    """Evaluate the compact PCA coefficient vector for one SMPL-X pose."""
    if validate_pack:
        pack.validate()
    pose = _pose55_v1(pose_axis_angle)
    feature = pose[np.asarray(pack.driver_joint_ids, dtype=np.int64)]
    values = _gaussian_rbf_values_v1(feature, pack.rbf_centers, pack.rbf_widths)
    coefficients = values @ np.asarray(pack.center_coefficients, dtype=np.float64)
    return np.asarray(coefficients, dtype=np.float32)


def evaluate_tube_pose_corrective_local_v1(
    pack: TubePoseCorrectivePackV1,
    pose_axis_angle: Any,
    *,
    validate_pack: bool = True,
) -> np.ndarray:
    """Evaluate bounded rest-local corrections for the sparse tube domain."""
    coefficients = evaluate_tube_pose_corrective_coefficients_v1(
        pack, pose_axis_angle, validate_pack=validate_pack
    )
    local = np.einsum(
        "vck,k->vc",
        np.asarray(pack.local_displacement_basis, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
        optimize=True,
    )
    maximum = float(pack.maximum_displacement_m)
    if maximum == 0.0:
        return np.zeros(local.shape, dtype=np.float32)
    length = np.linalg.norm(local, axis=1)
    over_limit = length > maximum
    if np.any(over_limit):
        local[over_limit] *= (maximum / length[over_limit])[:, None]
    return np.asarray(local, dtype=np.float32)


def _selected_14_slot_weights_v1(
    driver_indices: Any,
    driver_weights: Any,
    *,
    sparse_vertex_count: int,
    global_vertex_ids: np.ndarray,
    posed_vertex_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and validate the authored 14-slot tube weights.

    Callers normally pass the tube coupling pack's already selected ``[V,14]``
    arrays.  Passing full global ``[N,14]`` asset arrays is also accepted for
    integration convenience and is selected by ``vertex_ids`` here.
    """
    indices = np.asarray(driver_indices)
    weights = np.asarray(driver_weights)
    if indices.ndim != 2 or weights.ndim != 2:
        raise ValueError("tube corrective driver arrays must be two-dimensional")
    if indices.shape[1] != TUBE_POSE_CORRECTIVE_INFLUENCE_SLOTS_V1 or weights.shape[1] != TUBE_POSE_CORRECTIVE_INFLUENCE_SLOTS_V1:
        raise ValueError("tube corrective requires exactly 14 sparse Armature slots")
    if indices.shape != weights.shape:
        raise ValueError("tube corrective driver indices and weights shape mismatch")
    if indices.dtype.kind not in {"i", "u"}:
        raise ValueError("tube corrective driver_indices must have integer dtype")
    if weights.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("tube corrective driver_weights must have numeric dtype")
    if indices.shape[0] == int(sparse_vertex_count):
        selected_indices = np.asarray(indices, dtype=np.int64)
        selected_weights = np.asarray(weights, dtype=np.float64)
    elif indices.shape[0] == int(posed_vertex_count):
        selected_indices = np.asarray(indices[global_vertex_ids], dtype=np.int64)
        selected_weights = np.asarray(weights[global_vertex_ids], dtype=np.float64)
    else:
        raise ValueError(
            "tube corrective driver arrays must be selected [V,14] or global [N,14]"
        )
    if (
        np.any(selected_indices < 0)
        or np.any(selected_weights < 0.0)
        or not np.all(np.isfinite(selected_weights))
        or not np.allclose(
            selected_weights.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0
        )
    ):
        raise ValueError("tube corrective contains invalid authored 14-slot weights")
    return selected_indices, selected_weights


def transform_tube_pose_corrective_local_v1(
    local_displacements_m: Any,
    source_transforms: Any,
    driver_indices: Any,
    driver_weights: Any,
    *,
    vertex_ids: Any | None = None,
    posed_vertex_count: int | None = None,
) -> np.ndarray:
    """Carry local corrections through exact per-vertex 14-slot LBS linears.

    Only the 3x3 linear component is applied, because the input is a local
    displacement vector rather than a point.  This preserves the exact matrix
    blend used by the authoritative tube LBS path while excluding translation.
    """
    local = np.asarray(local_displacements_m, dtype=np.float64)
    if local.ndim != 2 or local.shape[1:] != (3,) or not np.all(np.isfinite(local)):
        raise ValueError("tube corrective local displacements must be finite [V,3]")
    count = int(len(local))
    if vertex_ids is None:
        ids = np.arange(count, dtype=np.int64)
        vertex_count = count if posed_vertex_count is None else int(posed_vertex_count)
    else:
        ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
        if len(ids) != count or np.any(ids < 0):
            raise ValueError("tube corrective vertex_ids do not match local displacements")
        vertex_count = (
            int(np.max(ids)) + 1 if posed_vertex_count is None else int(posed_vertex_count)
        )
    if vertex_count <= 0 or (len(ids) and np.any(ids >= vertex_count)):
        raise ValueError("tube corrective vertex_ids are outside the posed vertex buffer")
    indices, weights = _selected_14_slot_weights_v1(
        driver_indices,
        driver_weights,
        sparse_vertex_count=count,
        global_vertex_ids=ids,
        posed_vertex_count=vertex_count,
    )
    transforms = np.asarray(source_transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
        raise ValueError("tube corrective source transforms must be [B,4,4]")
    if not np.all(np.isfinite(transforms)):
        raise ValueError("tube corrective source transforms contain non-finite values")
    affine_row = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    if not np.allclose(transforms[:, 3, :], affine_row, atol=1.0e-7, rtol=0.0):
        raise ValueError("tube corrective source transforms must be affine")
    if np.any(indices >= len(transforms)):
        raise ValueError("tube corrective driver index is outside source transforms")
    selected = transforms[indices]
    blended = np.sum(selected * weights[..., None, None], axis=1)
    result = np.einsum("vij,vj->vi", blended[:, :3, :3], local, optimize=True)
    return np.asarray(result, dtype=np.float32)


def apply_tube_pose_corrective_v1(
    tube_lbs_vertices: Any,
    pack: TubePoseCorrectivePackV1,
    *,
    pose_axis_angle: Any,
    source_transforms: Any,
    driver_indices: Any,
    driver_weights: Any,
    validate_pack: bool = True,
) -> np.ndarray:
    """Scatter-add a frozen corrective after authoritative tube LBS.

    ``tube_lbs_vertices`` must already include ``apply_tube_coupling_v8`` (or
    the equivalent strict LBS result).  This function deliberately does not
    replace, re-skin, or spatially project any tube vertices.
    """
    if validate_pack:
        pack.validate()
    posed = np.asarray(tube_lbs_vertices)
    if posed.ndim != 2 or posed.shape[1:] != (3,) or not np.all(np.isfinite(posed)):
        raise ValueError("tube LBS vertices must be finite [N,3]")
    ids = np.asarray(pack.vertex_ids, dtype=np.int64)
    if np.any(ids >= len(posed)):
        raise ValueError("tube pose corrective vertex_ids are outside tube LBS vertices")
    local = evaluate_tube_pose_corrective_local_v1(
        pack, pose_axis_angle, validate_pack=False
    )
    posed_offset = transform_tube_pose_corrective_local_v1(
        local,
        source_transforms,
        driver_indices,
        driver_weights,
        vertex_ids=ids,
        posed_vertex_count=len(posed),
    )
    result = np.array(posed, dtype=np.float32, copy=True)
    # np.add.at makes the ownership explicit and remains correct even if a
    # future schema relaxes the currently unique sparse vertex-id invariant.
    np.add.at(result, ids, posed_offset)
    return result


def _pose_samples55_v1(pose_axis_angle_samples: Any) -> np.ndarray:
    samples = np.asarray(pose_axis_angle_samples, dtype=np.float64)
    if samples.ndim >= 2 and samples.shape[-2:] == (SMPLX_JOINT_COUNT_V1, 3):
        result = samples.reshape(-1, SMPLX_JOINT_COUNT_V1, 3)
    elif samples.ndim >= 1 and samples.shape[-1] == SMPLX_JOINT_COUNT_V1 * 3:
        result = samples.reshape(-1, SMPLX_JOINT_COUNT_V1, 3)
    else:
        raise ValueError(
            "tube pose corrective samples must be [S,55,3] or [S,165]"
        )
    if not len(result) or not np.all(np.isfinite(result)):
        raise ValueError("tube pose corrective pose samples must be finite and non-empty")
    return result


def _choose_pca_rank_v1(
    singular_values: np.ndarray,
    *,
    maximum_components: int | None,
    explained_variance_ratio: float,
) -> int:
    if maximum_components is not None and int(maximum_components) <= 0:
        raise ValueError("maximum_components must be positive when provided")
    if not 0.0 < float(explained_variance_ratio) <= 1.0:
        raise ValueError("explained_variance_ratio must be in (0,1]")
    limit = len(singular_values)
    if maximum_components is not None:
        limit = min(limit, int(maximum_components))
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(np.sum(energy))
    if total <= np.finfo(np.float64).eps:
        return 1
    cumulative = np.cumsum(energy) / total
    needed = int(np.searchsorted(cumulative, float(explained_variance_ratio)) + 1)
    return max(1, min(limit, needed))


def _derive_rbf_widths_v1(
    features: np.ndarray,
    *,
    rbf_width_scale: float,
    minimum_rbf_width_rad: float,
) -> np.ndarray:
    if rbf_width_scale <= 0.0 or not np.isfinite(rbf_width_scale):
        raise ValueError("rbf_width_scale must be finite and positive")
    if minimum_rbf_width_rad <= 0.0 or not np.isfinite(minimum_rbf_width_rad):
        raise ValueError("minimum_rbf_width_rad must be finite and positive")
    flat = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    if len(flat) == 1:
        return np.asarray((max(1.0, minimum_rbf_width_rad),), dtype=np.float64)
    delta = flat[:, None, :] - flat[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1)
    finite = nearest[np.isfinite(nearest) & (nearest > 1.0e-12)]
    fallback = float(np.median(finite)) if len(finite) else 1.0
    nearest = np.where(np.isfinite(nearest) & (nearest > 1.0e-12), nearest, fallback)
    return np.maximum(
        nearest * float(rbf_width_scale), float(minimum_rbf_width_rad)
    )


def bake_tube_pose_corrective_v1(
    vertex_ids: Any,
    pose_axis_angle_samples: Any,
    local_displacement_samples_m: Any,
    driver_joint_ids: Any,
    *,
    maximum_components: int | None = None,
    explained_variance_ratio: float = 0.995,
    rbf_width_scale: float = 1.0,
    minimum_rbf_width_rad: float = 1.0e-3,
    rbf_ridge: float = 1.0e-8,
    maximum_displacement_m: float | None = None,
) -> tuple[TubePoseCorrectivePackV1, dict[str, Any]]:
    """Bake PCA/RBF tube corrections from offline local-space pose samples.

    ``local_displacement_samples_m`` is ``[S,V,3]`` relative to the strict
    tube LBS result.  The PCA is intentionally uncentered: the zero correction
    remains representable without an unlisted runtime mean field, so neutral
    poses can remain exactly neutral when supplied that way.  The RBF fit uses
    a small ridge term and only NumPy linear algebra.
    """
    poses = _pose_samples55_v1(pose_axis_angle_samples)
    ids = np.asarray(vertex_ids)
    if ids.dtype.kind not in {"i", "u"}:
        raise ValueError("tube pose corrective vertex_ids must have integer dtype")
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    joints = np.asarray(driver_joint_ids)
    if joints.dtype.kind not in {"i", "u"}:
        raise ValueError("tube pose corrective driver_joint_ids must have integer dtype")
    joints = np.asarray(joints, dtype=np.int16).reshape(-1)
    if (
        not len(ids)
        or np.any(ids < 0)
        or not np.array_equal(ids, np.unique(ids))
    ):
        raise ValueError("tube pose corrective vertex_ids must be unique and ordered")
    if (
        not len(joints)
        or np.any(joints < 0)
        or np.any(joints >= SMPLX_JOINT_COUNT_V1)
        or not np.array_equal(joints, np.unique(joints))
    ):
        raise ValueError(
            "tube pose corrective driver_joint_ids must be unique SMPL-X joint ids"
        )
    displacements = np.asarray(local_displacement_samples_m, dtype=np.float64)
    if (
        displacements.shape != (len(poses), len(ids), 3)
        or not np.all(np.isfinite(displacements))
    ):
        raise ValueError(
            "tube pose corrective local displacement samples must be finite [S,V,3]"
        )
    if not np.isfinite(rbf_ridge) or float(rbf_ridge) < 0.0:
        raise ValueError("rbf_ridge must be finite and non-negative")

    sample_maximum = (
        float(np.max(np.linalg.norm(displacements, axis=2))) if displacements.size else 0.0
    )
    if maximum_displacement_m is None:
        displacement_limit = sample_maximum
    else:
        displacement_limit = float(maximum_displacement_m)
        if not np.isfinite(displacement_limit) or displacement_limit < 0.0:
            raise ValueError("maximum_displacement_m must be finite and non-negative")
        if sample_maximum > displacement_limit + 1.0e-9:
            raise ValueError(
                "sampled local displacement exceeds maximum_displacement_m"
            )

    flat = displacements.reshape(len(poses), -1)
    _left, singular, right = np.linalg.svd(flat, full_matrices=False)
    rank = _choose_pca_rank_v1(
        singular,
        maximum_components=maximum_components,
        explained_variance_ratio=explained_variance_ratio,
    )
    basis_flat = np.asarray(right[:rank].T, dtype=np.float64)
    coordinates = flat @ basis_flat
    basis = basis_flat.reshape(len(ids), 3, rank)

    # The constructor below is also the canonical semantic validation for the
    # sparse vertex and driver-joint domains before fitting any coefficients.
    provisional = TubePoseCorrectivePackV1(
        vertex_ids=ids,
        local_displacement_basis=basis,
        driver_joint_ids=joints,
        rbf_centers=np.asarray(poses[:, joints, :], dtype=np.float64),
        rbf_widths=np.ones(len(poses), dtype=np.float64),
        center_coefficients=np.zeros((len(poses), rank), dtype=np.float64),
        maximum_displacement_m=displacement_limit,
    )
    features = np.asarray(poses[:, provisional.driver_joint_ids, :], dtype=np.float64)
    widths = _derive_rbf_widths_v1(
        features,
        rbf_width_scale=float(rbf_width_scale),
        minimum_rbf_width_rad=float(minimum_rbf_width_rad),
    )
    design = _gaussian_rbf_matrix_v1(features, features, widths)
    ridge = float(rbf_ridge)
    normal = design.T @ design + ridge * np.eye(len(design), dtype=np.float64)
    right_hand = design.T @ coordinates
    try:
        center_coefficients = np.linalg.solve(normal, right_hand)
    except np.linalg.LinAlgError:
        augmented_design = np.concatenate(
            (design, np.sqrt(ridge) * np.eye(len(design), dtype=np.float64)), axis=0
        )
        augmented_target = np.concatenate(
            (coordinates, np.zeros_like(coordinates)), axis=0
        )
        center_coefficients = np.linalg.lstsq(
            augmented_design, augmented_target, rcond=None
        )[0]
    pack = TubePoseCorrectivePackV1(
        vertex_ids=provisional.vertex_ids,
        local_displacement_basis=provisional.local_displacement_basis,
        driver_joint_ids=provisional.driver_joint_ids,
        rbf_centers=features,
        rbf_widths=widths,
        center_coefficients=center_coefficients,
        maximum_displacement_m=displacement_limit,
    )
    reconstructed = np.empty_like(displacements)
    for index, pose in enumerate(poses):
        reconstructed[index] = evaluate_tube_pose_corrective_local_v1(
            pack, pose, validate_pack=False
        )
    error = np.linalg.norm(reconstructed - displacements, axis=2)
    total_energy = float(np.sum(np.square(singular)))
    retained_energy = float(np.sum(np.square(singular[:rank])))
    report = {
        "available": True,
        "passed": True,
        "schema": "tube_pose_corrective_v1",
        "sample_count": int(len(poses)),
        "vertex_count": int(len(ids)),
        "driver_joint_count": int(len(joints)),
        "component_count": int(rank),
        "rbf_center_count": int(len(widths)),
        "maximum_displacement_m": float(displacement_limit),
        "sample_maximum_displacement_m": float(sample_maximum),
        "retained_variance_ratio": (
            1.0 if total_energy <= np.finfo(np.float64).eps else retained_energy / total_energy
        ),
        "sample_reconstruction_rms_m": float(np.sqrt(np.mean(np.square(error)))),
        "sample_reconstruction_max_m": float(np.max(error)),
        "runtime_spatial_query": False,
        "runtime_graph_solve": False,
        "runtime_collision": False,
        "content_digest": pack.content_digest(),
    }
    return pack, report


# The short aliases make the isolated pack convenient for artifact integration
# while retaining the explicit V1 version on every public callable.
tube_pose_corrective_to_runtime_fields_v1 = tube_pose_corrective_pack_to_runtime_fields_v1
tube_pose_corrective_from_runtime_fields_v1 = tube_pose_corrective_pack_from_runtime_fields_v1
evaluate_tube_pose_corrective_v1 = evaluate_tube_pose_corrective_local_v1


__all__ = [
    "SMPLX_JOINT_COUNT_V1",
    "TUBE_POSE_CORRECTIVE_ARTIFACT_KIND_V1",
    "TUBE_POSE_CORRECTIVE_INFLUENCE_SLOTS_V1",
    "TUBE_POSE_CORRECTIVE_RUNTIME_PREFIX_V1",
    "TUBE_POSE_CORRECTIVE_SCHEMA_V1",
    "TubePoseCorrectivePackV1",
    "apply_tube_pose_corrective_v1",
    "bake_tube_pose_corrective_v1",
    "evaluate_tube_pose_corrective_coefficients_v1",
    "evaluate_tube_pose_corrective_local_v1",
    "evaluate_tube_pose_corrective_v1",
    "has_tube_pose_corrective_v1",
    "transform_tube_pose_corrective_local_v1",
    "tube_pose_corrective_from_runtime_fields_v1",
    "tube_pose_corrective_pack_from_runtime_fields_v1",
    "tube_pose_corrective_pack_to_runtime_fields_v1",
    "tube_pose_corrective_to_runtime_fields_v1",
]
