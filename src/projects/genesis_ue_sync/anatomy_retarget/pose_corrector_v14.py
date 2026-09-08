"""Compact-support pose-local SE(3) corrections for the V14 runtime.

The corrector is deliberately a small, geometry-independent component.  A
caller fits it from poses and local twists once, then evaluates the saved
interpolant at run time.  Pose locations are represented by selected SMPL-X
joint rotation matrices, rather than by the input axis-angle coordinates;
therefore equivalent ``r`` and ``r + 2*pi*a`` representations have the same
feature whenever they describe the same rotation.

The interpolant is a compactly supported Wendland C2 RBF.  Its radius is a
fixed fit parameter and is persisted in the NPZ artifact.  Runtime queries
outside every kernel support fail closed with :class:`PoseCorrectionSupportError`.
No geometry, optimization, or per-frame re-fitting belongs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


POSE_CORRECTOR_SCHEMA_VERSION = 14
POSE_CORRECTOR_KIND = "PoseCorrectorV14"
CONTROLLER_COUNT = 235
SMPLX_JOINT_COUNT = 55
DEFAULT_MAX_ROTATION_NORM = float(np.deg2rad(5.0))
DEFAULT_MAX_TRANSLATION_NORM = 5.0e-3

# Matrix features are computed in float64.  This tolerance also covers a
# float32 2*pi axis-angle input while remaining far below any meaningful pose
# change.  It is used only to identify equivalent rotations and the neutral
# feature, never to expand kernel support.
FEATURE_EQ_ATOL = 1.0e-6
TARGET_EQ_ATOL = 1.0e-12
NODE_MATCH_ATOL = 1.0e-8


class PoseCorrectionSupportError(ValueError):
    """Raised when a pose has no positive-weight RBF center in its support."""


class PoseCorrectionAmplitudeError(ValueError):
    """Raised when a fitted or interpolated twist exceeds its configured bound."""


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive_scalar(value: Any, *, name: str) -> float:
    result = _as_finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _validate_bound(value: Any, *, name: str) -> float:
    result = _as_finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _index_vector(value: Any, *, name: str, upper: int, allow_empty: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not allow_empty and len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    result = np.asarray(array, dtype=np.int64).copy()
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    if np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} contains an out-of-bounds index for [0, {upper})")
    return result


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _pose_array(value: Any, *, name: str) -> np.ndarray:
    result = _finite_array(value, name=name, ndim=3)
    if result.shape[-2:] != (SMPLX_JOINT_COUNT, 3):
        raise ValueError(f"{name} must have shape [N, 55, 3]")
    if len(result) == 0:
        raise ValueError(f"{name} must contain at least one pose")
    return result


def _rotation_features(poses: np.ndarray, selected_joint_ids: np.ndarray) -> np.ndarray:
    """Return flattened selected-joint 3x3 matrices for ``poses``."""
    values = np.asarray(poses, dtype=np.float64)
    matrices = Rotation.from_rotvec(values.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(len(values), SMPLX_JOINT_COUNT, 3, 3)
    return np.ascontiguousarray(matrices[:, selected_joint_ids].reshape(len(values), -1))


def _neutral_feature(selected_count: int) -> np.ndarray:
    return np.tile(np.eye(3, dtype=np.float64), (selected_count, 1, 1)).reshape(-1)


def _wendland_c2(distance: np.ndarray, radius: float) -> np.ndarray:
    """Evaluate the compactly-supported Wendland C2 kernel.

    ``phi(q) = (1-q)^4 * (4q+1)`` for ``0 <= q < 1`` and zero otherwise.
    The strict boundary is intentional: a query exactly at the support edge
    has zero influence and consequently fails closed if no other center helps.
    """
    q = np.asarray(distance, dtype=np.float64) / float(radius)
    result = np.zeros_like(q, dtype=np.float64)
    inside = q < 1.0
    if np.any(inside):
        one_minus = 1.0 - q[inside]
        result[inside] = one_minus**4 * (4.0 * q[inside] + 1.0)
    return result


def _kernel_matrix(features: np.ndarray, radius: float) -> np.ndarray:
    delta = features[:, None, :] - features[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    return _wendland_c2(distance, radius)


def _validate_equivalent_samples(features: np.ndarray, twists: np.ndarray) -> None:
    """Reject equivalent rotations with conflicting local corrections."""
    for first in range(len(features)):
        equivalent = np.linalg.norm(features[first + 1:] - features[first], axis=1)
        for offset in np.flatnonzero(equivalent <= FEATURE_EQ_ATOL):
            other = first + 1 + int(offset)
            if not np.allclose(twists[first], twists[other], atol=TARGET_EQ_ATOL, rtol=0.0):
                raise ValueError(
                    "equivalent pose samples have different local_twists "
                    f"(rows {first} and {other})"
                )


def _deduplicate_samples(
    features: np.ndarray, twists: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one representative of duplicate equivalent pose samples."""
    keep: list[int] = []
    for row in range(len(features)):
        if keep:
            distances = np.linalg.norm(features[keep] - features[row], axis=1)
            if np.any(distances <= FEATURE_EQ_ATOL):
                # Conflicting targets were checked before this function.  A
                # same-target duplicate adds no interpolation information.
                continue
        keep.append(row)
    ids = np.asarray(keep, dtype=np.int64)
    return features[ids], twists[ids]


def _validate_twist_bounds(
    twists: np.ndarray, *, max_rotation_norm: float, max_translation_norm: float,
    name: str,
) -> None:
    rotation_norm = np.linalg.norm(twists[..., :3], axis=-1)
    translation_norm = np.linalg.norm(twists[..., 3:], axis=-1)
    # A small relative tolerance accepts a value that was rounded to float32;
    # values beyond it are rejected, never clipped.
    rotation_limit = max_rotation_norm + max(1.0e-12, abs(max_rotation_norm) * 1.0e-10)
    translation_limit = max_translation_norm + max(1.0e-12, abs(max_translation_norm) * 1.0e-10)
    over_rotation = np.argwhere(rotation_norm > rotation_limit)
    over_translation = np.argwhere(translation_norm > translation_limit)
    if len(over_rotation):
        row = tuple(int(v) for v in over_rotation[0])
        raise PoseCorrectionAmplitudeError(
            f"{name} rotation norm exceeds max_rotation_norm at index {row}: "
            f"{rotation_norm[row]:.9g} > {max_rotation_norm:.9g}"
        )
    if len(over_translation):
        row = tuple(int(v) for v in over_translation[0])
        raise PoseCorrectionAmplitudeError(
            f"{name} translation norm exceeds max_translation_norm at index {row}: "
            f"{translation_norm[row]:.9g} > {max_translation_norm:.9g}"
        )


def _as_scalar_string(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "SU":
        raise ValueError(f"{name} must be a scalar string field")
    return str(array.item())


@dataclass
class PoseCorrectorV14:
    """Offline-fitted local correction field for all 235 controllers.

    ``sample_features`` stores one flattened 3x3 rotation matrix per selected
    joint.  ``coefficients`` are the solved Wendland RBF coefficients with
    shape ``[S, M, 6]``; ``sample_twists`` retains the input targets for
    authority/audit checks.  A twist is ``[rotvec_x, rotvec_y, rotvec_z,
    translation_x, translation_y, translation_z]`` in radians and metres.
    """

    selected_joint_ids: np.ndarray
    controller_ids: np.ndarray
    sample_features: np.ndarray
    sample_twists: np.ndarray
    coefficients: np.ndarray
    radius: float
    max_rotation_norm: float = DEFAULT_MAX_ROTATION_NORM
    max_translation_norm: float = DEFAULT_MAX_TRANSLATION_NORM

    def __post_init__(self) -> None:
        self.selected_joint_ids = _index_vector(
            self.selected_joint_ids, name="selected_joint_ids", upper=SMPLX_JOINT_COUNT,
            allow_empty=False,
        )
        self.controller_ids = _index_vector(
            self.controller_ids, name="controller_ids", upper=CONTROLLER_COUNT,
            allow_empty=True,
        )
        self.radius = _validate_positive_scalar(self.radius, name="radius")
        self.max_rotation_norm = _validate_bound(
            self.max_rotation_norm, name="max_rotation_norm"
        )
        self.max_translation_norm = _validate_bound(
            self.max_translation_norm, name="max_translation_norm"
        )

        features = _finite_array(self.sample_features, name="sample_features", ndim=2)
        twists = _finite_array(self.sample_twists, name="sample_twists", ndim=3)
        coefficients = _finite_array(self.coefficients, name="coefficients", ndim=3)
        expected_dim = 9 * len(self.selected_joint_ids)
        expected_shape = (len(self.controller_ids), 6)
        if features.shape[1:] != (expected_dim,):
            raise ValueError(
                f"sample_features must have shape [S, {expected_dim}], got {features.shape}"
            )
        if len(features) == 0:
            raise ValueError("at least one RBF sample is required")
        if twists.shape != (len(features), *expected_shape):
            raise ValueError(
                "sample_twists must have shape "
                f"[S, {expected_shape[0]}, 6], got {twists.shape}"
            )
        if coefficients.shape != twists.shape:
            raise ValueError("coefficients must have the same shape as sample_twists")
        rotations = features.reshape(len(features), len(self.selected_joint_ids), 3, 3)
        gram = np.einsum("nkij,nklj->nkil", rotations, rotations)
        if not np.allclose(gram, np.eye(3)[None, None], atol=3.0e-6, rtol=0.0):
            raise ValueError("sample_features must contain proper rotation matrices")
        determinants = np.linalg.det(rotations)
        if np.any(determinants <= 0.0):
            raise ValueError("sample_features must contain proper rotations")
        _validate_twist_bounds(
            twists, max_rotation_norm=self.max_rotation_norm,
            max_translation_norm=self.max_translation_norm, name="sample_twists",
        )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must contain only finite values")
        _validate_equivalent_samples(features, twists)
        neutral = _neutral_feature(len(self.selected_joint_ids))
        neutral_rows = np.linalg.norm(features - neutral[None], axis=1) <= FEATURE_EQ_ATOL
        if not np.any(neutral_rows):
            raise ValueError("fit must include a neutral identity pose sample")
        if not np.allclose(twists[neutral_rows], 0.0, atol=TARGET_EQ_ATOL, rtol=0.0):
            raise ValueError("neutral identity pose samples must have zero correction")

        # Check that persisted coefficients are actually the interpolant for
        # the persisted samples.  This catches incomplete or mismatched NPZ
        # fields while allowing the tiny roundoff from solving an ill-scaled
        # but otherwise valid center system.
        kernel = _kernel_matrix(features, self.radius)
        reconstructed = kernel @ coefficients.reshape(len(features), -1)
        if not np.allclose(
            reconstructed.reshape(twists.shape), twists, atol=2.0e-8, rtol=2.0e-8
        ):
            raise ValueError("coefficients do not reproduce the saved sample_twists")

        self.sample_features = np.ascontiguousarray(features)
        self.sample_twists = np.ascontiguousarray(twists)
        self.coefficients = np.ascontiguousarray(coefficients)

    @classmethod
    def fit(
        cls,
        poses: Any,
        selected_joint_ids: Any,
        controller_ids: Any,
        local_twists: Any,
        *,
        radius: float = 1.0,
        max_rotation_norm: float = DEFAULT_MAX_ROTATION_NORM,
        max_translation_norm: float = DEFAULT_MAX_TRANSLATION_NORM,
    ) -> "PoseCorrectorV14":
        """Fit the fixed-radius Wendland C2 field from caller-provided samples.

        This method only solves the interpolation coefficients.  It does not
        split fit/validation data, inspect geometry, or optimize any pose or
        mesh quantity; the caller owns those policy decisions.
        """
        pose_values = _pose_array(poses, name="poses")
        selected = _index_vector(
            selected_joint_ids, name="selected_joint_ids", upper=SMPLX_JOINT_COUNT,
            allow_empty=False,
        )
        controllers = _index_vector(
            controller_ids, name="controller_ids", upper=CONTROLLER_COUNT,
            allow_empty=True,
        )
        radius_value = _validate_positive_scalar(radius, name="radius")
        max_rotation = _validate_bound(max_rotation_norm, name="max_rotation_norm")
        max_translation = _validate_bound(
            max_translation_norm, name="max_translation_norm"
        )
        twists = _finite_array(local_twists, name="local_twists", ndim=3)
        expected_shape = (len(pose_values), len(controllers), 6)
        if twists.shape != expected_shape:
            raise ValueError(
                f"local_twists must have shape {expected_shape}, got {twists.shape}"
            )
        _validate_twist_bounds(
            twists, max_rotation_norm=max_rotation,
            max_translation_norm=max_translation, name="local_twists",
        )
        features = _rotation_features(pose_values, selected)
        _validate_equivalent_samples(features, twists)
        neutral = _neutral_feature(len(selected))
        neutral_rows = np.linalg.norm(features - neutral[None], axis=1) <= FEATURE_EQ_ATOL
        if not np.any(neutral_rows):
            raise ValueError("fit requires a neutral identity pose sample")
        if not np.allclose(twists[neutral_rows], 0.0, atol=TARGET_EQ_ATOL, rtol=0.0):
            raise ValueError("neutral identity pose samples must have zero correction")
        features, twists = _deduplicate_samples(features, twists)

        kernel = _kernel_matrix(features, radius_value)
        rhs = twists.reshape(len(features), -1)
        if rhs.shape[1] == 0 or np.all(rhs == 0.0):
            coefficients = np.zeros_like(rhs)
        else:
            try:
                coefficients = np.linalg.solve(kernel, rhs)
            except np.linalg.LinAlgError as exc:
                # Least squares is only a deterministic fallback for a
                # singular center system.  It is accepted only if it still
                # interpolates the supplied nodes; no optimization or clamp
                # is hidden here.
                coefficients, _, _, _ = np.linalg.lstsq(kernel, rhs, rcond=None)
                if not np.allclose(kernel @ coefficients, rhs, atol=2.0e-8, rtol=2.0e-8):
                    raise ValueError(
                        "RBF sample centers are singular and cannot interpolate exactly"
                    ) from exc
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("RBF solve produced non-finite coefficients")
        return cls(
            selected_joint_ids=selected,
            controller_ids=controllers,
            sample_features=features,
            sample_twists=twists,
            coefficients=coefficients.reshape(twists.shape),
            radius=radius_value,
            max_rotation_norm=max_rotation,
            max_translation_norm=max_translation,
        )

    def _query_feature(self, pose: Any) -> np.ndarray:
        values = _finite_array(pose, name="pose", ndim=2)
        if values.shape != (SMPLX_JOINT_COUNT, 3):
            raise ValueError("pose must have shape [55, 3]")
        return _rotation_features(values[None], self.selected_joint_ids)[0]

    def _interpolated_twists(self, feature: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(self.sample_features - feature[None], axis=1)
        support = _wendland_c2(distances, self.radius)
        active = support > 0.0
        if not np.any(active):
            nearest = float(np.min(distances))
            raise PoseCorrectionSupportError(
                "pose correction has no Wendland C2 kernel support: "
                f"nearest feature distance {nearest:.9g} >= radius {self.radius:.9g}"
            )
        # Preserve exact node targets when the incoming rotation matrix is an
        # equivalent representation of a stored node.  Away from nodes the
        # ordinary RBF evaluation is smooth and compactly supported.
        nodes = np.flatnonzero(distances <= NODE_MATCH_ATOL)
        if len(nodes):
            return self.sample_twists[int(nodes[0])].copy()
        values = support @ self.coefficients.reshape(len(self.sample_features), -1)
        return values.reshape(len(self.controller_ids), 6)

    def evaluate(self, pose: Any) -> np.ndarray:
        """Evaluate local SE(3) corrections for all 235 controllers."""
        feature = self._query_feature(pose)
        neutral = _neutral_feature(len(self.selected_joint_ids))
        if np.linalg.norm(feature - neutral) <= FEATURE_EQ_ATOL:
            # Avoid even the last floating-point residue at neutral.  This is
            # a contract: the identity pose has an exact identity correction.
            return np.tile(np.eye(4, dtype=np.float64), (CONTROLLER_COUNT, 1, 1))
        twists = self._interpolated_twists(feature)
        _validate_twist_bounds(
            twists[None], max_rotation_norm=self.max_rotation_norm,
            max_translation_norm=self.max_translation_norm, name="interpolated twists",
        )
        result = np.tile(np.eye(4, dtype=np.float64), (CONTROLLER_COUNT, 1, 1))
        if len(self.controller_ids):
            rotations = Rotation.from_rotvec(twists[:, :3]).as_matrix()
            result[self.controller_ids, :3, :3] = rotations
            result[self.controller_ids, :3, 3] = twists[:, 3:]
        return result

    def save(self, path: str | Path) -> None:
        """Save a self-contained, pickle-free NPZ artifact."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            schema_version=np.asarray(POSE_CORRECTOR_SCHEMA_VERSION, dtype=np.int64),
            artifact_kind=np.asarray(POSE_CORRECTOR_KIND),
            kernel=np.asarray("wendland_c2"),
            selected_joint_ids=self.selected_joint_ids,
            controller_ids=self.controller_ids,
            sample_features=self.sample_features,
            sample_twists=self.sample_twists,
            coefficients=self.coefficients,
            radius=np.asarray(self.radius, dtype=np.float64),
            max_rotation_norm=np.asarray(self.max_rotation_norm, dtype=np.float64),
            max_translation_norm=np.asarray(self.max_translation_norm, dtype=np.float64),
        )
        # A file handle prevents numpy from silently changing ``path`` to
        # ``path + '.npz'`` when a caller intentionally chose another name.
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "PoseCorrectorV14":
        """Load and validate a pure-NPZ V14 pose corrector."""
        required = {
            "schema_version", "artifact_kind", "kernel", "selected_joint_ids",
            "controller_ids", "sample_features", "sample_twists", "coefficients",
            "radius", "max_rotation_norm", "max_translation_norm",
        }
        try:
            with np.load(Path(path), allow_pickle=False) as data:
                names = set(data.files)
                if names != required:
                    missing = sorted(required - names)
                    extra = sorted(names - required)
                    raise ValueError(
                        f"incomplete pose corrector fields (missing={missing}, extra={extra})"
                    )
                version = np.asarray(data["schema_version"])
                if version.ndim != 0 or version.dtype.kind not in "iu" or int(version) != POSE_CORRECTOR_SCHEMA_VERSION:
                    raise ValueError("unsupported pose corrector schema version")
                if _as_scalar_string(data["artifact_kind"], name="artifact_kind") != POSE_CORRECTOR_KIND:
                    raise ValueError("not a PoseCorrectorV14 artifact")
                if _as_scalar_string(data["kernel"], name="kernel") != "wendland_c2":
                    raise ValueError("unsupported pose corrector kernel")
                values = {
                    name: data[name].copy()
                    for name in required
                    if name not in {"schema_version", "artifact_kind", "kernel"}
                }
        except ValueError:
            raise
        except (OSError, KeyError, TypeError) as exc:
            raise ValueError(f"cannot load PoseCorrectorV14 NPZ: {exc}") from exc
        return cls(**values)


def fit_pose_corrector_v14(
    poses: Any,
    selected_joint_ids: Any,
    controller_ids: Any,
    local_twists: Any,
    *,
    radius: float = 1.0,
    max_rotation_norm: float = DEFAULT_MAX_ROTATION_NORM,
    max_translation_norm: float = DEFAULT_MAX_TRANSLATION_NORM,
) -> PoseCorrectorV14:
    """Functional alias for :meth:`PoseCorrectorV14.fit`."""
    return PoseCorrectorV14.fit(
        poses,
        selected_joint_ids,
        controller_ids,
        local_twists,
        radius=radius,
        max_rotation_norm=max_rotation_norm,
        max_translation_norm=max_translation_norm,
    )


def load_pose_corrector_v14(path: str | Path) -> PoseCorrectorV14:
    """Functional alias for :meth:`PoseCorrectorV14.load`."""
    return PoseCorrectorV14.load(path)


__all__ = [
    "CONTROLLER_COUNT",
    "DEFAULT_MAX_ROTATION_NORM",
    "DEFAULT_MAX_TRANSLATION_NORM",
    "POSE_CORRECTOR_KIND",
    "POSE_CORRECTOR_SCHEMA_VERSION",
    "PoseCorrectionAmplitudeError",
    "PoseCorrectionSupportError",
    "PoseCorrectorV14",
    "fit_pose_corrector_v14",
    "load_pose_corrector_v14",
]
