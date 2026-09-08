"""Bounded cap preserving axial shaft fields for anatomy retargeting (V14).

This module contains the small, geometry independent part of the rest fit.  A
single scalar field translates the distal cap of a segment and smoothly
transports the shaft between two frozen cap domains.  The field can be sampled
on bone vertices, controller origins, and soft tissue points alike, which is
important for keeping those objects in one material coordinate system.

The map is deliberately not a similarity transform.  It has the form

    ``F(x) = x + delta * phi(dot(x - origin, axis)) * axis``

where ``phi`` is zero on the proximal cap, one on the distal cap, and a cubic
smoothstep in between.  Thus the caps are rigid translations and every radial
vector about the axis is unchanged.  The only non-unit Jacobian eigenvalue is
the analytic axial value ``1 + delta * phi'``; a non-positive value is rejected
when the field is built.

No Blender, SciPy, or project-specific geometry dependency is used here.  The
array-only serialization is suitable for embedding this field in a compiled
per-subject runtime package.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math
from typing import Any, Mapping

import numpy as np


AXIAL_CAPS_V14_SCHEMA = "axial_caps_v14"
MIN_TOTAL_LENGTH_SCALE_V14 = 0.90
MAX_TOTAL_LENGTH_SCALE_V14 = 1.10
_EPS = 1.0e-12
_ROTATION_EPS = 1.0e-10


def _as_points(value: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate an ``[N, 3]`` point array without silently flattening it."""

    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {result.shape}")
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return result


def _as_input_points(value: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate points while retaining the caller's numeric dtype.

    Rest identity is a byte-level contract for compiled assets.  The field's
    calculations use float64, but an identity ``map_points`` call must not
    silently turn a float32 source array into float64.
    """

    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {result.shape}")
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{name} must be non-empty")
    try:
        finite = np.isfinite(result)
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if not np.all(finite):
        raise ValueError(f"{name} contains non-finite coordinates")
    return result


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape [3], got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _as_scalar(value: Any, *, name: str) -> float:
    result = np.asarray(value)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    scalar = float(result)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _normalize_axis(value: Any, *, name: str) -> np.ndarray:
    axis = _as_vector(value, name=name)
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError(f"{name} is degenerate")
    return axis / norm


def _validate_scale(value: Any) -> float:
    scale = _as_scalar(value, name="total_length_scale")
    if not MIN_TOTAL_LENGTH_SCALE_V14 <= scale <= MAX_TOTAL_LENGTH_SCALE_V14:
        raise ValueError(
            "total_length_scale must lie in "
            f"[{MIN_TOTAL_LENGTH_SCALE_V14}, {MAX_TOTAL_LENGTH_SCALE_V14}]"
        )
    return scale


def _validate_ids(value: Any, *, name: str, count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional integer array")
    # Index arrays are part of the frozen topology contract.  Accepting a
    # float such as 1.5 and truncating it would make a different cap silently.
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer IDs")
    ids = raw.astype(np.int64, copy=True)
    if np.any(ids < 0) or np.any(ids >= int(count)):
        raise ValueError(f"{name} contains an out-of-range vertex ID")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate vertex IDs")
    return ids


def _axis_and_length(
    points: np.ndarray,
    *,
    axis_origin: np.ndarray,
    axis: Any | None,
    normalized_axis: Any | None,
    endpoints: Any | None,
    source_length_m: Any | None,
    total_length_m: Any | None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Resolve axis and source length, returning point scalar coordinates."""

    if axis is not None and normalized_axis is not None:
        raise ValueError("provide only one of axis and normalized_axis")
    if axis is not None or normalized_axis is not None:
        if endpoints is not None:
            raise ValueError("provide either an axis or endpoints, not both")
        direction = _normalize_axis(
            axis if axis is not None else normalized_axis,
            name="axis",
        )
    elif endpoints is not None:
        endpoint_array = np.asarray(endpoints, dtype=np.float64)
        if endpoint_array.shape != (2, 3):
            raise ValueError(
                f"endpoints must have shape [2, 3], got {endpoint_array.shape}"
            )
        if not np.all(np.isfinite(endpoint_array)):
            raise ValueError("endpoints contain non-finite coordinates")
        vector = endpoint_array[1] - endpoint_array[0]
        length_from_endpoints = float(np.linalg.norm(vector))
        if length_from_endpoints <= _EPS:
            raise ValueError("endpoints are degenerate")
        direction = vector / length_from_endpoints
    else:
        raise ValueError("provide axis/normalized_axis or endpoints")

    supplied_lengths = [
        value for value in (source_length_m, total_length_m) if value is not None
    ]
    if len(supplied_lengths) > 1:
        first = _as_scalar(supplied_lengths[0], name="source_length_m")
        second = _as_scalar(supplied_lengths[1], name="total_length_m")
        if not math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("source_length_m and total_length_m disagree")
        length = first
    elif supplied_lengths:
        length = _as_scalar(
            supplied_lengths[0],
            name="source_length_m" if source_length_m is not None else "total_length_m",
        )
    elif endpoints is not None:
        length = length_from_endpoints
    else:
        scalar = (points - axis_origin[None, :]) @ direction
        length = float(np.max(scalar) - np.min(scalar))

    if not math.isfinite(length) or length <= _EPS:
        raise ValueError("source length must be finite and positive")
    scalar = (points - axis_origin[None, :]) @ direction
    if not np.all(np.isfinite(scalar)):
        raise ValueError("axis scalar coordinates are non-finite")
    return direction, scalar, length, np.asarray(scalar, dtype=np.float64)


def _parse_bounds(
    value: Any,
    *,
    source_scalars: np.ndarray,
    name: str = "cap_scalar_bounds",
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse either two transition boundaries or four outer cap bounds.

    A two-value sequence is ``(proximal_end, distal_start)``.  Its outer
    bounds are taken from the points used to compile the field.  A four-value
    sequence is ``(proximal_start, proximal_end, distal_start, distal_end)``.
    A mapping may use ``proximal``/``distal`` (or the corresponding
    ``*_cap`` names) for two-value ranges.
    """

    if isinstance(value, Mapping):
        proximal_value = value.get("proximal", value.get("proximal_cap"))
        distal_value = value.get("distal", value.get("distal_cap"))
        if proximal_value is not None and distal_value is not None:
            proximal = np.asarray(proximal_value, dtype=np.float64)
            distal = np.asarray(distal_value, dtype=np.float64)
            if proximal.shape != (2,) or distal.shape != (2,):
                raise ValueError(f"{name} mapping cap ranges must each have two values")
            raw = np.concatenate((proximal, distal))
        elif "proximal_end" in value and "distal_start" in value:
            raw = np.asarray(
                (
                    float(np.asarray(source_scalars).min()),
                    value["proximal_end"],
                    value["distal_start"],
                    float(np.asarray(source_scalars).max()),
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"{name} mapping requires proximal/distal ranges or "
                "proximal_end/distal_start"
            )
    else:
        raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if raw.shape == (2,):
        p_end, d_start = (float(raw[0]), float(raw[1]))
        p_start = float(np.min(source_scalars))
        d_end = float(np.max(source_scalars))
    elif raw.shape == (4,):
        p_start, p_end, d_start, d_end = (float(item) for item in raw)
    else:
        raise ValueError(f"{name} must contain two or four scalar bounds")
    bounds = (p_start, p_end, d_start, d_end)
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} must be finite")
    if p_start > p_end or d_start > d_end:
        raise ValueError(f"{name} cap bounds must be ordered")
    if p_end >= d_start:
        raise ValueError(f"{name} proximal and distal cap domains overlap")
    if d_start - p_end <= _EPS:
        raise ValueError(f"{name} leaves no non-degenerate shaft transition")
    return (p_start, p_end), (d_start, d_end)


def _cap_bounds_from_ids(
    scalar: np.ndarray,
    proximal_ids: np.ndarray,
    distal_ids: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    proximal_values = scalar[proximal_ids]
    distal_values = scalar[distal_ids]
    proximal = (float(np.min(proximal_values)), float(np.max(proximal_values)))
    distal = (float(np.min(distal_values)), float(np.max(distal_values)))
    if proximal[1] >= distal[0]:
        raise ValueError("proximal and distal cap ID domains overlap")
    if distal[0] - proximal[1] <= _EPS:
        raise ValueError("cap ID domains leave no non-degenerate shaft transition")
    return proximal, distal


def _validate_cap_ids_pair(
    points: np.ndarray,
    scalar: np.ndarray,
    proximal_cap_ids: Any | None,
    distal_cap_ids: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    if (proximal_cap_ids is None) != (distal_cap_ids is None):
        raise ValueError("proximal_cap_ids and distal_cap_ids must be supplied together")
    if proximal_cap_ids is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    proximal = _validate_ids(
        proximal_cap_ids,
        name="proximal_cap_ids",
        count=len(points),
    )
    distal = _validate_ids(
        distal_cap_ids,
        name="distal_cap_ids",
        count=len(points),
    )
    if np.intersect1d(proximal, distal).size:
        raise ValueError("proximal and distal cap IDs overlap")
    # Indexing has already checked finite scalar coordinates; this explicit
    # check keeps the failure mode local if this helper is reused later.
    if not np.all(np.isfinite(scalar[proximal])) or not np.all(
        np.isfinite(scalar[distal])
    ):
        raise ValueError("cap ID scalar coordinates are non-finite")
    return proximal, distal


def _smoothstep(scalar: np.ndarray, start: float, stop: float) -> tuple[np.ndarray, np.ndarray]:
    width = float(stop - start)
    t = np.clip((scalar - float(start)) / width, 0.0, 1.0)
    phi = t * t * (3.0 - 2.0 * t)
    derivative = np.where(
        (scalar > float(start)) & (scalar < float(stop)),
        6.0 * t * (1.0 - t) / width,
        0.0,
    )
    # Explicit saturation makes cap values exact even if a caller samples a
    # point just outside the source segment.  It also avoids relying on clip's
    # floating point comparison at the exact transition boundary.
    phi = np.where(scalar <= float(start), 0.0, phi)
    phi = np.where(scalar >= float(stop), 1.0, phi)
    derivative = np.where(
        (scalar <= float(start)) | (scalar >= float(stop)),
        0.0,
        derivative,
    )
    return phi, derivative


def polar_rotation_v14(linear: Any) -> np.ndarray:
    """Return the nearest proper rotation of a finite, full-rank 3x3 matrix.

    Singular values are deliberately discarded.  This is the only interface
    provided for turning a local deformation into a frame rotation, so axial
    length adaptation can never be accidentally written into a bind rotation.
    Reflections are projected to the nearest determinant-positive rotation;
    rank-deficient inputs fail closed because their polar factor is not unique.
    """

    matrix = np.asarray(linear, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"linear must have shape [3, 3], got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("linear contains non-finite values")
    u, singular, vt = np.linalg.svd(matrix)
    if float(np.min(singular)) <= _ROTATION_EPS:
        raise ValueError("linear is rank-deficient; polar rotation is undefined")
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        rotation = u @ vt
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise ValueError("polar extraction did not produce a proper rotation")
    return rotation


def extract_proper_rotation_v14(linear: Any) -> np.ndarray:
    """Named alias for :func:`polar_rotation_v14` used by frame mappers."""

    return polar_rotation_v14(linear)


@dataclass(frozen=True)
class AxialCapsFieldV14:
    """Immutable axial material field shared by all linked geometry."""

    axis_origin: np.ndarray
    axis_direction: np.ndarray
    source_length_m: float
    total_length_scale: float
    proximal_cap_bounds: tuple[float, float]
    distal_cap_bounds: tuple[float, float]
    proximal_cap_ids: np.ndarray = dataclass_field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    distal_cap_ids: np.ndarray = dataclass_field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        origin = _as_vector(self.axis_origin, name="axis_origin")
        direction = _normalize_axis(self.axis_direction, name="axis_direction")
        length = _as_scalar(self.source_length_m, name="source_length_m")
        if length <= _EPS:
            raise ValueError("source_length_m must be positive")
        scale = _validate_scale(self.total_length_scale)
        try:
            proximal = tuple(float(item) for item in self.proximal_cap_bounds)
            distal = tuple(float(item) for item in self.distal_cap_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("cap bounds must each contain two values") from exc
        if len(proximal) != 2 or len(distal) != 2:
            raise ValueError("cap bounds must each contain two values")
        if not np.all(np.isfinite(proximal + distal)):
            raise ValueError("cap bounds must be finite")
        if proximal[0] > proximal[1] or distal[0] > distal[1]:
            raise ValueError("cap bounds must be ordered")
        if proximal[1] >= distal[0]:
            raise ValueError("proximal and distal cap domains overlap")
        if distal[0] - proximal[1] <= _EPS:
            raise ValueError("cap bounds leave no shaft transition")
        proximal_ids = np.asarray(self.proximal_cap_ids)
        distal_ids = np.asarray(self.distal_cap_ids)
        for ids, name in (
            (proximal_ids, "proximal_cap_ids"),
            (distal_ids, "distal_cap_ids"),
        ):
            if ids.ndim != 1 or ids.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a one-dimensional integer array")
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"{name} contains duplicate IDs")
        if bool(len(proximal_ids)) != bool(len(distal_ids)):
            raise ValueError(
                "proximal_cap_ids and distal_cap_ids must be supplied together"
            )
        if np.intersect1d(proximal_ids, distal_ids).size:
            raise ValueError("proximal and distal cap IDs overlap")
        # The global analytic Jacobian is independent of sampled vertices.  A
        # field that folds anywhere in its transition is unusable for a
        # materialized rest fit, so reject it at construction time.
        minimum = self._analytic_jacobian_minimum(
            length=length,
            scale=scale,
            start=float(proximal[1]),
            stop=float(distal[0]),
        )
        if minimum <= 0.0:
            raise ValueError(
                "axial caps field has non-positive analytic Jacobian: "
                f"minimum={minimum:.9g}"
            )
        object.__setattr__(self, "axis_origin", _readonly(origin, np.float64))
        object.__setattr__(self, "axis_direction", _readonly(direction, np.float64))
        object.__setattr__(self, "source_length_m", length)
        object.__setattr__(self, "total_length_scale", scale)
        object.__setattr__(self, "proximal_cap_bounds", proximal)
        object.__setattr__(self, "distal_cap_bounds", distal)
        object.__setattr__(self, "proximal_cap_ids", _readonly(proximal_ids, np.int64))
        object.__setattr__(self, "distal_cap_ids", _readonly(distal_ids, np.int64))

    @staticmethod
    def _analytic_jacobian_minimum(*, length: float, scale: float, start: float | None = None, stop: float | None = None) -> float:
        # ``start``/``stop`` are supplied by the instance method below.  The
        # static form keeps the formula obvious and independently testable.
        if start is None or stop is None:
            raise ValueError("analytic Jacobian bounds are required")
        delta = float(scale - 1.0) * float(length)
        peak_derivative = 1.5 / float(stop - start)
        return 1.0 + min(0.0, delta * peak_derivative)

    def _minimum_jacobian(self) -> float:
        return self._analytic_jacobian_minimum(
            length=self.source_length_m,
            scale=self.total_length_scale,
            start=self.transition_start_m,
            stop=self.transition_end_m,
        )

    @property
    def axis(self) -> np.ndarray:
        """Compatibility alias for callers that use ``axis`` terminology."""

        return self.axis_direction

    @property
    def transition_start_m(self) -> float:
        return float(self.proximal_cap_bounds[1])

    @property
    def transition_end_m(self) -> float:
        return float(self.distal_cap_bounds[0])

    @property
    def transition_width_m(self) -> float:
        return self.transition_end_m - self.transition_start_m

    @property
    def delta_m(self) -> float:
        return float(self.source_length_m * (self.total_length_scale - 1.0))

    @property
    def target_length_m(self) -> float:
        return float(self.source_length_m * self.total_length_scale)

    @property
    def profile_peak_derivative(self) -> float:
        return float(1.5 / self.transition_width_m)

    @property
    def analytic_jacobian_minimum(self) -> float:
        return float(self._minimum_jacobian())

    @property
    def analytic_jacobian_maximum(self) -> float:
        signed = self.delta_m * self.profile_peak_derivative
        return float(1.0 + max(0.0, signed))

    @property
    def global_min_jacobian(self) -> float:
        return self.analytic_jacobian_minimum

    @property
    def identity(self) -> bool:
        return self.total_length_scale == 1.0

    def scalar_coordinate(self, points: Any) -> np.ndarray:
        value = _as_points(points, name="points", allow_empty=True)
        return (value - self.axis_origin[None, :]) @ self.axis_direction

    def profile(self, points: Any) -> np.ndarray:
        scalar = self.scalar_coordinate(points)
        return _smoothstep(scalar, self.transition_start_m, self.transition_end_m)[0]

    def profile_derivative(self, points: Any) -> np.ndarray:
        scalar = self.scalar_coordinate(points)
        return _smoothstep(scalar, self.transition_start_m, self.transition_end_m)[1]

    def displacement(self, points: Any) -> np.ndarray:
        value = _as_points(points, name="points", allow_empty=True)
        if self.identity:
            return np.zeros(value.shape, dtype=np.float64)
        phi = self.profile(value)
        return (self.delta_m * phi)[:, None] * self.axis_direction[None, :]

    def map_points(self, points: Any) -> np.ndarray:
        """Map arbitrary points with this field, preserving input topology."""

        raw = _as_input_points(points, name="points", allow_empty=True)
        # Avoid even a multiply/add round trip in the identity case.  This is
        # useful for a compiled package where identity rest fits must be
        # byte-for-byte reproducible.
        if self.identity:
            return np.array(raw, copy=True)
        value = np.asarray(raw, dtype=np.float64)
        mapped = value + self.displacement(value)
        if not np.all(np.isfinite(mapped)):
            raise ValueError("axial caps map produced non-finite points")
        return mapped

    def sample(self, points: Any) -> np.ndarray:
        """Alias used by callers sampling controller origins or soft points."""

        return self.map_points(points)

    def jacobian(self, points: Any) -> np.ndarray:
        """Return the pointwise 3x3 Jacobian without hiding axial scale."""

        value = _as_points(points, name="points", allow_empty=True)
        derivative = self.profile_derivative(value)
        coefficient = self.delta_m * derivative
        result = np.broadcast_to(np.eye(3, dtype=np.float64), (len(value), 3, 3)).copy()
        result += coefficient[:, None, None] * np.outer(
            self.axis_direction,
            self.axis_direction,
        )[None, :, :]
        return result

    def map_with_fields(self, points: Any) -> "AxialCapsSampleV14":
        value = np.asarray(
            _as_input_points(points, name="points", allow_empty=True),
            dtype=np.float64,
        )
        mapped = self.map_points(value)
        phi, derivative = _smoothstep(
            self.scalar_coordinate(value),
            self.transition_start_m,
            self.transition_end_m,
        )
        return AxialCapsSampleV14(
            points=mapped,
            displacement=mapped - value,
            phi=phi,
            profile_derivative=derivative,
            jacobian=self.jacobian(value),
        )

    def map_local_frame(self, frame: Any) -> np.ndarray:
        """Map a homogeneous local frame using a scale-free polar rotation.

        Input and output frames use column-vector homogeneous convention.  The
        translation is sampled from this field; the local deformation's polar
        rotation is composed with the input orientation, while its singular
        values (including axial length scale) are discarded.
        """

        matrix = np.asarray(frame, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"frame must have shape [4, 4], got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("frame contains non-finite values")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-10, rtol=0.0):
            raise ValueError("frame bottom row must be [0, 0, 0, 1]")
        if self.identity:
            return np.array(matrix, copy=True)
        origin = matrix[:3, 3][None, :]
        mapped_origin = self.map_points(origin)[0]
        local_rotation = polar_rotation_v14(self.jacobian(origin)[0])
        input_rotation = polar_rotation_v14(matrix[:3, :3])
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = local_rotation @ input_rotation
        result[:3, 3] = mapped_origin
        return result

    def report(self) -> dict[str, Any]:
        return {
            "schema": AXIAL_CAPS_V14_SCHEMA,
            "identity": bool(self.identity),
            "axis_origin": self.axis_origin.tolist(),
            "axis_direction": self.axis_direction.tolist(),
            "source_length_m": float(self.source_length_m),
            "target_length_m": float(self.target_length_m),
            "total_length_scale": float(self.total_length_scale),
            "delta_m": float(self.delta_m),
            "proximal_cap_bounds": [float(x) for x in self.proximal_cap_bounds],
            "distal_cap_bounds": [float(x) for x in self.distal_cap_bounds],
            "transition_width_m": float(self.transition_width_m),
            "radial_scale": 1.0,
            "profile_peak_derivative": float(self.profile_peak_derivative),
            "analytic_jacobian_minimum": float(self.analytic_jacobian_minimum),
            "analytic_jacobian_maximum": float(self.analytic_jacobian_maximum),
            "provenance_cap_ids_frozen": bool(
                len(self.proximal_cap_ids) and len(self.distal_cap_ids)
            ),
        }

    def to_array_dict(self) -> dict[str, np.ndarray]:
        """Serialize the field using NumPy arrays only."""

        return {
            "schema": np.asarray(AXIAL_CAPS_V14_SCHEMA),
            "axis_origin": np.array(self.axis_origin, copy=True),
            "axis_direction": np.array(self.axis_direction, copy=True),
            "source_length_m": np.asarray(self.source_length_m, dtype=np.float64),
            "total_length_scale": np.asarray(self.total_length_scale, dtype=np.float64),
            "proximal_cap_bounds": np.asarray(self.proximal_cap_bounds, dtype=np.float64),
            "distal_cap_bounds": np.asarray(self.distal_cap_bounds, dtype=np.float64),
            "proximal_cap_ids": np.array(self.proximal_cap_ids, copy=True),
            "distal_cap_ids": np.array(self.distal_cap_ids, copy=True),
        }

    @classmethod
    def from_array_dict(cls, arrays: Mapping[str, Any]) -> "AxialCapsFieldV14":
        return axial_caps_field_from_array_dict_v14(arrays)


@dataclass(frozen=True)
class AxialCapsSampleV14:
    """Mapped points and shared field samples for one point set."""

    points: np.ndarray
    displacement: np.ndarray
    phi: np.ndarray
    profile_derivative: np.ndarray
    jacobian: np.ndarray

    def __post_init__(self) -> None:
        points = _as_points(self.points, name="sample.points", allow_empty=True)
        displacement = _as_points(
            self.displacement,
            name="sample.displacement",
            allow_empty=True,
        )
        if displacement.shape != points.shape:
            raise ValueError("sample displacement shape does not match points")
        phi = np.asarray(self.phi, dtype=np.float64).reshape(-1)
        derivative = np.asarray(self.profile_derivative, dtype=np.float64).reshape(-1)
        jacobian = np.asarray(self.jacobian, dtype=np.float64)
        if len(phi) != len(points) or len(derivative) != len(points):
            raise ValueError("sample scalar fields do not match point count")
        if jacobian.shape != (len(points), 3, 3):
            raise ValueError("sample Jacobian must have shape [N, 3, 3]")
        if not all(np.all(np.isfinite(value)) for value in (phi, derivative, jacobian)):
            raise ValueError("sample fields must be finite")
        object.__setattr__(self, "points", _readonly(points, np.float64))
        object.__setattr__(self, "displacement", _readonly(displacement, np.float64))
        object.__setattr__(self, "phi", _readonly(phi, np.float64))
        object.__setattr__(self, "profile_derivative", _readonly(derivative, np.float64))
        object.__setattr__(self, "jacobian", _readonly(jacobian, np.float64))


@dataclass(frozen=True)
class AxialCapsMapResultV14:
    """Convenience result containing mapped points and their reusable field."""

    points: np.ndarray
    field: AxialCapsFieldV14
    sample: AxialCapsSampleV14

    @property
    def mapped_points(self) -> np.ndarray:
        return self.points

    @property
    def vertices(self) -> np.ndarray:
        """Compatibility alias for rest-fit callers that call points vertices."""

        return self.points


def build_axial_caps_field_v14(
    points: Any,
    *,
    axis_origin: Any,
    axis: Any | None = None,
    normalized_axis: Any | None = None,
    endpoints: Any | None = None,
    source_length_m: Any | None = None,
    total_length_m: Any | None = None,
    total_length_scale: Any = 1.0,
    proximal_cap_ids: Any | None = None,
    distal_cap_ids: Any | None = None,
    proximal_ids: Any | None = None,
    distal_ids: Any | None = None,
    cap_scalar_bounds: Any | None = None,
    cap_bounds: Any | None = None,
) -> AxialCapsFieldV14:
    """Compile one bounded cap-preserving field from rest geometry.

    ``points`` are used to infer source length when no endpoints/length is
    supplied and to resolve cap IDs.  The returned field can subsequently be
    sampled on any other point array with the same coordinate frame.
    """

    source = _as_points(points, name="points")
    origin = _as_vector(axis_origin, name="axis_origin")
    direction, scalar, length, _ = _axis_and_length(
        source,
        axis_origin=origin,
        axis=axis,
        normalized_axis=normalized_axis,
        endpoints=endpoints,
        source_length_m=source_length_m,
        total_length_m=total_length_m,
    )
    scale = _validate_scale(total_length_scale)
    if proximal_ids is not None:
        if proximal_cap_ids is not None:
            raise ValueError("provide only one of proximal_cap_ids and proximal_ids")
        proximal_cap_ids = proximal_ids
    if distal_ids is not None:
        if distal_cap_ids is not None:
            raise ValueError("provide only one of distal_cap_ids and distal_ids")
        distal_cap_ids = distal_ids
    proximal, distal = _validate_cap_ids_pair(
        source,
        scalar,
        proximal_cap_ids,
        distal_cap_ids,
    )
    if len(proximal):
        if cap_scalar_bounds is not None or cap_bounds is not None:
            raise ValueError("provide cap IDs or cap_scalar_bounds, not both")
        proximal_bounds, distal_bounds = _cap_bounds_from_ids(scalar, proximal, distal)
    else:
        if cap_scalar_bounds is not None and cap_bounds is not None:
            raise ValueError("provide only one of cap_scalar_bounds and cap_bounds")
        bounds_value = cap_scalar_bounds if cap_scalar_bounds is not None else cap_bounds
        if bounds_value is None:
            raise ValueError("provide cap IDs or cap_scalar_bounds")
        proximal_bounds, distal_bounds = _parse_bounds(
            bounds_value,
            source_scalars=scalar,
        )
    # Constructing first validates the analytic full-domain Jacobian.  This is
    # intentionally before any mapped points are returned.
    return AxialCapsFieldV14(
        axis_origin=origin,
        axis_direction=direction,
        source_length_m=length,
        total_length_scale=scale,
        proximal_cap_bounds=proximal_bounds,
        distal_cap_bounds=distal_bounds,
        proximal_cap_ids=proximal,
        distal_cap_ids=distal,
    )


def build_axial_caps_v14(
    points: Any,
    **kwargs: Any,
) -> AxialCapsFieldV14:
    """Short alias for :func:`build_axial_caps_field_v14`."""

    return build_axial_caps_field_v14(points, **kwargs)


def apply_axial_caps_v14(
    points: Any,
    **kwargs: Any,
) -> AxialCapsMapResultV14:
    """Compile a field and map its source points in one call."""

    source = _as_points(points, name="points")
    field = build_axial_caps_field_v14(source, **kwargs)
    sample = field.map_with_fields(source)
    return AxialCapsMapResultV14(points=sample.points, field=field, sample=sample)


def map_axial_caps_v14(points: Any, **kwargs: Any) -> AxialCapsMapResultV14:
    """Compatibility alias for :func:`apply_axial_caps_v14`."""

    return apply_axial_caps_v14(points, **kwargs)


def map_local_frame_v14(field: AxialCapsFieldV14, frame: Any) -> np.ndarray:
    """Top-level local-frame mapping helper."""

    if not isinstance(field, AxialCapsFieldV14):
        raise TypeError("field must be an AxialCapsFieldV14")
    return field.map_local_frame(frame)


def axial_caps_field_to_array_dict_v14(
    field: AxialCapsFieldV14,
) -> dict[str, np.ndarray]:
    if not isinstance(field, AxialCapsFieldV14):
        raise TypeError("field must be an AxialCapsFieldV14")
    return field.to_array_dict()


def axial_caps_field_from_array_dict_v14(
    arrays: Mapping[str, Any],
) -> AxialCapsFieldV14:
    required = {
        "schema",
        "axis_origin",
        "axis_direction",
        "source_length_m",
        "total_length_scale",
        "proximal_cap_bounds",
        "distal_cap_bounds",
        "proximal_cap_ids",
        "distal_cap_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"serialized axial caps field is missing {missing}")
    schema = np.asarray(arrays["schema"])
    if schema.size != 1:
        raise ValueError("serialized axial caps schema must be scalar")
    schema_value = schema.reshape(-1)[0]
    if str(schema_value) != AXIAL_CAPS_V14_SCHEMA:
        raise ValueError("serialized axial caps schema is not axial_caps_v14")
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"serialized field value {key!r} must be a NumPy array")
    return AxialCapsFieldV14(
        axis_origin=arrays["axis_origin"],
        axis_direction=arrays["axis_direction"],
        source_length_m=arrays["source_length_m"],
        total_length_scale=arrays["total_length_scale"],
        proximal_cap_bounds=arrays["proximal_cap_bounds"],
        distal_cap_bounds=arrays["distal_cap_bounds"],
        proximal_cap_ids=arrays["proximal_cap_ids"],
        distal_cap_ids=arrays["distal_cap_ids"],
    )


def serialize_axial_caps_v14(field: AxialCapsFieldV14) -> dict[str, np.ndarray]:
    return axial_caps_field_to_array_dict_v14(field)


def deserialize_axial_caps_v14(
    arrays: Mapping[str, Any],
) -> AxialCapsFieldV14:
    return axial_caps_field_from_array_dict_v14(arrays)


__all__ = [
    "AXIAL_CAPS_V14_SCHEMA",
    "MIN_TOTAL_LENGTH_SCALE_V14",
    "MAX_TOTAL_LENGTH_SCALE_V14",
    "AxialCapsFieldV14",
    "AxialCapsMapResultV14",
    "AxialCapsSampleV14",
    "apply_axial_caps_v14",
    "axial_caps_field_from_array_dict_v14",
    "axial_caps_field_to_array_dict_v14",
    "build_axial_caps_field_v14",
    "build_axial_caps_v14",
    "deserialize_axial_caps_v14",
    "extract_proper_rotation_v14",
    "map_axial_caps_v14",
    "map_local_frame_v14",
    "polar_rotation_v14",
    "serialize_axial_caps_v14",
]
