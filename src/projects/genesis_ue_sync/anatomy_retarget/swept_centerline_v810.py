"""Cross-section-preserving swept-centerline rest warp for V8.10.

The primitive bends a straight rest-space segment onto a C2 center curve.
Each source cross section is carried by the minimum proper rotation from the
source axis to the curve tangent.  There is no radial scale, topology change,
runtime dependency, or pose-time deformation in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _unit(vector: Any, *, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-10:
        raise ValueError(f"{label} must be non-degenerate")
    return value / norm


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )


def _minimum_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the least-angle proper rotation taking ``source`` to ``target``."""

    first = _unit(source, label="minimum-rotation source")
    second = _unit(target, label="minimum-rotation target")
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    if sine <= 1.0e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(first)))]
        axis = _unit(
            np.cross(first, basis),
            label="minimum-rotation antipodal axis",
        )
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    skew = _skew(cross)
    rotation = (
        np.eye(3, dtype=np.float64)
        + skew
        + skew @ skew * ((1.0 - cosine) / (sine * sine))
    )
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10, rtol=0.0)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-10)
    ):
        raise ValueError("minimum rotation is not a proper rigid frame")
    return rotation


@dataclass(frozen=True)
class SweptCenterlineRestWarpV810:
    """A C2 centerline sweep with unit-scale cross-section frames.

    ``target_center_offsets_m`` are offsets from the straight proximal-to-distal
    axis at ``station_fractions``.  Endpoint offsets must be zero.  ``blend`` is
    a scalar in [0, 1], allowing a caller to retain a measured residual without
    rebuilding the curve or changing its topology.
    """

    proximal_m: np.ndarray
    distal_m: np.ndarray
    station_fractions: np.ndarray
    target_center_offsets_m: np.ndarray
    blend: float = 1.0
    _axis_vector: np.ndarray = field(init=False, repr=False, compare=False)
    _axis_direction: np.ndarray = field(init=False, repr=False, compare=False)
    _axis_length_m: float = field(init=False, repr=False, compare=False)
    _offset_spline: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from scipy.interpolate import CubicSpline

        proximal = np.asarray(self.proximal_m, dtype=np.float64).reshape(3)
        distal = np.asarray(self.distal_m, dtype=np.float64).reshape(3)
        stations = np.asarray(self.station_fractions, dtype=np.float64).reshape(-1)
        offsets = np.asarray(
            self.target_center_offsets_m,
            dtype=np.float64,
        )
        blend = float(self.blend)
        if (
            not np.all(np.isfinite(proximal))
            or not np.all(np.isfinite(distal))
            or not np.all(np.isfinite(stations))
            or not np.all(np.isfinite(offsets))
        ):
            raise ValueError("swept centerline inputs must be finite")
        if offsets.shape != (len(stations), 3):
            raise ValueError("target center offsets must have shape [station, 3]")
        if len(stations) < 3:
            raise ValueError("swept centerline needs at least three stations")
        if not np.all(np.diff(stations) > 0.0):
            raise ValueError("station fractions must be strictly increasing")
        if (
            not math.isclose(float(stations[0]), 0.0, abs_tol=1.0e-12)
            or not math.isclose(float(stations[-1]), 1.0, abs_tol=1.0e-12)
        ):
            raise ValueError("station fractions must start at 0 and end at 1")
        if (
            float(np.linalg.norm(offsets[0])) > 1.0e-10
            or float(np.linalg.norm(offsets[-1])) > 1.0e-10
        ):
            raise ValueError("proximal and distal center offsets must be zero")
        if not math.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError("swept centerline blend must be in [0, 1]")

        axis_vector = distal - proximal
        axis_length = float(np.linalg.norm(axis_vector))
        if axis_length <= 1.0e-8:
            raise ValueError("swept centerline axis must be non-degenerate")
        axis_direction = axis_vector / axis_length
        spline = CubicSpline(
            stations,
            offsets,
            axis=0,
            bc_type=(
                (1, np.zeros(3, dtype=np.float64)),
                (1, np.zeros(3, dtype=np.float64)),
            ),
            extrapolate=False,
        )

        object.__setattr__(self, "proximal_m", _readonly(proximal))
        object.__setattr__(self, "distal_m", _readonly(distal))
        object.__setattr__(self, "station_fractions", _readonly(stations))
        object.__setattr__(self, "target_center_offsets_m", _readonly(offsets))
        object.__setattr__(self, "blend", blend)
        object.__setattr__(self, "_axis_vector", _readonly(axis_vector))
        object.__setattr__(self, "_axis_direction", _readonly(axis_direction))
        object.__setattr__(self, "_axis_length_m", axis_length)
        object.__setattr__(self, "_offset_spline", spline)

    @property
    def axis_direction(self) -> np.ndarray:
        return self._axis_direction

    @property
    def axis_length_m(self) -> float:
        return self._axis_length_m

    def _offset_values(self, fractions: Any, *, order: int = 0) -> np.ndarray:
        values = np.asarray(fractions, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("centerline fractions must be finite")
        clipped = np.clip(values, 0.0, 1.0)
        result = np.asarray(
            self._offset_spline(clipped, nu=int(order)),
            dtype=np.float64,
        )
        if int(order) > 0:
            outside = (values < 0.0) | (values > 1.0)
            if np.any(outside):
                result = result.copy()
                result[outside] = 0.0
        return result

    def center(self, fractions: Any, *, full_target: bool = False) -> np.ndarray:
        """Evaluate applied or full-target centers at normalized stations."""

        values = np.asarray(fractions, dtype=np.float64)
        base = (
            self.proximal_m
            + values[..., None] * self._axis_vector
        )
        factor = 1.0 if full_target else self.blend
        return base + factor * self._offset_values(values)

    def center_derivative(
        self,
        fractions: Any,
        *,
        order: int = 1,
        full_target: bool = False,
    ) -> np.ndarray:
        """Evaluate center derivatives with respect to normalized station."""

        derivative_order = int(order)
        if derivative_order not in (1, 2):
            raise ValueError("center derivative order must be 1 or 2")
        values = np.asarray(fractions, dtype=np.float64)
        factor = 1.0 if full_target else self.blend
        offset = factor * self._offset_values(values, order=derivative_order)
        if derivative_order == 1:
            offset = offset + self._axis_vector
        return offset

    def frame_rotations(self, fractions: Any) -> np.ndarray:
        """Return minimum-rotation source-axis-to-tangent frames."""

        values = np.asarray(fractions, dtype=np.float64)
        derivatives = self.center_derivative(values, order=1).reshape(-1, 3)
        rotations = np.asarray(
            [
                _minimum_rotation(self._axis_direction, derivative)
                for derivative in derivatives
            ],
            dtype=np.float64,
        )
        return rotations.reshape(values.shape + (3, 3))

    def apply(self, points: Any) -> np.ndarray:
        """Sweep points while preserving every radial cross-section vector."""

        source = np.asarray(points, dtype=np.float64)
        if source.ndim < 1 or source.shape[-1:] != (3,):
            raise ValueError("swept centerline points must end with a 3-vector")
        if not np.all(np.isfinite(source)):
            raise ValueError("swept centerline points must be finite")
        flat = source.reshape(-1, 3)
        relative = flat - self.proximal_m
        fractions = (relative @ self._axis_direction) / self._axis_length_m
        straight_centers = (
            self.proximal_m
            + fractions[:, None] * self._axis_vector
        )
        radial = flat - straight_centers
        rotations = self.frame_rotations(fractions).reshape(-1, 3, 3)
        mapped_radial = np.einsum(
            "nij,nj->ni",
            rotations,
            radial,
            optimize=True,
        )
        mapped = self.center(fractions) + mapped_radial
        return mapped.reshape(source.shape)

    def report(self, *, sample_count: int = 513) -> dict[str, Any]:
        """Return frame rigidity, station residual, and centerline strain."""

        count = int(sample_count)
        if count < 33:
            raise ValueError("swept centerline report needs at least 33 samples")
        samples = np.linspace(0.0, 1.0, count, dtype=np.float64)
        derivatives_s = self.center_derivative(samples, order=1)
        derivatives_h = derivatives_s / self._axis_length_m
        axial_jacobian = np.linalg.norm(derivatives_h, axis=1)
        second_h = (
            self.center_derivative(samples, order=2)
            / (self._axis_length_m * self._axis_length_m)
        )
        curvature = np.linalg.norm(
            np.cross(derivatives_h, second_h),
            axis=1,
        ) / np.maximum(axial_jacobian**3, 1.0e-12)
        rotations = self.frame_rotations(samples)
        determinants = np.linalg.det(rotations)
        singular_values = np.linalg.svd(rotations, compute_uv=False)

        station_applied = self.center(self.station_fractions)
        station_target = self.center(
            self.station_fractions,
            full_target=True,
        )
        station_expected = (
            self.proximal_m
            + self.station_fractions[:, None] * self._axis_vector
            + self.blend * self.target_center_offsets_m
        )
        interpolation_error = np.linalg.norm(
            station_applied - station_expected,
            axis=1,
        )
        station_residual = station_applied - station_target
        station_residual_norm = np.linalg.norm(station_residual, axis=1)
        arc_length = self._axis_length_m * float(
            np.sum(
                0.5
                * (axial_jacobian[:-1] + axial_jacobian[1:])
                * np.diff(samples)
            )
        )
        arc_ratio = arc_length / self._axis_length_m
        axial_strain = axial_jacobian - 1.0
        rotation_cosine = np.clip(
            np.einsum(
                "nij,j->ni",
                rotations,
                self._axis_direction,
            )
            @ self._axis_direction,
            -1.0,
            1.0,
        )
        return {
            "schema_version": 810,
            "method": "c2_minimum_rotation_swept_centerline_rest_v810",
            "center_curve": "clamped_c2_cubic_spline",
            "frame_method": "pointwise_minimum_rotation",
            "blend": self.blend,
            "axis_length_m": self._axis_length_m,
            "station_fractions": self.station_fractions.tolist(),
            "applied_station_centers_m": station_applied.tolist(),
            "target_station_centers_m": station_target.tolist(),
            "station_residuals_m": station_residual.tolist(),
            "station_residual_norms_m": station_residual_norm.tolist(),
            "station_residual_max_m": float(np.max(station_residual_norm)),
            "station_residual_rms_m": float(
                np.sqrt(np.mean(station_residual_norm**2))
            ),
            "station_interpolation_error_max_m": float(
                np.max(interpolation_error)
            ),
            "frame_determinant_min": float(np.min(determinants)),
            "frame_determinant_max": float(np.max(determinants)),
            "frame_scale_min": float(np.min(singular_values)),
            "frame_scale_max": float(np.max(singular_values)),
            "cross_section_scale": 1.0,
            "rotation_angle_max_deg": float(
                np.degrees(np.max(np.arccos(rotation_cosine)))
            ),
            "axial_jacobian_min": float(np.min(axial_jacobian)),
            "axial_jacobian_max": float(np.max(axial_jacobian)),
            "axial_jacobian_at_stations": (
                np.linalg.norm(
                    self.center_derivative(
                        self.station_fractions,
                        order=1,
                    )
                    / self._axis_length_m,
                    axis=1,
                ).tolist()
            ),
            "axial_strain_q99_abs": float(
                np.quantile(np.abs(axial_strain), 0.99)
            ),
            "axial_strain_max_abs": float(np.max(np.abs(axial_strain))),
            "centerline_arc_length_m": arc_length,
            "centerline_arc_length_ratio": arc_ratio,
            "centerline_arc_strain": arc_ratio - 1.0,
            "curvature_max_inverse_m": float(np.max(curvature)),
        }


def swept_centerline_rest_warp_v810(
    points: Any,
    *,
    proximal_m: Any,
    distal_m: Any,
    station_fractions: Any,
    target_center_offsets_m: Any,
    lambda_: float = 1.0,
    report_sample_count: int = 513,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one swept-centerline rest warp and return its audit report."""

    warp = SweptCenterlineRestWarpV810(
        proximal_m=np.asarray(proximal_m, dtype=np.float64),
        distal_m=np.asarray(distal_m, dtype=np.float64),
        station_fractions=np.asarray(station_fractions, dtype=np.float64),
        target_center_offsets_m=np.asarray(
            target_center_offsets_m,
            dtype=np.float64,
        ),
        blend=float(lambda_),
    )
    return warp.apply(points), warp.report(sample_count=report_sample_count)


__all__ = [
    "SweptCenterlineRestWarpV810",
    "swept_centerline_rest_warp_v810",
]
