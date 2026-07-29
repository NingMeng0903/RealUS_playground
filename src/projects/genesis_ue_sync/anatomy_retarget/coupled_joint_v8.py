"""Compact three-dimensional joint-state response fields for V8.

The coefficients in this module are baked offline from frozen material
contact domains.  Runtime evaluation is deliberately limited to fixed-size
array arithmetic: no spatial query, collision solve, or pose cache is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def coupled_state_centers_v8(
    *,
    support_radius_rad: float,
    inner_fraction: float = 0.45,
) -> np.ndarray:
    """Return a deterministic 3-D sampling design inside an axis-angle ball.

    The outer shell includes axis, two-axis, and three-axis directions.  The
    inner axial shell stabilizes the response near bind.  This is intentionally
    not a tensor product of three one-dimensional curves.
    """

    radius = float(support_radius_rad)
    fraction = float(inner_fraction)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("support_radius_rad must be finite and positive")
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("inner_fraction must lie strictly inside (0, 1)")
    directions: list[np.ndarray] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            direction = np.zeros(3, dtype=np.float64)
            direction[axis] = sign
            directions.append(direction)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for sign_first in (-1.0, 1.0):
            for sign_second in (-1.0, 1.0):
                direction = np.zeros(3, dtype=np.float64)
                direction[first] = sign_first
                direction[second] = sign_second
                direction /= np.linalg.norm(direction)
                directions.append(direction)
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            for sign_z in (-1.0, 1.0):
                direction = np.asarray(
                    (sign_x, sign_y, sign_z), dtype=np.float64
                )
                direction /= np.linalg.norm(direction)
                directions.append(direction)
    inner_directions = [
        direction * (fraction * radius)
        for direction in directions
    ]
    result = np.asarray(
        [
            np.zeros(3, dtype=np.float64),
            *inner_directions,
            *(direction * radius for direction in directions),
        ],
        dtype=np.float64,
    )
    if result.shape != (53, 3):
        raise AssertionError("V8 coupled state design must contain 53 samples")
    return result


def bake_coupled_rbf_response_v8(
    *,
    states_rotvec_rad: Any,
    translations_parent_local_m: Any,
    smplx_joint: int,
    joint_kind: str,
    support_radius_rad: float,
    maximum_translation_m: float,
    kernel_width: float = 8.0,
    ridge: float = 1.0e-12,
) -> dict[str, Any]:
    """Bake a compact Gaussian RBF map from one 3-D state to translation."""

    states = np.asarray(states_rotvec_rad, dtype=np.float64)
    translations = np.asarray(translations_parent_local_m, dtype=np.float64)
    radius = float(support_radius_rad)
    width = float(kernel_width)
    regularization = float(ridge)
    maximum = float(maximum_translation_m)
    if (
        states.ndim != 2
        or states.shape[1] != 3
        or len(states) < 8
        or translations.shape != states.shape
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(translations))
    ):
        raise ValueError("coupled RBF samples must be finite [N, 3] arrays")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("support_radius_rad must be finite and positive")
    if float(np.max(np.linalg.norm(states, axis=1))) > radius + 1.0e-9:
        raise ValueError("coupled RBF state lies outside its support ball")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("kernel_width must be finite and positive")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_translation_m must be finite and positive")
    normalized = states / radius
    zero_kernel = np.exp(
        -(width * width) * np.sum(normalized * normalized, axis=1)
    )
    zero_offset = (
        zero_kernel @ translations
        / max(float(np.sum(zero_kernel)), regularization)
    )
    return {
        "schema": "coupled_normalized_gaussian_rbf_translation_v8",
        "joint_kind": str(joint_kind),
        "smplx_joint": int(smplx_joint),
        "state_centers_rotvec_rad": states.tolist(),
        "rbf_values_parent_local_m": translations.tolist(),
        "rbf_zero_parent_local_m": zero_offset.tolist(),
        "support_radius_rad": radius,
        "kernel_width": width,
        "maximum_translation_m": maximum,
        "training_sample_count": int(len(states)),
        "independent_axis_sum": False,
    }


def evaluate_coupled_rbf_response_v8(
    response: Any,
    state_rotvec_rad: Any,
) -> np.ndarray:
    """Evaluate one baked response with fixed-size runtime array operations."""

    if not isinstance(response, dict):
        raise ValueError("V8 coupled response must be a mapping")
    if response.get("schema") != "coupled_normalized_gaussian_rbf_translation_v8":
        raise ValueError("V8 coupled response has an unsupported schema")
    centers = np.asarray(
        response.get("state_centers_rotvec_rad", []), dtype=np.float64
    )
    values = np.asarray(
        response.get("rbf_values_parent_local_m", []), dtype=np.float64
    )
    zero_offset = np.asarray(
        response.get("rbf_zero_parent_local_m", []), dtype=np.float64
    )
    state = np.asarray(state_rotvec_rad, dtype=np.float64).reshape(3)
    radius = float(response.get("support_radius_rad", np.nan))
    width = float(response.get("kernel_width", np.nan))
    maximum = float(response.get("maximum_translation_m", np.nan))
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or len(centers) < 8
        or values.shape != centers.shape
        or zero_offset.shape != (3,)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(zero_offset))
        or not np.all(np.isfinite(state))
        or not np.isfinite(radius)
        or radius <= 0.0
        or not np.isfinite(width)
        or width <= 0.0
        or not np.isfinite(maximum)
        or maximum <= 0.0
    ):
        raise ValueError("V8 coupled response has invalid coefficients")
    state_norm = float(np.linalg.norm(state))
    if state_norm > radius + 1.0e-7:
        raise ValueError(
            f"V8 coupled response state {state_norm:.6f} rad exceeds "
            f"its {radius:.6f} rad support"
        )
    delta = (centers - state[None, :]) / radius
    kernel = np.exp(-(width * width) * np.sum(delta * delta, axis=1))
    denominator = float(np.sum(kernel))
    if not np.isfinite(denominator) or denominator <= 1.0e-12:
        raise ValueError("V8 coupled response has no local support")
    translation = kernel @ values / denominator - zero_offset
    if (
        not np.all(np.isfinite(translation))
        or float(np.linalg.norm(translation)) > maximum + 1.0e-7
    ):
        raise ValueError("V8 coupled response exceeds its baked translation bound")
    if state_norm <= 1.0e-12:
        translation = np.zeros(3, dtype=np.float64)
    return np.asarray(translation, dtype=np.float64)


__all__ = [
    "bake_coupled_rbf_response_v8",
    "coupled_state_centers_v8",
    "evaluate_coupled_rbf_response_v8",
]
