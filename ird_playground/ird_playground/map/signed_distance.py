"""Anisotropic weighted 5-D signed distance from a flange occupancy grid."""

from __future__ import annotations

import numpy as np

from ird_playground.ird.metric import LAMBDA_M_PER_RAD

AXIS_TILT = 2
AXIS_AZIMUTH = 3
AXIS_GAMMA = 4


def _edt_1d_squared(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Felzenszwalb–Huttenlocher 1-D squared distance transform."""
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n = f.shape[0]
    if n == 0:
        return f.copy()
    v = np.zeros(n, dtype=np.int64)
    z = np.zeros(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + (q * w[q]) ** 2) - (f[v[k]] + (v[k] * w[v[k]]) ** 2)) / (
            2.0 * (q * w[q] - v[k] * w[v[k]]) + 1.0e-18
        )
        while s <= z[k]:
            k -= 1
            s = ((f[q] + (q * w[q]) ** 2) - (f[v[k]] + (v[k] * w[v[k]]) ** 2)) / (
                2.0 * (q * w[q] - v[k] * w[v[k]]) + 1.0e-18
            )
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    d = np.empty(n, dtype=np.float64)
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        d[q] = f[v[k]] + (dq * w[v[k]]) ** 2
    return d


def _line_weights(
    axis: int,
    spacings: np.ndarray,
    *,
    n: int,
    tilt_axis: np.ndarray | None,
    tilt_index: int | None,
) -> np.ndarray:
    if axis in (AXIS_TILT, AXIS_GAMMA):
        base = float(spacings[axis]) * float(LAMBDA_M_PER_RAD)
        return np.full(n, base, dtype=np.float64)
    if axis == AXIS_AZIMUTH:
        base = float(spacings[axis]) * float(LAMBDA_M_PER_RAD)
        if tilt_axis is not None and tilt_index is not None:
            base *= max(float(np.sin(tilt_axis[tilt_index])), 1.0e-3)
        return np.full(n, base, dtype=np.float64)
    return np.full(n, float(spacings[axis]), dtype=np.float64)


def _edt_axis(
    dist_sq: np.ndarray,
    axis: int,
    *,
    spacings: np.ndarray,
    periodic: bool,
    tilt_axis: np.ndarray | None,
) -> np.ndarray:
    moved = np.moveaxis(dist_sq, axis, 0)
    n = moved.shape[0]
    if periodic:
        moved = np.concatenate([moved, moved, moved], axis=0)
        n_work = moved.shape[0]
    else:
        n_work = n
    out = np.empty_like(moved)
    line_shape = moved.shape[1:]
    n_lines = int(np.prod(line_shape)) if line_shape else 1
    flat = moved.reshape(n_work, n_lines)
    flat_out = out.reshape(n_work, n_lines)
    for line in range(n_lines):
        idx = np.unravel_index(line, line_shape) if line_shape else ()
        tilt_index = idx[AXIS_TILT] if len(idx) > AXIS_TILT else None
        weights = _line_weights(
            axis,
            spacings,
            n=n_work,
            tilt_axis=tilt_axis,
            tilt_index=tilt_index,
        )
        line_sq = flat[:, line]
        inf = np.where(np.isfinite(line_sq), line_sq, np.inf)
        flat_out[:, line] = _edt_1d_squared(inf, weights)
    result = np.moveaxis(out, 0, axis)
    if periodic:
        start = (result.shape[axis] - n) // 2
        sl = [slice(None)] * result.ndim
        sl[axis] = slice(start, start + n)
        result = result[tuple(sl)]
    return result


def _distance_to_false(occ: np.ndarray, spacings: np.ndarray, periodic_axes: tuple[int, ...], tilt_axis: np.ndarray) -> np.ndarray:
    f = np.where(occ, np.inf, 0.0)
    out = f
    for axis in range(occ.ndim):
        out = _edt_axis(
            out,
            axis,
            spacings=spacings,
            periodic=axis in periodic_axes,
            tilt_axis=tilt_axis,
        )
    return np.sqrt(np.maximum(out, 0.0))


def signed_distance_from_occupancy(
    occ: np.ndarray,
    spacings: np.ndarray | tuple[float, ...],
    periodic_axes: tuple[int, ...] = (3, 4),
    *,
    tilt_axis_values: np.ndarray | None = None,
) -> np.ndarray:
    """Return signed distance in metres (+ inside reachable occupancy).

    ``spacings`` are native chart steps: metres for ``p_z``/``r``, radians for
    angles.  Angular axes are scaled by ``λ`` so 1 deg ≡ 1 cm.  Azimuth spacing
    is additionally weighted by ``sin(tilt)`` near the poles.
    """
    occ = np.asarray(occ, dtype=bool)
    sp = np.asarray(spacings, dtype=np.float64).reshape(-1)
    if sp.shape[0] != occ.ndim:
        raise ValueError(f"expected {occ.ndim} spacings, got {sp.shape[0]}")
    if tilt_axis_values is None:
        tilt_axis_values = np.linspace(0.0, np.pi, occ.shape[AXIS_TILT], dtype=np.float64)
    dist_in = _distance_to_false(occ, sp, periodic_axes, tilt_axis_values)
    dist_out = _distance_to_false(~occ, sp, periodic_axes, tilt_axis_values)
    return np.where(occ, dist_in, -dist_out).astype(np.float32)


__all__ = ["signed_distance_from_occupancy"]
