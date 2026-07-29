"""Anisotropic weighted 5-D signed distance from a flange occupancy grid.

Distance is chart-coordinate weighted L2 (metres under the declared metric),
**not** an SE(3) geodesic.  Use only as a far-field prior; near-field labels
come from mm-scale stencil bisection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ird_playground.ird.metric import LAMBDA_M_PER_RAD, metric_manifest

AXIS_TILT = 2
AXIS_AZIMUTH = 3
AXIS_GAMMA = 4

EDT_WARNING = (
    "chart-coordinate weighted L2, not an SE(3) geodesic; far-field prior only"
)

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None  # type: ignore


def _edt_1d_squared_py(f: np.ndarray, w: float) -> np.ndarray:
    """Felzenszwalb–Huttenlocher 1-D squared DT with constant spacing ``w``."""
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    n = int(f.shape[0])
    if n == 0:
        return f.copy()
    w = float(w)
    w2 = w * w
    v = np.empty(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        fq = f[q]
        if not np.isfinite(fq):
            # Skip non-sites / unreachable seeds; keep envelope unchanged.
            continue
        while True:
            r = int(v[k])
            fr = f[r]
            # Intersection of parabolas at sites q and r (index domain).
            s = ((fq - fr) / (w2 * (q - r) + 1.0e-30) + (q + r)) * 0.5
            if s > z[k]:
                break
            k -= 1
            if k < 0:
                k = 0
                s = -np.inf
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    # If the first site was +inf, locate the first finite seed.
    if not np.isfinite(f[int(v[0])]):
        first = -1
        for q in range(n):
            if np.isfinite(f[q]):
                first = q
                break
        if first < 0:
            return np.full(n, np.inf, dtype=np.float64)
        k = 0
        v[0] = first
        z[0] = -np.inf
        z[1] = np.inf
        for q in range(first + 1, n):
            fq = f[q]
            if not np.isfinite(fq):
                continue
            while True:
                r = int(v[k])
                fr = f[r]
                s = ((fq - fr) / (w2 * (q - r) + 1.0e-30) + (q + r)) * 0.5
                if s > z[k]:
                    break
                k -= 1
                if k < 0:
                    k = 0
                    s = -np.inf
                    break
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - int(v[k])
        d[q] = f[int(v[k])] + (dq * w) ** 2
    return d


if njit is not None:

    @njit(cache=True)
    def _edt_1d_squared_numba(f: np.ndarray, w: float) -> np.ndarray:  # pragma: no cover
        n = f.shape[0]
        d = np.empty(n, dtype=np.float64)
        if n == 0:
            return d
        w2 = w * w
        v = np.empty(n, dtype=np.int64)
        z = np.empty(n + 1, dtype=np.float64)
        # Find first finite site.
        first = -1
        for q in range(n):
            if np.isfinite(f[q]):
                first = q
                break
        if first < 0:
            for q in range(n):
                d[q] = np.inf
            return d
        k = 0
        v[0] = first
        z[0] = -np.inf
        z[1] = np.inf
        for q in range(first + 1, n):
            fq = f[q]
            if not np.isfinite(fq):
                continue
            while True:
                r = v[k]
                fr = f[r]
                s = ((fq - fr) / (w2 * (q - r) + 1.0e-30) + (q + r)) * 0.5
                if s > z[k]:
                    break
                k -= 1
                if k < 0:
                    k = 0
                    s = -np.inf
                    break
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            dq = q - v[k]
            d[q] = f[v[k]] + (dq * w) * (dq * w)
        return d

    def _edt_1d_squared(f: np.ndarray, w: float) -> np.ndarray:
        return _edt_1d_squared_numba(np.asarray(f, dtype=np.float64), float(w))

else:  # pragma: no cover

    def _edt_1d_squared(f: np.ndarray, w: float) -> np.ndarray:
        return _edt_1d_squared_py(f, w)


def _axis_spacing_m(
    axis: int,
    spacings: np.ndarray,
    *,
    tilt_rad: float | None,
) -> float:
    """Native chart step → metres under the declared metric."""
    if axis in (AXIS_TILT, AXIS_GAMMA):
        return float(spacings[axis]) * float(LAMBDA_M_PER_RAD)
    if axis == AXIS_AZIMUTH:
        base = float(spacings[axis]) * float(LAMBDA_M_PER_RAD)
        if tilt_rad is not None:
            base *= max(float(np.sin(tilt_rad)), 1.0e-3)
        return base
    return float(spacings[axis])


def _remaining_axis_index(moved_axis: int, original_axis: int) -> int | None:
    """Map an original axis id onto the index tuple after ``moveaxis(..., 0)``."""
    if original_axis == moved_axis:
        return None
    if original_axis > moved_axis:
        return original_axis - 1
    return original_axis


def _edt_axis(
    dist_sq: np.ndarray,
    axis: int,
    *,
    spacings: np.ndarray,
    periodic: bool,
    tilt_axis: np.ndarray | None,
) -> np.ndarray:
    # ``moveaxis`` views are often non-contiguous; reshape+column writes then
    # silently corrupt the buffer. Force C order before flattening lines.
    moved = np.ascontiguousarray(np.moveaxis(dist_sq, axis, 0))
    n = int(moved.shape[0])
    if periodic:
        moved = np.ascontiguousarray(np.concatenate([moved, moved, moved], axis=0))
    n_work = int(moved.shape[0])
    out = np.empty(moved.shape, dtype=np.float64)
    line_shape = moved.shape[1:]
    n_lines = int(np.prod(line_shape)) if line_shape else 1
    flat = moved.reshape(n_work, n_lines)
    flat_out = out.reshape(n_work, n_lines)
    tilt_rem = _remaining_axis_index(axis, AXIS_TILT)
    for line in range(n_lines):
        idx = np.unravel_index(line, line_shape) if line_shape else ()
        tilt_rad = None
        if tilt_axis is not None and tilt_rem is not None and len(idx) > tilt_rem:
            tilt_rad = float(tilt_axis[int(idx[tilt_rem])])
        w = _axis_spacing_m(axis, spacings, tilt_rad=tilt_rad)
        flat_out[:, line] = _edt_1d_squared(flat[:, line], w)
    result = np.moveaxis(out, 0, axis)
    if periodic:
        start = (result.shape[axis] - n) // 2
        sl = [slice(None)] * result.ndim
        sl[axis] = slice(start, start + n)
        result = np.ascontiguousarray(result[tuple(sl)])
    else:
        result = np.ascontiguousarray(result)
    return result


def _distance_to_false(
    occ: np.ndarray,
    spacings: np.ndarray,
    periodic_axes: tuple[int, ...],
    tilt_axis: np.ndarray,
) -> np.ndarray:
    # Sites (0) are False cells; True cells start at +inf.
    f = np.where(occ, np.inf, 0.0).astype(np.float64)
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

    Non-occupied cells (including FK-budget unknowns) are treated as exterior
    for this far-field prior only — see :data:`EDT_WARNING`.
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


@dataclass(frozen=True)
class FlangeEdtConfig:
    occupancy_npz: str = "data/maps/flange_occupancy_smoke.npz"
    output_npz: str = "data/maps/flange_sdf_smoke.npz"
    periodic_axes: tuple[int, ...] = (AXIS_AZIMUTH, AXIS_GAMMA)


def build_flange_edt(
    cfg: FlangeEdtConfig | None = None,
    *,
    occupancy: np.ndarray | None = None,
    axes: tuple[np.ndarray, ...] | None = None,
    occupancy_meta: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute anisotropic 5-D EDT and write ``sdf`` + manifest."""
    from ird_playground.map.build_flange_tensor import CHART_NAMES

    cfg = cfg or FlangeEdtConfig()
    if occupancy is None or axes is None:
        blob = np.load(cfg.occupancy_npz, allow_pickle=False)
        occupancy = np.asarray(blob["occupancy"])
        axes = tuple(np.asarray(blob[name]) for name in CHART_NAMES)
        meta_path = Path(cfg.occupancy_npz).with_suffix(".meta.json")
        if occupancy_meta is None and meta_path.is_file():
            occupancy_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert occupancy is not None and axes is not None
    occ_bool = occupancy.astype(bool)
    spacings = np.array(
        [
            float(axes[0][1] - axes[0][0]) if len(axes[0]) > 1 else 0.03,
            float(axes[1][1] - axes[1][0]) if len(axes[1]) > 1 else 0.03,
            float(axes[2][1] - axes[2][0]) if len(axes[2]) > 1 else np.deg2rad(12.0),
            float(axes[3][1] - axes[3][0]) if len(axes[3]) > 1 else np.deg2rad(12.0),
            float(axes[4][1] - axes[4][0]) if len(axes[4]) > 1 else np.deg2rad(12.0),
        ],
        dtype=np.float64,
    )
    sdf = signed_distance_from_occupancy(
        occ_bool,
        spacings,
        periodic_axes=tuple(cfg.periodic_axes),
        tilt_axis_values=axes[AXIS_TILT],
    )
    arrays = {
        "sdf": sdf,
        "occupancy": occupancy.astype(np.uint8),
        **{name: np.asarray(axis, dtype=np.float32) for name, axis in zip(CHART_NAMES, axes)},
    }
    finite = np.isfinite(sdf)
    meta = {
        "schema": "flange_sdf_v1",
        "config": asdict(cfg),
        "warning": EDT_WARNING,
        "distance_definition": EDT_WARNING,
        "metric": metric_manifest(),
        "spacings_native": spacings.tolist(),
        "periodic_axes": list(cfg.periodic_axes),
        "periodic_axis_names": [CHART_NAMES[i] for i in cfg.periodic_axes],
        "pole_weight": "azimuth spacing *= max(sin(tilt), 1e-3)",
        "unknown_policy": (
            "Non-occupied voxels (FK unknowns) are treated as exterior for this "
            "far-field EDT prior only; they are not proven unreachable."
        ),
        "sdf_stats": {
            "min": float(sdf[finite].min()) if finite.any() else None,
            "max": float(sdf[finite].max()) if finite.any() else None,
            "mean": float(sdf[finite].mean()) if finite.any() else None,
            "n_positive": int((sdf > 0).sum()),
            "n_negative": int((sdf < 0).sum()),
            "n_zero": int((sdf == 0).sum()),
            "occupied_fraction": float(occ_bool.mean()),
        },
        "source_occupancy_meta": occupancy_meta,
    }
    out_path = Path(cfg.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return arrays, meta


__all__ = [
    "AXIS_AZIMUTH",
    "AXIS_GAMMA",
    "AXIS_TILT",
    "EDT_WARNING",
    "FlangeEdtConfig",
    "build_flange_edt",
    "signed_distance_from_occupancy",
]
