"""Public material-chart position features for the recovery experiment.

The feature vector in this module is deliberately small and causal.  It uses
only the state available to a controller at a decision time: the current and
candidate chart locations, the pending query interval, the last trusted
location, a short list of posterior modes, predicted field-of-view
probabilities, and anonymous packet responsibilities.  It never consumes a
``Scene`` object, truth labels, or evaluator-only fields.

The older runs use the longitudinal/lateral chart proxy. When a frozen
material surface is supplied, encode the harmonic-atlas surface address and
the lateral/depth bounds of the probe window along the sampled path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline


DEFAULT_S_RANGE = (2.0, 31.4)
DEFAULT_X_RANGE = (-30.0, 0.0)
DEFAULT_ANGLE_REF_DEG = 15.0
DEFAULT_TOP_K = 8
DEFAULT_PATH_SAMPLES = 4
DEFAULT_MAX_PACKETS = 3


@dataclass(frozen=True)
class EmbeddingLayout:
    """Offsets and dimensions of :func:`position_embedding` output."""

    path_samples: int = DEFAULT_PATH_SAMPLES
    top_k: int = DEFAULT_TOP_K
    max_packets: int = DEFAULT_MAX_PACKETS

    @property
    def dimension(self) -> int:
        # Current/endpoint (s, x): 4; candidate path: 2*path_samples;
        # query interval and last trusted delta/mask: 4; angle sin/cos: 2.
        # Each posterior mode
        # contributes weight, validity, relative (s,x), predicted visibility:
        # 5 * top_k.  Packet responsibilities are padded with a validity mask.
        return 10 + 2 * self.path_samples + 5 * self.top_k + 2 * self.max_packets + 1


class MaterialSurface:
    """Fast evaluation of the archived ScanTemplate surface splines."""

    def __init__(self, knots):
        knots = np.asarray(knots, dtype=float)
        self.spline = CubicSpline(knots[:, 0], knots[:, 1:], bc_type="natural", extrapolate=False)

    def address(self, s, x):
        h, theta, transverse = np.moveaxis(self.spline(s), -1, 0)
        theta = theta + .001 * np.asarray(x) / transverse
        return np.stack((h, np.sin(theta), np.cos(theta)), axis=-1)

    def window_features(self, s, x, fov):
        lower, upper = np.asarray(fov)
        # The two skin addresses describe the nominal lateral span. Depth
        # limits stay in the calibrated image coordinates.
        return np.r_[self.address(s, x).ravel(),
                     self.address(s, x + lower[0]).ravel(),
                     self.address(s, x + upper[0]).ravel(), [lower[1] / 60., upper[1] / 60.]]


EMBEDDING_DEFINITION: Mapping[str, Any] = {
    "name": "causal_chart_position_embedding_v1",
    "coordinates": "registered material-chart (s, x) proxy; not full harmonic 3-D coordinates",
    "uses": (
        "current probe chart position",
        "candidate trajectory samples",
        "query interval and last trusted chart location",
        "top posterior modes with their predicted target locations and FOV probability",
        "anonymous observed-packet target responsibilities",
    ),
    "does_not_use": ("Scene truth", "target roles", "hidden hypothesis", "evaluator labels"),
    "mode_order": "posterior mass descending, then public hypothesis index for ties",
}


def _finite(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scale(value: np.ndarray | float, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("normalization range must have high > low")
    return 2.0 * (np.asarray(value, dtype=float) - low) / (high - low) - 1.0


def _path_samples(values: np.ndarray, count: int) -> np.ndarray:
    values = _finite(values, name="candidate path")
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("candidate path arrays must be one-dimensional and non-empty")
    if count < 1:
        raise ValueError("path sample count must be positive")
    if len(values) == count:
        return values
    positions = np.linspace(0.0, len(values) - 1.0, count)
    return np.interp(positions, np.arange(len(values), dtype=float), values)


def _posterior_modes(
    belief: np.ndarray,
    target_positions: np.ndarray,
    visibility: np.ndarray,
    *,
    anchor_x: float,
    top_k: int,
) -> np.ndarray:
    """Return explicit top posterior modes without averaging hypotheses.

    The output columns are ``[weight, valid, rel_lateral, depth,
    fov_probability]``.  ``target_positions`` are the public two-dimensional
    predicted image/chart-plane locations at the candidate endpoint, so the
    first coordinate is made relative to the candidate's lateral position and
    the second is retained as a normalized depth feature.  Sorting is
    deterministic and does not change the underlying belief used by the
    Bayesian filter.
    """

    belief = _finite(belief, name="belief").reshape(-1)
    positions = _finite(target_positions, name="target_positions")
    vis = _finite(visibility, name="visibility").reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("target_positions must have shape (H, 2)")
    if len(belief) != len(positions) or len(vis) != len(belief):
        raise ValueError("belief, target_positions and visibility must share H")
    if len(belief) == 0 or float(np.sum(belief)) <= 0.0:
        raise ValueError("belief must contain positive mass")
    posterior = np.clip(belief, 0.0, None)
    posterior /= float(np.sum(posterior))
    order = np.lexsort((np.arange(len(posterior)), -posterior))[:top_k]
    out = np.zeros((top_k, 5), dtype=float)
    out[:, 1] = 0.0
    for row, index in enumerate(order):
        # Relative location is a public prediction at the current probe
        # position.  It is intentionally not replaced by a posterior mean.
        out[row] = (
            posterior[index],
            1.0,
            float(np.clip(_scale(positions[index, 0] - anchor_x, -30.0, 30.0), -2.0, 2.0)),
            float(np.clip(_scale(positions[index, 1], 0.0, 60.0), -2.0, 2.0)),
            float(np.clip(vis[index], 0.0, 1.0)),
        )
    return out


def position_embedding(
    *,
    current_s: float,
    current_x: float,
    candidate_s: np.ndarray,
    candidate_x: np.ndarray,
    query_interval: tuple[float, float],
    last_trusted_s: float | None,
    belief: np.ndarray,
    target_positions: np.ndarray,
    target_visibility: np.ndarray,
    packet_responsibilities: np.ndarray | None = None,
    angle_deg: float | None = None,
    s_range: tuple[float, float] = DEFAULT_S_RANGE,
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    angle_ref_deg: float = DEFAULT_ANGLE_REF_DEG,
    layout: EmbeddingLayout = EmbeddingLayout(),
    material_surface: MaterialSurface | None = None,
    field_of_view: tuple | None = None,
) -> np.ndarray:
    """Build a fixed-dimensional, public position/belief feature vector.

    Parameters are deliberately primitive arrays so this module can be used
    by the existing controller without importing its private ``Scene`` type.
    ``target_positions[h]`` and ``target_visibility[h]`` must be the public
    hypothesis predictions for the current decision state.  The resulting
    vector is deterministic, finite, and suitable for a small regressor or a
    ProDMP skill selector.
    """

    if not isinstance(layout, EmbeddingLayout):
        raise TypeError("layout must be an EmbeddingLayout")
    current_s = float(_finite(current_s, name="current_s"))
    current_x = float(_finite(current_x, name="current_x"))
    if not (np.isfinite(s_range[0]) and np.isfinite(s_range[1])
            and np.isfinite(x_range[0]) and np.isfinite(x_range[1])):
        raise ValueError("normalization ranges must be finite")

    path_s = _path_samples(candidate_s, layout.path_samples)
    path_x = _path_samples(candidate_x, layout.path_samples)
    if len(candidate_s) != len(candidate_x):
        raise ValueError("candidate_s and candidate_x must have the same length")
    q = _finite(query_interval, name="query_interval").reshape(-1)
    if q.shape != (2,):
        raise ValueError("query_interval must be a pair (start, end)")
    if q[1] < q[0]:
        raise ValueError("query_interval end must be >= start")

    features: list[float] = [
        float(np.clip(_scale(current_s, *s_range), -2.0, 2.0)),
        float(np.clip(_scale(current_x, *x_range), -2.0, 2.0)),
        float(np.clip(_scale(path_s[-1] - current_s, -20.0, 20.0), -2.0, 2.0)),
        float(np.clip(_scale(path_x[-1] - current_x, -30.0, 30.0), -2.0, 2.0)),
    ]
    for s_value, x_value in zip(path_s, path_x):
        features.extend((
            float(np.clip(_scale(s_value - current_s, -20.0, 20.0), -2.0, 2.0)),
            float(np.clip(_scale(x_value - current_x, -30.0, 30.0), -2.0, 2.0)),
        ))
    dimension = layout.dimension
    if material_surface is not None:
        if field_of_view is None:
            raise ValueError("material addresses require the calibrated field of view")
        chart = material_surface.window_features(np.r_[current_s, path_s], np.r_[current_x, path_x], field_of_view)
        dimension += len(chart) - len(features)
        features = chart.tolist()
    # The four path samples already include the endpoint displacement.  The
    # first four entries are current position plus endpoint displacement; the
    # explicit path contributes 2*path_samples entries below.
    features.extend((
        float(np.clip(_scale(q[0] - current_s, -20.0, 20.0), -2.0, 2.0)),
        float(np.clip(_scale(q[1] - current_s, -20.0, 20.0), -2.0, 2.0)),
        float(np.clip(_scale((last_trusted_s if last_trusted_s is not None else current_s - 10.0)
                            - current_s, -20.0, 20.0), -2.0, 2.0)),
        float(last_trusted_s is not None),
    ))
    angle = angle_ref_deg if angle_deg is None else float(angle_deg)
    angle_delta = np.deg2rad(angle - angle_ref_deg)
    features.extend((float(np.sin(angle_delta)), float(np.cos(angle_delta))))

    modes = _posterior_modes(
        belief,
        target_positions,
        target_visibility,
        anchor_x=float(path_x[-1]),
        top_k=layout.top_k,
    )
    features.extend(modes.reshape(-1).tolist())

    responsibilities = np.zeros(layout.max_packets, dtype=float)
    responsibility_mask = np.zeros(layout.max_packets, dtype=float)
    if packet_responsibilities is not None:
        values = _finite(packet_responsibilities, name="packet_responsibilities").reshape(-1)
        n = min(len(values), layout.max_packets)
        responsibilities[:n] = np.clip(values[:n], 0.0, 1.0)
        responsibility_mask[:n] = 1.0
    features.extend(responsibilities.tolist())
    features.extend(responsibility_mask.tolist())
    features.append(float(np.clip(-np.sum(np.asarray(belief) / np.sum(belief)
                                        * np.log(np.maximum(np.asarray(belief) / np.sum(belief), 1e-12)))
                             / np.log(max(len(np.asarray(belief)), 2)), 0.0, 1.0)))

    result = np.asarray(features, dtype=float)
    if result.shape != (dimension,) or np.any(~np.isfinite(result)):
        raise RuntimeError(f"embedding shape/finite invariant failed: {result.shape}, {layout.dimension}")
    return result


__all__ = ["EmbeddingLayout", "EMBEDDING_DEFINITION", "position_embedding"]

