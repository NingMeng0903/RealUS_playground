"""Paper-faithful colormaps.

Zacharias Fig 3 convention (per the published figures):

* **Low D(x)** (few reachable directions, poor) → **red**
* **High D(x)** (many directions, good) → **deep blue**
* **21 discrete bands** on an absolute **0 … 100 %** reachability index (Eq. 30/33)

Vahrenkamp palettes are unchanged (density / placement score).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

ZACHARIAS_ROBOT_GRAY: str = "#666666"
ZACHARIAS_DIR_FACE_REACHABLE: str = "#3faa3f"
ZACHARIAS_DIR_FACE_MISSING: str = "#c8c8c8"
ZACHARIAS_DIR_EDGE: str = "#000000"

VAHRENKAMP_BEST_GOLD: str = "#ffd700"
VAHRENKAMP_INFEASIBLE_GRAY: str = "#888888"

# Paper Fig 3: red (low D) → orange → green → cyan → blue (high D).
# Keep stops saturated/bright so dense glyphs stay readable under lighting.
_ZACHARIAS_D_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#e31a1c"),
    (0.18, "#fd8d3c"),
    (0.36, "#fed976"),
    (0.52, "#78c679"),
    (0.68, "#41b6c4"),
    (0.84, "#2b8cbe"),
    (1.00, "#0868ac"),
)

_VAHRENKAMP_IRM_STOPS: tuple[tuple[float, str], ...] = (
    (0.00, "#1a1a4b"),
    (0.25, "#3e6cb2"),
    (0.50, "#3fa87a"),
    (0.75, "#fbcf5a"),
    (1.00, "#c02020"),
)

ZACHARIAS_COLOR_LEVELS: int = 21
# Paper Eq. (30): D(g) = R(g)/n_p * 100  (percentage of sphere points reachable).
ZACHARIAS_COLORBAR_MAX: float = 100.0


def _cmap_from_stops(name: str, stops: Iterable[tuple[float, str]]) -> LinearSegmentedColormap:
    xs = [s[0] for s in stops]
    rgbs = [to_rgb(s[1]) for s in stops]
    seg = {
        "red":   [(x, r, r) for x, (r, _g, _b) in zip(xs, rgbs)],
        "green": [(x, g, g) for x, (_r, g, _b) in zip(xs, rgbs)],
        "blue":  [(x, b, b) for x, (_r, _g, b) in zip(xs, rgbs)],
    }
    return LinearSegmentedColormap(name, seg)  # type: ignore[arg-type]


def make_zacharias_d_cmap() -> LinearSegmentedColormap:
    return _cmap_from_stops("zacharias_d", _ZACHARIAS_D_STOPS)


def make_zacharias_d_cmap_discrete(n_levels: int = ZACHARIAS_COLOR_LEVELS) -> ListedColormap:
    """21-step discrete ramp matching the paper colorbar."""
    base = make_zacharias_d_cmap()
    n = max(2, int(n_levels))
    colors = [base(i / (n - 1))[:3] for i in range(n)]
    return ListedColormap(colors, name=f"zacharias_d_{n}")


def make_vahrenkamp_irm_cmap() -> LinearSegmentedColormap:
    return _cmap_from_stops("vahrenkamp_irm", _VAHRENKAMP_IRM_STOPS)


def sample_cmap(cmap: LinearSegmentedColormap | ListedColormap, n: int = 256) -> np.ndarray:
    """Return (n, 3) uint8 RGB samples for PyVista's ``cmap=`` API."""
    xs = np.linspace(0.0, 1.0, n)
    rgba = cmap(xs)
    return (rgba[:, :3] * 255.0).astype(np.uint8)


def discretize_d_for_display(
    d: np.ndarray,
    *,
    clim: tuple[float, float],
    n_levels: int = ZACHARIAS_COLOR_LEVELS,
    bar_max: float = ZACHARIAS_COLORBAR_MAX,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Map stored D(x) fraction to paper % index in [0, bar_max].

    Stored ``d_value`` is (# reachable dirs) / n_orient ∈ [0, 1].
    Paper reachability index D(g) = that ratio × 100 (Eq. 30).
    ``clim`` is in the same fraction units; default (0, 1) → full 0–100 % scale.
    """
    lo, hi = float(clim[0]), float(clim[1])
    span = max(hi - lo, 1e-12)
    t = np.clip((np.asarray(d, dtype=np.float64) - lo) / span, 0.0, 1.0)
    n = max(2, int(n_levels))
    bins = np.floor(t * (n - 1)).astype(np.int32)
    tick_vals = np.linspace(0.0, float(bar_max), n, dtype=np.float64)
    display = tick_vals[bins].astype(np.float32)
    return display, (0.0, float(bar_max))


def colorbar_tick_values(
    n_levels: int = ZACHARIAS_COLOR_LEVELS,
    bar_max: float = ZACHARIAS_COLORBAR_MAX,
) -> np.ndarray:
    """Tick positions for the vertical 21-step legend (0, 5, 10, …, 100)."""
    n = max(2, int(n_levels))
    return np.linspace(0.0, float(bar_max), n)
