"""IRD point-cloud visualization: GT vs network prediction.

Shows base-relative-to-TCP translations colored by reachability quality,
similar in spirit to Zacharias D(x) sphere glyphs (here as scatter / slices).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _scatter_panel(ax, xyz: np.ndarray, values: np.ndarray, title: str, clim=(0.0, 1.0)):
    v = np.clip(np.asarray(values, dtype=np.float64), clim[0], clim[1])
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=v,
        cmap="turbo",
        s=2,
        alpha=0.55,
        vmin=clim[0],
        vmax=clim[1],
        linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Δx (m)")
    ax.set_ylabel("Δy (m)")
    ax.set_zlabel("Δz (m)")
    return sc


def _slice_panel(ax, xyz: np.ndarray, values: np.ndarray, axis: str, title: str, clim=(0.0, 1.0)):
    """2D max-projection slice onto a coordinate plane."""
    v = np.asarray(values, dtype=np.float64)
    if axis == "xy":
        u, w = xyz[:, 0], xyz[:, 1]
        ax.set_xlabel("Δx")
        ax.set_ylabel("Δy")
    elif axis == "xz":
        u, w = xyz[:, 0], xyz[:, 2]
        ax.set_xlabel("Δx")
        ax.set_ylabel("Δz")
    else:
        u, w = xyz[:, 1], xyz[:, 2]
        ax.set_xlabel("Δy")
        ax.set_ylabel("Δz")
    # bin max projection
    n = 80
    u0, u1 = float(u.min()), float(u.max())
    w0, w1 = float(w.min()), float(w.max())
    if u1 <= u0 or w1 <= w0:
        ax.set_title(title)
        return None
    ui = np.clip(((u - u0) / (u1 - u0) * (n - 1e-9)).astype(int), 0, n - 1)
    wi = np.clip(((w - w0) / (w1 - w0) * (n - 1e-9)).astype(int), 0, n - 1)
    grid = np.full((n, n), np.nan, dtype=np.float64)
    flat = wi * n + ui
    order = np.argsort(v)  # low first so max overwrites
    for idx in order:
        grid[wi[idx], ui[idx]] = v[idx]
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[u0, u1, w0, w1],
        cmap="turbo",
        vmin=clim[0],
        vmax=clim[1],
        aspect="equal",
    )
    ax.set_title(title, fontsize=10)
    return im


def render_ird_comparison(
    *,
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray | None,
    out_path: str | Path,
    max_points: int = 40_000,
    seed: int = 0,
    value_name: str = "d",
) -> Path:
    """Write a multi-panel PNG: 3D GT [/ Pred / |err|] + XY/XZ/YZ slices."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    n = xyz.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        xyz, gt = xyz[idx], gt[idx]
        if pred is not None:
            pred = np.asarray(pred, dtype=np.float64).reshape(-1)[idx]

    has_pred = pred is not None
    ncols = 3 if has_pred else 1
    fig = plt.figure(figsize=(5.2 * ncols, 9.0), dpi=140)

    # Row 0: 3D scatters
    ax0 = fig.add_subplot(2, ncols, 1, projection="3d")
    sc = _scatter_panel(ax0, xyz, gt, f"GT {value_name}")
    if has_pred:
        ax1 = fig.add_subplot(2, ncols, 2, projection="3d")
        _scatter_panel(ax1, xyz, pred, f"Pred {value_name}")
        ax2 = fig.add_subplot(2, ncols, 3, projection="3d")
        err = np.abs(pred - gt)
        _scatter_panel(ax2, xyz, err, f"|Pred−GT|  MAE={err.mean():.3f}", clim=(0.0, max(0.2, float(np.percentile(err, 95)))))

    # Row 1: slices (XY for GT, and Pred/err if available)
    if has_pred:
        ax_s0 = fig.add_subplot(2, ncols, 4)
        _slice_panel(ax_s0, xyz, gt, "xy", "GT slice XY")
        ax_s1 = fig.add_subplot(2, ncols, 5)
        _slice_panel(ax_s1, xyz, pred, "xy", "Pred slice XY")
        ax_s2 = fig.add_subplot(2, ncols, 6)
        _slice_panel(ax_s2, xyz, np.abs(pred - gt), "xy", "Abs err XY", clim=(0.0, 0.5))
    else:
        ax_s0 = fig.add_subplot(2, 3, 4)
        _slice_panel(ax_s0, xyz, gt, "xy", "GT XY")
        ax_s1 = fig.add_subplot(2, 3, 5)
        _slice_panel(ax_s1, xyz, gt, "xz", "GT XZ")
        ax_s2 = fig.add_subplot(2, 3, 6)
        _slice_panel(ax_s2, xyz, gt, "yz", "GT YZ")

    fig.colorbar(sc, ax=fig.axes[:ncols], fraction=0.02, pad=0.02, label=value_name)
    fig.suptitle("IRD reachability (ΔT translation in TCP←base frame)", fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.05, wspace=0.22, hspace=0.28)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def features_to_xyz(features: np.ndarray) -> np.ndarray:
    """First 3 dims of IRD feature vector = ΔT translation."""
    f = np.asarray(features, dtype=np.float64)
    return f[:, :3]
