"""Zacharias-style sphere + robot visualization for **IRD** fields.

Colour meaning (do not confuse):

* **IRD ``d``** (default for this module) = ``p_reach * q_comfort`` ∈ [0, 1].
  On the capability map GT, positives use ``p_reach=1``, so voxel colour =
  the map comfort channel (same as training ``q_comfort`` / ``d``).
* **Zacharias ``D(x)``** = (# reachable orients) / ``n_orient`` — capability
  reachability index only. Use ``channel="reach_D"`` if you explicitly want that.

Colour bar is a **fixed** clim ``(0, 1)`` → 0–100 display ticks for cross-figure
compare (same convention as Zacharias bars; the *quantity* on the bar is IRD ``d``
unless ``reach_D`` is selected).

Robot default: horizontal ultrasound probe (``probe_default.yaml``), **not** the
stock ``z=0.220`` straight TCP.
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from ird_playground.viz.rm75_ns import ensure_rm75_namespace
from ird_playground.viz.viz_style import PROBE_COMPARE_CLIM

# Fixed absolute scale for cross-figure compare (fraction → bar ticks via sphere_glyphs).
FIXED_IRD_CLIM: tuple[float, float] = PROBE_COMPARE_CLIM
# Back-compat alias
FIXED_D_CLIM = FIXED_IRD_CLIM


def _load_cm(map_dir: str | Path):
    ensure_rm75_namespace()
    from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap

    return CapabilityMap.load(Path(map_dir), mmap=True)


def _with_scalar(cm, values: np.ndarray):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] != cm.d_value.shape[0]:
        raise ValueError(f"scalar length {values.shape[0]} != map rows {cm.d_value.shape[0]}")
    try:
        return replace(cm, d_value=values)
    except TypeError:
        out = copy(cm)
        setattr(out, "d_value", values)
        return out


def voxel_ird_d_gt(cm, *, comfort_from: str = "auto") -> np.ndarray:
    """Per-voxel IRD GT colour = training comfort (``d`` when ``p_reach=1``)."""
    d_reach = np.asarray(cm.d_value, dtype=np.float64)
    mu = None if cm.mu_mean is None else np.asarray(cm.mu_mean, dtype=np.float64)
    if comfort_from == "d_value":
        return d_reach.astype(np.float32)
    if comfort_from == "mu_mean":
        if mu is None:
            return d_reach.astype(np.float32)
        return np.clip(mu, 0.0, 1.0).astype(np.float32)
    # auto — matches ird.export_gt._comfort_channel
    if mu is not None and np.isfinite(mu).any():
        q = np.clip(mu / (np.abs(mu) + 1.0), 0.0, 1.0) * d_reach
        return q.astype(np.float32)
    return d_reach.astype(np.float32)


def default_probe_urdf(playground_root: str | Path | None = None) -> Path:
    from ird_playground.probe.transform import ensure_probe_visual_urdf

    root = Path(playground_root) if playground_root else Path(__file__).resolve().parents[2]
    return ensure_probe_visual_urdf(playground_root=root)


def render_ird_spheres(
    map_dir: str | Path,
    out_path: str | Path,
    *,
    scalars: np.ndarray | None = None,
    channel: str = "ird_d",
    comfort_from: str = "auto",
    d_min: float = 0.02,
    clim: tuple[float, float] = FIXED_IRD_CLIM,
    clim_auto: bool = False,
    robot_urdf: str | Path | None = None,
    size: tuple[int, int] = (2800, 1000),
    sphere_radius_m: float | None = None,
) -> Path:
    """Render RealMan (horizontal probe) + spheres coloured by IRD / optional D."""
    ensure_rm75_namespace()
    from rm75_control.tools.reachability.viz.sphere_glyphs import render_reachability_index

    cm = _load_cm(map_dir)
    if scalars is not None:
        field = np.asarray(scalars, dtype=np.float32)
    elif channel == "reach_D":
        field = np.asarray(cm.d_value, dtype=np.float32)
    else:
        field = voxel_ird_d_gt(cm, comfort_from=comfort_from)
    cm = _with_scalar(cm, field)

    if robot_urdf is None:
        robot_urdf = default_probe_urdf()

    return render_reachability_index(
        cm,
        Path(out_path),
        robot_urdf=robot_urdf,
        d_min=float(d_min),
        clim=None if clim_auto else (float(clim[0]), float(clim[1])),
        clim_auto=bool(clim_auto),
        size=size,
        sphere_radius_m=sphere_radius_m,
    )


# Alias kept for older imports
render_capability_spheres = render_ird_spheres


def predict_voxel_ird_d(
    map_dir: str | Path,
    net,
    *,
    n_orients: int = 8,
    batch_size: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Per-voxel predicted IRD ``d = p_reach * q_comfort`` (mean over sampled orients)."""
    from ird_playground.probe.se3 import (
        batch_features_from_delta_T,
        complete_frame_from_tool_axis,
        delta_T_tcp_inv_base,
        mat4_from_Rt,
    )

    cm = _load_cm(map_dir)
    rng = np.random.default_rng(seed)
    orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
    centres = cm.grid.center_of(cm.voxel_ids)
    m = int(centres.shape[0])
    k = max(1, min(int(n_orients), int(orients.shape[0])))
    oi = rng.choice(orients.shape[0], size=k, replace=False)

    feats = np.empty((m * k, 9), dtype=np.float32)
    row = 0
    for p in centres:
        for j in oi:
            R = complete_frame_from_tool_axis(orients[int(j)])
            dT = delta_T_tcp_inv_base(mat4_from_Rt(R, p))
            feats[row] = batch_features_from_delta_T(dT[None, ...])[0]
            row += 1

    preds = []
    for i in range(0, feats.shape[0], batch_size):
        chunk = net.score_features_np(feats[i : i + batch_size])
        preds.append(chunk["d"])
    d = np.concatenate(preds, axis=0).reshape(m, k)
    return d.mean(axis=1).astype(np.float32)


def render_gt_vs_pred_spheres(
    map_dir: str | Path,
    out_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    channel: str = "ird_d",
    comfort_from: str = "auto",
    d_min: float = 0.02,
    clim: tuple[float, float] = FIXED_IRD_CLIM,
    clim_auto: bool = False,
    n_orients: int = 8,
    robot_urdf: str | Path | None = None,
    stem: str = "ird_spheres",
) -> dict[str, Path]:
    """Write GT IRD sphere figure; if checkpoint set, also pred + stacked compare."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if robot_urdf is None:
        robot_urdf = default_probe_urdf()

    paths: dict[str, Path] = {}
    qty = "IRD d=p_reach·q_comfort" if channel == "ird_d" else "Zacharias D(x)"
    gt_path = out_dir / f"{stem}_gt.png"
    render_ird_spheres(
        map_dir,
        gt_path,
        channel=channel,
        comfort_from=comfort_from,
        d_min=d_min,
        clim=clim,
        clim_auto=clim_auto,
        robot_urdf=robot_urdf,
    )
    paths["gt"] = gt_path

    if checkpoint is None:
        return paths

    from ird_playground.neural.model import NeuralIRD

    net = NeuralIRD.load(checkpoint, device=device)
    if channel == "reach_D":
        # Approximate D(x) ≈ mean p_reach over sampled orients
        from ird_playground.probe.se3 import (
            batch_features_from_delta_T,
            complete_frame_from_tool_axis,
            delta_T_tcp_inv_base,
            mat4_from_Rt,
        )

        cm = _load_cm(map_dir)
        rng = np.random.default_rng(0)
        orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
        centres = cm.grid.center_of(cm.voxel_ids)
        m = int(centres.shape[0])
        k = max(1, min(int(n_orients), int(orients.shape[0])))
        oi = rng.choice(orients.shape[0], size=k, replace=False)
        feats = []
        for p in centres:
            for j in oi:
                R = complete_frame_from_tool_axis(orients[int(j)])
                feats.append(
                    batch_features_from_delta_T(
                        delta_T_tcp_inv_base(mat4_from_Rt(R, p))[None, ...]
                    )[0]
                )
        feats = np.asarray(feats, dtype=np.float32)
        chunks = []
        for i in range(0, feats.shape[0], 4096):
            chunks.append(net.score_features_np(feats[i : i + 4096])["p_reach"])
        pred = np.concatenate(chunks).reshape(m, k).mean(axis=1).astype(np.float32)
    else:
        pred = predict_voxel_ird_d(map_dir, net, n_orients=n_orients)

    pred_path = out_dir / f"{stem}_pred.png"
    render_ird_spheres(
        map_dir,
        pred_path,
        scalars=pred,
        d_min=d_min,
        clim=clim,
        clim_auto=clim_auto,
        robot_urdf=robot_urdf,
    )
    paths["pred"] = pred_path

    gt_img = plt.imread(gt_path)
    pr_img = plt.imread(pred_path)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=120)
    axes[0].imshow(gt_img)
    axes[0].set_title(f"GT  {qty}   [fixed clim 0–1]")
    axes[0].axis("off")
    axes[1].imshow(pr_img)
    axes[1].set_title(f"Pred  {qty}   [same fixed clim]")
    axes[1].axis("off")
    fig.tight_layout()
    compare = out_dir / f"{stem}_gt_vs_pred.png"
    fig.savefig(compare, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["compare"] = compare
    return paths
