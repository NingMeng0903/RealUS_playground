"""Matplotlib overlay helpers for anatomy / SMPL preview figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SMPL_TPOSE_COLOR = "#3366cc"
SMPL_POSED_COLOR = "#d0a000"
ANATOMY_COLOR = "#cc3333"
LEG_BONE_COLOR = "#f5f0dc"
LEG_BONE_EDGE = "#333333"


def _preview_overlay_legend(*, include_bones: bool) -> list:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ANATOMY_COLOR,
            markersize=5,
            alpha=0.85,
            linestyle="None",
            label="anatomy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SMPL_TPOSE_COLOR,
            markersize=5,
            alpha=0.85,
            linestyle="None",
            label="smpl tpose",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SMPL_POSED_COLOR,
            markersize=5,
            alpha=0.85,
            linestyle="None",
            label="smpl fit posed",
        ),
    ]
    if include_bones:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LEG_BONE_COLOR,
                markeredgecolor=LEG_BONE_EDGE,
                markeredgewidth=0.6,
                markersize=6,
                linestyle="None",
                label="leg bones",
            )
        )
    return handles


def _overlay_legend_handles(*, anatomy_label: str, smpl_label: str, smpl_color: str, include_bones: bool) -> list:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ANATOMY_COLOR,
            markersize=5,
            alpha=0.85,
            linestyle="None",
            label=anatomy_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=smpl_color,
            markersize=5,
            alpha=0.85,
            linestyle="None",
            label=smpl_label,
        ),
    ]
    if include_bones:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LEG_BONE_COLOR,
                markeredgecolor=LEG_BONE_EDGE,
                markeredgewidth=0.6,
                markersize=6,
                linestyle="None",
                label="leg bones",
            )
        )
    return handles


def _leg_bone_legend_handle():
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=LEG_BONE_COLOR,
        markeredgecolor=LEG_BONE_EDGE,
        markeredgewidth=0.6,
        markersize=6,
        linestyle="None",
        label="leg bones",
    )


def is_leg_vein_centerline_label(label: str) -> bool:
    """True for exported leg vein centerline names (SUPFEM/POP omit trailing _V)."""
    s = str(label)
    return s.endswith("_V") or s.endswith("SUPFEMV") or s.endswith("POPV")


def _centerline_legend_handle(label: str, rgb: tuple[int, int, int]):
    from matplotlib.lines import Line2D

    color = tuple(v / 255.0 for v in rgb)
    return Line2D([0], [0], color=color, linewidth=2.2, label=label)


def _polyline_arc_params(line: np.ndarray) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return np.zeros(pts.shape[0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if total < 1.0e-8:
        return np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float32)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return (cum / total).astype(np.float32)


def _polyline_rest_length(line: np.ndarray) -> float:
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1.0e-8 or nb < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    a /= na
    b /= nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, ortho)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
        vx = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return (np.eye(3, dtype=np.float64) - 2.0 * (vx @ vx)).astype(np.float32)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return (np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))).astype(np.float32)


def _transfer_polyline_rigid(rest_line: np.ndarray, posed_start: np.ndarray, posed_end: np.ndarray) -> np.ndarray:
    """Rigidly move a rest polyline between skinned endpoints (preserves local shape)."""
    rest = np.asarray(rest_line, dtype=np.float32).reshape(-1, 3)
    p0 = np.asarray(posed_start, dtype=np.float32).reshape(3)
    p1 = np.asarray(posed_end, dtype=np.float32).reshape(3)
    if rest.shape[0] <= 1:
        return rest.copy()
    dr = rest[-1] - rest[0]
    dp = p1 - p0
    lr = float(np.linalg.norm(dr))
    lp = float(np.linalg.norm(dp))
    if lr < 1.0e-8:
        return np.stack([p0 + (p1 - p0) * t for t in _polyline_arc_params(rest)], axis=0).astype(np.float32)
    rot = _rotation_from_vectors(dr, dp)
    scale = lp / lr
    out = ((rest - rest[0]) @ rot.T) * scale + p0
    out[0] = p0
    out[-1] = p1
    return out.astype(np.float32)


def _snap_polyline_to_segment_mesh(
    line: np.ndarray,
    posed_vertices: np.ndarray,
    segment_mask: np.ndarray,
    *,
    max_dist: float = 0.018,
    blend: float = 0.25,
) -> np.ndarray:
    """Softly pull interior points toward the posed segment mesh (same label)."""
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    mask = np.asarray(segment_mask, dtype=bool).reshape(-1)
    seg_pts = np.asarray(posed_vertices, dtype=np.float32).reshape(-1, 3)[mask]
    if seg_pts.shape[0] == 0 or pts.shape[0] <= 2:
        return pts.copy()
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(seg_pts)
    except Exception:
        return pts.copy()
    out = pts.copy()
    alpha = float(np.clip(blend, 0.0, 1.0))
    for i in range(1, pts.shape[0] - 1):
        d, j = tree.query(out[i], k=1)
        if float(d) <= float(max_dist):
            target = seg_pts[int(j)]
            out[i] = ((1.0 - alpha) * out[i] + alpha * target).astype(np.float32)
    return out


def refresh_short_rigid_centerlines(
    centerlines_posed: dict[str, np.ndarray],
    centerlines_rest: dict[str, np.ndarray],
    *,
    labels: tuple[str, ...] = ("L_POPV", "R_POPV"),
) -> None:
    for label in labels:
        rest = centerlines_rest.get(label)
        posed = centerlines_posed.get(label)
        if rest is None or posed is None or np.asarray(rest).shape[0] < 2:
            continue
        centerlines_posed[label] = _transfer_polyline_rigid(
            np.asarray(rest, dtype=np.float32),
            np.asarray(posed[0], dtype=np.float32),
            np.asarray(posed[-1], dtype=np.float32),
        )


def skin_centerlines_to_posed(
    asset: AnatomyRiggedAsset,
    centerlines_rest: dict[str, np.ndarray],
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
    anchor_vertices: np.ndarray | None = None,
    anchor_weights: np.ndarray | None = None,
    posed_segment_vertices: np.ndarray | None = None,
    posed_segment_labels: np.ndarray | None = None,
    short_segment_length_m: float = 0.11,
) -> dict[str, np.ndarray]:
    """Skin each rest centerline point with asset (Blender-exported) LBS weights."""
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_points

    anchor_kw = {
        "anchor_vertices": anchor_vertices,
        "anchor_weights": anchor_weights,
        "neighbor_k": 6,
    }
    skin_kw = {"pose_axis_angle": pose_axis_angle, "transl": transl, **anchor_kw}
    out: dict[str, np.ndarray] = {}
    for label, line in centerlines_rest.items():
        arr = np.asarray(line, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] < 2:
            continue
        out[label] = skin_points(asset, arr, **skin_kw)
    return out


def _align_branch_exit_rotation(
    upstream: np.ndarray,
    downstream: np.ndarray,
    *,
    blend: float = 0.58,
) -> None:
    """Rotate downstream interior points toward upstream exit direction (endpoints unchanged)."""
    up = np.asarray(upstream, dtype=np.float32).reshape(-1, 3)
    down = np.asarray(downstream, dtype=np.float32).reshape(-1, 3)
    if up.shape[0] < 2 or down.shape[0] < 4:
        return
    up_dir = up[-1] - up[-2]
    nu = float(np.linalg.norm(up_dir))
    if nu < 1.0e-6:
        return
    up_dir = (up_dir / nu).astype(np.float32)
    anchor = down[0].copy()
    alpha = float(np.clip(blend, 0.0, 1.0))
    for idx in range(1, down.shape[0] - 1):
        vec = down[idx] - anchor
        ln = float(np.linalg.norm(vec))
        if ln < 1.0e-6:
            continue
        cur_dir = (vec / ln).astype(np.float32)
        target_dir = (1.0 - alpha) * cur_dir + alpha * up_dir
        tn = float(np.linalg.norm(target_dir))
        if tn < 1.0e-6:
            continue
        target_dir = (target_dir / tn).astype(np.float32)
        down[idx] = anchor + target_dir * ln


def align_centerline_junction_tangents(centerlines: dict[str, np.ndarray], *, blend: float = 0.68) -> None:
    """Nudge downstream branches at shared junctions (calf splits from POP)."""
    for up_label, down_label in (
        ("L_POPV", "L_POST_TIB_V"),
        ("L_POPV", "L_PERONEAL_V"),
        ("R_POPV", "R_POST_TIB_V"),
        ("R_POPV", "R_PERONEAL_V"),
    ):
        up = centerlines.get(up_label)
        down = centerlines.get(down_label)
        if up is None or down is None or up.shape[0] < 2 or down.shape[0] < 3:
            continue
        up_dir = up[-1] - up[-2]
        nu = float(np.linalg.norm(up_dir))
        if nu < 1.0e-6:
            continue
        up_dir = (up_dir / nu).astype(np.float32)
        anchor = np.asarray(down[0], dtype=np.float32).reshape(3)
        seg = np.asarray(down[1], dtype=np.float32).reshape(3) - anchor
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1.0e-6:
            continue
        target = anchor + up_dir * seg_len
        alpha = float(np.clip(blend, 0.0, 1.0))
        down[1] = ((1.0 - alpha) * down[1] + alpha * target).astype(np.float32)


def align_sup_pop_exit_rotations(centerlines: dict[str, np.ndarray]) -> None:
    for up_label, down_label in (("L_SUPFEMV", "L_POPV"), ("R_SUPFEMV", "R_POPV")):
        up = centerlines.get(up_label)
        down = centerlines.get(down_label)
        if up is None or down is None:
            continue
        _align_branch_exit_rotation(up, down, blend=0.58)


def pin_centerline_junctions(centerlines: dict[str, np.ndarray]) -> None:
    """Re-pin shared junction endpoints so adjacent segments meet exactly."""
    groups = (
        (("L_COM_FEM_V", -1), ("L_SUPFEMV", 0), ("L_SAPH_V", 0)),
        (("R_COM_FEM_V", -1), ("R_SUPFEMV", 0), ("R_SAPH_V", 0)),
        (("L_SUPFEMV", -1), ("L_POPV", 0)),
        (("R_SUPFEMV", -1), ("R_POPV", 0)),
        (("L_POPV", -1), ("L_POST_TIB_V", 0), ("L_PERONEAL_V", 0)),
        (("R_POPV", -1), ("R_POST_TIB_V", 0), ("R_PERONEAL_V", 0)),
    )
    for group in groups:
        pts: list[np.ndarray] = []
        refs: list[tuple[str, int]] = []
        for label, end in group:
            line = centerlines.get(label)
            if line is None or line.shape[0] < 1:
                continue
            idx = 0 if end == 0 else -1
            pts.append(np.asarray(line[idx], dtype=np.float32).reshape(3))
            refs.append((label, end))
        if len(pts) < 2:
            continue
        junction = np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)
        for label, end in refs:
            if end == 0:
                centerlines[label][0] = junction
            else:
                centerlines[label][-1] = junction


def _bbox_from_centerlines(centerlines: dict[str, np.ndarray], *, pad: float = 0.07) -> tuple[np.ndarray, np.ndarray] | None:
    lines = [
        np.asarray(line, dtype=np.float32).reshape(-1, 3)
        for line in centerlines.values()
        if np.asarray(line, dtype=np.float32).size >= 6
    ]
    if not lines:
        return None
    pts = np.concatenate(lines, axis=0)
    span = np.ptp(pts, axis=0)
    margin = np.maximum(span * float(pad), 0.035).astype(np.float32)
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


def _clip_points_near_centerlines(
    points: np.ndarray,
    centerlines: dict[str, np.ndarray],
    *,
    radius: float = 0.11,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    chunks = [np.asarray(line, dtype=np.float32).reshape(-1, 3) for line in centerlines.values() if np.asarray(line).size]
    if not chunks:
        return pts
    anchors = np.concatenate(chunks, axis=0)
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(anchors).query(pts, k=1)
        return pts[dist <= float(radius)]
    except Exception:
        return pts


def _clip_points_aabb(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mask = np.all(pts >= lo[None, :], axis=1) & np.all(pts <= hi[None, :], axis=1)
    return pts[mask]


def _fit_axis_limits(ax, clouds: list[np.ndarray], i: int, j: int, *, pad: float = 0.08) -> None:
    chunks = [np.asarray(c, dtype=np.float32).reshape(-1, 3) for c in clouds if c is not None and np.asarray(c).size]
    if not chunks:
        return
    pts = np.concatenate(chunks, axis=0)
    xs = pts[:, i]
    ys = pts[:, j]
    if xs.size == 0:
        return
    xr = float(np.ptp(xs))
    yr = float(np.ptp(ys))
    mx = max(xr, yr, 1.0e-3) * pad
    ax.set_xlim(float(xs.min()) - mx, float(xs.max()) + mx)
    ax.set_ylim(float(ys.min()) - mx, float(ys.max()) + mx)


def dense_body_cloud(vertices: np.ndarray, *, step: int = 1) -> np.ndarray:
    pts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    return pts[:: max(1, int(step))]


def anatomy_cloud(vertices: np.ndarray, *, step: int = 8) -> np.ndarray:
    pts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    return pts[:: max(1, int(step))]


def _bone_axis_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].astype(np.float32)
    u = vh[1].astype(np.float32) if vh.shape[0] > 1 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = vh[2].astype(np.float32) if vh.shape[0] > 2 else np.cross(axis, u).astype(np.float32)
    return axis, u, v, pts.mean(axis=0).astype(np.float32)


def _bone_marker_station_layout(pts: np.ndarray) -> tuple[int, int]:
    """Return (stations_along_axis, points_per_cross_section) from bone length."""
    n_verts = int(pts.shape[0])
    if n_verts <= 8:
        return max(2, n_verts // 2), max(2, min(4, n_verts))
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    length = max(float(np.ptp(centered @ vh[0])), 1.0e-4)
    if length < 0.08:
        return 4, 4
    lo, hi = 0.08, 0.445
    t = float(np.clip((length - lo) / max(hi - lo, 1.0e-4), 0.0, 1.0))
    stations = int(round(8 + t * 20))
    ring = 5 if length < 0.30 else 6
    stations = int(np.clip(stations, 6, 28))
    ring = int(np.clip(ring, 4, min(8, max(4, n_verts // max(stations, 1)))))
    return stations, ring


def _pick_cross_section_ring(
    slab: np.ndarray,
    *,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n_ring: int,
) -> np.ndarray:
    if slab.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if slab.shape[0] <= n_ring:
        return slab.astype(np.float32)
    rel = slab - origin[None, :]
    ang = np.arctan2(rel @ v, rel @ u)
    targets = np.linspace(-np.pi, np.pi, int(n_ring), endpoint=False)
    picks: list[np.ndarray] = []
    used: set[int] = set()
    for target in targets:
        delta = np.angle(np.exp(1j * (ang - float(target))))
        order = np.argsort(np.abs(delta))
        for idx in order.tolist():
            if idx not in used:
                used.add(idx)
                picks.append(slab[idx])
                break
    if not picks:
        return slab[: int(n_ring)].astype(np.float32)
    return np.stack(picks, axis=0).astype(np.float32)


def _sample_bone_marker_points(pts: np.ndarray) -> np.ndarray:
    """Multi-point cross-sections along the bone axis so projections read as solid shafts."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    if pts.shape[0] <= 12:
        return pts.copy()
    axis, u, v, origin = _bone_axis_frame(pts)
    centered = pts - origin[None, :]
    t = centered @ axis
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1.0e-4)
    n_stations, n_ring = _bone_marker_station_layout(pts)
    band = max(span / max(2 * n_stations, 1), 0.004)
    picks: list[np.ndarray] = []
    for frac in np.linspace(0.04, 0.96, int(n_stations)):
        t_cut = t_min + float(frac) * span
        slab = pts[np.abs(t - t_cut) <= band]
        if slab.shape[0] < 3:
            idx = int(np.argmin(np.abs(t - t_cut)))
            picks.append(pts[idx : idx + 1])
            continue
        ring = _pick_cross_section_ring(slab, origin=origin, u=u, v=v, n_ring=n_ring)
        picks.append(ring)
    if not picks:
        return pts.copy()
    out = np.concatenate(picks, axis=0).astype(np.float32)
    _, uniq_idx = np.unique(np.round(out, 5), axis=0, return_index=True)
    return out[np.sort(uniq_idx)]


def _bone_marker_sample_count(
    pts: np.ndarray,
    *,
    min_samples: int = 4,
    max_samples: int = 42,
) -> int:
    """Legacy helper: approximate total marker count for reporting."""
    if pts.shape[0] <= min_samples:
        return int(pts.shape[0])
    stations, ring = _bone_marker_station_layout(pts)
    return int(min(pts.shape[0], stations * ring))


def sparse_leg_bone_vertices(
    vertices: np.ndarray,
    raw: "np.lib.npyio.NpzFile",
    mesh_names: set[str] | frozenset[str],
    *,
    samples_per_mesh: int | None = None,
    min_samples: int = 4,
    max_samples: int = 42,
) -> np.ndarray:
    """Sample leg bones as cross-section rings along each shaft (length-adaptive density)."""
    names = [str(v) for v in raw["source_mesh_names"].reshape(-1).tolist()]
    ranges = np.asarray(raw["source_vertex_ranges"], dtype=np.int64).reshape(-1, 2)
    chunks: list[np.ndarray] = []
    for mesh_name, (start, end) in zip(names, ranges, strict=True):
        if mesh_name not in mesh_names:
            continue
        pts = np.asarray(vertices[int(start) : int(end)], dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        if samples_per_mesh is not None:
            # Back-compat: treat as stations and use 5 points per ring.
            centered = pts - pts.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            order = np.argsort(centered @ axis)
            picks: list[np.ndarray] = []
            for q in np.linspace(0.02, 0.98, int(samples_per_mesh)):
                idx = int(order[int(round(float(q) * (order.shape[0] - 1)))])
                picks.append(pts[idx])
            chunks.append(np.stack(picks, axis=0))
            continue
        chunks.append(_sample_bone_marker_points(pts))
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def draw_preview_overlay(
    path: Path,
    *,
    smpl_tpose: np.ndarray,
    anatomy_tpose: np.ndarray,
    smpl_posed: np.ndarray,
    anatomy_posed: np.ndarray,
    leg_bones_tpose: np.ndarray | None = None,
    leg_bones_posed: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    views = [(0, 1, "XY front"), (2, 1, "ZY side"), (0, 2, "XZ top")]

    st = dense_body_cloud(smpl_tpose, step=1)
    sp = dense_body_cloud(smpl_posed, step=1)
    at = anatomy_cloud(anatomy_tpose, step=8)
    ap = anatomy_cloud(anatomy_posed, step=8)

    for col, (i, j, name) in enumerate(views):
        ax = axes[0, col]
        ax.scatter(at[:, i], at[:, j], s=0.35, c=ANATOMY_COLOR, alpha=0.32)
        ax.scatter(st[:, i], st[:, j], s=0.45, c=SMPL_TPOSE_COLOR, alpha=0.42)
        if leg_bones_tpose is not None and leg_bones_tpose.size:
            ax.scatter(
                leg_bones_tpose[:, i],
                leg_bones_tpose[:, j],
                s=10.0,
                c=LEG_BONE_COLOR,
                edgecolors=LEG_BONE_EDGE,
                linewidths=0.35,
                alpha=0.9,
                marker="o",
            )
        ax.set_title(f"T-pose {name}")
        ax.set_aspect("equal")

        ax = axes[1, col]
        ax.scatter(ap[:, i], ap[:, j], s=0.35, c=ANATOMY_COLOR, alpha=0.32)
        ax.scatter(sp[:, i], sp[:, j], s=0.45, c=SMPL_POSED_COLOR, alpha=0.42)
        if leg_bones_posed is not None and leg_bones_posed.size:
            ax.scatter(
                leg_bones_posed[:, i],
                leg_bones_posed[:, j],
                s=10.0,
                c=LEG_BONE_COLOR,
                edgecolors=LEG_BONE_EDGE,
                linewidths=0.35,
                alpha=0.9,
                marker="o",
            )
        ax.set_title(f"Posed {name}")
        ax.set_aspect("equal")

    include_bones = leg_bones_tpose is not None and leg_bones_tpose.size > 0
    fig.legend(
        handles=_preview_overlay_legend(include_bones=include_bones),
        loc="outside lower center",
        ncol=3 + int(include_bones),
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(path, dpi=120)
    plt.close(fig)


def draw_vein_on_body_pose_figure(
    path: Path,
    *,
    smpl_tpose: np.ndarray,
    smpl_posed: np.ndarray,
    tpose_centerlines: dict[str, np.ndarray],
    posed_centerlines: dict[str, np.ndarray],
    segment_colors: dict[str, tuple[int, int, int]],
    leg_bones_tpose: np.ndarray | None = None,
    leg_bones_posed: np.ndarray | None = None,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # Seated posed: leg side = XY (view from +Z), top = XZ (view from +Y).
    panel_specs = (
        (0, 0, smpl_tpose, SMPL_TPOSE_COLOR, tpose_centerlines, leg_bones_tpose, 0, 1, "T-pose front"),
        (0, 1, smpl_tpose, SMPL_TPOSE_COLOR, tpose_centerlines, leg_bones_tpose, 2, 1, "T-pose side"),
        (1, 0, smpl_posed, SMPL_POSED_COLOR, posed_centerlines, leg_bones_posed, 0, 1, "Posed leg side"),
        (1, 1, smpl_posed, SMPL_POSED_COLOR, posed_centerlines, leg_bones_posed, 0, 2, "Posed top"),
    )
    legend_handles: list = []
    legend_labels: list[str] = []
    for row, col, body, body_color, centerlines, bones, i, j, panel_title in panel_specs:
        ax = axes[row, col]
        body_show = np.asarray(body, dtype=np.float32).reshape(-1, 3)
        if row == 1:
            body_show = _clip_points_near_centerlines(body_show, centerlines, radius=0.11)
        else:
            bbox = _bbox_from_centerlines(centerlines)
            if bbox is not None:
                body_show = _clip_points_aabb(body_show, bbox[0], bbox[1])
        body_pts = dense_body_cloud(body_show, step=1)
        ax.scatter(
            body_pts[:, i],
            body_pts[:, j],
            s=0.45,
            c=body_color,
            alpha=0.40,
        )
        for label in sorted(centerlines):
            if not is_leg_vein_centerline_label(label):
                continue
            line = np.asarray(centerlines[label], dtype=np.float32).reshape(-1, 3)
            if line.shape[0] < 2:
                continue
            rgb = segment_colors.get(str(label), (170, 170, 170))
            color = tuple(v / 255.0 for v in rgb)
            ax.plot(
                line[:, i],
                line[:, j],
                "-",
                color=color,
                linewidth=2.4,
                alpha=0.96,
                solid_capstyle="round",
            )
            if label not in legend_labels:
                legend_handles.append(_centerline_legend_handle(str(label), rgb))
                legend_labels.append(str(label))
        if bones is not None and bones.size:
            ax.scatter(
                bones[:, i],
                bones[:, j],
                s=7.0,
                c=LEG_BONE_COLOR,
                edgecolors=LEG_BONE_EDGE,
                linewidths=0.25,
                alpha=0.82,
                marker="o",
            )
        line_clouds = [line for line in centerlines.values() if np.asarray(line).size]
        _fit_axis_limits(ax, [body_show, *line_clouds, bones], i, j)
        ax.set_aspect("equal")
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.10)
    fig.suptitle(title, fontsize=12, y=0.98)
    if leg_bones_tpose is not None and leg_bones_tpose.size:
        legend_handles.append(_leg_bone_legend_handle())
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4, fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.16, hspace=0.52, wspace=0.22)
    plt.savefig(path, dpi=150)
    plt.close(fig)
