"""Publication figure: Orbbec RGB with planned paths, plus a dense rail-base cloud."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

STUDY = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
UNBIASED_DIR = STUDY / "ultrapoc" / "009"
BIASED_DIR = STUDY / "ultrapoc" / "010"
DEFAULT_CAPTURE = STUDY / "figures" / "orbbec_live_plan_capture.npz"
DEFAULT_OUT = STUDY / "figures" / "scan_path"

UNBIASED_COLOR = "#56B4E9"
BIASED_COLOR = "#CC79A7"
UNBIASED_FRAME = "#2171B5"
BIASED_FRAME = "#8C2D6B"
INK = "#272727"
TICK_M = 0.018
TICK_STRIDE = 8
SCAN_SPEED_M_S = 0.010
TIME_CMAP = "coolwarm"

RC = {
    "font.family": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 20,
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 20,
    "svg.fonttype": "none",
}


def _read_ascii_ply(path: Path):
    text = Path(path).read_text(encoding="utf-8").splitlines()
    n = 0
    header = 0
    has_color = False
    for i, line in enumerate(text):
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        if line.startswith("property uchar red"):
            has_color = True
        if line == "end_header":
            header = i + 1
            break
    rows = [ln.split() for ln in text[header:header + n] if ln.strip()]
    xyz = np.asarray([row[:3] for row in rows], dtype=np.float64)
    rgb = None
    if has_color:
        rgb = np.clip(np.asarray([row[3:6] for row in rows], dtype=np.float64) / 255.0, 0, 1)
    return xyz, rgb


def load_plan(directory: Path):
    summary = json.loads((Path(directory) / "plan.json").read_text(encoding="utf-8"))
    poses = np.asarray(summary["poses"], dtype=np.float64)
    normals = np.asarray(summary["waypoint_normals_outward"], dtype=np.float64)
    command = summary.get("waypoint_command_normals_outward")
    command = normals if command is None else np.asarray(command, dtype=np.float64)
    return SimpleNamespace(
        poses=poses,
        normals=normals,
        command_normals=command,
        outline=np.asarray(summary["outline_corners"], dtype=np.float64),
        right=np.asarray(summary["long_axis"], dtype=np.float64),
        far=np.asarray(summary["across_short_axis"], dtype=np.float64),
        q8=np.asarray(summary["q8_at_detection"], dtype=np.float64),
        offset_deg=float(summary.get("normal_offset_deg") or 0.0),
        directory=Path(directory),
    )


def camera_from_rail(q8: np.ndarray):
    from peirastic.DEMO.phathom_scanning.cloud import resolve_urdf
    from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import (
        RailBaseLink7FK,
        load_T_link7_cam,
    )

    T_rb_cam = RailBaseLink7FK(resolve_urdf()).T_railbase_link7(q8) @ load_T_link7_cam()
    return T_rb_cam, np.linalg.inv(T_rb_cam)


def project_points(xyz_rb, K, T_cam_rb, dist=None):
    from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import transform_points

    xyz = transform_points(T_cam_rb, np.asarray(xyz_rb, dtype=np.float64).reshape(-1, 3))
    z = xyz[:, 2]
    ok = np.isfinite(xyz).all(axis=1) & (z > 0.04)
    uv = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    if dist is None or not np.any(np.asarray(dist, dtype=np.float64)):
        uv[ok, 0] = K[0, 0] * xyz[ok, 0] / z[ok] + K[0, 2]
        uv[ok, 1] = K[1, 1] * xyz[ok, 1] / z[ok] + K[1, 2]
        return uv, ok, xyz
    import cv2

    pix, _ = cv2.projectPoints(
        xyz[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1),
    )
    uv[ok] = pix.reshape(-1, 2)
    return uv, ok, xyz


def resample_polyline(xyz, n=280):
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return pts
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(ds)))
    if arc[-1] <= 1e-9:
        return pts
    s = np.linspace(0.0, arc[-1], int(n))
    out = np.column_stack([np.interp(s, arc, pts[:, i]) for i in range(3)])
    return out


def path_time_s(xyz, speed_m_s=SCAN_SPEED_M_S):
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(pts) == 1:
        return np.zeros(1, dtype=np.float64)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(ds)))
    return arc / max(float(speed_m_s), 1e-9)


def path_s(xyz):
    t = path_time_s(xyz, speed_m_s=1.0)
    if t.size == 0 or float(t[-1]) <= 1e-12:
        return t
    return t / t[-1]


def tool_partial_axes(plan):
    """Tool X, Y, Z at each waypoint (Z into the phantom)."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", np.asarray(plan.poses)[:, 3:6]).as_matrix()
    return rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def rehome_to_live(plan, hit):
    """Put the planned Lissajous on the currently detected phantom top."""
    src = plan.poses[:, :3]
    center = src.mean(axis=0)
    n = _unit(hit.normal)
    right = _unit(plan.right - np.dot(plan.right, n) * n)
    far = _unit(np.cross(n, right))
    top = np.asarray(hit.points, dtype=np.float64)
    u_top = (top - hit.centroid) @ right
    v_top = (top - hit.centroid) @ far
    live_center = (
        hit.centroid
        + 0.5 * (float(u_top.min()) + float(u_top.max())) * right
        + 0.5 * (float(v_top.min()) + float(v_top.max())) * far
    )
    u = (src - center) @ right
    v = (src - center) @ far
    xyz = live_center + u[:, None] * right + v[:, None] * far
    return xyz, right, far, n


def live_hit(xyz_rb, colors, q8):
    from peirastic.DEMO.phathom_scanning.detect import detect_phantom_top
    from peirastic.DEMO.phathom_scanning.run import _toward_mount

    toward = _toward_mount(xyz_rb.mean(axis=0), rail_m=float(q8[0]))
    return detect_phantom_top(xyz_rb, colors, toward)


def camera_elev_azim(T_rb_cam, center, right=None):
    """3D frame whose +X is camera-right, viewed from the wrist camera."""
    del right
    z = np.array([0.0, 0.0, 1.0])
    x = _unit(np.array([T_rb_cam[0, 0], T_rb_cam[1, 0], 0.0]))
    y = _unit(np.cross(z, x))
    R = np.column_stack([x, y, z])
    cam = (np.asarray(T_rb_cam)[:3, 3] - np.asarray(center)) @ R
    elev = float(np.degrees(np.arctan2(cam[2], np.hypot(cam[0], cam[1]))))
    azim = 90.0 if cam[1] >= 0.0 else -90.0
    return elev, azim, R


def live_q8():
    from peirastic.api.arm import OK, PeirasticArm

    ret, q = PeirasticArm().get_joint_radian()
    if ret != OK or not q or len(q) < 8:
        raise RuntimeError("no live 8-DOF q")
    return np.asarray(q[:8], dtype=np.float64)


def grab_orbbec(path: Path = DEFAULT_CAPTURE, *, stride: int = 2) -> Path:
    """One dense D2C frame. Needs the camera_calib env on sys.path."""
    from multicam_calib.calib.orbbec_depth_ray import apply_depth_ray_scale, load_depth_ray_coeff
    from multicam_calib.calib.orbbec_rgbd import unproject_aligned_depth
    from multicam_calib.devices.orbbec import OrbbecRGBDSession
    from multicam_calib.io.config import load_orbbec

    session = OrbbecRGBDSession(load_orbbec())
    try:
        params = session.open()
        frame = None
        for _ in range(8):
            frame = session.read(timeout_ms=2000)
            if frame is not None and getattr(frame, "color_bgr", None) is not None:
                if getattr(frame, "depth_m", None) is not None:
                    break
        if frame is None:
            raise RuntimeError("Orbbec returned no RGB-D frame")
        K = np.asarray(params.color.K, dtype=np.float64).reshape(3, 3)
        xyz, rgb = unproject_aligned_depth(
            frame.depth_m, K, color_bgr=frame.color_bgr, stride=int(stride),
            min_m=0.2, max_m=3.0,
        )
        ray_c, _ = load_depth_ray_coeff()
        xyz = apply_depth_ray_scale(xyz, ray_c)
    finally:
        session.close()
    q8 = live_q8()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        color_bgr=frame.color_bgr,
        xyz_cam=np.asarray(xyz, dtype=np.float32),
        rgb=np.asarray(rgb, dtype=np.uint8),
        K=K,
        dist=np.asarray(params.color.dist, dtype=np.float64),
        q8=q8,
    )
    return path


def load_capture(path: Path):
    packed = np.load(Path(path), allow_pickle=False)
    q8 = packed["q8"] if "q8" in packed.files else live_q8()
    rgb = np.asarray(packed["rgb"])
    image = packed["color_bgr"][:, :, ::-1] if "color_bgr" in packed.files else None
    if rgb.ndim == 3:
        colors = None
    else:
        raw = rgb.reshape(-1, 3).astype(np.float64)
        if raw.max() > 1.0 + 1e-6:
            raw = raw / 255.0
        # unproject_aligned_depth samples color_bgr.
        colors = np.clip(raw[:, ::-1], 0.0, 1.0)
    return SimpleNamespace(
        image=image,
        xyz_cam=np.asarray(packed["xyz_cam"], dtype=np.float64),
        colors=colors,
        K=np.asarray(packed["K"], dtype=np.float64).reshape(3, 3),
        dist=None if "dist" not in packed.files else np.asarray(packed["dist"], dtype=np.float64),
        q8=np.asarray(q8, dtype=np.float64),
    )


def scene_from_run(directory: Path):
    """Fallback when there is no live RGB: splat the saved detect cloud."""
    plan = load_plan(directory)
    xyz, colors = _read_ascii_ply(Path(directory) / "detect_cloud.ply")
    K = np.array(
        [[454.8604490327609, 0.0, 328.38649725335137],
         [0.0, 454.92853069709406, 246.95987800518532],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    _, T_cam_rb = camera_from_rail(plan.q8)
    uv, ok, _ = project_points(xyz, K, T_cam_rb)
    image = _splat_image(uv[ok], colors[ok], (480, 640))
    return SimpleNamespace(image=image, xyz_rb=xyz, colors=colors, K=K, q8=plan.q8)


def _splat_image(uv, colors, shape, radius=4):
    h, w = shape
    acc = np.zeros((h, w, 3), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-0.5 * (xx * xx + yy * yy) / max(radius / 2.2, 0.4) ** 2)
    for (u, v), color in zip(uv, colors):
        cx, cy = int(round(u)), int(round(v))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        patch = kernel[y0 - (cy - radius): y1 - (cy - radius), x0 - (cx - radius): x1 - (cx - radius)]
        acc[y0:y1, x0:x1] += patch[:, :, None] * color
        weight[y0:y1, x0:x1] += patch
    out = np.ones((h, w, 3), dtype=np.float64)
    hit = weight > 1e-6
    out[hit] = acc[hit] / weight[hit, None]
    return np.clip(out, 0, 1)


def _draw_points(ax, uv, color, *, size=18, z=6, edge="#ffffff"):
    keep = np.isfinite(uv).all(axis=1)
    if not keep.any():
        return
    ax.scatter(
        uv[keep, 0], uv[keep, 1], s=size, c=color, zorder=z,
        edgecolors=edge, linewidths=0.35, alpha=0.95,
    )


def _draw_ticks(ax, origin_uv, tip_uv, color):
    keep = np.isfinite(origin_uv).all(axis=1) & np.isfinite(tip_uv).all(axis=1)
    for a, b in zip(origin_uv[keep], tip_uv[keep]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6, solid_capstyle="round", zorder=6)


def _crop(image, uv_list, pad=90, extra_right=0, extra_bottom=0):
    pts = np.vstack([uv[np.isfinite(uv).all(axis=1)] for uv in uv_list if len(uv)])
    h, w = image.shape[:2]
    x0 = max(int(np.floor(pts[:, 0].min()) - pad), 0)
    x1 = min(int(np.ceil(pts[:, 0].max()) + pad + extra_right), w)
    y0 = max(int(np.floor(pts[:, 1].min()) - pad), 0)
    y1 = min(int(np.ceil(pts[:, 1].max()) + pad + extra_bottom), h)
    return (x0, x1, y0, y1)


def _square_lr(box):
    """Trim left and right equally so the window is square."""
    x0, x1, y0, y1 = (int(v) for v in box)
    width, height = x1 - x0, y1 - y0
    if width > height:
        extra = width - height
        x0 += extra // 2
        x1 -= extra - extra // 2
    return (x0, x1, y0, y1)


def _time_line_collection(uv, t, *, lw=2.8, z=7):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    keep = np.isfinite(uv).all(axis=1)
    pts = np.asarray(uv, dtype=np.float64)[keep]
    coord = np.asarray(t, dtype=np.float64)[keep]
    if len(pts) < 2:
        return None
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(
        segs, cmap=TIME_CMAP, norm=Normalize(0.0, 1.0), linewidths=lw,
        capstyle="round", joinstyle="round", zorder=z,
    )
    lc.set_array(0.5 * (coord[:-1] + coord[1:]))
    return lc


def _enlarge_3d(ax, factor=1.65):
    """Matplotlib 3D leaves a large empty margin; scale the projection to fill it."""
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    def _proj():
        return np.dot(Axes3D.get_proj(ax), np.diag([factor, factor, factor, 1.0]))

    ax.get_proj = _proj


def _place_square_panels(fig, ax3, ax, cax=None):
    """Give both content axes the same on-page square and park the colorbar beside them."""
    fig_w, fig_h = fig.get_size_inches()
    margin_l, margin_r, margin_b, margin_t, gap, cbar_w, cbar_pad = (
        0.82, 0.60, 0.80, 0.44, 0.28, 0.22, 0.10,
    )
    side = min((fig_w - margin_l - margin_r - gap - cbar_w - cbar_pad) / 2.0,
               fig_h - margin_b - margin_t)
    y0 = margin_b + 0.5 * (fig_h - margin_b - margin_t - side)
    x3 = margin_l
    x2 = margin_l + side + gap
    ax3.set_position([x3 / fig_w, y0 / fig_h, side / fig_w, side / fig_h])
    ax.set_position([x2 / fig_w, y0 / fig_h, side / fig_w, side / fig_h])
    if cax is not None:
        cax.set_position([
            (x2 + side + cbar_pad) / fig_w,
            (y0 + 0.06 * side) / fig_h,
            cbar_w / fig_w,
            0.88 * side / fig_h,
        ])
    return side


def _arrow3d(ax3, start, direction, length_m, color, *, zorder=12, lw=1.35):
    start = np.asarray(start, dtype=np.float64).reshape(3)
    d = _unit(direction)
    tip = start + float(length_m) * d
    ax3.plot(*np.vstack([start, tip]).T * 1000, color=color, lw=lw, zorder=zorder, solid_capstyle="round")
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = _unit(np.cross(d, ref))
    b = _unit(np.cross(d, a))
    head_len = 0.30 * float(length_m)
    head_w = 0.13 * float(length_m)
    base = tip - head_len * d
    for wing in (a, -a, b, -b):
        ax3.plot(*np.vstack([tip, base + head_w * wing]).T * 1000,
                 color=color, lw=max(lw - 0.15, 0.9), zorder=zorder + 1, solid_capstyle="round")


def _draw_partial_frames(ax3, origins, x_axis, y_axis, z_axis, color, *, length_m=0.022, zorder=12):
    """Tool triad: X, Y, and outward Z (displayed out of the phantom)."""
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    x_dir = np.asarray(x_axis, dtype=np.float64).reshape(-1, 3)
    y_dir = np.asarray(y_axis, dtype=np.float64).reshape(-1, 3)
    z_out = -np.asarray(z_axis, dtype=np.float64).reshape(-1, 3)
    for p, x, y, z in zip(o, x_dir, y_dir, z_out):
        _arrow3d(ax3, p, x, length_m, color, zorder=zorder, lw=1.25)
        _arrow3d(ax3, p, y, length_m, color, zorder=zorder, lw=1.25)
        _arrow3d(ax3, p, z, length_m, color, zorder=zorder + 1, lw=1.45)


def _surface_cloud(plan):
    xyz, colors = _read_ascii_ply(plan.directory / "detect_cloud.ply")
    top, _ = _read_ascii_ply(plan.directory / "detect_top.ply")
    return xyz, colors, top


def _smooth_nan(field, mask, sigma):
    from scipy.ndimage import gaussian_filter

    weight = mask.astype(np.float64)
    filled = np.where(mask, field, 0.0)
    num = gaussian_filter(filled, sigma, mode="nearest")
    den = gaussian_filter(weight, sigma, mode="nearest")
    out = num / np.maximum(den, 1e-8)
    out[~mask] = np.nan
    return out


def _inside_support(u, v, samples_u, samples_v, *, radius_m):
    from matplotlib.path import Path
    from scipy.ndimage import binary_closing
    from scipy.spatial import ConvexHull, cKDTree

    pts = np.column_stack([np.asarray(samples_u, dtype=np.float64).ravel(),
                           np.asarray(samples_v, dtype=np.float64).ravel()])
    pts = pts[np.isfinite(pts).all(axis=1)]
    uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    query = np.column_stack([uu.ravel(), vv.ravel()])
    if len(pts) < 3:
        return np.ones(uu.shape, dtype=bool)
    dist, _ = cKDTree(pts).query(query, k=1)
    mask = dist.reshape(uu.shape) <= float(radius_m)
    try:
        ring = pts[ConvexHull(pts).vertices]
        mask |= Path(ring, closed=True).contains_points(query, radius=1e-6).reshape(uu.shape)
    except Exception:
        pass
    return binary_closing(mask, iterations=2)


def colored_surface_grid(xyz, colors, right, far, normal, *,
                         pitch_m=0.0032, smooth_sigma=1.7, min_count=4):
    """Interpolate a tangent-plane height/RGB surface and Gaussian-smooth it."""
    del min_count
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rgb = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    if len(pts) != len(rgb):
        raise ValueError("xyz and colors must have the same length")
    keep = np.isfinite(pts).all(axis=1) & np.isfinite(rgb).all(axis=1)
    pts, rgb = pts[keep], rgb[keep]
    if len(pts) < 16:
        raise ValueError("need a dense colored cloud to fit a surface")
    if len(pts) > 8000:
        pick = np.linspace(0, len(pts) - 1, 8000, dtype=int)
        pts, rgb = pts[pick], rgb[pick]
    origin = pts.mean(axis=0)
    axis_u, axis_v, axis_n = _unit(right), _unit(far), _unit(normal)
    uvh = (pts - origin) @ np.column_stack([axis_u, axis_v, axis_n])
    u, v, h = uvh[:, 0], uvh[:, 1], uvh[:, 2]
    u0, u1 = np.quantile(u, [0.012, 0.988])
    v0, v1 = np.quantile(v, [0.012, 0.988])
    core = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    if int(core.sum()) >= 16:
        u, v, h, rgb = u[core], v[core], h[core], rgb[core]
    nu = max(12, int(np.ceil((u1 - u0) / float(pitch_m))) + 1)
    nv = max(12, int(np.ceil((v1 - v0) / float(pitch_m))) + 1)
    uc = np.linspace(u0, u1, nu)
    vc = np.linspace(v0, v1, nv)
    uu, vv = np.meshgrid(uc, vc, indexing="ij")
    mask = _inside_support(uu, vv, u, v, radius_m=1.6 * float(pitch_m))
    sample = np.column_stack([u, v])
    height = LinearNDInterpolator(sample, h)(uu, vv)
    hole = mask & ~np.isfinite(height)
    if hole.any():
        height[hole] = NearestNDInterpolator(sample, h)(uu[hole], vv[hole])
    shade = np.empty(uu.shape + (3,))
    for ch in range(3):
        field = LinearNDInterpolator(sample, rgb[:, ch])(uu, vv)
        miss = mask & ~np.isfinite(field)
        if miss.any():
            field[miss] = NearestNDInterpolator(sample, rgb[:, ch])(uu[miss], vv[miss])
        shade[..., ch] = field
    height = _smooth_nan(height, mask, smooth_sigma)
    for ch in range(3):
        shade[..., ch] = _smooth_nan(shade[..., ch], mask, smooth_sigma)
    mask = mask & np.isfinite(height) & np.isfinite(shade).all(axis=2)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size < 2 or cols.size < 2:
        raise ValueError("need a dense colored cloud to fit a surface")
    sl = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    uu, vv, height, shade, mask = uu[sl], vv[sl], height[sl], shade[sl], mask[sl]
    fill_h = float(np.nanmedian(height[mask])) if mask.any() else 0.0
    height = np.where(mask, height, fill_h)
    xyz_grid = (origin
                + uu[..., None] * axis_u
                + vv[..., None] * axis_v
                + height[..., None] * axis_n)
    vertex = np.clip(np.nan_to_num(shade, nan=0.75), 0.0, 1.0)
    vertex = np.concatenate([vertex, np.ones(mask.shape + (1,))], axis=2)
    face = 0.25 * (vertex[:-1, :-1] + vertex[1:, :-1] + vertex[:-1, 1:] + vertex[1:, 1:])
    face[..., 3] = 1.0
    return SimpleNamespace(xyz=xyz_grid, facecolors=face, mask=mask, origin=origin)


def render(fig, *, image, xyz_rb, colors, K, T_rb_cam, T_cam_rb, unbiased, biased,
           hit=None, dist=None, annotated_rgb=False):
    from matplotlib.patches import Patch

    if hit is not None:
        xyz_u0, right, far, normal = rehome_to_live(unbiased, hit)
        xyz_b0, _, _, _ = rehome_to_live(biased, hit)
        top = hit.points
        center = hit.centroid
    else:
        xyz_u0, xyz_b0 = unbiased.poses[:, :3], biased.poses[:, :3]
        right, far, normal = biased.right, biased.far, _unit(biased.normals.mean(axis=0))
        top = None
        top_ply = Path(biased.directory) / "detect_top.ply"
        if top_ply.is_file():
            top, _ = _read_ascii_ply(top_ply)
        center = xyz_b0.mean(axis=0)
    xyz_line = resample_polyline(xyz_u0, 360)
    xyz_u = resample_polyline(xyz_u0, 220)
    xyz_b = resample_polyline(xyz_b0, 220)
    kw = dict(dist=dist)
    uv_line = None
    if annotated_rgb:
        faded = np.clip(np.asarray(image, dtype=np.float64) / (
            255.0 if np.asarray(image).max() > 1.5 else 1.0), 0, 1)
        h_img, w_img = faded.shape[:2]
        x0, x1, y0, y1 = 0, w_img, 0, h_img
    else:
        uv_line, _, _ = project_points(xyz_line, K, T_cam_rb, **kw)
        uv_top = None
        if top is not None and len(top):
            uv_top, _, _ = project_points(top[::max(1, len(top) // 2500)], K, T_cam_rb, **kw)
        crop_uv = (uv_line,) if uv_top is None else (uv_line, uv_top)
        x0, x1, y0, y1 = _square_lr(_crop(image, crop_uv, pad=22, extra_right=14, extra_bottom=10))
        faded = np.clip(0.22 + 0.78 * np.asarray(image, dtype=np.float64) / (
            255.0 if np.asarray(image).max() > 1.5 else 1.0), 0, 1)

    if top is not None and len(top):
        lo, hi = top.min(axis=0) - 0.010, top.max(axis=0) + 0.010
        lo[2] -= 0.018
        keep = np.all((xyz_rb >= lo) & (xyz_rb <= hi), axis=1)
        near, near_col = xyz_rb[keep], np.asarray(colors)[keep]
        pts, col = (near, near_col) if len(near) else (xyz_rb, np.asarray(colors))
    else:
        pts, col = xyz_rb, np.asarray(colors)
    elev, azim, R = camera_elev_azim(T_rb_cam, center, right)
    pts_v = (pts - center) @ R
    path_u = (xyz_u - center) @ R
    path_b = (xyz_b - center) @ R
    n_v = normal @ R
    lift = 0.006 * n_v

    ax3 = fig.add_subplot(121, projection="3d")
    ax = fig.add_subplot(122)
    cax = fig.add_axes([0.9, 0.2, 0.015, 0.6])

    ax3.computed_zorder = False
    surface_xyz = None
    if len(pts):
        try:
            mesh = colored_surface_grid(pts, col, right, far, normal)
            surface_xyz = (mesh.xyz.reshape(-1, 3)[mesh.mask.ravel()] - center) @ R
            verts = (mesh.xyz - center) @ R
            ax3.plot_surface(
                verts[:, :, 0] * 1000, verts[:, :, 1] * 1000, verts[:, :, 2] * 1000,
                facecolors=mesh.facecolors, linewidth=0, antialiased=True,
                shade=False, zorder=1, rstride=1, cstride=1,
            )
        except ValueError:
            ax3.scatter(
                *(pts_v * 1000).T, c=np.clip(col, 0, 1), s=5.5, alpha=0.62,
                linewidths=0, depthshade=False, zorder=1,
            )
            surface_xyz = pts_v
    ax3.plot(*(path_u + lift).T * 1000, color=UNBIASED_COLOR, lw=2.6, zorder=8, solid_capstyle="round")
    ax3.plot(*(path_b + lift).T * 1000, color=BIASED_COLOR, lw=2.2, zorder=9, solid_capstyle="round")
    idx = np.linspace(0, len(xyz_u0) - 1, min(9, len(xyz_u0)), dtype=int)
    x_u, y_u, z_u = tool_partial_axes(unbiased)
    x_b, y_b, z_b = tool_partial_axes(biased)
    frame_len = 0.022
    _draw_partial_frames(
        ax3, (xyz_u0[idx] - center) @ R + lift,
        x_u[idx] @ R, y_u[idx] @ R, z_u[idx] @ R, UNBIASED_FRAME, length_m=frame_len, zorder=12,
    )
    _draw_partial_frames(
        ax3, (xyz_b0[idx] - center) @ R + lift,
        x_b[idx] @ R, y_b[idx] @ R, z_b[idx] @ R, BIASED_FRAME, length_m=frame_len, zorder=15,
    )
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_zlabel("")
    ax3.view_init(elev=26.0, azim=-90.0)
    try:
        ax3.dist = 7.4
    except Exception:
        pass
    ax3.xaxis.pane.set_facecolor("#f4f4f4")
    ax3.yaxis.pane.set_facecolor("#f4f4f4")
    ax3.zaxis.pane.set_facecolor("#f4f4f4")
    frame_tips = []
    for xyz0, x_ax, y_ax, z_ax in (
        (xyz_u0[idx], x_u[idx], y_u[idx], z_u[idx]),
        (xyz_b0[idx], x_b[idx], y_b[idx], z_b[idx]),
    ):
        o = (xyz0 - center) @ R + lift
        frame_tips.append(o - frame_len * (z_ax @ R))
        frame_tips.append(o + frame_len * (x_ax @ R))
        frame_tips.append(o + frame_len * (y_ax @ R))
    cloud_bounds = surface_xyz if surface_xyz is not None and len(surface_xyz) else pts_v
    bounds = np.vstack([cloud_bounds, path_u, path_b, *frame_tips]) if len(cloud_bounds) else np.vstack([path_u, path_b, *frame_tips])
    span = np.maximum(np.ptp(bounds, axis=0), 0.04)
    ax3.set_box_aspect(np.maximum(span, 0.05))
    pad = 0.003
    ax3.set_xlim((bounds[:, 0].min() - pad) * 1000, (bounds[:, 0].max() + pad) * 1000)
    ax3.set_ylim((bounds[:, 1].min() - pad) * 1000, (bounds[:, 1].max() + pad) * 1000)
    ax3.set_zlim((bounds[:, 2].min() - pad) * 1000, (bounds[:, 2].max() + 0.008) * 1000)
    ax3.tick_params(labelsize=15, pad=3)
    ax3.set_xticks([-50, 0, 50])
    ax3.set_yticks([])
    ax3.set_zticks([-40, -20, 0, 20, 40])
    _enlarge_3d(ax3, 1.52)

    ax.imshow(faded, origin="upper")
    lc = None if uv_line is None else _time_line_collection(uv_line, path_s(xyz_line), lw=3.2, z=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if lc is not None:
        ax.add_collection(lc)
        mappable = lc
    else:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        mappable = ScalarMappable(cmap=TIME_CMAP, norm=Normalize(0.0, 1.0))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$s$", fontsize=20, labelpad=6)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=15, pad=2)
    cbar.outline.set_linewidth(1.0)
    _place_square_panels(fig, ax3, ax, cax)
    pos3 = ax3.get_position()
    fig.text(pos3.x0 + 0.52 * pos3.width, pos3.y0 - 0.055, "X (mm)",
             ha="center", va="top", fontsize=22)
    fig.text(pos3.x1 - 0.010, pos3.y0 + 0.10 * pos3.height, "Y (mm)",
             ha="right", va="center", fontsize=22)
    fig.text(pos3.x1 - 0.048, pos3.y0 + 0.88 * pos3.height, "Z (mm)",
             ha="right", va="center", fontsize=22)
    fig.legend(
        handles=[
            Patch(facecolor=UNBIASED_COLOR, edgecolor=UNBIASED_COLOR, alpha=0.6, label="unbiased"),
            Patch(facecolor=BIASED_COLOR, edgecolor=BIASED_COLOR, alpha=0.6, label="biased"),
        ],
        loc="lower center", bbox_to_anchor=(pos3.x0 + 0.5 * pos3.width, pos3.y1 + 0.012),
        ncols=2, fontsize=20, handlelength=1.5, columnspacing=1.8,
        handletextpad=0.55, frameon=False, borderaxespad=0.0,
    )


def write_figure(path: Path, *, unbiased, biased, capture=None, run_fallback=None,
                 annotated_rgb=None):
    from matplotlib import pyplot as plt
    from peirastic.DEMO.phathom_scanning.cloud import cloud_in_rail_base

    plt.rcParams.update(RC)
    hit = None
    dist = None
    if capture is not None:
        scene = load_capture(capture)
        T_rb_cam, T_cam_rb = camera_from_rail(scene.q8)
        xyz_rb = cloud_in_rail_base(scene.xyz_cam, scene.q8)
        colors = scene.colors
        image = scene.image
        K = scene.K
        dist = scene.dist
        try:
            hit = live_hit(xyz_rb, colors, scene.q8)
        except Exception:
            hit = None
    else:
        scene = scene_from_run(run_fallback or biased.directory)
        xyz_rb, colors, image, K = scene.xyz_rb, scene.colors, scene.image, scene.K
        T_rb_cam, T_cam_rb = camera_from_rail(scene.q8)

    use_annotated = False
    rgb_path = Path(annotated_rgb) if annotated_rgb is not None else STUDY / "figures" / "scan_path_rgb.png"
    if capture is None and rgb_path.is_file():
        from matplotlib.image import imread

        image = imread(rgb_path)
        use_annotated = True

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.8, 6.70))
    render(
        fig, image=image, xyz_rb=xyz_rb, colors=colors, K=K,
        T_rb_cam=T_rb_cam, T_cam_rb=T_cam_rb,
        unbiased=unbiased, biased=biased, hit=hit, dist=dist,
        annotated_rgb=use_annotated,
    )
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    from matplotlib.transforms import Bbox

    box = Bbox.from_extents(
        tight.x0 - 0.02, tight.y0 - 0.02, tight.x1 + 0.03, tight.y1 + 0.02,
    )
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches=box)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches=box)
    fig.savefig(path.with_suffix(".svg"), bbox_inches=box)
    plt.close(fig)
    return path.with_suffix(".png")
