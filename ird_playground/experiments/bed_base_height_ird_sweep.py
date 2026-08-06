"""Bed-surface IRD reachability vs base_link height (MP4).

Sweeps ``base_link`` world Z at fixed XY from slider_rail world_calib, queries
the trained probe45 signed IRD with TaskCone (tip ±30°, roll ±20°) over a
downward TCP on the bed footprint at leg heights 10/15/20 cm above the bed,
max over rail_y ∈ [0, 0.8], and writes a top-down heatmap video.

Outputs (default)::

    data/reports/bed_base_height_ird_sweep/
      bed_reachability_vs_base_z.mp4
      frames/zb_XXcm.png
      summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_EXPERIMENTS = Path(__file__).resolve().parent
_ROOT = _EXPERIMENTS.parent
for _p in (_ROOT, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cylinder_region_ird_demo import load_conformal_threshold  # noqa: E402
from ird_playground.ird.robot_model import load_robot_model_spec  # noqa: E402
from ird_playground.neural.signed_field import ReachabilitySDF  # noqa: E402
from ird_playground.region.operator import base_from_rail_torch  # noqa: E402
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability  # noqa: E402


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rsc

    w, x, y, z = _normalize_quat_wxyz(quat_wxyz)
    return Rsc.from_quat([x, y, z, w]).as_matrix()


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def load_base_xy_quat(slider_yaml: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy_m, quat_wxyz) from slider_rail world_calib (Z ignored)."""
    raw = yaml.safe_load(slider_yaml.read_text(encoding="utf-8"))
    wc = (raw.get("slider_rail") or raw).get("world_calib") or {}
    pos = np.asarray(wc.get("base_pos_m", [-0.05, 0.5, 0.09]), dtype=np.float64).reshape(3)
    quat = _normalize_quat_wxyz(
        np.asarray(
            wc.get(
                "base_quat_wxyz",
                [0.7071067811865476, 0.0, 0.0, -0.7071067811865476],
            ),
            dtype=np.float64,
        )
    )
    return pos[:2].copy(), quat


def load_bed_footprint(bundle_yaml: Path) -> dict[str, float | np.ndarray]:
    raw = yaml.safe_load(bundle_yaml.read_text(encoding="utf-8"))
    bed = raw["bed"]
    size = np.asarray(bed["size_m"], dtype=np.float64).reshape(2)
    center = np.asarray(bed.get("center_on_floor", [0.0, 0.0, 0.0]), dtype=np.float64)
    return {
        "height_m": float(bed["height_m"]),
        "size_xy_m": size,
        "center_xy_m": center[:2].copy(),
        "x_min": float(center[0] - 0.5 * size[0]),
        "x_max": float(center[0] + 0.5 * size[0]),
        "y_min": float(center[1] - 0.5 * size[1]),
        "y_max": float(center[1] + 0.5 * size[1]),
    }


def rail_base_to_base_link(spec) -> np.ndarray:
    """``T_rail_base_link`` at ``rail_y = rail_locked_at_m`` (IRD URDF)."""
    import pinocchio as pin

    urdf = Path(spec.kinematics_urdf)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q = pin.neutral(model)
    # Lock rail to contract value if the joint exists.
    try:
        jid = model.getJointId(spec.rail_joint)
        # pinocchio joint idx → q index
        jq = model.joints[jid].idx_q
        if jq >= 0:
            q[jq] = float(spec.rail_locked_at_m)
    except Exception:
        pass
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId("base_link")
    return np.asarray(data.oMf[fid].homogeneous, dtype=np.float64)


def world_rail_from_base_link(
    base_xy: np.ndarray,
    base_z: float,
    quat_wxyz: np.ndarray,
    T_rail_base_link: np.ndarray,
) -> np.ndarray:
    """``T_world_rail`` so base_link @ rail_y=0 matches calibrated pose."""
    T_world_base = _pose_matrix(
        np.array([float(base_xy[0]), float(base_xy[1]), float(base_z)], dtype=np.float64),
        quat_wxyz,
    )
    return T_world_base @ np.linalg.inv(T_rail_base_link)


def downward_tcp_batch(
    xs: torch.Tensor,
    ys: torch.Tensor,
    zs: torch.Tensor,
) -> torch.Tensor:
    """Medical TCP [b,t,n] with n = -world Z (vertical scan).

    Parameters are broadcastable 1-D or matching grids; returns ``[..., 4, 4]``.
    """
    # Broadcast to a common leading shape.
    x, y, z = torch.broadcast_tensors(xs, ys, zs)
    # n = -Z, t = +Y, b = t × n = (-1, 0, 0) → columns [b, t, n]
    R = torch.zeros(*x.shape, 3, 3, dtype=x.dtype, device=x.device)
    R[..., 0, 0] = -1.0
    R[..., 1, 1] = 1.0
    R[..., 2, 2] = -1.0
    T = torch.eye(4, dtype=x.dtype, device=x.device).expand(*x.shape, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., 0, 3] = x
    T[..., 1, 3] = y
    T[..., 2, 3] = z
    return T


@torch.no_grad()
def max_rail_taskcone_clearance(
    field,
    task_cone: TaskConeReachability,
    T_tcp: torch.Tensor,
    *,
    T_world_rail: torch.Tensor,
    T_rail_axis0: torch.Tensor,
    rails: torch.Tensor,
    chunk: int = 512,
) -> torch.Tensor:
    """For each TCP, max over rail of TaskCone best_clearance. Shape ``T_tcp[:-2]``."""
    lead = T_tcp.shape[:-2]
    flat = T_tcp.reshape(-1, 4, 4)
    n = flat.shape[0]
    out = torch.full((n,), -1.0e9, dtype=flat.dtype, device=flat.device)
    for i0 in range(0, n, max(1, int(chunk))):
        i1 = min(n, i0 + max(1, int(chunk)))
        tcp = flat[i0:i1]  # [B, 4, 4]
        best = torch.full((tcp.shape[0],), -1.0e9, dtype=tcp.dtype, device=tcp.device)
        for r in rails:
            axis = base_from_rail_torch(
                r, T_world_rail, T_rail_axis0, axis=1
            )  # [4, 4]
            axis_b = axis.expand(tcp.shape[0], 4, 4)
            c = task_cone(field, tcp, axis_b).best_clearance
            best = torch.maximum(best, c)
        out[i0:i1] = best
    return out.reshape(lead)


def bed_quality_map(
    field,
    task_cone: TaskConeReachability,
    *,
    T_world_rail: torch.Tensor,
    T_rail_axis0: torch.Tensor,
    rails: torch.Tensor,
    xs: np.ndarray,
    ys: np.ndarray,
    bed_top: float,
    leg_heights_m: list[float],
    chunk: int,
) -> np.ndarray:
    """Return (ny, nx) mean max-rail clearance over leg-height layers."""
    device = T_world_rail.device
    dtype = T_world_rail.dtype
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    accum = np.zeros(X.shape, dtype=np.float64)
    for h in leg_heights_m:
        z = float(bed_top) + float(h)
        tcp = downward_tcp_batch(
            torch.as_tensor(X, device=device, dtype=dtype),
            torch.as_tensor(Y, device=device, dtype=dtype),
            torch.full(X.shape, z, device=device, dtype=dtype),
        )
        C = max_rail_taskcone_clearance(
            field,
            task_cone,
            tcp,
            T_world_rail=T_world_rail,
            T_rail_axis0=T_rail_axis0,
            rails=rails,
            chunk=chunk,
        )
        accum += C.detach().cpu().numpy()
    return accum / max(len(leg_heights_m), 1)


def rail_world_direction_xy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Unit world-XY direction of +rail_y (URDF rail axis = local +Y)."""
    R = _quat_wxyz_to_R(quat_wxyz)
    d = R @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    d_xy = d[:2]
    n = float(np.linalg.norm(d_xy))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return d_xy / n


def rail_track_endpoints(
    base_xy: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    travel_m: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (start_xy, end_xy, dir_xy) for base_link path over rail travel."""
    d = rail_world_direction_xy(quat_wxyz)
    p0 = np.asarray(base_xy, dtype=np.float64).reshape(2)
    p1 = p0 + float(travel_m) * d
    return p0, p1, d


def render_heatmap_frame(
    quality: np.ndarray,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    base_xy: np.ndarray,
    base_quat: np.ndarray,
    rail_travel_m: float,
    zb_cm: float,
    tip_deg: float,
    roll_deg: float,
    m_safe: float | None,
    vmin: float,
    vmax: float,
    out_path: Path | None = None,
) -> np.ndarray:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=100)
    canvas = FigureCanvasAgg(fig)
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    im = ax.imshow(
        quality,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdYlBu",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if m_safe is not None:
        ax.contour(
            xs,
            ys,
            quality,
            levels=[float(m_safe)],
            colors="black",
            linewidths=1.2,
        )
    p0, p1, _d = rail_track_endpoints(
        base_xy, base_quat, travel_m=float(rail_travel_m)
    )
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color="#212121",
        lw=2.8,
        solid_capstyle="round",
        zorder=4,
        label=f"rail 800 mm  X[{p0[0]:+.2f},{p1[0]:+.2f}]",
    )
    ax.scatter(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        c=["#00e676", "#ff6f00"],
        edgecolors="black",
        s=55,
        zorder=5,
    )
    ax.annotate(
        "rail_y=0",
        (p0[0], p0[1]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=7,
        color="#1b5e20",
    )
    ax.annotate(
        "rail_y=0.8",
        (p1[0], p1[1]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=7,
        color="#e65100",
    )
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    zb_mm = float(zb_cm) * 10.0
    if abs(zb_mm - round(zb_mm)) < 1e-6:
        ax.set_title(f"base_link z = {zb_mm:.0f} mm")
    else:
        ax.set_title(f"base_link z = {zb_cm:.2f} cm")
    ax.text(
        0.01,
        1.02,
        f"tip±{tip_deg:.0f}°  roll±{roll_deg:.0f}°  |  "
        f"rail X {p0[0]*1000:.0f}→{p1[0]*1000:.0f} mm @ Y={p0[1]*1000:.0f} mm",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        color="#37474f",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="IRD clearance")
    fig.tight_layout()
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return buf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoints/rm4d_signed/selected.pt"),
    )
    parser.add_argument(
        "--robot-spec",
        type=Path,
        default=Path("configs/robot_probe45.yaml"),
    )
    parser.add_argument(
        "--conformal",
        type=Path,
        default=Path("data/calib/conformal_rm4d_signed.json"),
    )
    parser.add_argument(
        "--slider-yaml",
        type=Path,
        default=Path(
            "../rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml"
        ),
    )
    parser.add_argument(
        "--bed-bundle",
        type=Path,
        default=Path("../camera_calibration/calibration_results/genesis_bundle.yaml"),
    )
    parser.add_argument(
        "--base-x-m",
        type=float,
        default=None,
        help="Override world_calib base X (m); default from slider_rail.yaml",
    )
    parser.add_argument(
        "--base-y-m",
        type=float,
        default=None,
        help="Override world_calib base Y (m); default from slider_rail.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/reports/bed_base_height_ird_sweep"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zb-min-cm", type=float, default=8.0)
    parser.add_argument("--zb-max-cm", type=float, default=60.0)
    parser.add_argument(
        "--zb-step-cm",
        type=float,
        default=1.0,
        help="base_link Z step in cm (1 = original; 0.1 = 1 mm)",
    )
    parser.add_argument("--grid-cm", type=float, default=3.0)
    parser.add_argument("--tip-half-angle-deg", type=float, default=30.0)
    parser.add_argument("--roll-half-range-deg", type=float, default=20.0)
    parser.add_argument(
        "--rail-samples",
        type=int,
        default=17,
        help="rail_y samples over [0, 0.8] m",
    )
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--fps", type=int, default=12, help="MP4 playback FPS")
    parser.add_argument("--task-cone-samples", type=int, default=64)
    parser.add_argument(
        "--png-every-cm",
        type=float,
        default=1.0,
        help="also save PNG keyframes every N cm (0 = every query)",
    )
    args = parser.parse_args(argv)

    root = _ROOT
    out = _resolve(root, args.out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for bed IRD sweep (same constraint as ellipse demo)."
        )

    slider_yaml = _resolve(root, args.slider_yaml)
    bed_bundle = _resolve(root, args.bed_bundle)
    base_xy, base_quat = load_base_xy_quat(slider_yaml)
    if args.base_x_m is not None:
        base_xy[0] = float(args.base_x_m)
    if args.base_y_m is not None:
        base_xy[1] = float(args.base_y_m)
    bed = load_bed_footprint(bed_bundle)
    m_safe = load_conformal_threshold(_resolve(root, args.conformal))

    robot_spec = load_robot_model_spec(_resolve(root, args.robot_spec))
    T_rail_base_link = rail_base_to_base_link(robot_spec)
    T_rail_axis0_np = robot_spec.root_to_j1_axis().astype(np.float32)

    sdf = ReachabilitySDF.load(
        _resolve(root, args.checkpoint),
        device=str(device),
        expected_robot=robot_spec,
        allow_stale=True,
    )
    field = sdf.model
    task_cone = TaskConeReachability(
        TaskConeConfig(
            tip_half_angle_deg=float(args.tip_half_angle_deg),
            roll_half_range_deg=float(args.roll_half_range_deg),
            samples=int(args.task_cone_samples),
            seed=17,
        )
    ).to(device)

    T_rail_axis0 = torch.as_tensor(T_rail_axis0_np, device=device)
    rails = torch.linspace(
        0.0, 0.8, max(2, int(args.rail_samples)), device=device, dtype=torch.float32
    )

    step = float(args.grid_cm) * 0.01
    xs = np.arange(float(bed["x_min"]), float(bed["x_max"]) + 0.5 * step, step)
    ys = np.arange(float(bed["y_min"]), float(bed["y_max"]) + 0.5 * step, step)
    leg_heights = [0.10, 0.15, 0.20]

    zb_step = float(args.zb_step_cm)
    zb_cm_list = np.arange(
        float(args.zb_min_cm),
        float(args.zb_max_cm) + 0.5 * zb_step,
        zb_step,
    )
    # Snap to mm grid to avoid float drift (8.0, 8.1, …).
    zb_cm_list = np.round(zb_cm_list, 4)
    n_zb = int(len(zb_cm_list))
    rail_travel_m = 0.80
    rail_p0, rail_p1, rail_dir = rail_track_endpoints(
        base_xy, base_quat, travel_m=rail_travel_m
    )
    print(
        f"[bed-ird] base XY=({base_xy[0]:+.3f}, {base_xy[1]:+.3f}) m  "
        f"bed_top={float(bed['height_m']):.3f} m  grid={args.grid_cm:.1f} cm "
        f"({len(xs)}×{len(ys)})  zb={zb_cm_list[0]:.1f}→{zb_cm_list[-1]:.1f} cm "
        f"step={zb_step*10:.1f} mm ({n_zb} queries)  "
        f"rail_samples={int(args.rail_samples)}  "
        f"tip±{args.tip_half_angle_deg:.0f} roll±{args.roll_half_range_deg:.0f}  "
        f"fps={int(args.fps)}",
        flush=True,
    )
    print(
        f"[bed-ird] rail 800 mm → world dir=({rail_dir[0]:+.3f},{rail_dir[1]:+.3f})  "
        f"X {rail_p0[0]*1000:.1f}→{rail_p1[0]*1000:.1f} mm  "
        f"Y {rail_p0[1]*1000:.1f}→{rail_p1[1]*1000:.1f} mm",
        flush=True,
    )

    maps: list[np.ndarray] = []
    rows: list[dict] = []
    log_every = max(1, n_zb // 40)
    for i_zb, zb_cm in enumerate(zb_cm_list):
        zb = float(zb_cm) * 0.01
        T_wr_np = world_rail_from_base_link(base_xy, zb, base_quat, T_rail_base_link)
        T_wr = torch.as_tensor(T_wr_np, device=device, dtype=torch.float32)
        Q = bed_quality_map(
            field,
            task_cone,
            T_world_rail=T_wr,
            T_rail_axis0=T_rail_axis0,
            rails=rails,
            xs=xs,
            ys=ys,
            bed_top=float(bed["height_m"]),
            leg_heights_m=leg_heights,
            chunk=int(args.chunk),
        )
        maps.append(Q)
        mean_c = float(np.nanmean(Q))
        if m_safe is not None:
            coverage = float(np.mean(Q > float(m_safe)))
        else:
            coverage = float(np.mean(Q > 0.0))
        rows.append(
            {
                "base_link_z_cm": float(zb_cm),
                "base_link_z_mm": float(zb_cm) * 10.0,
                "base_link_z_m": zb,
                "mean_clearance": mean_c,
                "coverage": coverage,
                "p50_clearance": float(np.nanpercentile(Q, 50)),
                "p90_clearance": float(np.nanpercentile(Q, 90)),
            }
        )
        if i_zb % log_every == 0 or i_zb + 1 == n_zb:
            print(
                f"[bed-ird] {i_zb + 1}/{n_zb}  zb={float(zb_cm)*10:.0f} mm  "
                f"mean_C={mean_c:+.3f}  coverage={coverage:.3f}",
                flush=True,
            )

    all_vals = np.concatenate([m.ravel() for m in maps])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-6:
        vmin, vmax = -1.0, 1.0

    import imageio.v2 as imageio

    png_every = float(args.png_every_cm)
    video_frames: list[np.ndarray] = []
    print(f"[bed-ird] rendering {n_zb} video frames @ {int(args.fps)} fps …", flush=True)
    for i_zb, (zb_cm, Q) in enumerate(zip(zb_cm_list, maps)):
        zb_mm = int(round(float(zb_cm) * 10.0))
        save_png = png_every <= 0.0 or abs(
            (float(zb_cm) / max(png_every, 1e-9))
            - round(float(zb_cm) / max(png_every, 1e-9))
        ) < 1e-6
        png = frames_dir / f"zb_{zb_mm:04d}mm.png" if save_png else None
        frame = render_heatmap_frame(
            Q,
            xs=xs,
            ys=ys,
            base_xy=base_xy,
            base_quat=base_quat,
            rail_travel_m=rail_travel_m,
            zb_cm=float(zb_cm),
            tip_deg=float(args.tip_half_angle_deg),
            roll_deg=float(args.roll_half_range_deg),
            m_safe=m_safe,
            vmin=vmin,
            vmax=vmax,
            out_path=png,
        )
        video_frames.append(frame)
        if i_zb % max(1, n_zb // 20) == 0 or i_zb + 1 == n_zb:
            print(f"[bed-ird] render {i_zb + 1}/{n_zb}", flush=True)

    mp4_path = out / "bed_reachability_vs_base_z.mp4"
    imageio.mimwrite(
        mp4_path,
        video_frames,
        fps=max(1, int(args.fps)),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    # Prefer coverage, break ties with mean clearance.
    best_i = int(
        max(
            range(len(rows)),
            key=lambda i: (rows[i]["coverage"], rows[i]["mean_clearance"]),
        )
    )
    best = rows[best_i]
    summary = {
        "config": {
            "base_xy_m": [float(base_xy[0]), float(base_xy[1])],
            "base_quat_wxyz": [float(x) for x in base_quat],
            "rail_travel_m": float(rail_travel_m),
            "rail_world_dir_xy": [float(rail_dir[0]), float(rail_dir[1])],
            "rail_track_xy_m": {
                "start": [float(rail_p0[0]), float(rail_p0[1])],
                "end": [float(rail_p1[0]), float(rail_p1[1])],
                "x_mm": [float(rail_p0[0] * 1000.0), float(rail_p1[0] * 1000.0)],
                "y_mm": [float(rail_p0[1] * 1000.0), float(rail_p1[1] * 1000.0)],
            },
            "zb_cm": [float(x) for x in zb_cm_list],
            "grid_cm": float(args.grid_cm),
            "leg_heights_m": leg_heights,
            "bed_top_m": float(bed["height_m"]),
            "bed_size_xy_m": [float(x) for x in bed["size_xy_m"]],
            "tip_half_angle_deg": float(args.tip_half_angle_deg),
            "roll_half_range_deg": float(args.roll_half_range_deg),
            "rail_samples": int(args.rail_samples),
            "m_safe": None if m_safe is None else float(m_safe),
            "checkpoint": str(_resolve(root, args.checkpoint)),
            "robot_spec": str(_resolve(root, args.robot_spec)),
        },
        "per_height": rows,
        "best": best,
        "outputs": {
            "mp4": str(mp4_path),
            "frames_dir": str(frames_dir),
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nbest base_link z = {best['base_link_z_cm']:.0f} cm "
        f"(coverage={best['coverage']:.3f}, mean_C={best['mean_clearance']:+.3f})",
        flush=True,
    )
    print(f"MP4 → {mp4_path}", flush=True)
    print(f"summary → {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
