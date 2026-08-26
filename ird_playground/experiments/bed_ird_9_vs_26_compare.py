"""Compare base_link z=9 cm vs 26 cm: IRD (tip±30/roll±30) + near-rail ROI + QP-IK/J4.

Shows why IRD can paint near-bed-edge reachable at 26 cm while sim feels bad
(J4 near limits): IRD has no joint-margin head; QP-IK reports J4 distance to limit.

Outputs::

    data/reports/bed_ird_9_vs_26_tip30_roll30/
      heatmap_zb09cm.png
      heatmap_zb26cm.png
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

from bed_base_height_ird_sweep import (  # noqa: E402
    _resolve,
    bed_quality_map,
    downward_tcp_batch,
    load_base_xy_quat,
    load_bed_footprint,
    rail_base_to_base_link,
    rail_track_endpoints,
    render_heatmap_frame,
    world_rail_from_base_link,
)
from cylinder_region_ird_demo import load_conformal_threshold  # noqa: E402
from ellipse_vessel_ird_demo import tcp_to_pose6  # noqa: E402
from ird_playground.ird.robot_model import load_robot_model_spec  # noqa: E402
from ird_playground.neural.signed_field import ReachabilitySDF  # noqa: E402
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability  # noqa: E402

# q = [rail, j1, j2, j3, j4, j5, j6, j7]
J4_INDEX = 4


def near_rail_mask(
    X: np.ndarray,
    Y: np.ndarray,
    base_xy: np.ndarray,
    *,
    y_half_m: float = 0.35,
    x_pad_m: float = 0.10,
    travel_m: float = 0.80,
) -> np.ndarray:
    x0, y0 = float(base_xy[0]), float(base_xy[1])
    return (
        (np.abs(Y - y0) < float(y_half_m))
        & (X > x0 - float(x_pad_m))
        & (X < x0 + float(travel_m) + float(x_pad_m))
    )


def bed_edge_mask(
    X: np.ndarray,
    Y: np.ndarray,
    base_xy: np.ndarray,
    bed: dict,
    *,
    edge_band_m: float = 0.12,
) -> np.ndarray:
    """Strip of bed closest to the rail (toward +Y base side)."""
    y0 = float(base_xy[1])
    y_bed_max = float(bed["y_max"])
    # Near-arm edge of bed = high-Y edge of bed footprint
    return (Y >= y_bed_max - float(edge_band_m)) & (Y <= y_bed_max + 0.02)


def roi_stats(Q: np.ndarray, mask: np.ndarray, m_safe: float | None) -> dict:
    thr = float(m_safe) if m_safe is not None else 0.0
    vals = Q[mask]
    if vals.size == 0:
        return {"n": 0, "mean_clearance": float("nan"), "coverage": float("nan")}
    return {
        "n": int(vals.size),
        "mean_clearance": float(np.mean(vals)),
        "coverage": float(np.mean(vals > thr)),
        "p10_clearance": float(np.percentile(vals, 10)),
        "p50_clearance": float(np.percentile(vals, 50)),
    }


def sample_near_edge_points(
    base_xy: np.ndarray,
    bed: dict,
    *,
    n: int = 20,
    leg_heights_m: list[float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return (N, 3) world XYZ on bed near the rail-side edge."""
    rng = np.random.default_rng(seed)
    heights = leg_heights_m or [0.10, 0.15]
    x0 = float(base_xy[0])
    # Along rail travel = world +X under rail-aligned Stage 2. y_max is the
    # axis-aligned bed edge; a skewed bed makes this an approximation.
    xs = rng.uniform(x0 + 0.05, x0 + 0.75, size=n)
    y_edge = float(bed["y_max"]) - rng.uniform(0.02, 0.12, size=n)
    zs = float(bed["height_m"]) + rng.choice(heights, size=n)
    return np.stack([xs, y_edge, zs], axis=1)


@torch.no_grad()
def ird_at_points(
    field,
    task_cone: TaskConeReachability,
    pts: np.ndarray,
    *,
    T_world_rail: torch.Tensor,
    T_rail_axis0: torch.Tensor,
    rails: torch.Tensor,
) -> np.ndarray:
    from bed_base_height_ird_sweep import max_rail_taskcone_clearance

    device = T_world_rail.device
    dtype = T_world_rail.dtype
    tcp = downward_tcp_batch(
        torch.as_tensor(pts[:, 0], device=device, dtype=dtype),
        torch.as_tensor(pts[:, 1], device=device, dtype=dtype),
        torch.as_tensor(pts[:, 2], device=device, dtype=dtype),
    )
    C = max_rail_taskcone_clearance(
        field,
        task_cone,
        tcp,
        T_world_rail=T_world_rail,
        T_rail_axis0=T_rail_axis0,
        rails=rails,
        chunk=64,
    )
    return C.detach().cpu().numpy()


def qpik_spotcheck(
    pts: np.ndarray,
    *,
    yaml_path: Path,
    tip_samples: list[np.ndarray] | None = None,
) -> dict:
    """8DOF ProxQP IK at each point; report J4 margin and sigma_min."""
    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
    from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik

    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    ik_cfg = build_joint_ik_config(raw)
    kin = RobotKinematics(euler_order=ik_cfg.euler_order)
    q_nom_deg = np.asarray(raw["inner"]["nullspace"]["q_nominal_deg"], dtype=float)
    q0 = np.zeros(8, dtype=float)
    q0[0] = 0.4
    arm = np.deg2rad(q_nom_deg[1:] if len(q_nom_deg) == 8 else q_nom_deg)
    q0[1 : 1 + len(arm)] = arm
    q0 = np.clip(q0, kin.q_lower, kin.q_upper)

    lo = kin.q_lower.copy()
    hi = kin.q_upper.copy()
    j4_lo, j4_hi = float(lo[J4_INDEX]), float(hi[J4_INDEX])

    # Nominal downward + a few tip/roll free samples (identity + ± offsets on R)
    if tip_samples is None:
        tip_samples = [np.eye(3)]
        for deg, axis in (
            (20.0, np.array([1.0, 0.0, 0.0])),  # tip toward ±b in local → about t
            (-20.0, np.array([1.0, 0.0, 0.0])),
            (20.0, np.array([0.0, 1.0, 0.0])),
            (-20.0, np.array([0.0, 1.0, 0.0])),
            (25.0, np.array([0.0, 0.0, 1.0])),  # roll about n
            (-25.0, np.array([0.0, 0.0, 1.0])),
        ):
            from scipy.spatial.transform import Rotation as Rsc

            tip_samples.append(Rsc.from_rotvec(np.deg2rad(deg) * axis).as_matrix())

    rows = []
    q = q0.copy()
    for i, p in enumerate(pts):
        T0 = np.eye(4)
        # same as downward_tcp: columns [-X, +Y, -Z]
        T0[:3, :3] = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        T0[:3, 3] = p
        best = None
        for dR in tip_samples:
            T = T0.copy()
            T[:3, :3] = T0[:3, :3] @ dR
            pose = tcp_to_pose6(T, euler_order=ik_cfg.euler_order)
            q_try, converged, report = solve_pose_ik(
                kin,
                q_seed=q,
                pose_target=pose,
                qp_cfg=ik_cfg.qp,
                nullspace_cfg=ik_cfg.nullspace,
                max_iters=250,
                pos_tol_m=1.0e-3,
                rot_tol_rad=0.05,
            )
            j4 = float(q_try[J4_INDEX])
            j4_margin = float(min(j4 - j4_lo, j4_hi - j4))
            cand = {
                "ok": bool(converged),
                "j4_rad": j4,
                "j4_margin_rad": j4_margin,
                "j4_margin_deg": float(np.rad2deg(j4_margin)),
                "sigma_min": float(report.sigma_min),
                "pos_err_mm": float(report.pos_err_mm),
                "rot_err_deg": float(report.rot_err_deg),
                "rail_m": float(q_try[0]),
                "q": q_try.tolist(),
            }
            if best is None:
                best = cand
                q_best = q_try
            else:
                # Prefer converged; then larger J4 margin; then larger sigma
                def key(c):
                    return (int(c["ok"]), c["j4_margin_rad"], c["sigma_min"])

                if key(cand) > key(best):
                    best = cand
                    q_best = q_try
        assert best is not None
        q = np.clip(q_best, lo, hi)
        best["xyz_m"] = [float(p[0]), float(p[1]), float(p[2])]
        rows.append(best)
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"[qpik] {i + 1}/{len(pts)} ok={best['ok']} "
                f"J4_margin={best['j4_margin_deg']:.1f}deg "
                f"sigma={best['sigma_min']:.3f}",
                flush=True,
            )

    ok = np.array([r["ok"] for r in rows], dtype=bool)
    j4m = np.array([r["j4_margin_deg"] for r in rows], dtype=float)
    sig = np.array([r["sigma_min"] for r in rows], dtype=float)
    near_lim = ok & (j4m < 10.0)  # <10° to J4 limit
    return {
        "j4_limit_rad": [j4_lo, j4_hi],
        "j4_limit_deg": [float(np.rad2deg(j4_lo)), float(np.rad2deg(j4_hi))],
        "ok_fraction": float(ok.mean()) if ok.size else 0.0,
        "mean_j4_margin_deg": float(j4m.mean()) if j4m.size else float("nan"),
        "mean_j4_margin_deg_ok_only": float(j4m[ok].mean()) if ok.any() else float("nan"),
        "frac_ok_j4_margin_lt_10deg": float(near_lim.mean()) if ok.size else 0.0,
        "mean_sigma_min": float(sig.mean()) if sig.size else float("nan"),
        "mean_sigma_min_ok_only": float(sig[ok].mean()) if ok.any() else float("nan"),
        "points": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    ap.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    ap.add_argument("--conformal", type=Path, default=Path("data/calib/conformal_rm4d_signed.json"))
    ap.add_argument(
        "--slider-yaml",
        type=Path,
        default=Path(
            "../rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml"
        ),
    )
    ap.add_argument(
        "--bed-bundle",
        type=Path,
        default=Path("../camera_calibration/calibration_results/genesis_bundle.yaml"),
    )
    ap.add_argument(
        "--ik-yaml",
        type=Path,
        default=Path("../rm75_control/configs/joint_admittance_8dof.yaml"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/reports/bed_ird_9_vs_26_tip30_roll30"),
    )
    ap.add_argument("--base-x-m", type=float, default=-0.05)
    ap.add_argument("--base-y-m", type=float, default=0.41)
    ap.add_argument("--grid-cm", type=float, default=3.0)
    ap.add_argument("--rail-samples", type=int, default=17)
    ap.add_argument("--ik-points", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-ik", action="store_true")
    args = ap.parse_args(argv)

    root = _ROOT
    out = _resolve(root, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    base_xy, base_quat = load_base_xy_quat(_resolve(root, args.slider_yaml))
    base_xy = np.array([float(args.base_x_m), float(args.base_y_m)], dtype=np.float64)
    bed = load_bed_footprint(_resolve(root, args.bed_bundle))
    m_safe = load_conformal_threshold(_resolve(root, args.conformal))

    robot_spec = load_robot_model_spec(_resolve(root, args.robot_spec))
    T_bl = rail_base_to_base_link(robot_spec)
    T_axis = torch.as_tensor(robot_spec.root_to_j1_axis().astype(np.float32), device=device)
    sdf = ReachabilitySDF.load(
        _resolve(root, args.checkpoint),
        device=str(device),
        expected_robot=robot_spec,
        allow_stale=True,
    )
    cone = TaskConeReachability(
        TaskConeConfig(
            tip_half_angle_deg=30.0,
            roll_half_range_deg=30.0,
            samples=64,
            seed=17,
        )
    ).to(device)
    rails = torch.linspace(0.0, 0.8, max(2, int(args.rail_samples)), device=device)

    step = float(args.grid_cm) * 0.01
    xs = np.arange(float(bed["x_min"]), float(bed["x_max"]) + 0.5 * step, step)
    ys = np.arange(float(bed["y_min"]), float(bed["y_max"]) + 0.5 * step, step)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    near = near_rail_mask(X, Y, base_xy)
    edge = bed_edge_mask(X, Y, base_xy, bed)
    leg_heights = [0.10, 0.15]

    rail_p0, rail_p1, rail_dir = rail_track_endpoints(base_xy, base_quat, travel_m=0.80)
    print(
        f"[compare] BASE XY=({base_xy[0]:+.3f},{base_xy[1]:+.3f})  "
        f"tip±30 roll±30  legs={leg_heights}  "
        f"rail X {rail_p0[0]*1000:.0f}→{rail_p1[0]*1000:.0f} mm",
        flush=True,
    )

    heights_cm = [9.0, 26.0]
    maps: dict[float, np.ndarray] = {}
    ird_summary: dict[str, dict] = {}
    for zb_cm in heights_cm:
        zb = zb_cm * 0.01
        Twr = torch.as_tensor(
            world_rail_from_base_link(base_xy, zb, base_quat, T_bl),
            device=device,
            dtype=torch.float32,
        )
        print(f"[compare] IRD map zb={zb_cm:.0f} cm …", flush=True)
        Q = bed_quality_map(
            sdf.model,
            cone,
            T_world_rail=Twr,
            T_rail_axis0=T_axis,
            rails=rails,
            xs=xs,
            ys=ys,
            bed_top=float(bed["height_m"]),
            leg_heights_m=leg_heights,
            chunk=256,
        )
        maps[zb_cm] = Q
        ird_summary[f"zb_{int(zb_cm):02d}cm"] = {
            "whole_bed": roi_stats(Q, np.ones_like(Q, dtype=bool), m_safe),
            "near_rail": roi_stats(Q, near, m_safe),
            "bed_edge_near_arm": roi_stats(Q, edge, m_safe),
        }
        s = ird_summary[f"zb_{int(zb_cm):02d}cm"]
        print(
            f"[compare] zb={zb_cm:.0f}  whole mean={s['whole_bed']['mean_clearance']:+.3f} "
            f"cov={s['whole_bed']['coverage']:.3f}  |  near mean={s['near_rail']['mean_clearance']:+.3f} "
            f"cov={s['near_rail']['coverage']:.3f}  |  edge mean={s['bed_edge_near_arm']['mean_clearance']:+.3f} "
            f"cov={s['bed_edge_near_arm']['coverage']:.3f}",
            flush=True,
        )

    all_vals = np.concatenate([m.ravel() for m in maps.values()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))
    if abs(vmax - vmin) < 1e-6:
        vmin, vmax = -1.0, 1.0

    for zb_cm, Q in maps.items():
        png = out / f"heatmap_zb{int(zb_cm):02d}cm.png"
        render_heatmap_frame(
            Q,
            xs=xs,
            ys=ys,
            base_xy=base_xy,
            base_quat=base_quat,
            rail_travel_m=0.80,
            zb_cm=zb_cm,
            tip_deg=30.0,
            roll_deg=30.0,
            m_safe=m_safe,
            vmin=vmin,
            vmax=vmax,
            out_path=png,
        )
        print(f"[compare] wrote {png}", flush=True)

    # Shared edge sample points for IRD + IK
    pts = sample_near_edge_points(base_xy, bed, n=int(args.ik_points), seed=7)
    ik_by_z: dict[str, dict] = {}
    ird_edge_pts: dict[str, dict] = {}
    for zb_cm in heights_cm:
        zb = zb_cm * 0.01
        Twr = torch.as_tensor(
            world_rail_from_base_link(base_xy, zb, base_quat, T_bl),
            device=device,
            dtype=torch.float32,
        )
        C_pts = ird_at_points(
            sdf.model,
            cone,
            pts,
            T_world_rail=Twr,
            T_rail_axis0=T_axis,
            rails=rails,
        )
        thr = float(m_safe) if m_safe is not None else 0.0
        ird_edge_pts[f"zb_{int(zb_cm):02d}cm"] = {
            "mean_clearance": float(C_pts.mean()),
            "coverage": float(np.mean(C_pts > thr)),
            "clearances": C_pts.tolist(),
        }
        print(
            f"[compare] IRD@edge_pts zb={zb_cm:.0f} mean={C_pts.mean():+.3f} "
            f"cov={np.mean(C_pts > thr):.3f}",
            flush=True,
        )

    if not args.skip_ik:
        # QP-IK does not depend on base_z in world if TCP is absolute world —
        # WAIT: TCP is in world frame at bed height; robot base_link Z changes
        # the arm mount height. solve_pose_ik uses RobotKinematics with rail_base
        # at identity / URDF frame, NOT the calibrated world placement!
        #
        # Ellipse demo used T_world_rail=I and TCP in that same frame.
        # For fair IK at different base heights we must express TCP in the
        # arm/rail frame that kin uses (rail_base at origin), OR transform
        # targets into base_link frame.
        #
        # RobotKinematics FK is in model frame with rail_y prismatic.
        # base_link at rail_y=0 is at [0,-0.4,0] in rail_base (IRD URDF).
        # World: T_world_rail @ T_rail_tcp = T_world_tcp
        # => T_rail_tcp = inv(T_world_rail) @ T_world_tcp
        #
        # We solve IK in rail_base frame with transformed TCP.

        from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
        from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik
        from scipy.spatial.transform import Rotation as Rsc

        ik_yaml = _resolve(root, args.ik_yaml)
        raw = yaml.safe_load(ik_yaml.read_text(encoding="utf-8"))
        ik_cfg = build_joint_ik_config(raw)
        kin = RobotKinematics(euler_order=ik_cfg.euler_order)
        q_nom_deg = np.asarray(raw["inner"]["nullspace"]["q_nominal_deg"], dtype=float)

        for zb_cm in heights_cm:
            zb = zb_cm * 0.01
            T_wr = world_rail_from_base_link(base_xy, zb, base_quat, T_bl)
            T_rw = np.linalg.inv(T_wr)
            # Transform world TCP samples into rail_base frame
            pts_rail = []
            for p in pts:
                Tw = np.eye(4)
                Tw[:3, :3] = np.array(
                    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
                )
                Tw[:3, 3] = p
                Tr = T_rw @ Tw
                pts_rail.append(Tr)
            pts_rail_np = np.stack(pts_rail, axis=0)

            print(f"[compare] QP-IK spotcheck zb={zb_cm:.0f} cm (TCP in rail_base) …", flush=True)
            # Reuse qpik but with poses already in rail frame — call local loop
            q0 = np.zeros(8, dtype=float)
            q0[0] = 0.4
            arm = np.deg2rad(q_nom_deg[1:] if len(q_nom_deg) == 8 else q_nom_deg)
            q0[1 : 1 + len(arm)] = arm
            q0 = np.clip(q0, kin.q_lower, kin.q_upper)
            lo, hi = kin.q_lower, kin.q_upper
            j4_lo, j4_hi = float(lo[J4_INDEX]), float(hi[J4_INDEX])

            dRs = [np.eye(3)]
            for deg, axis in (
                (20.0, np.array([1.0, 0.0, 0.0])),
                (-20.0, np.array([1.0, 0.0, 0.0])),
                (20.0, np.array([0.0, 1.0, 0.0])),
                (-20.0, np.array([0.0, 1.0, 0.0])),
                (25.0, np.array([0.0, 0.0, 1.0])),
                (-25.0, np.array([0.0, 0.0, 1.0])),
            ):
                dRs.append(Rsc.from_rotvec(np.deg2rad(deg) * axis).as_matrix())

            rows = []
            q = q0.copy()
            for i, Tr0 in enumerate(pts_rail_np):
                best = None
                q_best = q
                for dR in dRs:
                    Tr = Tr0.copy()
                    Tr[:3, :3] = Tr0[:3, :3] @ dR
                    pose = tcp_to_pose6(Tr, euler_order=ik_cfg.euler_order)
                    q_try, converged, report = solve_pose_ik(
                        kin,
                        q_seed=q,
                        pose_target=pose,
                        qp_cfg=ik_cfg.qp,
                        nullspace_cfg=ik_cfg.nullspace,
                        max_iters=250,
                        pos_tol_m=1.0e-3,
                        rot_tol_rad=0.05,
                    )
                    j4 = float(q_try[J4_INDEX])
                    j4_margin = float(min(j4 - j4_lo, j4_hi - j4))
                    cand = {
                        "ok": bool(converged),
                        "j4_rad": j4,
                        "j4_margin_rad": j4_margin,
                        "j4_margin_deg": float(np.rad2deg(j4_margin)),
                        "sigma_min": float(report.sigma_min),
                        "pos_err_mm": float(report.pos_err_mm),
                        "rot_err_deg": float(report.rot_err_deg),
                        "rail_m": float(q_try[0]),
                        "xyz_world_m": pts[i].tolist(),
                    }

                    def key(c):
                        return (int(c["ok"]), c["j4_margin_rad"], c["sigma_min"])

                    if best is None or key(cand) > key(best):
                        best = cand
                        q_best = q_try
                q = np.clip(q_best, lo, hi)
                rows.append(best)
                if (i + 1) % 5 == 0 or i == 0:
                    print(
                        f"[qpik zb={zb_cm:.0f}] {i + 1}/{len(pts)} ok={best['ok']} "
                        f"J4_margin={best['j4_margin_deg']:.1f}deg sigma={best['sigma_min']:.3f}",
                        flush=True,
                    )

            ok = np.array([r["ok"] for r in rows], dtype=bool)
            j4m = np.array([r["j4_margin_deg"] for r in rows], dtype=float)
            sig = np.array([r["sigma_min"] for r in rows], dtype=float)
            near_lim = ok & (j4m < 10.0)
            ik_by_z[f"zb_{int(zb_cm):02d}cm"] = {
                "j4_limit_deg": [
                    float(np.rad2deg(j4_lo)),
                    float(np.rad2deg(j4_hi)),
                ],
                "ok_fraction": float(ok.mean()),
                "mean_j4_margin_deg": float(j4m.mean()),
                "mean_j4_margin_deg_ok_only": float(j4m[ok].mean()) if ok.any() else float("nan"),
                "frac_ok_with_j4_margin_lt_10deg": float(near_lim.sum() / max(ok.size, 1)),
                "mean_sigma_min": float(sig.mean()),
                "mean_sigma_min_ok_only": float(sig[ok].mean()) if ok.any() else float("nan"),
                "points": rows,
            }

    # Verdict
    e9 = ird_summary["zb_09cm"]["bed_edge_near_arm"]
    e26 = ird_summary["zb_26cm"]["bed_edge_near_arm"]
    verdict = {
        "ird_edge_mean_prefers": "9cm"
        if e9["mean_clearance"] >= e26["mean_clearance"]
        else "26cm",
        "note": (
            "IRD has no J4 margin head; if IK shows 26cm ok but J4_margin small, "
            "that explains IRD 'fake good' near the arm."
        ),
    }
    if ik_by_z:
        i9 = ik_by_z["zb_09cm"]
        i26 = ik_by_z["zb_26cm"]
        verdict["ik_ok_fraction"] = {
            "zb_09cm": i9["ok_fraction"],
            "zb_26cm": i26["ok_fraction"],
        }
        verdict["ik_mean_j4_margin_deg_ok"] = {
            "zb_09cm": i9["mean_j4_margin_deg_ok_only"],
            "zb_26cm": i26["mean_j4_margin_deg_ok_only"],
        }
        verdict["ik_frac_j4_lt_10deg"] = {
            "zb_09cm": i9["frac_ok_with_j4_margin_lt_10deg"],
            "zb_26cm": i26["frac_ok_with_j4_margin_lt_10deg"],
        }
        # Prefer higher ok, then larger J4 margin
        score = lambda d: (d["ok_fraction"], d["mean_j4_margin_deg_ok_only"])
        verdict["ik_prefers"] = (
            "9cm" if score(i9) >= score(i26) else "26cm"
        )

    summary = {
        "config": {
            "base_xy_m": base_xy.tolist(),
            "tip_half_angle_deg": 30.0,
            "roll_half_range_deg": 30.0,
            "leg_heights_m": leg_heights,
            "grid_cm": float(args.grid_cm),
            "m_safe": None if m_safe is None else float(m_safe),
            "rail_track_x_mm": [float(rail_p0[0] * 1000), float(rail_p1[0] * 1000)],
            "rail_track_y_mm": [float(rail_p0[1] * 1000), float(rail_p1[1] * 1000)],
        },
        "ird_maps": ird_summary,
        "ird_edge_sample_points": ird_edge_pts,
        "qpik_edge_spotcheck": ik_by_z,
        "verdict": verdict,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n======== VERDICT ========", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"summary → {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
