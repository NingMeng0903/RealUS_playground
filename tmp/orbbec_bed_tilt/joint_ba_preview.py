#!/usr/bin/env python3
"""Offline Stage-5 + bed-plane joint hand-eye preview. Does not write yaml.

  source camera_calibration/env.sh
  python tmp/orbbec_bed_tilt/joint_ba_preview.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rsc

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
for _p in (_REPO / "rm75_control", _REPO / "camera_calibration" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from multicam_calib.board.apriltag_board import build_board_geometry  # noqa: E402
from multicam_calib.calib.orbbec_handeye import (  # noqa: E402
    _reproj_rmse,
    load_orbbec_color_intrinsics,
    load_orbbec_handeye_captures,
    observations_from_captures,
    orbbec_handeye_captures_path,
    orbbec_handeye_path,
)
from multicam_calib.calib.pose_graph import se3_exp, se3_inv, se3_log  # noqa: E402
from multicam_calib.calib.urdf_fk import UrdfFK  # noqa: E402
from multicam_calib.io.config import load_board, load_robot  # noqa: E402

TILT_ROOT = Path(__file__).resolve().parent
LAMBDAS = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
EZ = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def _tilts(n: np.ndarray) -> tuple[float, float, float]:
    n = _unit(n)
    if n[2] < 0.0:
        n = -n
    tot = float(np.rad2deg(np.arccos(np.clip(n[2], -1.0, 1.0))))
    rx = float(np.rad2deg(np.arctan2(-n[1], n[2])))
    ry = float(np.rad2deg(np.arctan2(n[0], n[2])))
    return tot, rx, ry


def _fit_n_cam(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 20:
        raise RuntimeError("too few xyz_cam points")
    zmed = float(np.median(pts[:, 2]))
    band = np.abs(pts[:, 2] - zmed) < 0.08
    if int(np.count_nonzero(band)) >= 20:
        pts = pts[band]
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = _unit(vh[-1])
    if float(n @ c) > 0.0:
        n = -n
    return n


def _n_world(T_wl7: np.ndarray, T_lc: np.ndarray, n_cam: np.ndarray) -> np.ndarray:
    n = (T_wl7 @ T_lc)[:3, :3] @ n_cam
    n = _unit(n)
    if n[2] < 0.0:
        n = -n
    return n


@dataclass
class PlaneObs:
    name: str
    T_wl7: np.ndarray
    n_cam: np.ndarray


@dataclass
class SolveOut:
    tag: str
    T_lc: np.ndarray
    T_rb: np.ndarray
    ba_rmse: float
    plane_rms: float
    tilts: list[float]
    dR_deg: float
    dZ_deg: float
    dt_mm: float
    lam: float | None = None


def _load_planes(root: Path) -> list[PlaneObs]:
    out: list[PlaneObs] = []
    for gdir in sorted(root.glob("g0*")):
        z = np.load(gdir / "cloud.npz")
        T_wl7 = z["T_world_railbase"] @ z["T_railbase_link7"]
        out.append(PlaneObs(name=gdir.name, T_wl7=T_wl7, n_cam=_fit_n_cam(z["xyz_cam"])))
    if len(out) < 3:
        raise RuntimeError(f"need ≥3 bed groups in {root}")
    return out


def _load_baseline_T() -> tuple[np.ndarray, np.ndarray]:
    data = yaml.safe_load(orbbec_handeye_path().read_text(encoding="utf-8")) or {}
    T_lc = np.asarray(data["T_link7_cam"], dtype=np.float64).reshape(4, 4)
    T_rb = np.asarray(data["T_railbase_board"], dtype=np.float64).reshape(4, 4)
    return T_lc, T_rb


def _metrics(T_lc: np.ndarray, T_rb: np.ndarray, T0: np.ndarray, obs, planes, K, dist) -> tuple:
    ba = _reproj_rmse(obs, T_lc, T_rb, K, dist)
    tilts = [_tilts(_n_world(p.T_wl7, T_lc, p.n_cam))[0] for p in planes]
    prms = float(np.sqrt(np.mean(np.square(tilts)))) if tilts else float("nan")
    Rerr = T0[:3, :3].T @ T_lc[:3, :3]
    dR = float(np.degrees(np.arccos(np.clip((np.trace(Rerr) - 1.0) * 0.5, -1.0, 1.0))))
    dZ = _ang_deg(T0[:3, 2], T_lc[:3, 2])
    dt = float(np.linalg.norm(T_lc[:3, 3] - T0[:3, 3]) * 1000.0)
    return ba, prms, tilts, dR, dZ, dt


def _pack(tag, T_lc, T_rb, T0, obs, planes, K, dist, lam=None) -> SolveOut:
    ba, prms, tilts, dR, dZ, dt = _metrics(T_lc, T_rb, T0, obs, planes, K, dist)
    return SolveOut(tag, T_lc, T_rb, ba, prms, tilts, dR, dZ, dt, lam)


def _tag_chunks(obs, T_lc, T_rb, K, dist) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for o in obs:
        T_cam_board = se3_inv(o.T_railbase_link7 @ T_lc) @ T_rb
        rvec, _ = cv2_rodrigues(T_cam_board[:3, :3])
        tvec = T_cam_board[:3, 3].reshape(3, 1)
        proj, _ = project_points(o.object_pts, rvec, tvec, K, dist)
        chunks.append((proj.reshape(-1, 2) - o.image_pts).reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0)


def cv2_rodrigues(R: np.ndarray):
    import cv2

    return cv2.Rodrigues(R)


def project_points(obj, rvec, tvec, K, dist):
    import cv2

    return cv2.projectPoints(obj, rvec, tvec, K, dist)


def _cauchy_map(r: np.ndarray, f_scale: float = 1.0) -> np.ndarray:
    """Map pixel residuals so linear LS matches Cauchy cost (f_scale=1 like Stage 5)."""
    c = float(f_scale)
    return c * np.sqrt(np.log1p(np.square(r / c))) * np.sign(r)


def _plane_deg(T_lc: np.ndarray, planes: list[PlaneObs]) -> np.ndarray:
    out = []
    for p in planes:
        n = _n_world(p.T_wl7, T_lc, p.n_cam)
        out.extend(np.rad2deg(n[:2]).tolist())
    return np.asarray(out, dtype=np.float64)


def _refit_board(obs, T_lc, T_rb0, K, dist) -> np.ndarray:
    x0 = se3_log(T_rb0)

    def residual(x: np.ndarray) -> np.ndarray:
        return _tag_chunks(obs, T_lc, se3_exp(x), K, dist)

    res = least_squares(residual, x0, loss="cauchy", f_scale=1.0, max_nfev=80)
    return se3_exp(res.x)


def solve_A(T0: np.ndarray, T_rb0, obs, planes, K, dist) -> SolveOut:
    def residual(p: np.ndarray) -> np.ndarray:
        R = Rsc.from_euler("xyz", [p[0], p[1], 0.0]).as_matrix()
        T = T0.copy()
        T[:3, :3] = T0[:3, :3] @ R
        return _plane_deg(T, planes)

    sol = least_squares(residual, np.zeros(2), method="lm")
    R = Rsc.from_euler("xyz", [sol.x[0], sol.x[1], 0.0]).as_matrix()
    T_lc = T0.copy()
    T_lc[:3, :3] = T0[:3, :3] @ R
    T_rb = _refit_board(obs, T_lc, T_rb0, K, dist)
    return _pack("A plane-only rxy", T_lc, T_rb, T0, obs, planes, K, dist)


def solve_joint(
    T0: np.ndarray,
    T_rb0: np.ndarray,
    obs,
    planes: list[PlaneObs],
    K,
    dist,
    lam: float,
    *,
    lock_t: bool,
    max_nfev: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    t_lock = T0[:3, 3].copy()
    if lock_t:
        x0 = np.concatenate([se3_log(T0)[:3], se3_log(T_rb0)])
    else:
        x0 = np.concatenate([se3_log(T0), se3_log(T_rb0)])

    def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if lock_t:
            T_lc = se3_exp(np.concatenate([x[:3], t_lock]))
            T_rb = se3_exp(x[3:])
        else:
            T_lc = se3_exp(x[:6])
            T_rb = se3_exp(x[6:])
        return T_lc, T_rb

    def residual(x: np.ndarray) -> np.ndarray:
        T_lc, T_rb = split(x)
        tags = _cauchy_map(_tag_chunks(obs, T_lc, T_rb, K, dist), 1.0)
        if lam <= 0.0 or not planes:
            return tags
        return np.concatenate([tags, float(lam) * _plane_deg(T_lc, planes)])

    res = least_squares(residual, x0, loss="linear", max_nfev=max_nfev)
    return split(res.x)


def _print_row(s: SolveOut) -> None:
    tilts = " ".join(f"{v:.2f}" for v in s.tilts)
    lam = "-" if s.lam is None else f"{s.lam:g}"
    print(
        f"{s.tag:<22} λ={lam:<5} ba={s.ba_rmse:5.3f} plane_rms={s.plane_rms:5.2f} "
        f"dR={s.dR_deg:5.2f} dZ={s.dZ_deg:5.2f} dt={s.dt_mm:5.2f}mm  tilts[{tilts}]"
    )


def _gates(s: SolveOut, ba0: float) -> list[str]:
    lines = []
    lines.append(f"  plane_rms ≤1.2° : {'PASS' if s.plane_rms <= 1.2 else 'FAIL'} ({s.plane_rms:.2f})")
    lines.append(
        f"  ba_rmse ≤ {ba0:.2f}+0.25 : {'PASS' if s.ba_rmse <= ba0 + 0.25 else 'FAIL'} ({s.ba_rmse:.3f})"
    )
    lines.append(f"  Δt < 2 mm        : {'PASS' if s.dt_mm < 2.0 else 'FAIL'} ({s.dt_mm:.2f})")
    lines.append(f"  optical Z 2–3°   : {'PASS' if 2.0 <= s.dZ_deg <= 3.2 else 'NOTE'} ({s.dZ_deg:.2f})")
    fight = s.ba_rmse >= 3.0 or s.plane_rms > 2.0
    lines.append(
        f"  fight (ba≥3 or plane>2 after opt): {'YES — do not write yaml' if fight else 'no'}"
    )
    return lines


def main() -> int:
    captures = load_orbbec_handeye_captures(orbbec_handeye_captures_path())
    if len(captures) < 6:
        print(f"need Stage 5 captures, got {len(captures)}", file=sys.stderr)
        return 2
    board = build_board_geometry(load_board())
    robot_cfg = load_robot()
    fk = UrdfFK(robot_cfg.wbc_urdf_path())
    size = tuple(int(x) for x in (captures[0].get("image_size") or (640, 480))[:2])
    K_intr = load_orbbec_color_intrinsics(image_size=size)
    obs = observations_from_captures(
        captures,
        board_geom=board,
        intrinsics=K_intr,
        fk=fk,
        min_tags=8,
        offsets_j16_rad=None,
    )
    K = K_intr.K
    dist = K_intr.dist
    planes = _load_planes(TILT_ROOT)
    T0, Trb0 = _load_baseline_T()
    n_tag = 0
    for o in obs:
        n_tag += int(o.image_pts.shape[0]) * 2
    print(
        f"views={len(obs)} tag_residuals={n_tag} planes={len(planes)} "
        f"K fx={float(K[0, 0]):.2f} source={K_intr.source} dq=0"
    )
    print("no yaml / URDF writes\n")

    base = _pack("0 baseline yaml", T0, Trb0, T0, obs, planes, K, dist)
    _print_row(base)

    a = solve_A(T0, Trb0, obs, planes, K, dist)
    _print_row(a)

    print("\n-- B tag + λ plane, R free, t locked --")
    rows_b: list[SolveOut] = []
    for lam in LAMBDAS:
        T_lc, T_rb = solve_joint(T0, Trb0, obs, planes, K, dist, lam, lock_t=True)
        s = _pack(f"B lock-t", T_lc, T_rb, T0, obs, planes, K, dist, lam=lam)
        rows_b.append(s)
        _print_row(s)

    print("\n-- C tag + λ plane, SE3 free --")
    rows_c: list[SolveOut] = []
    for lam in (50.0, 100.0, 200.0):
        T_lc, T_rb = solve_joint(T0, Trb0, obs, planes, K, dist, lam, lock_t=False)
        s = _pack(f"C free-t", T_lc, T_rb, T0, obs, planes, K, dist, lam=lam)
        rows_c.append(s)
        _print_row(s)

    def score(s: SolveOut) -> tuple:
        ba_ok = 1 if s.ba_rmse <= base.ba_rmse + 0.25 else 0
        pl_ok = 1 if s.plane_rms <= 1.2 else 0
        return (ba_ok + pl_ok, -s.plane_rms, -(s.ba_rmse - base.ba_rmse))

    best_b = max(rows_b, key=score)
    print(f"\n-- pick B λ={best_b.lam:g} for LOO --")

    print("26-view LOO (held-out tag RMSE, all planes, B λ):")
    loo_ba = []
    for i in range(len(obs)):
        kept = [o for j, o in enumerate(obs) if j != i]
        T_lc, T_rb = solve_joint(T0, Trb0, kept, planes, K, dist, float(best_b.lam), lock_t=True, max_nfev=50)
        held = _reproj_rmse([obs[i]], T_lc, T_rb, K, dist)
        loo_ba.append(held)
        print(f"  hold view {i:02d}: {held:.3f} px")
    print(f"  LOO ba rms={float(np.sqrt(np.mean(np.square(loo_ba)))):.3f}  mean={float(np.mean(loo_ba)):.3f}")

    print("9-plane LOO (held-out tilt, all tags, B λ):")
    loo_pl = []
    for k in range(len(planes)):
        kept = [p for j, p in enumerate(planes) if j != k]
        T_lc, T_rb = solve_joint(T0, Trb0, obs, kept, K, dist, float(best_b.lam), lock_t=True, max_nfev=50)
        tot = _tilts(_n_world(planes[k].T_wl7, T_lc, planes[k].n_cam))[0]
        loo_pl.append(tot)
        print(f"  hold {planes[k].name}: {tot:.2f}°")
    loo_pl_rms = float(np.sqrt(np.mean(np.square(loo_pl))))
    print(f"  LOO plane rms={loo_pl_rms:.2f}  max={max(loo_pl):.2f}")

    print("\n-- gates vs baseline ba={:.3f} --".format(base.ba_rmse))
    print("A:")
    print("\n".join(_gates(a, base.ba_rmse)))
    print(f"B λ={best_b.lam:g}:")
    print("\n".join(_gates(best_b, base.ba_rmse)))
    print(f"  9-plane LOO ≤2° : {'PASS' if loo_pl_rms <= 2.0 and max(loo_pl) <= 2.6 else 'FAIL'} (rms {loo_pl_rms:.2f})")
    print("C λ=100:")
    c100 = next((s for s in rows_c if s.lam == 100.0), rows_c[0])
    print("\n".join(_gates(c100, base.ba_rmse)))

    write = (
        best_b.plane_rms <= 1.2
        and best_b.ba_rmse <= base.ba_rmse + 0.25
        and best_b.dt_mm < 2.0
        and loo_pl_rms <= 2.0
        and best_b.ba_rmse < 3.0
    )
    print("\nverdict: " + ("numbers pass the write-yaml gate (still not writing)" if write else "do NOT write yaml"))
    if not write:
        print("use as twin preview rotation only, or drop the plane term.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
