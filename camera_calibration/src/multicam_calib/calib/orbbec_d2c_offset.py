"""Depth-to-color residual rotation after factory D2C (camera frame).

Stage 5 ``T_link7_cam`` is the color optical frame. Unprojected D2C depth is
tilted a couple of degrees from the AprilTag board plane. This module fits a
single rotation ``R`` such that ``x_color = R @ x_depth`` and stores it in
``orbbec_d2c_offset.yaml``. It does not change hand-eye.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from multicam_calib.io.config import RESULTS_DIR

SCHEMA = "orbbec_d2c_offset_v1"
_REPO = Path(__file__).resolve().parents[4]
DEFAULT_BOARD_RGBD_ROOT = _REPO / "tmp" / "orbbec_bed_tilt" / "board_rgbd"


def orbbec_d2c_offset_path() -> Path:
    return RESULTS_DIR / "orbbec_d2c_offset.yaml"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def _wahba(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    H = src.T @ dst
    u, _, vt = np.linalg.svd(H)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        R = vt.T @ u.T
    return R


def _tag_plane(T_cam_board: np.ndarray) -> tuple[np.ndarray, float]:
    T = np.asarray(T_cam_board, dtype=np.float64).reshape(4, 4)
    n = _unit(T[:3, 2])
    t = T[:3, 3]
    if float(n @ t) > 0.0:
        n = -n
    return n, float(n @ t)


def apply_R_depth_to_color(xyz: np.ndarray, R: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float32)
    if R is None or pts.size == 0:
        return pts
    M = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return (pts.reshape(-1, 3) @ M.T).astype(np.float32, copy=False)


def load_R_depth_to_color(path: Path | str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (R, meta). Missing/invalid file → identity."""
    p = Path(path) if path is not None else orbbec_d2c_offset_path()
    ident = np.eye(3, dtype=np.float64)
    if not p.is_file():
        return ident, {"source": "identity", "path": str(p), "R_depth_to_color_rpy_xyz_deg": [0.0, 0.0, 0.0]}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = data.get("R_depth_to_color")
    try:
        R = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    except Exception:
        return ident, {"source": "identity_bad_yaml", "path": str(p)}
    if not np.isfinite(R).all() or abs(float(np.linalg.det(R)) - 1.0) > 0.05:
        return ident, {"source": "identity_bad_R", "path": str(p)}
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0.0:
        u[:, -1] *= -1.0
        R = u @ vt
    meta = dict(data.get("metadata") or {})
    meta["source"] = str(p)
    rpy = data.get("R_depth_to_color_rpy_xyz_deg")
    if rpy is None:
        rpy = np.rad2deg(Rsc.from_matrix(R).as_euler("xyz"))
    meta["R_depth_to_color_rpy_xyz_deg"] = [float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(3)]
    return R, meta


@dataclass
class D2COffsetFit:
    R: np.ndarray
    R_wahba: np.ndarray
    per_view_deg_before: list[float]
    per_view_deg_after: list[float]
    loo_deg: list[float]
    views: list[str]
    n_points: int
    serial: str
    ptp_rms_mm_before: float
    ptp_rms_mm_after: float

    @property
    def rms_before(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.per_view_deg_before))))

    @property
    def rms_after(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.per_view_deg_after))))

    @property
    def loo_rms(self) -> float:
        if not self.loo_deg:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(self.loo_deg))))


def _collect_views(root: Path) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for gdir in sorted(root.glob("g*")):
        npz = gdir / "cloud.npz"
        summary = gdir / "summary.json"
        if not npz.is_file() or not summary.is_file():
            continue
        z = np.load(npz)
        s = json.loads(summary.read_text(encoding="utf-8"))
        T = np.asarray(z["T_cam_board"] if "T_cam_board" in z.files else s.get("T_cam_board"), dtype=np.float64)
        if T.shape != (4, 4) or float(np.linalg.norm(T[:3, 3])) < 1e-6:
            continue
        n_tag, d_tag = _tag_plane(T)
        xyz = np.asarray(z["xyz_board"] if "xyz_board" in z.files else z["xyz_all"], dtype=np.float64).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.shape[0] < 50:
            continue
        n_dep = np.asarray(s.get("n_depth_board_cam") or z.get("n_depth_board"), dtype=np.float64)
        if n_dep.size != 3 or float(np.linalg.norm(n_dep)) < 0.1:
            c = xyz.mean(axis=0)
            _, _, vh = np.linalg.svd(xyz - c, full_matrices=False)
            n_dep = _unit(vh[-1])
            if float(n_dep @ c) > 0.0:
                n_dep = -n_dep
        if float(n_dep @ n_tag) < 0.0:
            n_dep = -n_dep
        if xyz.shape[0] > 2500:
            xyz = xyz[:: int(np.ceil(xyz.shape[0] / 2500))]
        views.append(
            {
                "name": gdir.name,
                "n_tag": n_tag,
                "d_tag": d_tag,
                "n_dep": _unit(n_dep),
                "xyz": xyz,
                "serial": str(s.get("serial") or ""),
            }
        )
    return views


def _ptp_rms_mm(views: list[dict[str, Any]], R: np.ndarray | None = None) -> float:
    chunks: list[np.ndarray] = []
    for v in views:
        pts = v["xyz"] if R is None else v["xyz"] @ np.asarray(R, dtype=np.float64).T
        chunks.append(pts @ v["n_tag"] - float(v["d_tag"]))
    r = np.concatenate(chunks) if chunks else np.zeros(0)
    return float(np.sqrt(np.mean(np.square(r))) * 1000.0) if r.size else float("nan")


def fit_R_depth_to_color(board_rgbd_root: Path | str) -> D2COffsetFit:
    """Wahba: one rotation mapping depth-board normals onto tag-board normals.

    Locked-distance point-to-plane is *not* used for ``R``. Depth has a mm-scale
    bias; absorbing it into a rotation tilts the optical axis the wrong way.
    """
    views = _collect_views(Path(board_rgbd_root))
    if len(views) < 3:
        raise RuntimeError(f"need ≥3 board RGB-D views in {board_rgbd_root}")
    n_dep = np.stack([v["n_dep"] for v in views], axis=0)
    n_tag = np.stack([v["n_tag"] for v in views], axis=0)
    R = _wahba(n_dep, n_tag)
    before = [_ang_deg(v["n_dep"], v["n_tag"]) for v in views]
    after = [_ang_deg(R @ v["n_dep"], v["n_tag"]) for v in views]
    loo: list[float] = []
    for k in range(len(views)):
        kept = [v for i, v in enumerate(views) if i != k]
        Rloo = _wahba(
            np.stack([v["n_dep"] for v in kept]),
            np.stack([v["n_tag"] for v in kept]),
        )
        hold = views[k]
        loo.append(_ang_deg(Rloo @ hold["n_dep"], hold["n_tag"]))
    n_pts = int(sum(int(v["xyz"].shape[0]) for v in views))
    serials = {str(v.get("serial") or "") for v in views if v.get("serial")}
    return D2COffsetFit(
        R=R,
        R_wahba=R,
        per_view_deg_before=before,
        per_view_deg_after=after,
        loo_deg=loo,
        views=[v["name"] for v in views],
        n_points=n_pts,
        serial=next(iter(serials)) if len(serials) == 1 else ",".join(sorted(serials)),
        ptp_rms_mm_before=_ptp_rms_mm(views, None),
        ptp_rms_mm_after=_ptp_rms_mm(views, R),
    )


def save_d2c_offset(fit: D2COffsetFit, path: Path | None = None) -> Path:
    p = path or orbbec_d2c_offset_path()
    rpy = Rsc.from_matrix(fit.R).as_euler("xyz")
    rpy_w = Rsc.from_matrix(fit.R_wahba).as_euler("xyz")
    payload = {
        "schema": SCHEMA,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "serial": fit.serial,
            "n_views": len(fit.views),
            "views": list(fit.views),
            "n_points": int(fit.n_points),
            "angle_rms_deg_before": fit.rms_before,
            "angle_rms_deg_after": fit.rms_after,
            "per_view_deg_before": [float(v) for v in fit.per_view_deg_before],
            "per_view_deg_after": [float(v) for v in fit.per_view_deg_after],
            "leave_one_out_deg": [float(v) for v in fit.loo_deg],
            "leave_one_out_rms_deg": fit.loo_rms,
            "point_to_plane_rms_mm_before": fit.ptp_rms_mm_before,
            "point_to_plane_rms_mm_after": fit.ptp_rms_mm_after,
            "method": "wahba_board_normals",
            "optical_axis_tilt_deg": float(_ang_deg(fit.R @ np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))),
            "applies_to": "xyz_cam after D2C unproject; x_color = R @ x_depth",
            "does_not_change": ["orbbec_handeye.yaml", "genesis_bundle.yaml"],
            "note": "Rotation only. Locked-distance point-to-plane is diagnostic (depth bias); it is not used for R.",
        },
        "R_depth_to_color": [[float(x) for x in row] for row in fit.R],
        "R_depth_to_color_rpy_xyz_rad": [float(v) for v in rpy],
        "R_depth_to_color_rpy_xyz_deg": [float(v) for v in np.rad2deg(rpy)],
        "R_wahba_rpy_xyz_deg": [float(v) for v in np.rad2deg(rpy_w)],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_BOARD_RGBD_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: calibration_results/orbbec_d2c_offset.yaml")
    args = ap.parse_args(argv)
    fit = fit_R_depth_to_color(args.root)
    path = save_d2c_offset(fit, args.out)
    rpy = np.rad2deg(Rsc.from_matrix(fit.R).as_euler("xyz"))
    print(
        f"wrote {path}\n"
        f"serial={fit.serial!r} views={fit.views}\n"
        f"R_rpy_xyz_deg=[{rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}]\n"
        f"angle_rms {fit.rms_before:.3f}° → {fit.rms_after:.3f}°  "
        f"loo_rms={fit.loo_rms:.3f}°\n"
        f"ptp_rms {fit.ptp_rms_mm_before:.2f} → {fit.ptp_rms_mm_after:.2f} mm",
        flush=True,
    )
    for name, a, b, lo in zip(fit.views, fit.per_view_deg_before, fit.per_view_deg_after, fit.loo_deg):
        print(f"  {name}: {a:.3f}° → {b:.3f}°  loo={lo:.3f}°", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
