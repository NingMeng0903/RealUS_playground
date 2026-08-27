"""Ray-preserving depth scale (factory D2C + color K stay on the color ray).

Dense depth-board planes sit ~2° off the AprilTag plane. A rotation about the
color optical center would fix that angle but knocks same-pixel corners off
the ray (~1 cm lateral). This model only scales each point along its ray:

    xyz' = xyz * (c0 + c1 * x/z + c2 * y/z)

``xn=x/z``, ``yn=y/z`` are normalized image coordinates. Stage 5 / hand-eye
are not written.
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

from multicam_calib.io.config import RESULTS_DIR

SCHEMA = "orbbec_depth_ray_v1"
_REPO = Path(__file__).resolve().parents[4]
DEFAULT_BOARD_RGBD_ROOT = _REPO / "tmp" / "orbbec_bed_tilt" / "board_rgbd"


def orbbec_depth_ray_path() -> Path:
    return RESULTS_DIR / "orbbec_depth_ray.yaml"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def _tag_plane(T_cam_board: np.ndarray) -> tuple[np.ndarray, float]:
    T = np.asarray(T_cam_board, dtype=np.float64).reshape(4, 4)
    n = _unit(T[:3, 2])
    t = T[:3, 3]
    if float(n @ t) > 0.0:
        n = -n
    return n, float(n @ t)


def _plane_of(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = _unit(vh[-1])
    if float(n @ c) > 0.0:
        n = -n
    return n


def _features(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    z = np.clip(pts[:, 2], 1e-6, None)
    xn = pts[:, 0] / z
    yn = pts[:, 1] / z
    return np.column_stack([np.ones(len(pts)), xn, yn])


def apply_depth_ray_scale(xyz: np.ndarray, coeff: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float32)
    if coeff is None or pts.size == 0:
        return pts
    c = np.asarray(coeff, dtype=np.float64).reshape(-1)
    if c.size < 3:
        return pts
    pts3 = pts.reshape(-1, 3)
    mult = (_features(pts3) @ c[:3]).astype(np.float32)
    return (pts3 * mult.reshape(-1, 1)).astype(np.float32, copy=False)


def load_depth_ray_coeff(path: Path | str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (c0,c1,c2), meta. Missing file → (1,0,0) identity."""
    ident = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    p = Path(path) if path is not None else orbbec_depth_ray_path()
    if not p.is_file():
        return ident, {"source": "identity", "path": str(p)}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = data.get("scale_poly_xn_yn")
    try:
        c = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception:
        return ident, {"source": "identity_bad_yaml", "path": str(p)}
    if c.size < 3 or not np.isfinite(c[:3]).all() or abs(float(c[0]) - 1.0) > 0.2:
        return ident, {"source": "identity_bad_coeff", "path": str(p)}
    meta = dict(data.get("metadata") or {})
    meta["source"] = str(p)
    meta["scale_poly_xn_yn"] = [float(v) for v in c[:3]]
    return c[:3].copy(), meta


@dataclass
class DepthRayFit:
    coeff: np.ndarray
    views: list[str]
    per_view_deg_before: list[float]
    per_view_deg_after: list[float]
    loo_deg: list[float]
    n_points: int
    serial: str

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


def _collect(root: Path) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for gdir in sorted(root.glob("g*")):
        npz = gdir / "cloud.npz"
        summary = gdir / "summary.json"
        if not npz.is_file() or not summary.is_file():
            continue
        z = np.load(npz)
        s = json.loads(summary.read_text(encoding="utf-8"))
        T = np.asarray(s.get("T_cam_board") if "T_cam_board" in s else z.get("T_cam_board"), dtype=np.float64)
        if T.shape != (4, 4) or float(np.linalg.norm(T[:3, 3])) < 1e-6:
            continue
        n_tag, d_tag = _tag_plane(T)
        xyz = np.asarray(z["xyz_board"] if "xyz_board" in z.files else z["xyz_all"], dtype=np.float64).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.shape[0] < 50:
            continue
        if xyz.shape[0] > 8000:
            xyz = xyz[:: int(np.ceil(xyz.shape[0] / 8000))]
        nx = xyz @ n_tag
        scale = np.divide(d_tag, nx, out=np.ones(len(xyz)), where=np.abs(nx) > 1e-9)
        views.append(
            {
                "name": gdir.name,
                "xyz": xyz,
                "n_tag": n_tag,
                "scale": scale,
                "serial": str(s.get("serial") or ""),
            }
        )
    return views


def _fit_coeff(views: list[dict[str, Any]]) -> np.ndarray:
    F = np.concatenate([_features(v["xyz"]) for v in views], axis=0)
    y = np.concatenate([v["scale"] for v in views], axis=0)
    ok = np.isfinite(y) & (y > 0.90) & (y < 1.10)
    w, *_ = np.linalg.lstsq(F[ok], y[ok], rcond=None)
    return np.asarray(w, dtype=np.float64).reshape(3)


def fit_depth_ray(board_rgbd_root: Path | str) -> DepthRayFit:
    views = _collect(Path(board_rgbd_root))
    if len(views) < 3:
        raise RuntimeError(f"need ≥3 board RGB-D views in {board_rgbd_root}")
    w = _fit_coeff(views)
    before = [_ang_deg(_plane_of(v["xyz"]), v["n_tag"]) for v in views]
    after = [_ang_deg(_plane_of(apply_depth_ray_scale(v["xyz"], w)), v["n_tag"]) for v in views]
    loo: list[float] = []
    for k in range(len(views)):
        wk = _fit_coeff([v for i, v in enumerate(views) if i != k])
        hold = views[k]
        loo.append(_ang_deg(_plane_of(apply_depth_ray_scale(hold["xyz"], wk)), hold["n_tag"]))
    serials = {str(v.get("serial") or "") for v in views if v.get("serial")}
    return DepthRayFit(
        coeff=w,
        views=[v["name"] for v in views],
        per_view_deg_before=before,
        per_view_deg_after=after,
        loo_deg=loo,
        n_points=int(sum(int(v["xyz"].shape[0]) for v in views)),
        serial=next(iter(serials)) if len(serials) == 1 else ",".join(sorted(serials)),
    )


def save_depth_ray(fit: DepthRayFit, path: Path | None = None) -> Path:
    p = path or orbbec_depth_ray_path()
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
            "method": "along_ray_scale_poly_xn_yn",
            "applies_to": "xyz_cam after D2C unproject; xyz *= (c0 + c1*x/z + c2*y/z)",
            "does_not_change": ["orbbec_handeye.yaml", "genesis_bundle.yaml"],
            "note": "Preserves color rays. Replaces the discarded optical-center R in orbbec_d2c_offset.yaml.",
        },
        "scale_poly_xn_yn": [float(v) for v in fit.coeff],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_BOARD_RGBD_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    fit = fit_depth_ray(args.root)
    path = save_depth_ray(fit, args.out)
    print(
        f"wrote {path}\n"
        f"serial={fit.serial!r} coeff={fit.coeff.tolist()}\n"
        f"angle_rms {fit.rms_before:.3f}° → {fit.rms_after:.3f}°  loo_rms={fit.loo_rms:.3f}°",
        flush=True,
    )
    for name, a, b, lo in zip(fit.views, fit.per_view_deg_before, fit.per_view_deg_after, fit.loo_deg):
        print(f"  {name}: {a:.3f}° → {b:.3f}°  loo={lo:.3f}°", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
