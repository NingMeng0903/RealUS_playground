#!/usr/bin/env python3
"""Snapshot Orbbec wrist cloud + robot/TF for bed-tilt diagnosis.

  source camera_calibration/env.sh
  python tmp/orbbec_bed_tilt/capture.py --group 01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
for _p in (_REPO / "rm75_control", _REPO / "camera_calibration" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from multicam_calib.ingress.robot_state import RobotStateReader  # noqa: E402
from rm75_control.control.joint_admittance_8dof.param_model.generator import (  # noqa: E402
    compute_layout,
    load_spec,
)
from rm75_control.control.joint_admittance_8dof.param_model.paths import (  # noqa: E402
    DEFAULT_SPEC_YAML,
    GENERATED_URDF,
)
from rm75_control.control.joint_admittance_8dof.param_model.placement import (  # noqa: E402
    entity_pose_from_calib,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import (  # noqa: E402
    DEFAULT_ORBBEC_CLOUD_BIND,
    DEFAULT_ORBBEC_CLOUD_TOPIC,
    RailBaseLink7FK,
    T_from_pos_quat_wxyz,
    T_world_cam,
    load_T_link7_cam,
    transform_points,
    unpack_cloud_multipart,
)

OUT_ROOT = Path(__file__).resolve().parent


def expand_q8(q_deg: np.ndarray, rail_m: float) -> np.ndarray:
    q = np.asarray(q_deg, dtype=np.float64).reshape(-1)
    if q.size >= 8:
        return q[:8].copy()
    if float(np.max(np.abs(q[:7]))) > 2.0 * np.pi:
        q = np.deg2rad(q[:7])
    out = np.zeros(8, dtype=np.float64)
    out[0] = float(rail_m)
    out[1:8] = q[:7]
    return out


def _recv_cloud(subscribe: str, topic: str, timeout_s: float = 2.5):
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.connect(subscribe)
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    try:
        parts = None
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                parts = sock.recv_multipart()
                break
            except zmq.Again:
                continue
        if parts is None:
            raise TimeoutError(f"no cloud on {subscribe} topic={topic}")
    finally:
        sock.close(0)
    return unpack_cloud_multipart(parts)


def _fit_plane(pts: np.ndarray) -> dict:
    xyz = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(xyz).all(axis=1)
    xyz = xyz[ok]
    if xyz.shape[0] < 20:
        raise RuntimeError(f"too few points for plane: {xyz.shape[0]}")
    zmed0 = float(np.median(xyz[:, 2]))
    band = np.abs(xyz[:, 2] - zmed0) < 0.08
    if int(np.count_nonzero(band)) >= 20:
        xyz = xyz[band]
    c = xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(xyz - c, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    if float(n[2]) < 0.0:
        n = -n
    d = float(c @ n)
    resid = xyz @ n - d
    rms = float(np.sqrt(np.mean(resid * resid)))
    inlier = np.abs(resid) < 0.015
    return {
        "normal": n,
        "d": d,
        "rms_m": rms,
        "resid_med_m": float(np.median(np.abs(resid))),
        "n_pts": int(xyz.shape[0]),
        "n_inlier_15mm": int(np.count_nonzero(inlier)),
        "z_med": float(np.median(xyz[:, 2])),
        "z_p10": float(np.percentile(xyz[:, 2], 10)),
        "z_p90": float(np.percentile(xyz[:, 2], 90)),
        "xyz_min": xyz.min(axis=0),
        "xyz_max": xyz.max(axis=0),
        "centroid": c,
    }


def _tilts(n: np.ndarray) -> dict:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    total = float(np.rad2deg(np.arccos(np.clip(n[2], -1.0, 1.0))))
    about_x = float(np.rad2deg(np.arctan2(-n[1], n[2])))
    about_y = float(np.rad2deg(np.arctan2(n[0], n[2])))
    return {
        "normal": n.tolist(),
        "total_deg": total,
        "about_x_deg": about_x,
        "about_y_deg": about_y,
    }


def _T_list(T: np.ndarray) -> list:
    return np.asarray(T, dtype=np.float64).reshape(4, 4).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="e.g. 01")
    ap.add_argument("--note", default="")
    ap.add_argument("--subscribe", default=DEFAULT_ORBBEC_CLOUD_BIND)
    ap.add_argument("--topic", default=DEFAULT_ORBBEC_CLOUD_TOPIC)
    args = ap.parse_args()

    gid = str(args.group).strip().zfill(2)
    dest = OUT_ROOT / f"g{gid}"
    dest.mkdir(parents=True, exist_ok=True)

    spec = load_spec(DEFAULT_SPEC_YAML)
    layout = compute_layout(spec)
    calib = resolve_world_calib(spec, layout)
    entity = entity_pose_from_calib(calib)
    T_world_railbase = T_from_pos_quat_wxyz(entity["pos"], entity["quat_wxyz"])
    T_link7_cam = load_T_link7_cam()
    fk = RailBaseLink7FK(GENERATED_URDF)

    reader = RobotStateReader()
    snap = reader.read()
    if snap is None:
        print(f"rm75_state missing: {reader.last_error}", file=sys.stderr)
        return 2
    meta, xyz_cam, rgb = _recv_cloud(args.subscribe, args.topic)
    snap2 = reader.read() or snap
    q8 = expand_q8(snap2.q_deg, snap2.rail_m)
    T_rb_l7 = fk.T_railbase_link7(q8)
    T_wc = T_world_cam(T_world_railbase, T_rb_l7, T_link7_cam)
    xyz_w = transform_points(T_wc, xyz_cam)
    plane = _fit_plane(xyz_w)
    tilts = _tilts(plane["normal"])

    R = T_wc[:3, :3]
    cam_pos = T_wc[:3, 3]
    cam_z = R[:, 2]
    look_down = float(np.rad2deg(np.arccos(np.clip(-cam_z[2], -1.0, 1.0))))

    summary = {
        "group": gid,
        "note": str(args.note),
        "wall_time_ns": int(time.time_ns()),
        "robot": {
            "seq": int(snap2.seq),
            "t_s": float(snap2.t_s),
            "rail_m": float(snap2.rail_m),
            "q_deg": snap2.q_deg.tolist(),
            "q8_rad": q8.tolist(),
            "pose6_railbase_tcp": snap2.pose.tolist(),
        },
        "cloud_meta": {k: meta[k] for k in meta if k not in ("T_link7_cam",)},
        "n": int(xyz_cam.shape[0]),
        "T_world_railbase": _T_list(T_world_railbase),
        "T_railbase_link7": _T_list(T_rb_l7),
        "T_link7_cam": _T_list(T_link7_cam),
        "T_world_cam": _T_list(T_wc),
        "cam_pos_world": cam_pos.tolist(),
        "cam_axes_world": {"x": R[:, 0].tolist(), "y": R[:, 1].tolist(), "z": cam_z.tolist()},
        "look_down_from_world_negZ_deg": look_down,
        "world_calib": {
            "base_pos_m": np.asarray(calib["base_pos_m"], dtype=float).tolist(),
            "base_quat_wxyz": np.asarray(calib["base_quat_wxyz"], dtype=float).tolist(),
            "entity_pos": list(entity["pos"]),
            "entity_quat_wxyz": list(entity["quat_wxyz"]),
        },
        "plane": {
            **tilts,
            "d": plane["d"],
            "rms_mm": plane["rms_m"] * 1000.0,
            "resid_med_mm": plane["resid_med_m"] * 1000.0,
            "n_pts": plane["n_pts"],
            "n_inlier_15mm": plane["n_inlier_15mm"],
            "z_med_m": plane["z_med"],
            "z_p10_m": plane["z_p10"],
            "z_p90_m": plane["z_p90"],
            "centroid_m": plane["centroid"].tolist(),
            "xyz_min": plane["xyz_min"].tolist(),
            "xyz_max": plane["xyz_max"].tolist(),
        },
    }
    np.savez_compressed(
        dest / "cloud.npz",
        xyz_cam=xyz_cam.astype(np.float32),
        rgb=rgb.astype(np.float32),
        xyz_world=xyz_w.astype(np.float32),
        q8=q8,
        q_deg=snap2.q_deg,
        rail_m=np.asarray(snap2.rail_m, dtype=np.float64),
        pose6=snap2.pose,
        T_world_railbase=T_world_railbase,
        T_railbase_link7=T_rb_l7,
        T_link7_cam=T_link7_cam,
        T_world_cam=T_wc,
        plane_normal=np.asarray(plane["normal"], dtype=np.float64),
        plane_d=np.asarray(plane["d"], dtype=np.float64),
    )
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    p = summary["plane"]
    print(
        f"saved {dest} n={p['n_pts']} "
        f"tilt={p['total_deg']:.2f}° aboutX={p['about_x_deg']:.2f}° aboutY={p['about_y_deg']:.2f}° "
        f"z_med={p['z_med_m']:.4f} rms={p['rms_mm']:.2f}mm "
        f"rail={snap2.rail_m:.3f} q_deg={np.round(snap2.q_deg, 1).tolist()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
