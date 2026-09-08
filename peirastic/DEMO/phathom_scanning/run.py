#!/usr/bin/env python3
"""Orbbec cloud → phantom top-left standoff (PTP) → 4 N hybrid press.

Identifies the brown elevated top face, PTP (joint-space after IK) to a
pose 3–4 cm above the near-left corner with tool +Z into the surface,
then TRACK_HYBRID hold at F*=4 N.

    # Window A
    python -m peirastic.apps.run_controller

    # Orbbec publisher (USB; skip if already running)
    python perception/apps/run_orbbec_cloud_publisher.py

    # This script
    python -m peirastic.DEMO.phathom_scanning
    python -m peirastic.DEMO.phathom_scanning --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from scipy.spatial.transform import Rotation as Rsc

from peirastic.DEMO.cartesian import _fmt_pose, _live_pose
from peirastic.DEMO.phathom_scanning.cloud import (
    cloud_in_rail_base,
    recv_camera_cloud,
)
from peirastic.DEMO.phathom_scanning.detect import STANDOFF_M, detect_phantom_top
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK

FORCE_N = 4.0
FORCE_AXES = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
PTP_V = 0.30
HOLD_S = 12.0
CLOSE_TIMEOUT_S = 20.0
PRINT_S = 1.0


def _fmt_xyz(p) -> str:
    return " ".join(f"{float(v) * 1000.0:+7.1f}" for v in p[:3])


def _toward_mount(centroid, *, rail_m: float = 0.4) -> list[float]:
    c = [float(v) for v in centroid[:3]]
    # Arm mount translates with rail_y, whose fixed origin is y=-0.4 m.
    toward = [-c[0], float(rail_m) - 0.4 - c[1], 0.0]
    if math.hypot(toward[0], toward[1]) < 1e-6:
        toward = [0.0, -1.0, 0.0]
    return toward


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="detect and print poses, do not move")
    ap.add_argument("--force", type=float, default=FORCE_N, help="F* (N), default 4")
    ap.add_argument("--standoff-m", type=float, default=STANDOFF_M, help="air gap along the surface normal")
    ap.add_argument("--ptp-v", type=float, default=PTP_V, help="cartesian PTP speed scale (0, 1]")
    ap.add_argument("--hold-s", type=float, default=HOLD_S, help="seconds to press after contact")
    ap.add_argument("--cloud-timeout", type=float, default=4.0)
    ap.add_argument("--subscribe", type=str, default=None)
    return ap.parse_args(argv)


def _live_q8(arm: PeirasticArm) -> list[float]:
    ret, q = arm.get_joint_radian()
    if ret != OK or not q or len(q) < 8:
        raise RuntimeError("no live 8-DOF q from Window A")
    return [float(v) for v in q[:8]]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        arm = PeirasticArm()
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first:", flush=True)
        print("      python -m peirastic.apps.run_controller", flush=True)
        return 1

    recv_kw = {"timeout_s": float(args.cloud_timeout)}
    if args.subscribe:
        recv_kw["subscribe"] = str(args.subscribe)

    try:
        q8 = _live_q8(arm)
        print(f"[STATE] q8 rail={q8[0] * 1000.0:.1f} mm", flush=True)
        print("[CLOUD] waiting Orbbec …", flush=True)
        _meta, xyz_cam, rgb = recv_camera_cloud(**recv_kw)
        xyz = cloud_in_rail_base(xyz_cam, q8)
        live0 = _live_pose(arm, timeout_s=1.0)
        yaw_axis = None
        if live0 is not None:
            yaw_axis = Rsc.from_euler("xyz", live0[3:6], degrees=False).as_matrix()[:, 0]
        toward = _toward_mount(xyz.mean(axis=0), rail_m=q8[0])
        hit = detect_phantom_top(
            xyz,
            rgb,
            toward,
            standoff_m=float(args.standoff_m),
            yaw_axis=yaw_axis,
        )
        toward = _toward_mount(hit.centroid, rail_m=q8[0])
        hit = detect_phantom_top(
            xyz,
            rgb,
            toward,
            standoff_m=float(args.standoff_m),
            yaw_axis=yaw_axis,
        )
        lift_mm = (hit.centroid[2] - hit.table_z) * 1000.0
        print(
            f"[OK] phantom top  n={hit.n_points}  table_z={hit.table_z * 1000.0:.1f} mm  "
            f"top_z={hit.centroid[2] * 1000.0:.1f} mm  lift={lift_mm:.0f} mm  "
            f"n_hat=({hit.normal[0]:+.3f},{hit.normal[1]:+.3f},{hit.normal[2]:+.3f})",
            flush=True,
        )
        if lift_mm > 80.0:
            print(
                f"[WARN] top is {lift_mm:.0f} mm above table_z — check twin if the "
                "brown face height looks right (table estimate can sit low)",
                flush=True,
            )
        print(f"[STATE] corner   xyz_mm=[{_fmt_xyz(hit.corner)} ]", flush=True)
        print(f"[STATE] standoff {_fmt_pose(hit.standoff_pose)}", flush=True)
        print(f"[STATE] contact  {_fmt_pose(hit.contact_pose)}", flush=True)
        if args.dry_run:
            print("[OK] dry-run, no motion", flush=True)
            return 0

        print(
            f"[MODE] PTP standoff then HFPC hold  F*={float(args.force):.1f}N  Z force",
            flush=True,
        )
        ret = arm.cartesian(
            hit.standoff_pose.tolist(),
            v=float(args.ptp_v),
            r=0,
            connect=0,
            block=1,
            label="phathom_standoff",
        )
        if ret != OK:
            print(f"[ERR] cartesian -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        live = _live_pose(arm, timeout_s=1.0)
        if live is not None:
            print(f"[OK] PTP  {_fmt_pose(live)}", flush=True)
        else:
            print("[OK] PTP", flush=True)

        ret = arm.hfpc(
            [hit.standoff_pose.tolist()],
            reference="hold",
            law="tff",
            force=float(args.force),
            force_axes=FORCE_AXES,
            duration_s=None,
            label="phathom_press",
            soft_start=True,
            ramp_s=1.0,
            block=0,
        )
        if ret != OK:
            print(f"[ERR] hfpc -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        if arm.wait_contact(timeout_s=CLOSE_TIMEOUT_S, want_label="phathom_press") != OK:
            print("[ERR] no Fz contact", flush=True)
            arm.stop_force()
            return 1
        print(f"[OK] contact  holding {float(args.hold_s):.1f}s", flush=True)
        t0 = time.monotonic()
        next_print = t0
        while time.monotonic() - t0 < float(args.hold_s):
            now = time.monotonic()
            if now >= next_print:
                ret_s, live_s = arm.get_controller_state()
                fz = float(live_s.get("f_ext_z", float("nan"))) if ret_s == OK else float("nan")
                print(f"[HOLD] t={now - t0:5.1f}s  Fz={fz:+6.2f}N  F*={float(args.force):.1f}", flush=True)
                next_print = now + PRINT_S
            time.sleep(0.05)
        arm.stop_force()
        print("[OK] press done", flush=True)
        return 0
    except KeyboardInterrupt:
        try:
            arm.set_arm_stop()
        except Exception:
            pass
        print("[STOP] interrupted", flush=True)
        return 0
    except Exception as exc:
        print(f"[ERR] {exc}", flush=True)
        return 1
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
