#!/usr/bin/env python3
"""One-shot FK check: Pinocchio (same chain as Genesis) vs Realman UDP snapshot via SHM.

Reads one frame from the controller's state relay — no files written, no second UDP push.

  # Controller must be running with --state-relay rm75_state (can be holding @ D)
  python apps/joint_admittance_8dof/check_fk_once.py --subscribe rm75_state

  # Flange check (arm chain, tool-agnostic) needs active tool offset once over TCP:
  python apps/joint_admittance_8dof/check_fk_once.py --subscribe rm75_state --ip 192.168.1.18
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from rm75_control.control.admittance_common.state_relay import RelayStateBus
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.validation import (
    POS_TOL_MM,
    ROT_TOL_DEG,
    base_flange_from_tool,
    pose_diff,
)


def _wait_frame(bus: RelayStateBus, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snap = bus.read()
        if snap.ok and snap.q_deg is not None and snap.pose is not None:
            return snap
        time.sleep(0.05)
    return None


def _fetch_tool_offset(ip: str, port: int) -> tuple[str, np.ndarray, np.ndarray]:
    from rm75_control.core.session import RobotSession

    with RobotSession(ip=ip, port=port) as sess:
        ret, tf = sess.robot.rm_get_current_tool_frame()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_tool_frame failed: {ret}")
        name = str(tf.get("name", "?"))
        offset = np.asarray(tf["pose"][:6], dtype=float)
        ret_j, st = sess.robot.rm_get_current_arm_state()
        if ret_j != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: {ret_j}")
        q_deg = np.asarray(st["joint"][:7], dtype=float)
        rm_fk = np.asarray(
            sess.robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)[:6],
            dtype=float,
        )
    return name, offset, rm_fk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subscribe", default="rm75_state", help="SHM relay name from controller")
    ap.add_argument("--wait-s", type=float, default=8.0, help="Wait for first valid relay frame")
    ap.add_argument(
        "--ip",
        default=None,
        help="Optional robot IP: one short TCP read for tool frame + rm_algo FK (no UDP push)",
    )
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    bus = RelayStateBus(args.subscribe)
    try:
        print(f"waiting for relay {args.subscribe!r} ({args.wait_s:.0f}s max)...", flush=True)
        snap = _wait_frame(bus, args.wait_s)
        if snap is None:
            print(
                "FAIL: no frame (start controller with --state-relay first)",
                flush=True,
            )
            return 1

        q8 = bus.q_meas_8dof()
        if q8 is None:
            print("FAIL: no 8-DOF q from relay", flush=True)
            return 1

        kin = RobotKinematics()
        euler = kin.euler_order
        pose_pin_tcp = kin.fk_pose(q8)
        pose_pin_l7 = kin.frame_pose(q8, "link_7")
        pose_udp = np.asarray(snap.pose, dtype=float)

        print(f"\nURDF (Pinocchio / Genesis chain): {kin.urdf_path}", flush=True)
        print(f"relay seq={snap.seq}  rail_m={bus.last_rail_m:.4f}", flush=True)
        print(f"q_deg (arm):  {np.round(snap.q_deg, 3).tolist()}", flush=True)
        print(f"Pin tcp xyz:  {np.round(pose_pin_tcp[:3], 4).tolist()} m", flush=True)
        print(f"Pin link7:    {np.round(pose_pin_l7[:3], 4).tolist()} m", flush=True)
        print(f"UDP pose xyz: {np.round(pose_udp[:3], 4).tolist()} m  (active Realman tool)", flush=True)

        d_tcp_mm, d_tcp_deg = pose_diff(pose_pin_tcp, pose_udp, euler)
        print(
            f"\n[tcp] Pinocchio tcp vs UDP pose:  {d_tcp_mm:.2f} mm  {d_tcp_deg:.3f} deg",
            flush=True,
        )
        if d_tcp_mm > 50.0:
            print(
                "  (large gap is normal when active tool != URDF tcp, e.g. Arm_Tip vs +220mm tcp)",
                flush=True,
            )

        flange_ok = None
        rm_fk_mm = None
        if args.ip:
            try:
                tool_name, tool_offset, rm_fk = _fetch_tool_offset(args.ip, args.port)
                flange_meas = base_flange_from_tool(pose_udp, tool_offset, euler)
                d_fl_mm, d_fl_deg = pose_diff(pose_pin_l7, flange_meas, euler)
                rm_fk_mm, rm_fk_deg = pose_diff(pose_pin_tcp, rm_fk, euler)
                flange_ok = d_fl_mm < POS_TOL_MM and d_fl_deg < ROT_TOL_DEG
                print(f"\nactive tool: {tool_name!r}", flush=True)
                print(
                    f"[flange] Pin link_7 vs recovered flange:  {d_fl_mm:.3f} mm  {d_fl_deg:.4f} deg  "
                    f"-> {'PASS' if flange_ok else 'FAIL'}  (tol {POS_TOL_MM} mm / {ROT_TOL_DEG} deg)",
                    flush=True,
                )
                print(
                    f"[rm_fk]  Pin tcp vs rm_algo_forward_kinematics:  {rm_fk_mm:.3f} mm  {rm_fk_deg:.4f} deg",
                    flush=True,
                )
            except Exception as exc:
                print(f"\nWARN: could not read tool frame from {args.ip}: {exc}", flush=True)
        else:
            print(
                "\nTip: pass --ip ROBOT_IP for flange / rm_algo check (one TCP read, no UDP push).",
                flush=True,
            )

        print(
            "\nGenesis viewer uses the same q_deg + rail_m from this relay; "
            "if mesh aligns with the real arm, visual FK matches.",
            flush=True,
        )

        if flange_ok is not None:
            if not flange_ok:
                return 1
            print("\nRESULT: PASS (flange chain)", flush=True)
            return 0

        print("\nRESULT: frame OK (tcp vs UDP printed; use --ip for PASS/FAIL on arm chain)", flush=True)
        return 0
    finally:
        bus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
