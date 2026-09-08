#!/usr/bin/env python3
"""8DOF phantom scan: detect → centered path → PTP standoff → hybrid scan → home.

    python -m peirastic.apps.run_controller
    python -m peirastic.DEMO.phathom_scanning.s_scan

PHATHOM DETECT POSE is the taught 8DOF q in phathom_detect_pose.json (rail + 7 arm).
Start and finish MOVEJ to that setpoint. Detect from there.

Enter 1: detect the surface and plan a centered raster or Lissajous path.
Enter 2: Cartesian PTP (IK + MOVEJ) to the start, 4 cm along the outward normal.
Enter 3: TRACK_HYBRID, wait contact, follow the planned polyline.
Then position-lift 1 cm from the measured TCP along −tool Z and MOVEJ home.
All cloud points and targets are in the controller rail_base, in metres/radians.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

if __name__ == "__main__":
    # Keep the standalone scanner's numerical and GUI workers bounded.  The
    # guard preserves normal import behavior for hardware/controller tests.
    from rm75_control.control.admittance_common.observer_runtime import prepare_observer_process

    prepare_observer_process()

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from peirastic.DEMO.cartesian import _fmt_pose, _live_pose
from peirastic.core.ipc import Status
from peirastic.core.modes import Mode
from peirastic.DEMO.phathom_scanning.cloud import cloud_in_rail_base, recv_camera_cloud
from peirastic.DEMO.phathom_scanning.detect import detect_phantom_top
from peirastic.DEMO.phathom_scanning.preview import save_capture, save_failure, save_plan
from peirastic.DEMO.phathom_scanning.run import _fmt_xyz, _live_q8, _toward_mount
from peirastic.DEMO.phathom_scanning.s_plan import (
    LIFT_M,
    PROBE_WIDTH_M,
    STANDOFF_M,
    plan_s_scan,
)
from peirastic.DEMO.phathom_scanning.surface import MAX_FIT_RADIUS_M
from peirastic.DEMO.phathom_scanning.force_trace import ForceTraceRecorder
from peirastic.api import PeirasticArm
from peirastic.api.codes import CODE_NAMES, OK

SCAN_SPEED_M_S = 0.018
SCAN_LENGTH_M = 0.140
SCAN_WIDTH_M = 0.080
PTP_V = 0.30
CLOSE_SPEED_M_S = 0.012
MOVEJ_V = 0.35
APPROACH_ARRIVE_MM = 2.0
ARRIVE_DEG = 2.0
FORCE_AXES = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
CLOSE_TIMEOUT_S = 25.0
PRINT_S = 1.0


DETECT_POSE_NAME = "PHATHOM DETECT POSE"
DETECT_POSE_JSON = Path(__file__).resolve().parent / "phathom_detect_pose.json"


def load_detect_pose(path: Path = DETECT_POSE_JSON) -> list[float]:
    raw = json.loads(Path(path).read_text())
    q = [float(v) for v in raw["q"][:8]]
    if len(q) < 8 or not all(np.isfinite(q)):
        raise RuntimeError(f"{DETECT_POSE_NAME} needs 8 finite q in {path}")
    return q


def _fmt_q(q) -> str:
    vals = [float(v) for v in q[:8]]
    rail = f"rail={vals[0] * 1000.0:.1f}mm"
    arm = " ".join(f"{v:+.3f}" for v in vals[1:])
    return f"{rail}  q[1:7]=[{arm}]"


def _q_travel(q_a, q_b) -> tuple[float, float]:
    a = np.asarray(q_a, dtype=float).reshape(-1)[:8]
    b = np.asarray(q_b, dtype=float).reshape(-1)[:8]
    rail_mm = abs(float(a[0] - b[0])) * 1000.0
    arm_deg = float(np.max(np.abs(np.degrees(a[1:] - b[1:])))) if a.size >= 8 else 0.0
    return rail_mm, arm_deg


def _wait_enter(msg: str) -> None:
    try:
        input(f"[ENTER] {msg}")
    except EOFError as exc:
        raise RuntimeError("stdin closed; need a TTY for the enter steps") from exc


def _window_a_line(arm: PeirasticArm) -> str:
    snap = arm._snapshot() or {}
    st = int(snap.get("status", -1))
    try:
        name = Status(st).name
    except ValueError:
        name = f"status={st}"
    dof = snap.get("dof", "?")
    msg = str(snap.get("msg") or "").strip()
    extra = f"  {msg}" if msg else ""
    return f"window A  {name}  dof={dof}{extra}"


def _xyz_err_mm(live, target) -> float:
    a = np.asarray(live, dtype=float).reshape(-1)
    b = np.asarray(target, dtype=float).reshape(-1)
    return float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)


def _rot_err_deg(live, target) -> float:
    a = Rsc.from_euler("xyz", np.asarray(live)[3:6])
    b = Rsc.from_euler("xyz", np.asarray(target)[3:6])
    return float(np.degrees((a.inv() * b).magnitude()))


def _lift_target(live, distance_m: float) -> np.ndarray:
    pose = np.asarray(live, dtype=float).reshape(6).copy()
    # +tool Z points into the phantom; withdrawal is -tool Z.
    z_tool = Rsc.from_euler("xyz", pose[3:6]).as_matrix()[:, 2]
    pose[:3] -= float(distance_m) * z_tool
    return pose


def _latched(arm: PeirasticArm) -> bool:
    snap = arm._snapshot() or {}
    age = time.monotonic() - float(snap.get("t_mono", float("-inf")))
    if not np.isfinite(age) or not -0.05 <= age <= 0.5:
        raise RuntimeError(f"controller telemetry stale/unavailable (age={age:.3f}s)")
    st = int(snap.get("status", -1))
    return bool(snap.get("estop")) or st in (
        int(Status.ESTOP),
        int(Status.STOPPED),
        int(Status.ERROR),
    )


def _await_tcp(
    arm: PeirasticArm,
    target,
    *,
    timeout_s: float,
    tol_mm: float = APPROACH_ARRIVE_MM,
    want_mode: int | None = None,
    want_label: str | None = None,
    label: str = "approach",
) -> bool:
    t0 = time.monotonic()
    next_print = t0
    while time.monotonic() - t0 < float(timeout_s):
        if _latched(arm):
            print(f"[ERR] {_window_a_line(arm)}", flush=True)
            return False
        snap = arm._snapshot() or {}
        if want_mode is not None and int(snap.get("mode", -1)) != int(want_mode):
            time.sleep(0.05)
            continue
        if want_label and want_label not in str(snap.get("msg") or ""):
            time.sleep(0.05)
            continue
        live = _live_pose(arm, timeout_s=0.3)
        if live is None:
            time.sleep(0.05)
            continue
        err = _xyz_err_mm(live, target)
        rot = _rot_err_deg(live, target)
        now = time.monotonic()
        if now >= next_print:
            print(f"[STATE] {label}  {_fmt_pose(live)}  remain={err:.1f} mm/{rot:.1f} deg", flush=True)
            next_print = now + PRINT_S
        if err <= float(tol_mm) and rot <= ARRIVE_DEG:
            print(f"[OK] {label}  {_fmt_pose(live)}  remain={err:.1f} mm", flush=True)
            return True
        time.sleep(0.05)
    live = _live_pose(arm, timeout_s=0.3)
    extra = f"  {_fmt_pose(live)}" if live is not None else ""
    print(f"[ERR] {label} did not arrive{extra}", flush=True)
    return False


def _monitor_hybrid_scan(arm: PeirasticArm, plan, *, duration_s: float, seq: int,
                         force_trace: ForceTraceRecorder | None = None,
                         motion_writer=None, motion_flush=None) -> None:
    """Observe the one HFPC task; contact decisions stay in the controller."""
    t0 = next_print = time.monotonic()
    tracking_wall_t0 = None
    previous_motion = None
    label = "phathom_s_scan"
    while True:
        if force_trace is not None:
            force_trace.check_error()
        now = time.monotonic()
        snap = arm._snapshot()
        if snap is None:
            raise RuntimeError("controller telemetry lost during hybrid scan")
        stamp = float(snap.get("t_mono", float("nan")))
        if not np.isfinite(stamp) or not -0.1 <= now - stamp <= 0.5:
            raise RuntimeError("stale controller telemetry during hybrid scan")
        st = int(snap.get("status", -1))
        if _latched(arm) or st in (int(Status.ESTOP), int(Status.STOPPED), int(Status.ERROR)):
            raise RuntimeError(f"hybrid scan stopped: {snap.get('msg')}")
        if int(snap.get("done_seq", 0) or 0) >= seq:
            if tracking_wall_t0 is None:
                raise RuntimeError("hybrid task ended without confirmed contact/tracking")
            return
        msg = str(snap.get("msg") or "")
        parts = msg.split()
        stage = parts[0] if parts else ""
        active = (int(snap.get("mode", -1)) == int(Mode.TRACK_HYBRID)
                  and stage in (f"{label}:approach", f"{label}:tracking"))
        if not active:
            if tracking_wall_t0 is not None or now - t0 > 2.0:
                raise RuntimeError(f"contact-gated HFPC not active: {msg}; "
                                   "restart Window A to load the updated controller")
            time.sleep(0.05)
            continue
        fields = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
        elapsed = float(fields["t"])
        contact = int(fields["contact"])
        vz_cmd = float(fields["vz_cmd"])
        fz = float(snap.get("f_ext_z", float("nan")))
        if not all(np.isfinite(x) for x in (elapsed, vz_cmd, fz)):
            raise RuntimeError("invalid hybrid force/progress telemetry")
        tracking = stage.endswith(":tracking")
        gap = float("nan")
        live = None
        if tracking:
            if tracking_wall_t0 is None:
                tracking_wall_t0 = now
                print(f"[OK] controller confirmed contact — tracking original {getattr(plan, 'pattern', 'raster')} path; "
                      f"same HFPC force state  Fz={fz:+.2f}N", flush=True)
            # The controller finishes using governor-scaled path time. Allow
            # its existing minimum time scale (0.25) instead of charging seek
            # time against the scan's wall-clock deadline.
            if now - tracking_wall_t0 > 4.0 * duration_s + 12.0:
                raise RuntimeError("hybrid scan timed out")
        else:
            if now - t0 > CLOSE_TIMEOUT_S:
                raise RuntimeError("controller did not confirm contact (timeout)")
            live = _live_pose(arm, timeout_s=0.3)
            if live is None:
                raise RuntimeError("no measured TCP while seeking contact")
            gap = float(np.dot(np.asarray(live[:3]) - plan.poses[0, :3], _normal_at(plan, 0)))
            if gap < -0.010:
                raise RuntimeError("no contact within 10 mm below the detected surface")
        if motion_writer is not None:
            if live is None:
                live = _live_pose(arm, timeout_s=0.3)
            measured_vz = float("nan")
            if live is not None:
                pose = np.asarray(live, dtype=float)
                stamp_pose = time.monotonic()
                if previous_motion is not None:
                    old_time, old_pose = previous_motion
                    dt_pose = stamp_pose - old_time
                    if dt_pose > 0.0:
                        tool_z = Rsc.from_euler("xyz", pose[3:6]).as_matrix()[:, 2]
                        measured_vz = float(np.dot(pose[:3] - old_pose[:3], tool_z) / dt_pose * 1000)
                previous_motion = (stamp_pose, pose.copy())
            else:
                pose = np.full(6, np.nan)
            motion_writer.writerow([
                now, "tracking" if tracking else "approach", elapsed, fz,
                contact, vz_cmd, measured_vz, gap * 1000,
                snap.get("slack", float("nan")), *pose, *_live_q8(arm),
            ])
        if now >= next_print:
            progress = (f"[SCAN] t_ref={elapsed:5.1f}/{duration_s:.1f}s" if tracking
                        else f"[CLOSE] t={now - t0:5.1f}s  surface_gap={gap * 1000:+.1f}mm")
            print(f"{progress}  Fz={fz:+6.2f}N  contact={contact}  "
                  f"vz_cmd={vz_cmd:+.1f}mm/s (tool Z)", flush=True)
            next_print = now + PRINT_S
            if motion_flush is not None:
                motion_flush()
        time.sleep(0.05)


def _window_a_motion_ok(arm: PeirasticArm) -> bool:
    if _latched(arm):
        print(f"[ERR] {_window_a_line(arm)}", flush=True)
        print("[ERR] RESET Window A, then retry — motion is refused while latched", flush=True)
        return False
    return True


def _ok(arm: PeirasticArm, ret: int, what: str) -> bool:
    if ret == OK:
        return True
    print(f"[ERR] {what} -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
    try:
        arm.set_arm_stop()
    except Exception:
        pass
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="detect and plan only")
    ap.add_argument("--replay", type=Path, help="plan a saved capture.npz without attaching to the controller")
    ap.add_argument("--force", type=float, default=4.0, help="F* (N), default 4")
    ap.add_argument("--pattern", choices=("raster", "lissajous"), default="raster",
                    help="raster (default): parallel rows; lissajous: a closed two-lobe path")
    ap.add_argument("--scan-length-m", type=float, default=SCAN_LENGTH_M,
                    help="centered region length along phantom long edge, default 0.140 m")
    ap.add_argument("--scan-width-m", type=float, default=SCAN_WIDTH_M,
                    help="centered region width across phantom short edge, default 0.080 m")
    ap.add_argument("--standoff-m", type=float, default=STANDOFF_M)
    ap.add_argument("--lift-m", type=float, default=LIFT_M)
    ap.add_argument("--probe-width-m", type=float, default=PROBE_WIDTH_M)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--surface-radius-m", type=float, default=0.030,
                    help="local surface fitting radius; smooths depth and normals together")
    ap.add_argument("--speed-m-s", type=float, default=SCAN_SPEED_M_S)
    ap.add_argument("--ptp-v", type=float, default=PTP_V,
                    help="PTP joint speed scale (0, 1], same as DEMO.cartesian")
    ap.add_argument("--movej-v", type=float, default=MOVEJ_V)
    ap.add_argument("--cloud-timeout", type=float, default=4.0)
    ap.add_argument("--subscribe", type=str, default=None)
    ap.add_argument("--plan-dir", type=Path, default=_REPO / "peirastic" / "logs" / "phantom",
                    help="save the cloud, controller-frame plan and preview here")
    args = ap.parse_args(argv)
    for name in ("force", "standoff_m", "lift_m", "probe_width_m", "speed_m_s", "cloud_timeout", "surface_radius_m", "scan_length_m", "scan_width_m"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("ptp_v", "movej_v"):
        if not 0 < getattr(args, name) <= 1:
            ap.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if not 0 <= args.overlap < 1:
        ap.error("--overlap must be in [0, 1)")
    if args.surface_radius_m > MAX_FIT_RADIUS_M:
        ap.error(f"--surface-radius-m must be <= {MAX_FIT_RADIUS_M}")
    return args


def _detect(arm: PeirasticArm, args: argparse.Namespace):
    q8 = _live_q8(arm)
    recv_kw = {"timeout_s": float(args.cloud_timeout)}
    if args.subscribe:
        recv_kw["subscribe"] = str(args.subscribe)
    print("[CLOUD] waiting Orbbec …", flush=True)
    _meta, xyz_cam, rgb = recv_camera_cloud(**recv_kw)
    q_after = _live_q8(arm)
    rail_mm, arm_deg = _q_travel(q8, q_after)
    if rail_mm > 1.0 or arm_deg > 0.2:
        raise RuntimeError("robot moved during cloud capture; detect again from a stationary pose")
    wall_ns = _meta.get("wall_time_ns")
    if wall_ns is not None and not -0.1 <= time.time() - float(wall_ns) * 1e-9 <= 1.0:
        raise RuntimeError("Orbbec cloud is stale; wait for a fresh frame")
    q8 = q_after
    xyz = cloud_in_rail_base(xyz_cam, q8)
    live = _live_pose(arm, timeout_s=1.0)
    return _plan_capture(args, xyz_cam=xyz_cam, rgb=rgb, q8=q8, xyz=xyz, live=live)


def _plan_capture(args, *, xyz_cam, rgb, q8, xyz, live=None):
    directory = save_capture(args.plan_dir, xyz_cam=xyz_cam, rgb=rgb, q8=q8,
                             xyz=xyz, live_pose=live)
    print(f"[CLOUD] saved measured input: {directory / 'capture.npz'}", flush=True)
    yaw_axis = None
    if live is not None:
        yaw_axis = Rsc.from_euler("xyz", live[3:6], degrees=False).as_matrix()[:, 0]
    hit = None
    try:
        toward = _toward_mount(xyz.mean(axis=0), rail_m=q8[0])
        hit = detect_phantom_top(xyz, rgb, toward, yaw_axis=yaw_axis)
        toward = _toward_mount(hit.centroid, rail_m=q8[0])
        hit = detect_phantom_top(xyz, rgb, toward, yaw_axis=yaw_axis)
        print(f"[DETECT] top={hit.n_points} measured points  "
              f"span_mm={np.round(np.ptp(hit.points, axis=0) * 1000, 1).tolist()}", flush=True)
        plan = plan_s_scan(
            hit.points, hit.normal, toward,
            probe_width_m=float(args.probe_width_m), overlap=float(args.overlap),
            standoff_m=float(args.standoff_m), lift_m=float(args.lift_m),
            yaw_axis=yaw_axis, fit_radius_m=float(args.surface_radius_m),
            initial_rpy=None if live is None else np.asarray(live[3:6]),
            pattern=args.pattern,
            scan_length_m=float(args.scan_length_m),
            scan_width_m=float(args.scan_width_m),
        )
    except Exception as exc:
        failure = save_failure(directory, xyz=xyz, rgb=rgb, hit=hit, error=exc)
        print(f"[PLAN] rejected: {exc}\n[PLAN] diagnostic: {failure}", flush=True)
        raise
    preview = save_plan(args.plan_dir, xyz_cam=xyz_cam, rgb=rgb, q8=q8,
                        xyz=xyz, hit=hit, plan=plan, probe_width_m=args.probe_width_m,
                        capture_dir=directory)
    # Keep the force trace next to this exact cloud and planned trajectory.
    args.capture_dir = directory
    print(f"[PLAN] controller rail_base / m / rad; preview: {preview}", flush=True)
    print(f"[PLAN] scan=parallel to phantom long edge  "
          f"outline_LW_mm={np.round(plan.outline_dimensions_m * 1000, 1).tolist()}  "
          f"long_axis={np.round(plan.right, 4).tolist()}", flush=True)
    print(f"[PLAN] pattern={args.pattern}  centered region  rows={plan.n_rows}  "
          f"centerline_LW_mm=[{plan.u_span_m * 1000:.1f}, {plan.v_span_m * 1000:.1f}]  "
          f"spacing={plan.stride_m * 1000:.1f}mm", flush=True)
    angles = np.degrees(np.arccos(np.clip(plan.normals @ plan.normal, -1, 1)))
    spin = plan.orientation_diagnostics["max_tool_z_spin_deg"]
    print(f"[PLAN] orientation=minimum tilt from measured tool frame  "
          f"fit_radius={args.surface_radius_m * 1000:.0f}mm  "
          f"tilt_from_reference_max={angles.max():.1f}deg  "
          f"max_reference_tool_z_spin={spin:.6f}deg", flush=True)
    return hit, plan


def _normal_at(plan, index: int) -> np.ndarray:
    """Outward normal of the actual waypoint orientation, including curvature."""
    return -Rsc.from_euler("xyz", plan.poses[index, 3:6]).as_matrix()[:, 2]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.replay is not None:
        try:
            with np.load(args.replay, allow_pickle=False) as capture:
                xyz_cam, rgb, q8 = capture["xyz_cam"], capture["rgb"], capture["q8"]
                xyz = cloud_in_rail_base(xyz_cam, q8)
                live = capture["live_pose"] if "live_pose" in capture else None
            _, plan = _plan_capture(args, xyz_cam=xyz_cam, rgb=rgb, q8=q8, xyz=xyz, live=live)
            print(f"[OK] offline replay: {plan.n_rows} rows, {len(plan.poses)} waypoints; no motion", flush=True)
            return 0
        except Exception as exc:
            print(f"[ERR] replay: {exc}", flush=True)
            return 1
    try:
        arm = PeirasticArm()
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first:", flush=True)
        print("      python -m peirastic.apps.run_controller", flush=True)
        return 1

    force_trace = None
    trace_outcome = "aborted"
    try:
        if not args.dry_run and not _ok(arm, arm.set_dof(8, block=1), "set 8DOF"):
            return 1
        q_detect = load_detect_pose()
        q_live = _live_q8(arm)
        rail_mm, arm_deg = _q_travel(q_detect, q_live)
        print(f"[STATE] {DETECT_POSE_NAME}  {_fmt_q(q_detect)}", flush=True)
        print(f"[STATE] live  {_fmt_q(q_live)}  drail={rail_mm:.1f}mm  darm={arm_deg:.1f}deg", flush=True)
        print(f"[STATE] {_window_a_line(arm)}", flush=True)
        if args.dry_run:
            print(f"[OK] dry-run skips MOVEJ to {DETECT_POSE_NAME}", flush=True)
        elif not _window_a_motion_ok(arm):
            return 1
        elif not _ok(
            arm,
            arm.movej(q_detect, v=float(args.movej_v), block=1, label="phathom_detect_pose"),
            "movej detect pose",
        ):
            return 1
        else:
            q_now = _live_q8(arm)
            d_rail, d_arm = _q_travel(q_detect, q_now)
            if d_rail > 2.0 or d_arm > 0.5:
                raise RuntimeError(f"MOVEJ not at detect pose: rail={d_rail:.2f} mm, arm={d_arm:.2f} deg")
            print(
                f"[OK] MOVEJ {DETECT_POSE_NAME}  {_fmt_q(q_now)}  "
                f"drail={d_rail:.2f}mm  darm={d_arm:.2f}deg",
                flush=True,
            )

        if args.dry_run:
            hit, plan = _detect(arm, args)
        else:
            _wait_enter(f"detect at {DETECT_POSE_NAME}")
            hit, plan = _detect(arm, args)

        print(
            f"[OK] phantom top  n={hit.n_points}  "
            f"n_hat=({hit.normal[0]:+.3f},{hit.normal[1]:+.3f},{hit.normal[2]:+.3f})",
            flush=True,
        )
        print(f"[STATE] corner   xyz_mm=[{_fmt_xyz(hit.corner)} ]", flush=True)
        print(
            f"[OK] {args.pattern} scan  rows={plan.n_rows}  stride={plan.stride_m * 1000.0:.0f} mm  "
            f"span={plan.u_span_m * 1000.0:.0f}x{plan.v_span_m * 1000.0:.0f} mm  "
            f"waypoints={plan.poses.shape[0]}  length={plan.length_m * 1000.0:.0f} mm  "
            f"probe={float(args.probe_width_m) * 1000.0:.0f} mm",
            flush=True,
        )
        print(f"[STATE] start    {_fmt_pose(plan.poses[0])}", flush=True)
        print(f"[STATE] standoff {_fmt_pose(plan.standoff)}", flush=True)
        print(f"[STATE] lift     {_fmt_pose(plan.lift)}", flush=True)
        live0 = _live_pose(arm, timeout_s=1.0)
        if live0 is not None:
            print(
                f"[STATE] live     {_fmt_pose(live0)}  "
                f"remain={_xyz_err_mm(live0, plan.standoff):.1f} mm",
                flush=True,
            )
        if args.dry_run:
            print("[OK] dry-run, no motion", flush=True)
            return 0

        _wait_enter(f"PTP to scan start + {args.standoff_m * 1000:.0f} mm outward normal")
        if not _window_a_motion_ok(arm):
            return 1
        live = _live_pose(arm, timeout_s=1.0)
        if live is None:
            print("[ERR] no live TCP from Window A", flush=True)
            return 1
        remain = _xyz_err_mm(live, plan.standoff) / 1000.0
        print(
            f"[MODE] CARTESIAN PTP  remain={remain * 1000.0:.1f} mm  "
            f"delta_z={(plan.standoff[2] - live[2]) * 1000:+.1f} mm  v={args.ptp_v:.2f}",
            flush=True,
        )
        if not _ok(
            arm,
            arm.cartesian(
                plan.standoff.tolist(),
                v=float(args.ptp_v),
                label="phathom_s_standoff",
                block=1,
            ),
            "PTP standoff",
        ):
            return 1
        if not _await_tcp(
            arm,
            plan.standoff,
            timeout_s=6.0,
            label="standoff",
        ):
            arm.set_arm_stop()
            return 1

        fz = float(args.force)
        _wait_enter(f"hybrid contact + {args.pattern} scan  F*={fz:.1f}N")
        if not _window_a_motion_ok(arm):
            return 1
        # Recheck after the user pause: do not begin a contact seek from a moved TCP.
        if not _await_tcp(arm, plan.standoff, timeout_s=1.0, label="pre-contact standoff"):
            arm.set_arm_stop()
            return 1
        arm.set_force_control(force_axes=FORCE_AXES, control_frame="tool",
                              max_vz_tool_m_s=CLOSE_SPEED_M_S,
                              contact_enter_n=0.8, enter_confirm_s=0.05)
        arm.set_force_raw_override({"v_seek_free_m_s": CLOSE_SPEED_M_S})
        print(f"[MODE] tool Z force: target=+{fz:.1f}N  auto-seek=+tool Z  "
              f"Z speed cap={CLOSE_SPEED_M_S * 1000:.0f}mm/s (press/retract)", flush=True)
        duration = max(2.0, plan.length_m / max(float(args.speed_m_s), 1.0e-4) + 2.0)
        force_trace = ForceTraceRecorder(
            args.capture_dir, read_status=arm._snapshot, desired_force_n=fz,
        )
        force_trace.start()
        print(f"[LOG] force CSV: {args.capture_dir / 'force_trace.csv'}\n"
              f"[LOG] force plot after finish: {args.capture_dir / 'force_trace.png'}", flush=True)
        if not _ok(
            arm,
            arm.hfpc(
                plan.poses.tolist(),
                reference="polyline",
                law="tff",
                force=fz,
                force_axes=FORCE_AXES,
                speed_m_s=float(args.speed_m_s),
                duration_s=duration,
                label="phathom_s_scan",
                wait_for_contact=True,
                soft_start=True,
                ramp_s=0.4,
                block=0,
            ),
            "hfpc scan",
        ):
            return 1
        print(
            f"[MODE] TRACK_HYBRID auto-approach → polyline  {duration:.1f}s after contact  "
            f"v={float(args.speed_m_s) * 1000.0:.0f} mm/s",
            flush=True,
        )
        with (args.capture_dir / "motion_trace.csv").open("w", newline="") as motion_file:
            motion_writer = csv.writer(motion_file)
            motion_writer.writerow([
                "t_mono_s", "stage", "t_ref_s", "fz_n", "contact", "vz_cmd_mm_s",
                "vz_pose_diff_mm_s", "surface_gap_mm", "slack",
                "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad",
                "rail_m", *[f"q{i}_rad" for i in range(1, 8)],
            ])
            _monitor_hybrid_scan(arm, plan, duration_s=duration, seq=int(arm.last_seq),
                                 force_trace=force_trace, motion_writer=motion_writer,
                                 motion_flush=motion_file.flush)
        live_end = _live_pose(arm, timeout_s=1.0)
        if live_end is None:
            raise RuntimeError("no measured TCP after scan")
        delta = np.asarray(live_end[:3]) - plan.poses[-1, :3]
        end_normal = _normal_at(plan, -1)
        tangent_err = np.linalg.norm(delta - np.dot(delta, end_normal) * end_normal)
        if tangent_err > 0.005 or _rot_err_deg(live_end, plan.poses[-1]) > ARRIVE_DEG:
            raise RuntimeError(f"scan ended before endpoint: tangent error={tangent_err * 1000:.1f} mm")
        lift = _lift_target(live_end, args.lift_m)
        print(f"[OK] scan done — position lift {args.lift_m * 1000:.0f} mm along −tool Z", flush=True)
        arm.clear_force_control()
        # The finite HFPC has completed and the daemon has left force mode.
        # Stop sampling now; render after the lift and return motion finish.
        force_trace.stop()
        trace_outcome = "completed"
        if not _ok(
            arm,
            arm.cartesian_track(
                reference="polyline",
                poses=[list(live_end), lift.tolist()],
                speed_m_s=0.010,
                soft_start=True,
                ramp_s=0.3,
                label="phathom_s_lift",
                block=0,
            ),
            "track lift",
        ):
            return 1
        if not _await_tcp(
            arm,
            lift,
            timeout_s=max(6.0, args.lift_m / 0.010 + 4.0),
            tol_mm=1.0,
            want_mode=int(Mode.TRACK_CARTESIAN),
            want_label="phathom_s_lift",
            label="lift",
        ):
            arm.set_arm_stop()
            return 1
        if not _ok(
            arm,
            arm.movej(q_detect, v=float(args.movej_v), block=1, label="phathom_detect_pose"),
            "movej detect pose",
        ):
            return 1
        print(f"[OK] MOVEJ {DETECT_POSE_NAME}  {_fmt_q(q_detect)}", flush=True)
        return 0
    except KeyboardInterrupt:
        trace_outcome = "interrupted"
        if not args.dry_run:
            try:
                arm.set_arm_stop()
            except Exception:
                pass
        print("[STOP] interrupted", flush=True)
        return 0
    except Exception as exc:
        trace_outcome = f"aborted: {exc}"
        print(f"[ERR] {exc}", flush=True)
        if not args.dry_run:
            try:
                arm.set_arm_stop()
            except Exception:
                pass
        return 1
    finally:
        if force_trace is not None:
            try:
                artifacts = force_trace.finish(trace_outcome)
                print(f"[LOG] force CSV: {artifacts['csv']}\n"
                      f"[LOG] force plot: {artifacts['png']}", flush=True)
            except Exception as exc:
                print(f"[ERR] force log finalization failed: {exc}", flush=True)
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
