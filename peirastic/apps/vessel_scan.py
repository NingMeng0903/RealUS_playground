"""Xbox B supervisor: 8DOF QPIK to 5 cm standoff, hybrid close, then 10 cm scan.

Approach commands only the TCP standoff. Window A TRACK_CARTESIAN runs
coupled 8DOF QPIK, which allocates rail and arm each tick. No pre-picked
rail or MOVEJ q_target. Hybrid then tracks the mesh curve after contact.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from peirastic.api import OK, PeirasticArm
from peirastic.api.arm import poll_force_contact
from peirastic.api.payloads import TrackCartesianPayload
from peirastic.core.ipc import CommandClient, Status
from peirastic.core.modes import Mode
from peirastic.realman8dof.force.config import desired_z_n, load_force_raw
from perception.vessel_skin_plan import (
    STANDOFF_M,
    VesselPlan,
    load_ready_plan,
    load_T_smplx_from_tcp,
    smplx_poses_to_tcp,
)

SCAN_SPEED_M_S = 0.02
APPROACH_ARRIVE_MM = 15.0
APPROACH_TIMEOUT_S = 90.0
APPROACH_LOG_S = 2.0
ALIGN_ARRIVE_RAD = 0.20
CLOSE_TIMEOUT_S = 25.0
SCAN_TIMEOUT_PAD_S = 12.0
CONTACT_POLL_S = 0.02

_lock = threading.Lock()
_running = False


def program_is_running() -> bool:
    with _lock:
        return bool(_running)


def vessel_b_refuse_reason(*, repo: Path | None = None) -> str | None:
    """Capture overlay I/O does not block B. Only a live scan program is busy."""
    if program_is_running():
        return "busy"
    _plan, reason = load_ready_plan(repo=repo, build_if_missing=False)
    return reason or None


def contact_enter_n(raw: dict | None = None) -> float:
    src = raw if raw is not None else load_force_raw()
    hm = src.get("hybrid_motion") or {}
    pc = hm.get("physical_contact") or {}
    if pc.get("enter_n") is not None:
        return float(pc["enter_n"])
    return float(hm.get("contact_threshold_n", 0.8))


def contact_confirm_s(raw: dict | None = None) -> float:
    src = raw if raw is not None else load_force_raw()
    pc = (src.get("hybrid_motion") or {}).get("physical_contact") or {}
    return float(pc.get("enter_confirm_s", 0.02))


def robot_world_yaml_path() -> Path:
    from perception.vessel_skin_plan import robot_world_yaml_path as _path

    return _path()


def controller_T_world_from_rail_base() -> np.ndarray:
    return load_T_smplx_from_tcp()


def poses_world_to_rail_base(
    poses: np.ndarray,
    T_world_from_rail: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    return smplx_poses_to_tcp(poses, T=T_world_from_rail, euler_order=euler_order)


def standoff_pose_from_contact(contact_pose: np.ndarray, *, approach_dz_m: float = STANDOFF_M) -> np.ndarray:
    """Retract along −tool Z. Planned +Z goes into the skin, so air standoff is −Z."""
    contact = np.asarray(contact_pose, dtype=float).reshape(6)
    R = Rsc.from_euler("xyz", contact[3:6], degrees=False).as_matrix()
    out = contact.copy()
    out[:3] = contact[:3] + R @ np.array([0.0, 0.0, -float(approach_dz_m)])
    return out


def approach_cartesian_payload(
    pose_start: np.ndarray,
    standoff: np.ndarray,
    *,
    keep_live_rpy: bool = True,
    label: str = "vessel_approach",
) -> dict[str, Any]:
    """Hold a TCP pose. Coupled 8DOF QPIK allocates rail and arm; no q_target.

    Keep live RPY: a 100°+ planned yaw snap folds the arm (see run_20260831_214801).
    Tool +Z is already into the skin; only scan-axis yaw differs.
    """
    start = np.asarray(pose_start, dtype=float).reshape(6)
    hold = np.asarray(standoff, dtype=float).reshape(6).copy()
    if keep_live_rpy:
        hold[3:6] = start[3:6]
    payload = TrackCartesianPayload(
        reference="polyline",
        poses=[hold.tolist()],
        soft_start=False,
        duration_s=None,
        label=str(label),
    ).to_json()
    payload["duration_s"] = None
    return payload


def poses_keep_rpy(poses: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Copy XYZ from the plan, pin orientation to ``rpy`` (live TCP)."""
    out = np.asarray(poses, dtype=float).reshape(-1, 6).copy()
    out[:, 3:6] = np.asarray(rpy, dtype=float).reshape(3)
    return out


class _LiveFk:
    """Keep the state bus open while C waits for a real Cartesian arrival."""

    def __init__(self) -> None:
        from rm75_control.control.admittance_common.state_relay import RelayStateBus
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

        self.kin = RobotKinematics()
        self.bus = RelayStateBus()
        if not self.bus.ensure_attached():
            raise RuntimeError("no robot state")

    def pose(self) -> np.ndarray:
        q = self.bus.q_meas_8dof()
        if q is None or not np.isfinite(q).all():
            raise RuntimeError("no robot state")
        return np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)

    def close(self) -> None:
        self.bus.stop()


def _xyz(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=float).reshape(-1)[:3]


def _fmt_xyz(xyz: np.ndarray | None) -> str:
    if xyz is None:
        return "(none)"
    p = _xyz(xyz)
    return f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"


def _snapshot(client: CommandClient) -> dict[str, Any]:
    try:
        last = client.snapshot()
    except (TypeError, AttributeError) as exc:
        raise RuntimeError("interrupted") from exc
    if last is None:
        raise RuntimeError("interrupted")
    return last


def _rot_err_rad(a: np.ndarray, b: np.ndarray) -> float:
    ra = Rsc.from_euler("xyz", np.asarray(a, dtype=float).reshape(3), degrees=False)
    rb = Rsc.from_euler("xyz", np.asarray(b, dtype=float).reshape(3), degrees=False)
    return float(np.linalg.norm((ra.inv() * rb).as_rotvec()))


def _wait_status(
    client: CommandClient,
    *,
    want: set[int],
    timeout_s: float,
    fail: set[int] | None = None,
    on_tick: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    fail = fail or {int(Status.ERROR), int(Status.ESTOP)}
    deadline = time.monotonic() + float(timeout_s)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _snapshot(client)
        st = int(last.get("status", -1))
        if st in fail:
            raise RuntimeError(str(last.get("msg") or Status(st).name))
        if on_tick is not None and on_tick(last):
            return last
        if st in want:
            return last
        time.sleep(0.05)
    raise TimeoutError("mode wait timed out")


def wait_cartesian_arrival(
    client: CommandClient,
    *,
    goal_xyz: np.ndarray,
    pose_fn: Callable[[], np.ndarray],
    arrive_mm: float,
    timeout_s: float,
    want_mode: int | None = int(Mode.TRACK_CARTESIAN),
    want_label: str | None = "vessel_approach",
    goal_rpy: np.ndarray | None = None,
    arrive_rad: float = ALIGN_ARRIVE_RAD,
    on_progress: Callable[[float, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Wait until live FK is near the standoff. Do not use path ``track_err_mm``.

    A hold reference can still publish a small ``track_err_mm`` once the arm
    is sitting on an intermediate pose. Arrival is live FK vs the standoff.
    """
    goal = _xyz(goal_xyz)
    last_log = 0.0

    def _tick(tel: dict[str, Any]) -> bool:
        nonlocal last_log
        if want_mode is not None and int(tel.get("mode", -1)) != int(want_mode):
            return False
        if want_label and want_label not in str(tel.get("msg") or ""):
            return False
        live_pose = np.asarray(pose_fn(), dtype=float).reshape(-1)
        live = live_pose[:3]
        err_mm = float(np.linalg.norm(live - goal) * 1000.0)
        tel["goal_err_mm"] = err_mm
        tel["live_xyz"] = live.tolist()
        rot_ok = True
        if goal_rpy is not None and live_pose.size >= 6:
            rot_err = _rot_err_rad(live_pose[3:6], goal_rpy)
            tel["goal_err_rad"] = rot_err
            rot_ok = rot_err <= float(arrive_rad)
        now = time.monotonic()
        if on_progress is not None and (now - last_log) >= APPROACH_LOG_S:
            last_log = now
            on_progress(err_mm, live)
        return err_mm <= float(arrive_mm) and rot_ok

    return _wait_status(client, want=set(), timeout_s=timeout_s, on_tick=_tick)


def wait_contact(
    client: CommandClient,
    *,
    enter_n: float,
    confirm_s: float,
    timeout_s: float,
    want_mode: int | None = int(Mode.TRACK_HYBRID),
    want_label: str | None = "vessel_close",
) -> bool:
    """Latch Fz only after hybrid close is live and the probe has been in air."""
    return poll_force_contact(
        lambda: _snapshot(client),
        enter_n=enter_n,
        confirm_s=confirm_s,
        timeout_s=timeout_s,
        want_mode=want_mode,
        want_label=want_label,
    )


def wait_scan_hold(client: CommandClient, *, timeout_s: float) -> dict[str, Any]:
    """Hold hybrid scan for ``timeout_s``. Do not chase one-tick DONE."""
    fail = {int(Status.ERROR), int(Status.ESTOP)}
    deadline = time.monotonic() + float(timeout_s)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _snapshot(client)
        st = int(last.get("status", -1))
        if st in fail:
            raise RuntimeError(str(last.get("msg") or Status(st).name))
        time.sleep(0.05)
    return last


def _run_program(
    client: CommandClient,
    plan: VesselPlan,
    *,
    on_log: Callable[[str], None] | None = None,
) -> None:
    def log(msg: str) -> None:
        if on_log is not None:
            on_log(msg)

    force_raw = load_force_raw()
    enter_n = contact_enter_n(force_raw)
    confirm_s = contact_confirm_s(force_raw)
    fz_des = desired_z_n(force_raw)

    contact = np.asarray(plan.tcp_contact, dtype=float).reshape(6)
    scan_tcp = np.asarray(plan.tcp_poses, dtype=float).reshape(-1, 6)
    standoff = standoff_pose_from_contact(contact, approach_dz_m=plan.standoff_m)
    arm = PeirasticArm(client=client, attach=False)
    fk = _LiveFk()
    try:
        pose_now = fk.pose()
        approach = np.asarray(standoff, dtype=float).reshape(6).copy()
        approach[3:6] = pose_now[3:6]
        remain0 = float(np.linalg.norm(standoff[:3] - pose_now[:3]))
        log(
            f"[VESSEL] {plan.label} cartesian tcp {_fmt_xyz(contact)} "
            f"window={plan.window_m * 100.0:.1f}cm"
        )
        log(
            f"[VESSEL] standoff {_fmt_xyz(standoff)} live tcp {_fmt_xyz(pose_now)} "
            f"remain={remain0 * 100.0:.1f}cm 8dof QPIK hold xyz, keep live rpy"
        )
        arm.track_pose(approach, label="vessel_approach", block=0, soft_start=False)
        arrived = wait_cartesian_arrival(
            client,
            goal_xyz=standoff,
            pose_fn=fk.pose,
            arrive_mm=APPROACH_ARRIVE_MM,
            timeout_s=APPROACH_TIMEOUT_S,
            on_progress=lambda err_mm, live: log(
                f"[VESSEL] approach remain={err_mm:.0f}mm "
                f"tcp={_fmt_xyz(live)} goal={_fmt_xyz(standoff)}"
            ),
        )
        remain_mm = float(arrived.get("goal_err_mm", float("nan")))
        live_now = fk.pose()
        live_xyz = arrived.get("live_xyz")
        log(
            f"[VESSEL] standoff arrived remain={remain_mm:.1f}mm "
            f"tcp={_fmt_xyz(live_xyz if live_xyz is not None else live_now)} "
            f"keep live rpy (no yaw snap)"
        )

        close_pose = poses_keep_rpy(contact, live_now[3:6])[0]
        scan_held = poses_keep_rpy(scan_tcp, live_now[3:6])
        log(f"[VESSEL] HFPC close Fz*={fz_des:.2f}N enter={enter_n:.2f}N")
        arm.hfpc(
            [close_pose],
            speed_m_s=SCAN_SPEED_M_S,
            law="tff",
            block=0,
            label="vessel_close",
        )
        if arm.wait_contact(
            enter_n=enter_n, confirm_s=confirm_s, timeout_s=CLOSE_TIMEOUT_S, want_label="vessel_close"
        ) != OK:
            raise TimeoutError("no Fz contact")

        length = float(plan.window_m)
        duration = max(1.0, length / max(SCAN_SPEED_M_S, 1.0e-4) + 1.5)
        log(f"[VESSEL] HFPC scan {length * 100.0:.1f}cm")
        arm.hfpc(
            scan_held,
            speed_m_s=SCAN_SPEED_M_S,
            law="tff",
            duration_s=duration,
            block=0,
            label="vessel_scan",
        )
        wait_scan_hold(client, timeout_s=duration + SCAN_TIMEOUT_PAD_S)
        arm.stop_force()
        log("[VESSEL] done")
    finally:
        fk.close()


def try_start_vessel_scan(
    client: CommandClient,
    *,
    repo: Path | None = None,
    on_log: Callable[[str], None] | None = None,
) -> str | None:
    """Start the B program. Returns a one-line refuse reason, or None if started."""
    global _running
    plan, reason = load_ready_plan(repo=repo, build_if_missing=True)
    if plan is None:
        return reason or "no capture"
    with _lock:
        if _running:
            return "busy"
        _running = True

    def _worker() -> None:
        global _running
        try:
            _run_program(client, plan, on_log=on_log)
        except Exception as exc:
            if on_log is not None:
                on_log(f"[VESSEL] abort ({exc})")
            try:
                PeirasticArm(client=client, attach=False).stop_force()
            except Exception:
                pass
        finally:
            with _lock:
                _running = False

    threading.Thread(target=_worker, name="realus-vessel-scan", daemon=True).start()
    return None
