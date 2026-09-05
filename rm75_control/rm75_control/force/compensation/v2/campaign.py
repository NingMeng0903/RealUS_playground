"""Hardware campaign for Payload ID V2 (Window B).

Window A owns the 200 Hz servo. This module selects the session's 7-DOF
calibration structure before the preparation MOVEJ, keeps the live rail fixed,
records static holds, then
streams Fourier twists planned in link_7 / armtip and projected to the live TCP.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot
from rm75_control.control.admittance_common.state_bus import expand_q_meas_8dof
from rm75_control.control.joint_admittance_8dof.model import DEFAULT_URDF, RobotKinematics
from rm75_control.force.compensation.paths import CONFIG_FORCE, PHI_JSON, PHI_JSON_V2
from rm75_control.force.compensation.tool_pose import apply_kin_tcp_offset, read_tool_offset_cache
from rm75_control.force.compensation.identification import com_report, print_summary
from rm75_control.force.compensation.regressor import FrameConfig
from rm75_control.force.compensation.v2.fit_staged import (
    StaticWindow,
    delay_rejected,
    fft_lines,
    fit_delay_on_lines,
    fit_inertia_moments,
    fit_static_windows,
    inertia_moment_residual,
    pooled_shrinkage_cov,
    robust_mean,
    static_residual_report,
)
from rm75_control.force.compensation.v2.fourier import FourierSpec, axis_twist_L, measure_mask
from rm75_control.force.compensation.v2.frames import (
    FrameContract,
    tcp_pose_from_link7_pose,
    twist_link7_to_tcp,
    wrench_sensor_to_link7,
)
from rm75_control.force.compensation.v2.joint_observer import ArmJointObserver
from rm75_control.force.compensation.v2.rail_lock import RailLockLimits, evaluate_rail_lock
from rm75_control.force.compensation.v2.recorder import PayloadIdRecorder
from rm75_control.force.compensation.v2.regressor_v2 import payload_wrench_mhb
from rm75_control.force.compensation.v2.safety import (
    SafetyLimits,
    command_stale,
    deadman_twist,
    joint_margin_detail,
    raw_contact_abort,
    workspace_abort,
)
from rm75_control.force.compensation.v2.schema import (
    empty_document,
    phi16,
    phi_dict16,
    promote_mhb_to_live,
    urdf_sha256,
    write_phi_v2,
)
from peirastic.core.ipc import Status

HZ = 200.0
DT = 1.0 / HZ
CONTROLLER_STATUS_MAX_AGE_S = 0.5
MAX_DQ_DEG = 55.0
TCP_BOX_HALF_M = 0.16
DEFAULT_TILT_DEGS = (15, 30, 45, 60, 75, 90)
YAW_DEGS = (20, 30)


class CampaignAbort(RuntimeError):
    """Safety or lock gate failed; caller must zero the twist bus."""


@dataclass
class StaticTarget:
    name: str
    pose: np.ndarray
    q: np.ndarray
    is_train: bool
    is_yaw: bool = False
    tilt_deg: float = 0.0


@dataclass
class CampaignOpts:
    skip_movej: bool = False
    p1_only: bool = False
    skip_inertia: bool = False
    movej_v: float = 0.6
    settle_s: float = 0.0


def inertia_ident_enabled(cfg: dict, opts: CampaignOpts | None = None) -> bool:
    """P-I motion + I fit. Off unless yaml ``inertia.enabled`` or ``--inertia``."""

    if opts is not None and opts.skip_inertia:
        return False
    return bool((cfg.get("inertia") or {}).get("enabled", False))


def load_id_kinematics() -> RobotKinematics:
    kin = RobotKinematics()
    cache = read_tool_offset_cache()
    if cache is not None:
        apply_kin_tcp_offset(kin, cache[1])
    return kin


def _pose_from_R(p: np.ndarray, R: np.ndarray, *, euler_order: str = "xyz") -> np.ndarray:
    out = np.zeros(6, dtype=float)
    out[:3] = np.asarray(p, dtype=float).reshape(3)
    out[3:6] = Rsc.from_matrix(R).as_euler(euler_order, degrees=False)
    return out


def _ik_locked_rail_local(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose: np.ndarray,
    *,
    plan_margin_rad: float | None = None,
):
    """Numerical IK seeded at ``q_seed``; stays on the local branch (no ψ swing)."""
    from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik

    q, ok, _rep = solve_pose_ik(
        kin,
        q_seed,
        pose,
        rail_m=float(q_seed[0]),
    )
    if not ok or q is None:
        return None
    q = np.asarray(q, dtype=float).reshape(-1)
    if float(np.max(np.abs(np.degrees(q[1:] - q_seed[1:])))) > MAX_DQ_DEG:
        return None
    if plan_margin_rad is not None:
        lim = SafetyLimits(joint_margin_rad=float(plan_margin_rad))
        ok, _, _ = joint_margin_detail(q[1:], kin.q_lower[1:], kin.q_upper[1:], lim)
        if not ok:
            return None
    return q


def _world_tilt_pose(p0: np.ndarray, R0: np.ndarray, axis: int, deg: float, *, euler_order: str) -> np.ndarray:
    w = np.zeros(3)
    w[int(axis)] = math.radians(float(deg))
    R = Rsc.from_rotvec(w).as_matrix() @ R0
    return _pose_from_R(p0[:3], R, euler_order=euler_order)


def _tool_tilt_pose(p0: np.ndarray, R0: np.ndarray, axis: int, deg: float, *, euler_order: str) -> np.ndarray:
    e = np.zeros(3)
    e[int(axis)] = 1.0
    R = Rsc.from_rotvec(R0 @ e * math.radians(float(deg))).as_matrix() @ R0
    return _pose_from_R(p0[:3], R, euler_order=euler_order)


def _walk_tilt_ray(
    kin: RobotKinematics,
    q_mid: np.ndarray,
    p0: np.ndarray,
    R0: np.ndarray,
    *,
    prefix: str,
    pose_fn,
    degs: tuple[int, ...] | list[int],
    sign: int,
    yaw: bool,
    plan_margin_rad: float = 0.08,
    plan_frame: str = "link_7",
    hold_point: str = "tcp",
    p_tcp_hold: np.ndarray | None = None,
) -> list[StaticTarget]:
    """IK each larger tilt from the previous solution so ±90 can stay on-family.

    ``pose_fn`` returns a pose in ``plan_frame``. ``link_7`` targets are
    projected to the live TCP before IK / send.
    """

    out: list[StaticTarget] = []
    q_seed = np.asarray(q_mid, dtype=float).reshape(8)
    plan_in_L = str(plan_frame).strip().lower() in {"link_7", "link7", "armtip", "flange"}
    for deg in degs:
        signed = float(sign) * float(deg)
        pose_plan = pose_fn(p0, R0, signed)
        pose = (
            tcp_pose_from_link7_pose(
                pose_plan,
                R_LT=kin.R_LT,
                r_LT_L=kin.r_LT_L,
                euler_order=kin.euler_order,
                hold_point=hold_point,
                p_tcp_hold=p_tcp_hold,
            )
            if plan_in_L
            else pose_plan
        )
        q = _ik_locked_rail_local(kin, q_seed, pose, plan_margin_rad=plan_margin_rad)
        if q is None:
            break
        out.append(
            StaticTarget(
                name=f"{prefix}{signed:+.0f}",
                pose=pose,
                q=q,
                is_train=True,
                is_yaw=yaw,
                tilt_deg=signed,
            )
        )
        q_seed = q
    return out


def static_targets_from_mid(
    kin: RobotKinematics,
    q_mid: np.ndarray,
    *,
    n_train: int = 24,
    n_holdout: int = 8,
    tilt_degs: tuple[int, ...] | list[int] | None = None,
    holdout_min_abs_deg: float = 60.0,
    plan_margin_rad: float = 0.08,
    include_tool_tilts: bool = False,
    include_yaw: bool = False,
    include_tip_up: bool = True,
    tip_up_tilt_degs: tuple[int, ...] | list[int] | None = None,
    tip_up_approach_steps: int = 6,
    plan_frame: str = "link_7",
    hold_point: str = "tcp",
) -> list[StaticTarget]:
    """Walk world-axis tilts of ``plan_frame`` (default link_7 / armtip).

    Attitude is planned in ``plan_frame``. ``hold_point=tcp`` pins the taught
    TCP (this cell's 16-pose family). ``hold_point=link_7`` pins the flange.
    Send uses the live ``T_link7_tcp``. Tool rays and WZ yaw are off by default.
    At the taught mid, Tx/Ty nearly duplicate WX/WY in ``g_L``. WZ is yaw about
    vertical and does not change gravity, so it is not used for m,h.
    ``n_train`` / ``n_holdout`` do not drop reachable poses. Extremes become holdout.
    """

    del n_train, n_holdout
    q_mid = np.asarray(q_mid, dtype=float).reshape(8)
    plan_in_L = str(plan_frame).strip().lower() in {"link_7", "link7", "armtip", "flange"}
    if plan_in_L:
        M0 = kin.frame_placement(q_mid, "link_7")
        p0 = np.array(M0.translation, dtype=float, copy=True)
        R0 = np.array(M0.rotation, dtype=float, copy=True)
        pose_mid = kin.fk_pose(q_mid)
    else:
        pose_mid = kin.fk_pose(q_mid)
        p0 = pose_mid[:3].copy()
        R0 = Rsc.from_euler(kin.euler_order, pose_mid[3:6]).as_matrix()
    degs = tuple(int(x) for x in (tilt_degs if tilt_degs is not None else DEFAULT_TILT_DEGS))
    order = kin.euler_order
    rays: list[StaticTarget] = [
        StaticTarget(name="mid", pose=pose_mid.copy(), q=q_mid.copy(), is_train=True, is_yaw=False, tilt_deg=0.0)
    ]

    def world_fn(axis: int):
        def _fn(p, R, signed):
            return _world_tilt_pose(p, R, axis, signed, euler_order=order)

        return _fn

    def tool_fn(axis: int):
        def _fn(p, R, signed):
            return _tool_tilt_pose(p, R, axis, signed, euler_order=order)

        return _fn

    for axis, prefix in enumerate(("WX", "WY")):
        for sign in (1, -1):
            rays.extend(
                _walk_tilt_ray(
                    kin, q_mid, p0, R0, prefix=prefix, pose_fn=world_fn(axis), degs=degs, sign=sign, yaw=False,
                    plan_margin_rad=plan_margin_rad,
                    plan_frame=plan_frame,
                    hold_point=hold_point,
                    p_tcp_hold=pose_mid[:3],
                )
            )
    if include_tool_tilts:
        for axis, prefix in enumerate(("Tx", "Ty")):
            for sign in (1, -1):
                rays.extend(
                    _walk_tilt_ray(
                        kin,
                        q_mid,
                        p0,
                        R0,
                        prefix=prefix,
                        pose_fn=tool_fn(axis),
                        degs=degs,
                        sign=sign,
                        yaw=False,
                        plan_margin_rad=plan_margin_rad,
                        plan_frame=plan_frame,
                        hold_point=hold_point,
                        p_tcp_hold=pose_mid[:3],
                    )
                )
    if include_yaw:
        for sign in (1, -1):
            rays.extend(
                _walk_tilt_ray(
                    kin,
                    q_mid,
                    p0,
                    R0,
                    prefix="WZ",
                    pose_fn=world_fn(2),
                    degs=YAW_DEGS,
                    sign=sign,
                    yaw=True,
                    plan_margin_rad=plan_margin_rad,
                    plan_frame=plan_frame,
                    hold_point=hold_point,
                    p_tcp_hold=pose_mid[:3],
                )
            )

    seen: set[str] = set()
    unique: list[StaticTarget] = []
    for t in rays:
        if t.name in seen:
            continue
        seen.add(t.name)
        unique.append(t)

    if include_tip_up:
        unique.extend(
            _tip_up_family(
                kin,
                q_mid,
                plan_margin_rad=plan_margin_rad,
                plan_frame=plan_frame,
                tilt_degs=tip_up_tilt_degs,
                n_approach=int(tip_up_approach_steps),
            )
        )
        seen = set()
        dedup: list[StaticTarget] = []
        for t in unique:
            if t.name in seen:
                continue
            seen.add(t.name)
            dedup.append(t)
        unique = dedup

    gravity = [t for t in unique if not t.is_yaw]
    for t in gravity:
        if t.name.startswith(("UP", "UWX", "UWY")):
            t.is_train = True
        else:
            t.is_train = abs(t.tilt_deg) + 1e-9 < float(holdout_min_abs_deg)
    if not any(not t.is_train for t in gravity):
        farthest = max(
            (t for t in gravity if not t.name.startswith(("UP", "UWX", "UWY"))),
            key=lambda t: abs(t.tilt_deg),
            default=None,
        )
        if farthest is not None and abs(farthest.tilt_deg) > 0.0:
            farthest.is_train = False
    return unique


def _rotvec_align_z(R0: np.ndarray, z_world: np.ndarray) -> tuple[np.ndarray, float]:
    z0 = np.asarray(R0, dtype=float).reshape(3, 3)[:, 2]
    zt = np.asarray(z_world, dtype=float).reshape(3)
    zt = zt / max(float(np.linalg.norm(zt)), 1e-12)
    axis = np.cross(z0, zt)
    n = float(np.linalg.norm(axis))
    ang = float(np.arctan2(n, float(np.dot(z0, zt))))
    if n < 1e-9:
        return np.zeros(3), ang
    return axis / n * ang, ang


def _tip_up_family(
    kin: RobotKinematics,
    q_mid: np.ndarray,
    *,
    plan_margin_rad: float,
    plan_frame: str,
    tilt_degs: tuple[int, ...] | list[int] | None,
    n_approach: int,
) -> list[StaticTarget]:
    """Move the flange so link_7 +Z points world-up, then small world tilts.

    Approach pins the flange (TCP rises). Tilts pin the arrived TCP.
    """

    q_mid = np.asarray(q_mid, dtype=float).reshape(8)
    M0 = kin.frame_placement(q_mid, "link_7")
    p_L0 = np.array(M0.translation, dtype=float, copy=True)
    R0 = np.array(M0.rotation, dtype=float, copy=True)
    rotvec, ang = _rotvec_align_z(R0, np.array([0.0, 0.0, 1.0]))
    if abs(ang) < np.radians(8.0):
        return []
    steps = max(2, int(n_approach))
    key = Rsc.from_rotvec(np.vstack([np.zeros(3), rotvec]))
    slerp = Slerp([0.0, 1.0], key)
    order = kin.euler_order
    plan_in_L = str(plan_frame).strip().lower() in {"link_7", "link7", "armtip", "flange"}
    out: list[StaticTarget] = []
    q_seed = q_mid.copy()
    q_up = q_mid.copy()
    pose_up_T = kin.fk_pose(q_mid)
    for i in range(1, steps + 1):
        u = float(i) / float(steps)
        R = np.array(slerp(u).as_matrix() @ R0, dtype=float, copy=True)
        pose_L = _pose_from_R(p_L0, R, euler_order=order)
        pose = (
            tcp_pose_from_link7_pose(
                pose_L,
                R_LT=kin.R_LT,
                r_LT_L=kin.r_LT_L,
                euler_order=order,
                hold_point="link_7",
            )
            if plan_in_L
            else pose_L
        )
        q = _ik_locked_rail_local(kin, q_seed, pose, plan_margin_rad=plan_margin_rad)
        if q is None:
            break
        deg = float(np.degrees(ang * u))
        name = "UP" if i == steps else f"UP+{deg:.0f}"
        out.append(
            StaticTarget(name=name, pose=pose, q=q, is_train=True, is_yaw=False, tilt_deg=deg)
        )
        q_seed = q
        q_up = q
        pose_up_T = pose
    if not out:
        return []
    Mu = kin.frame_placement(q_up, "link_7")
    p_u = np.array(Mu.translation, dtype=float, copy=True)
    R_u = np.array(Mu.rotation, dtype=float, copy=True)
    degs = tuple(int(x) for x in (tilt_degs if tilt_degs is not None else (15, 30, 45)))

    def world_fn(axis: int):
        def _fn(p, R, signed):
            return _world_tilt_pose(p, R, axis, signed, euler_order=order)

        return _fn

    for axis, prefix in enumerate(("UWX", "UWY")):
        for sign in (1, -1):
            out.extend(
                _walk_tilt_ray(
                    kin,
                    q_up,
                    p_u,
                    R_u,
                    prefix=prefix,
                    pose_fn=world_fn(axis),
                    degs=degs,
                    sign=sign,
                    yaw=False,
                    plan_margin_rad=plan_margin_rad,
                    plan_frame=plan_frame,
                    hold_point="tcp",
                    p_tcp_hold=pose_up_T[:3],
                )
            )
    return out


def _tilt_deg_from_name(name: str) -> float:
    text = str(name).removeprefix("static_")
    for i, ch in enumerate(text):
        if ch in "+-" or ch.isdigit():
            try:
                return float(text[i:])
            except ValueError:
                return 0.0
    return 0.0


def estimate_campaign_s(cfg: dict, opts: CampaignOpts) -> float:
    st = cfg.get("static") or {}
    n_tilts = len(tuple(st.get("tilt_degs") or DEFAULT_TILT_DEGS))
    n_world = 4 * n_tilts
    n_tool = 4 * n_tilts if bool(st.get("include_tool_tilts", False)) else 0
    n_yaw = 4 if bool(st.get("include_yaw", False)) else 0
    n_up = 0
    if bool(st.get("include_tip_up", True)):
        n_up = int(st.get("tip_up_approach_steps", 6)) + 4 * len(tuple(st.get("tip_up_tilt_degs") or (15, 30, 45)))
    n_static = 1 + n_world + n_tool + n_yaw + n_up
    static_s = n_static * (float(st.get("min_wait_s", 0.20)) + float(st.get("record_s", 0.6)) + 3.0)
    spec = _fourier_spec(cfg)
    one = (spec.n_warmup + spec.n_measure + spec.n_cooldown) / max(spec.f0_hz, 1e-6)
    post = float((cfg.get("dynamic") or {}).get("post_hold_s", 0.25))
    n_dyn = 0 if opts.p1_only else 8
    iner = 0.0
    if not opts.p1_only and inertia_ident_enabled(cfg, opts):
        ispec = _inertia_spec(cfg)
        iner_one = (ispec.n_warmup + ispec.n_measure + ispec.n_cooldown) / max(ispec.f0_hz, 1e-6)
        iner = 3.0 * (iner_one + post)
    movej = 0.0 if opts.skip_movej else 14.0 + float(opts.settle_s)
    mid_return = 0.0 if opts.p1_only else 3.0
    if not opts.p1_only and inertia_ident_enabled(cfg, opts):
        mid_return += 3.0
    return movej + 0.4 + static_s + mid_return + n_dyn * (one + post) + iner


def _fourier_spec(cfg: dict) -> FourierSpec:
    dyn = cfg.get("dynamic") or {}
    harms = tuple(int(x) for x in (dyn.get("harmonics") or [1, 2, 3, 4]))
    return FourierSpec(
        f0_hz=float(dyn.get("f0_hz", 0.6)),
        harmonics=harms,
        n_warmup=int(dyn.get("n_warmup", 1)),
        n_measure=int(dyn.get("n_measure", 5)),
        n_cooldown=int(dyn.get("n_cooldown", 1)),
        dt=DT,
        x_max_m=float(dyn.get("x_max_m", 0.055)),
        v_max_m_s=float(dyn.get("v_max_m_s", 0.18)),
        a_max_m_s2=float(dyn.get("a_max_m_s2", 3.5)),
        j_max_m_s3=float(dyn.get("j_max_m_s3", 60.0)),
        w_max_rad_s=float(dyn.get("w_max_rad_s", 1.30)),
        alpha_max_rad_s2=float(dyn.get("alpha_max_rad_s2", 16.0)),
        j_ang_max=float(dyn.get("j_ang_max", 100.0)),
        ang_max_rad=float(dyn.get("ang_max_rad", 0.40)),
    )


def _inertia_spec(cfg: dict) -> FourierSpec:
    iner = cfg.get("inertia") or {}
    harms = tuple(int(x) for x in (iner.get("harmonics") or [1, 2, 3]))
    return FourierSpec(
        f0_hz=float(iner.get("f0_hz", 0.6)),
        harmonics=harms,
        n_warmup=int(iner.get("n_warmup", 1)),
        n_measure=int(iner.get("n_measure", 4)),
        n_cooldown=int(iner.get("n_cooldown", 1)),
        dt=DT,
        w_max_rad_s=float(iner.get("w_max_rad_s", 1.30)),
        alpha_max_rad_s2=float(iner.get("alpha_max_rad_s2", 16.0)),
        j_ang_max=float(iner.get("j_ang_max", 100.0)),
        ang_max_rad=float(iner.get("ang_max_rad", 0.40)),
    )


def _id_extra(cfg: dict) -> dict[str, Any]:
    # Stay on 8-DOF, but do not fall back to the session default ``track``
    # policy.  That re-enables centering/homotopy after rail lock and looks
    # like the old nullspace settle.
    return {
        "qp_aux": dict(cfg.get("qp_aux") or {}),
        "filter": False,
        "task_policy": "payload_id",
    }


class PayloadIdCampaign:
    def __init__(
        self,
        cfg: dict,
        log_csv: Path,
        *,
        out_json: Path,
        opts: CampaignOpts,
    ) -> None:
        from peirastic.api import PeirasticArm
        from peirastic.api.codes import CODE_NAMES, OK
        from peirastic.core.modes import Mode

        self.cfg = cfg
        self.opts = opts
        self.out_json = Path(out_json)
        self.OK = OK
        self.CODE_NAMES = CODE_NAMES
        self.Mode = Mode
        self.kin = load_id_kinematics()
        self.contract = FrameContract.from_yaml(CONFIG_FORCE)
        self.lim = SafetyLimits(m_max_kg=float((cfg.get("static") or {}).get("m_max", 2.5)))
        self.rail_lim = RailLockLimits(
            pos_err_max_m=float((cfg.get("rail") or {}).get("pos_err_max_m", 5e-4)),
            vel_p95_max_m_s=float((cfg.get("rail") or {}).get("vel_p95_max_m_s", 5e-4)),
        )
        self.arm = PeirasticArm()
        if self.arm.client is None or self.arm.twist is None:
            raise FileNotFoundError("peirastic SHM missing — start Window A first")
        if self.arm.state is None:
            raise RuntimeError("state relay missing — Window A must publish rm75_state")
        self.rec = PayloadIdRecorder(Path(log_csv))
        self.phase = "idle"
        self.record_enable = 0
        self.hold_ref_m = float((cfg.get("rail") or {}).get("target_m", 0.4))
        self.tcp_mid = np.full(3, np.nan)
        self.last_cmd_t = time.monotonic()
        self.last_twist = np.zeros(6)
        self._rail_hist: list[float] = []
        self._rail_t: list[float] = []
        self._abort = ""
        self._t_phase0 = time.monotonic()
        self._dof_before: int | None = None
        self._dof_changed = False
        self._explicit_7 = False
        self._restore_blocked = False

    def _hard_fault_active(self, reason: str = "") -> bool:
        """Classify restore safety from a fresh controller/arm snapshot.

        A normal operator interruption is recoverable after the stop boundary.
        Restore is blocked when the controller reports a fresh hard state, the
        arm feedback is freshly invalid, or the abort itself names a safety
        fault whose motion must not be resumed implicitly.
        """

        text = str(reason or "").strip().lower()
        hard_tokens = (
            "uncertified_brake",
            "feedback_stale",
            "rail_feedback_fault",
            "qpik_fault",
            "watchdog",
            "unexpected_contact",
            "raw_saturation",
            "workspace",
            "queue_overflow",
            "collision",
            "estop",
        )
        if any(text == token or text.startswith(token) for token in hard_tokens):
            return True

        client = getattr(self.arm, "client", None)
        snapshot = getattr(client, "snapshot", None)
        if callable(snapshot):
            try:
                status = dict(snapshot())
                stamp = float(status.get("t_mono", float("nan")))
                age = time.monotonic() - stamp
                fresh = np.isfinite(stamp) and -0.05 <= age <= CONTROLLER_STATUS_MAX_AGE_S
                state = int(status.get("status", -1))
                if fresh and (
                    bool(status.get("estop"))
                    or state
                    in (int(Status.ERROR), int(Status.STOPPED), int(Status.ESTOP))
                ):
                    return True
            except Exception:
                pass

        state_bus = getattr(self.arm, "state", None)
        read_state = getattr(state_bus, "read", None)
        if callable(read_state):
            try:
                state = read_state()
                stamp = float(getattr(state, "t_s", float("nan")))
                age = time.monotonic() - stamp
                if (
                    np.isfinite(stamp)
                    and -0.05 <= age <= CONTROLLER_STATUS_MAX_AGE_S
                    and not bool(getattr(state, "ok", False))
                ):
                    return True
            except Exception:
                pass
        return False

    def _enter_calibration_dof(self) -> None:
        """Keep the persistent session.  Default Payload ID is 8-DOF + rail HOLD."""

        ret, current = self.arm.get_dof()
        if ret != self.OK or int(current) not in (7, 8):
            raise CampaignAbort(f"read session DOF {ret} ({current})")
        self._dof_before = int(current)
        self._explicit_7 = self._dof_before == 7
        if self._dof_before == 8:
            return
        # An explicit 7-DOF caller session is left alone and pins live rail.

    def close(self, *, stop: bool = False) -> None:
        try:
            if self.arm.twist is not None:
                self.arm.twist.write(np.zeros(6), hz=float("nan"), connected=True)
        except Exception:
            pass
        # A zero SERVO command is intentionally continuous.  When calibration
        # changed 8→7, make an explicit stop boundary before restoring the
        # caller's session DOF; otherwise SET_DOF(after_current) would wait
        # forever for a task that is still live.
        stopped_for_restore = bool(self._dof_changed)
        restore_allowed = not bool(getattr(self, "_restore_blocked", False))
        stop_ret = self.OK
        if stop or stopped_for_restore:
            try:
                stop_fn = getattr(self.arm, "set_arm_stop", None)
                stop_ret = stop_fn() if callable(stop_fn) else self.OK
            except Exception:
                stop_ret = -1
            if stop_ret != self.OK:
                restore_allowed = False
                print(
                    f"[DOF] restore skipped: stop failed ({stop_ret})",
                    flush=True,
                )
        if self._dof_changed and self._dof_before in (7, 8) and restore_allowed:
            try:
                ret = self.arm.set_dof(int(self._dof_before), block=1)
                if ret != self.OK:
                    print(
                        f"[DOF] restore {self._dof_before} failed: {ret} "
                        f"({self.CODE_NAMES.get(ret, ret)})",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[DOF] restore {self._dof_before} failed: {exc}", flush=True)
        elif self._dof_changed and self._dof_before in (7, 8):
            print("[DOF] restore incomplete: calibration fault/stop failure", flush=True)
        if stop and not stopped_for_restore:
            # SET_DOF uses the same single-slot IPC command and clears its
            # stop bit; keep the caller's stop request authoritative.
            try:
                self.arm.set_arm_stop()
            except Exception:
                pass
        self.rec.stop()
        self.arm.close()

    def _live(self) -> tuple[AsyncStateSnapshot, np.ndarray, float]:
        snap = self.arm.state.read()
        rail = float(getattr(self.arm.state, "last_rail_m", float("nan")))
        q = None
        if snap.ok and snap.q_deg is not None and np.isfinite(rail):
            q = expand_q_meas_8dof(snap.q_deg, rail)
        return snap, q if q is not None else np.full(8, np.nan), rail

    def _safety(self, snap: AsyncStateSnapshot, q: np.ndarray) -> None:
        if snap.ok:
            st = raw_contact_abort(snap.force_raw, self.lim)
            if not st.ok:
                raise CampaignAbort(st.reason)
        transit = (
            self.phase.startswith("movej")
            or self.phase.startswith("ptp_")
            or self.phase.startswith("static_")
        )
        if snap.ok and snap.pose is not None and np.isfinite(self.tcp_mid).all() and not transit:
            lo = self.tcp_mid - TCP_BOX_HALF_M
            hi = self.tcp_mid + TCP_BOX_HALF_M
            boxed = SafetyLimits(
                workspace_p_min=tuple(lo.tolist()),
                workspace_p_max=tuple(hi.tolist()),
            )
            ws = workspace_abort(np.asarray(snap.pose[:3], dtype=float), boxed)
            if not ws.ok:
                raise CampaignAbort(ws.reason)
        if np.isfinite(q).all() and not transit:
            ok, ji, margin = joint_margin_detail(
                q[1:], self.kin.q_lower[1:], self.kin.q_upper[1:], self.lim
            )
            if not ok:
                raise CampaignAbort(f"joint_margin J{ji + 1} {np.degrees(margin):+.2f}deg")
        if self.rec.invalid:
            raise CampaignAbort("queue_overflow")

    def _tick(self, twist: np.ndarray | None = None, *, record: bool | None = None) -> AsyncStateSnapshot:
        now = time.monotonic()
        if twist is None:
            twist = np.zeros(6)
        twist = np.asarray(twist, dtype=float).reshape(6)
        if command_stale(now, self.last_cmd_t, self.lim):
            twist = deadman_twist(self.last_twist, dt=DT)
        self.arm.twist.write(twist, hz=float("nan"), connected=True)
        self.last_twist = twist
        self.last_cmd_t = now
        snap, q, rail = self._live()
        if np.isfinite(rail):
            self._rail_hist.append(rail)
            self._rail_t.append(now)
            if len(self._rail_hist) > 400:
                self._rail_hist = self._rail_hist[-400:]
                self._rail_t = self._rail_t[-400:]
        self._safety(snap, q)
        en = self.record_enable if record is None else (1 if record else 0)
        vcmd = twist
        self.rec.push(
            snap,
            phase_id=self.phase,
            record_enable=en,
            rail_pos_m=rail if np.isfinite(rail) else "",
            rail_hold_ref_m=self.hold_ref_m,
            cmd_t_mono_ns=int(now * 1e9),
            v_cmd_x=float(vcmd[0]),
            v_cmd_y=float(vcmd[1]),
            v_cmd_z=float(vcmd[2]),
            w_cmd_x=float(vcmd[3]),
            w_cmd_y=float(vcmd[4]),
            w_cmd_z=float(vcmd[5]),
        )
        return snap

    def _sleep_tick(self, twist: np.ndarray | None = None, *, record: bool | None = None) -> None:
        t0 = time.monotonic()
        self._tick(twist, record=record)
        leftover = DT - (time.monotonic() - t0)
        if leftover > 0.0:
            time.sleep(leftover)

    def _hold_seconds(self, seconds: float, *, record: bool) -> None:
        end = time.monotonic() + float(seconds)
        while time.monotonic() < end:
            self._sleep_tick(np.zeros(6), record=record)

    def _enter_servo(self, *, hold: bool, label: str, joint_hold: bool = True) -> None:
        extra = _id_extra(self.cfg)
        if joint_hold:
            extra["joint_hold"] = True
        ret = self.arm.cartesian_velocity(
            None,
            hold=hold,
            label=label,
            filter=False,
            extra=extra,
        )
        if ret != self.OK:
            raise CampaignAbort(f"servo enter {ret} ({self.CODE_NAMES.get(ret, ret)})")
        for _ in range(8):
            self._sleep_tick(np.zeros(6), record=False)

    def _wait_done(self, seq: int, *, timeout_s: float, twist: np.ndarray | None = None) -> int:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self._sleep_tick(twist, record=False)
            snap = self.arm.client.snapshot()
            from peirastic.core.ipc import Status

            st = int(snap.get("status", -1))
            if int(snap.get("estop") or 0) or st in (int(Status.ESTOP), int(Status.STOPPED)):
                return -6
            if int(snap.get("done_seq", 0)) >= int(seq):
                err = int(snap.get("err_code", 0))
                return 0 if err == 0 else int(err)
            if st == int(Status.ERROR):
                err = int(snap.get("err_code", 0))
                return -1 if err == 0 else int(err)
        return -5

    def movej_mid(self) -> None:
        from peirastic.DEMO.movej import _fmt_q, q_target_rad

        q_goal = np.asarray(q_target_rad(), dtype=float)
        rail = self.arm._current_rail_m()
        if bool(getattr(self, "_explicit_7", False)):
            if np.isfinite(rail):
                q_goal[0] = float(rail)
                self.hold_ref_m = float(rail)
        elif np.isfinite(float(q_goal[0])):
            self.hold_ref_m = float(q_goal[0])
        print(f"[MOVEJ] mid  {_fmt_q(q_goal)}  v={self.opts.movej_v:.2f}", flush=True)
        self.phase = "movej_mid"
        ret = self.arm.movej(
            q_goal,
            v=float(self.opts.movej_v),
            r=0,
            connect=0,
            block=0,
            label="payload_id_movej_mid",
        )
        if ret != self.OK:
            raise CampaignAbort(f"MOVEJ send {ret} ({self.CODE_NAMES.get(ret, ret)})")
        rc = self._wait_done(self.arm.last_seq, timeout_s=90.0)
        if rc != 0:
            raise CampaignAbort(f"MOVEJ {rc} ({self.CODE_NAMES.get(rc, rc)})")
        print("[OK] MOVEJ mid-stroke", flush=True)
        if self.opts.settle_s > 0.0:
            self.phase = "movej_settle"
            print(f"[SETTLE] {self.opts.settle_s:.1f}s after MOVEJ", flush=True)
            self._enter_servo(hold=True, label="payload_id_settle")
            self._hold_seconds(self.opts.settle_s, record=False)

    def _capture_mid(self) -> np.ndarray:
        snap, q, rail = self._live()
        if snap.pose is not None:
            self.tcp_mid = np.asarray(snap.pose[:3], dtype=float)
        if np.isfinite(rail):
            self.hold_ref_m = float(rail)
        if not np.isfinite(q).all():
            from peirastic.DEMO.movej import q_target_rad

            q = np.asarray(q_target_rad(), dtype=float)
        return q

    def rail_lock_gate(self) -> None:
        self.phase = "rail_lock"
        self._enter_servo(hold=True, label="payload_id_lock")
        self._rail_hist.clear()
        self._rail_t.clear()
        self._hold_seconds(0.3, record=False)
        meas = np.asarray(self._rail_hist, dtype=float)
        if meas.size < 8:
            raise CampaignAbort("rail_samples")
        t = np.asarray(self._rail_t, dtype=float)
        vel = np.gradient(meas, t) if t.size == meas.size else np.zeros_like(meas)
        st = evaluate_rail_lock(self.hold_ref_m, meas, vel, self.rail_lim)
        print(
            f"[RAIL] lock  err={1e3 * st.pos_err_m:.2f} mm  vel_p95={1e3 * st.vel_p95_m_s:.2f} mm/s  "
            f"{'OK' if st.ok else st.reason}",
            flush=True,
        )
        if not st.ok:
            self._hold_seconds(0.5, record=False)
            meas = np.asarray(self._rail_hist[-200:], dtype=float)
            t = np.asarray(self._rail_t[-200:], dtype=float)
            vel = np.gradient(meas, t) if t.size == meas.size else np.zeros_like(meas)
            st = evaluate_rail_lock(self.hold_ref_m, meas, vel, self.rail_lim)
            print(
                f"[RAIL] retry err={1e3 * st.pos_err_m:.2f} mm  vel_p95={1e3 * st.vel_p95_m_s:.2f} mm/s",
                flush=True,
            )
            if not st.ok:
                raise CampaignAbort(st.reason or "rail_lock")
        # After lock: 8-DOF SERVO_TWIST with payload_id (no track/centering).
        self._enter_servo(hold=False, label="payload_id_lock_hold")

    def _ptp(self, target: StaticTarget) -> None:
        self.phase = f"ptp_{target.name}"
        # Plan-time IK only. Rail stays at the lock ref; 7 arm joints do the pose.
        q_cmd = np.asarray(target.q, dtype=float).copy()
        q_cmd[0] = float(self.hold_ref_m)
        v = 0.15 if abs(target.tilt_deg) >= 45.0 else 0.25
        ret = self.arm.cartesian(
            target.pose,
            v=v,
            block=0,
            rail_m=self.hold_ref_m,
            q_target=q_cmd,
            label=f"payload_id_{target.name}",
            task_policy="payload_id",
        )
        if ret != self.OK:
            raise CampaignAbort(f"PTP send {target.name} {ret}")
        rc = self._wait_done(self.arm.last_seq, timeout_s=45.0)
        if rc != 0:
            raise CampaignAbort(f"PTP {target.name} {rc} ({self.CODE_NAMES.get(rc, rc)})")
        # Cover Window A's default track idle before it can recouple the rail.
        self._enter_servo(hold=False, label=f"payload_id_hold_{target.name}")

    def _wait_still(self, timeout_s: float, qdot_lim: float) -> None:
        end = time.monotonic() + float(timeout_s)
        last_q = None
        still_s = 0.0
        while time.monotonic() < end:
            self._sleep_tick(np.zeros(6), record=False)
            _, q, _ = self._live()
            if last_q is not None and np.isfinite(q).all() and np.isfinite(last_q).all():
                qd = float(np.linalg.norm((q[1:] - last_q[1:]) / DT))
                if qd < float(qdot_lim):
                    still_s += DT
                    if still_s >= 0.08:
                        return
                else:
                    still_s = 0.0
            last_q = q

    def _static_hold(self, target: StaticTarget) -> None:
        st = self.cfg.get("static") or {}
        self.phase = f"static_{target.name}"
        wait_s = float(st.get("min_wait_s", 0.20))
        rec_s = float(st.get("record_s", 0.6))
        qdot_lim = float(st.get("settle_qdot_rad_s", 0.03))
        self._wait_still(wait_s, qdot_lim)
        self.record_enable = 1
        self._hold_seconds(rec_s, record=True)
        self.record_enable = 0

    def run_static(self, q_mid: np.ndarray) -> list[StaticTarget]:
        st = self.cfg.get("static") or {}
        tilt = st.get("tilt_degs")
        print("[P1] planning static poses", flush=True)
        targets = static_targets_from_mid(
            self.kin,
            q_mid,
            tilt_degs=None if tilt is None else [int(x) for x in tilt],
            holdout_min_abs_deg=float(st.get("holdout_min_abs_deg", 60.0)),
            plan_margin_rad=float((self.cfg.get("safety") or {}).get("static_plan_margin_rad", 0.08)),
            include_tool_tilts=bool(st.get("include_tool_tilts", False)),
            include_yaw=bool(st.get("include_yaw", False)),
            include_tip_up=bool(st.get("include_tip_up", True)),
            tip_up_tilt_degs=None if st.get("tip_up_tilt_degs") is None else [int(x) for x in st["tip_up_tilt_degs"]],
            tip_up_approach_steps=int(st.get("tip_up_approach_steps", 6)),
            plan_frame=str(self.cfg.get("plan_frame") or st.get("plan_frame") or "link_7"),
            hold_point=str(st.get("hold_point") or self.cfg.get("hold_point") or "tcp"),
        )
        print(f"[P1] {len(targets)} static poses", flush=True)
        if len(targets) < 6:
            raise CampaignAbort("too_few_static_targets")
        mid = next((t for t in targets if t.name == "mid"), None)
        done: list[StaticTarget] = []
        last_ray = "mid"

        def _ray(t: StaticTarget) -> str:
            if t.name == "mid" or t.is_yaw:
                return "mid" if t.name == "mid" else f"yaw{'-' if t.tilt_deg < 0 else '+'}"
            if t.name.startswith(("UP", "UWX", "UWY")):
                return "up"
            pref = "".join(ch for ch in t.name if ch.isalpha())
            return f"{pref}{'-' if t.tilt_deg < 0 else '+'}"

        for tgt in targets:
            try:
                ray = _ray(tgt)
                if mid is not None and last_ray != "mid" and ray != last_ray:
                    self._ptp(mid)
                    last_ray = "mid"
                if tgt.name != "mid":
                    self._ptp(tgt)
                    last_ray = ray
                else:
                    self._enter_servo(hold=False, label="payload_id_hold_mid")
                    last_ray = "mid"
                self._static_hold(tgt)
                done.append(tgt)
            except CampaignAbort as exc:
                msg = str(exc)
                hard = (
                    "unexpected_contact",
                    "raw_saturation",
                    "workspace",
                    "queue_overflow",
                )
                if any(msg == k or msg.startswith(k) for k in hard):
                    raise
                print(f"[P1] skip {tgt.name}", flush=True)
                try:
                    self._enter_servo(hold=False, label="payload_id_hold_recover")
                except CampaignAbort:
                    raise
        print(f"[P1] recorded {len(done)}/{len(targets)}", flush=True)
        return done

    def _taught_mid_target(self, q_mid: np.ndarray) -> StaticTarget:
        q = np.asarray(q_mid, dtype=float).reshape(8)
        return StaticTarget(
            name="mid",
            pose=self.kin.fk_pose(q),
            q=q,
            is_train=True,
            is_yaw=False,
            tilt_deg=0.0,
        )

    def _return_to_mid(self, q_mid: np.ndarray, *, why: str) -> None:
        print(f"[MID] {why}", flush=True)
        self._ptp(self._taught_mid_target(q_mid))

    def _stream_traj(self, t: np.ndarray, twist_T: np.ndarray, phase: str) -> None:
        self.phase = phase
        self._enter_servo(hold=False, label=f"payload_id_{phase}", joint_hold=False)
        t0 = time.monotonic()
        n = t.size
        self.record_enable = 1
        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= float(t[-1]):
                break
            idx = min(int(elapsed / DT), n - 1)
            self._tick(twist_T[idx], record=True)
            leftover = t0 + (idx + 1) * DT - time.monotonic()
            if leftover > 0.0:
                time.sleep(leftover)
        self.record_enable = 0
        self._tick(np.zeros(6), record=False)
        self._enter_servo(hold=False, label=f"payload_id_hold_{phase}")
        post = float((self.cfg.get("dynamic") or {}).get("post_hold_s", 0.25))
        if post > 0.0:
            self._hold_seconds(post, record=False)

    def run_dynamic(self) -> None:
        spec = _fourier_spec(self.cfg)
        dyn = self.cfg.get("dynamic") or {}
        v_peak = float(dyn.get("v_peak_m_s", 0.15))
        w_peak = float(dyn.get("w_peak_rad_s", 1.10))
        R_LT = self.kin.R_LT
        r_LT = self.kin.r_LT_L
        axes = [
            ("dt_x", 0, False, v_peak),
            ("dt_y", 1, False, v_peak),
            ("dt_z", 2, False, v_peak),
            ("dr_x", 3, True, w_peak),
            ("dr_y", 4, True, w_peak),
            ("dr_z", 5, True, w_peak),
        ]
        for name, ax, rot, peak in axes:
            t, tw_L, _amps = axis_twist_L(spec, ax, peak=peak, rotational=rot)
            tw_T = np.vstack([twist_link7_to_tcp(row, R_LT=R_LT, r_LT_L=r_LT) for row in tw_L])
            print(f"[P2] {name}  {t[-1]:.1f}s", flush=True)
            self._stream_traj(t, tw_T, name)
        t, tw_L, _ = axis_twist_L(spec, 2, peak=0.7 * v_peak, rotational=False)
        tw_R = axis_twist_L(spec, 4, peak=0.7 * w_peak, rotational=True)[1]
        mix = tw_L + tw_R
        tw_T = np.vstack([twist_link7_to_tcp(row, R_LT=R_LT, r_LT_L=r_LT) for row in mix])
        print(f"[P2] dm_holdout  {t[-1]:.1f}s", flush=True)
        self._stream_traj(t, tw_T, "dm_holdout")

    def run_inertia(self) -> None:
        iner = self.cfg.get("inertia") or {}
        spec = _inertia_spec(self.cfg)
        peak = float(iner.get("w_peak_rad_s", 1.10))
        R_LT = self.kin.R_LT
        r_LT = self.kin.r_LT_L
        for ax, name in enumerate(("ix", "iy", "iz")):
            t, tw_L, _ = axis_twist_L(spec, 3 + ax, peak=peak, rotational=True)
            tw_T = np.vstack([twist_link7_to_tcp(row, R_LT=R_LT, r_LT_L=r_LT) for row in tw_L])
            print(f"[P-I] {name}  {t[-1]:.1f}s", flush=True)
            self._stream_traj(t, tw_T, f"inertia_{name}")

    def run(self) -> int:
        self.rec.start()
        print(f"[STATE] csv {self.rec.path}", flush=True)
        # Calibration is an arm-only experiment.  Commit the structure before
        # MOVEJ/settle preparation so no task policy can silently recouple the
        # rail; close() restores the caller's session DOF afterwards.
        self._enter_calibration_dof()
        if not self.opts.skip_movej:
            self.movej_mid()
        else:
            print("[MOVEJ] skipped", flush=True)
            self._enter_servo(hold=True, label="payload_id_settle")
            if self.opts.settle_s > 0.0:
                self._hold_seconds(min(self.opts.settle_s, 2.0), record=False)
        q_mid = self._capture_mid()
        print(
            f"[STATE] mid tcp={np.array2string(self.tcp_mid, precision=3)}  "
            f"rail={1000.0 * self.hold_ref_m:.1f} mm  tool=gripper2",
            flush=True,
        )
        self.rail_lock_gate()
        self.run_static(q_mid)
        if not self.opts.p1_only:
            self._return_to_mid(q_mid, why="P2")
            self.run_dynamic()
            if inertia_ident_enabled(self.cfg, self.opts):
                self._return_to_mid(q_mid, why="P-I")
                self.run_inertia()
        self.phase = "done"
        # Stay on commanded SERVO_TWIST at v*=0. HOLD would latch pose_d and
        # let 8-DOF idle steal the session (track + P-hold). joint_hold still
        # freezes q.
        self._enter_servo(hold=False, label="payload_id_done")
        self._hold_seconds(0.2, record=False)
        if self.rec.invalid:
            print("[WARN] recorder marked invalid (overflow/gaps) — sidecar still written", flush=True)
        return 0


def _fnum(row: dict, key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def qdot_rad_s_from_row(row: dict) -> np.ndarray | None:
    """CSV SDK joint speed in rad/s, or None if the row has no finite qdot."""

    vs = np.array([_fnum(row, f"qdot_sdk_deg_s_{j}") for j in range(1, 8)], dtype=float)
    if vs.size < 7 or not np.all(np.isfinite(vs)):
        return None
    return np.deg2rad(vs)


def _delay_reject_reason(fit) -> str:
    bits = []
    if fit.delay_hit_search_boundary:
        bits.append("boundary")
    if not fit.phase_linear:
        bits.append("phase")
    if fit.delay_ci95_s > 0.02:
        bits.append(f"ci95={1e3 * fit.delay_ci95_s:.1f}ms")
    spread = float(np.max(np.abs(np.asarray(fit.delay_per_axis_s) - fit.delay_sensor_vs_joint_s)))
    if spread > 0.005:
        bits.append(f"axis_spread={1e3 * spread:.1f}ms")
    return "reason=" + ",".join(bits) if bits else "reason=ok"


def windows_from_csv(
    path: Path,
    kin: RobotKinematics,
    contract: FrameContract,
) -> list[StaticWindow]:
    groups: dict[str, list[dict]] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("record_enable") or "") not in {"1", "1.0", "True"}:
                continue
            phase = str(row.get("phase_id") or "")
            if not phase.startswith("static_"):
                continue
            groups.setdefault(phase, []).append(row)
    g_base = contract.gravity_base()
    out: list[StaticWindow] = []
    names = sorted(groups)
    for i, name in enumerate(names):
        if name.startswith("static_WZ"):
            continue
        rows = groups[name]
        wrenches = []
        gs = []
        ts = []
        for row in rows:
            f = np.array(
                [_fnum(row, k) for k in (
                    "force_raw_fx", "force_raw_fy", "force_raw_fz",
                    "force_raw_mx", "force_raw_my", "force_raw_mz",
                )],
                dtype=float,
            )
            if not np.all(np.isfinite(f)):
                continue
            q_deg = np.array([_fnum(row, f"q_deg_{j}") for j in range(1, 8)], dtype=float)
            rail = _fnum(row, "rail_pos_m")
            if not np.all(np.isfinite(q_deg)) or not np.isfinite(rail):
                continue
            q8 = expand_q_meas_8dof(q_deg, rail)
            wrenches.append(wrench_sensor_to_link7(f, contract))
            gs.append(kin.gravity_link7(q8, g_base))
            ts.append(_fnum(row, "recv_t_mono_ns") * 1e-9)
        if len(wrenches) < 8:
            continue
        W = np.vstack(wrenches)
        tilt = abs(_tilt_deg_from_name(name))
        out.append(
            StaticWindow(
                g_L=robust_mean(np.vstack(gs)),
                wrench_L=robust_mean(W),
                t_s=float(np.median(ts)) if ts else 0.0,
                n_eff=float(len(wrenches)),
                samples=W,
                is_train=tilt < 60.0,
                is_anchor=(i % 4 == 0),
                block_id=i // 4,
                name=name.removeprefix("static_"),
            )
        )
    return out


def _motion_from_csv(
    path: Path,
    kin: RobotKinematics,
    contract: FrameContract,
    phase_prefix: str,
    *,
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    rows = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("record_enable") or "") not in {"1", "1.0", "True"}:
                continue
            phase = str(row.get("phase_id") or "")
            if exact:
                if phase != phase_prefix:
                    continue
            elif not phase.startswith(phase_prefix):
                continue
            rows.append(row)
    if len(rows) < 64:
        return None
    obs = ArmJointObserver(rail_locked=True)
    t, w_L, a_L, om_L, al_L, g_L = [], [], [], [], [], []
    g_base = contract.gravity_base()
    n_qdot_csv = 0
    for row in rows:
        f = np.array(
            [_fnum(row, k) for k in (
                "force_raw_fx", "force_raw_fy", "force_raw_fz",
                "force_raw_mx", "force_raw_my", "force_raw_mz",
            )],
            dtype=float,
        )
        q_deg = np.array([_fnum(row, f"q_deg_{j}") for j in range(1, 8)], dtype=float)
        rail = _fnum(row, "rail_pos_m")
        ts = _fnum(row, "recv_t_mono_ns") * 1e-9
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(q_deg)) or not np.isfinite(rail):
            continue
        q_arm = np.deg2rad(q_deg)
        qd = qdot_rad_s_from_row(row)
        if qd is not None:
            n_qdot_csv += 1
        q8, qd8, qdd8, _var = obs.step(ts, q_arm, qd, rail_q=rail)
        mot = kin.frame_classical_motion(q8, qd8, qdd8, "link_7")
        t.append(ts)
        w_L.append(wrench_sensor_to_link7(f, contract))
        a_L.append(mot.linear_acceleration)
        om_L.append(mot.angular_velocity)
        al_L.append(mot.angular_acceleration)
        g_L.append(kin.gravity_link7(q8, g_base))
    if len(t) < 64:
        return None
    if n_qdot_csv == 0:
        print(f"[FIT] {phase_prefix} no qdot in csv  n={len(t)}", flush=True)
    return (
        np.asarray(t),
        np.vstack(w_L),
        np.vstack(a_L),
        np.vstack(om_L),
        np.vstack(al_L),
        np.vstack(g_L),
    )


def _delay_from_motion(
    mot,
    fit,
    spec: FourierSpec,
) -> Any:
    t, w_meas, a_L, om_L, al_L, g_L = mot
    w_pay = np.zeros_like(w_meas)
    for i in range(t.size):
        w_pay[i] = payload_wrench_mhb(
            mass_kg=fit.mass_kg,
            h_L=fit.h_L,
            a_L=a_L[i],
            g_L=g_L[i],
            omega_L=om_L[i],
            alpha_L=al_L[i],
            bias=fit.bias0,
        )
    freqs = np.asarray(spec.harmonics, dtype=float) * spec.f0_hz
    mask = measure_mask(spec, t - float(t[0]))
    if int(np.count_nonzero(mask)) < 32:
        mask = np.ones(t.size, dtype=bool)
    return fit_delay_on_lines(
        fft_lines(t[mask], w_meas[mask], freqs),
        fft_lines(t[mask], w_pay[mask], freqs),
        freqs,
    )


def fit_hardware_log(
    path: Path,
    cfg: dict,
    *,
    out_json: Path,
    kin: RobotKinematics | None = None,
    contract: FrameContract | None = None,
) -> dict[str, Any]:
    kin = kin or load_id_kinematics()
    contract = contract or FrameContract.from_yaml(CONFIG_FORCE)
    windows = windows_from_csv(path, kin, contract)
    if len(windows) < 4:
        raise RuntimeError(f"{path} has only {len(windows)} static windows")
    samples = [w.samples for w in windows if w.samples is not None]
    Sigma = pooled_shrinkage_cov(samples) if samples else np.diag([0.04, 0.04, 0.04, 0.008, 0.008, 0.008]) ** 2
    st = cfg.get("static") or {}
    fit = fit_static_windows(
        windows,
        Sigma=Sigma,
        m_min=float(st.get("m_min", 0.05)),
        m_max=float(st.get("m_max", 5.0)),
        r_max_m=float(st.get("r_max_m", 0.12)),
        eps_b=float(st.get("drift_eps_b", 0.02)),
    )
    print(
        f"[FIT] static m={fit.mass_kg:.3f} kg  h={np.array2string(fit.h_L, precision=4)}  "
        f"rank={fit.rank_m0} cond={fit.cond_m0:.1f} drift={fit.drift_enabled}",
        flush=True,
    )
    residuals = static_residual_report(windows, fit)
    frame_cfg = FrameConfig.from_yaml(CONFIG_FORCE)
    phi_mhb = phi16(fit.mass_kg, fit.h_L, fit.bias0, None)
    com = com_report(phi_mhb, frame_cfg)
    rms_all = float((residuals.get("all") or {}).get("rms_all", float("nan")))
    summary_poses = {k: residuals[k] for k in ("all", "train", "holdout") if k in residuals}

    spec = _fourier_spec(cfg)
    delay = None
    axis_delays = []
    for pref in ("dt_x", "dt_y", "dt_z"):
        mot = _motion_from_csv(path, kin, contract, pref, exact=True)
        if mot is None:
            continue
        try:
            d_i = _delay_from_motion(mot, fit, spec)
        except Exception as exc:
            print(f"[FIT] delay {pref} skipped: {exc}", flush=True)
            continue
        axis_delays.append(d_i)
    if axis_delays:
        vals = np.array([d.delay_sensor_vs_joint_s for d in axis_delays], dtype=float)
        delay = axis_delays[int(np.argmin(np.abs(vals - np.median(vals))))]
        delay.delay_online_effective_s = float(np.median(vals))
        delay.delay_sensor_vs_joint_s = float(np.median(vals))
        if float(np.max(np.abs(vals - np.median(vals)))) > 0.008:
            delay.delay_hit_search_boundary = True
        print(
            f"[FIT] delay {1e3 * delay.delay_sensor_vs_joint_s:.1f} ms  "
            f"axes={[round(1e3 * v, 1) for v in vals]}  reject={delay_rejected(delay)}"
            f"{'' if not delay_rejected(delay) else '  ' + _delay_reject_reason(delay)}",
            flush=True,
        )

    iner = None
    iner_m = None
    if inertia_ident_enabled(cfg):
        iner_m = _motion_from_csv(path, kin, contract, "inertia_")
    if iner_m is not None:
        _t, w_meas, a_L, om, al, g_L = iner_m
        tau = np.zeros((w_meas.shape[0], 3))
        for i in range(w_meas.shape[0]):
            tau[i] = inertia_moment_residual(
                w_meas[i],
                mass_kg=fit.mass_kg,
                h_L=fit.h_L,
                a_L=a_L[i],
                g_L=g_L[i],
                omega_L=om[i],
                bias=fit.bias0,
            )
        n = tau.shape[0]
        split = max(8, n * 3 // 4)
        try:
            iner = fit_inertia_moments(
                al[:split],
                om[:split],
                tau[:split],
                mass_kg=fit.mass_kg,
                r_max_m=float(st.get("r_max_m", 0.12)),
                sigma_M=0.002,
                holdout_tau=np.mean(tau[split:], axis=0) if n > split else None,
                holdout_pred_mhb=np.zeros(3) if n > split else None,
                holdout_pred_I=np.mean(tau[split:], axis=0) if n > split else None,
                snr_min=float((cfg.get("inertia") or {}).get("snr_min", 3.0)),
            )
            print(
                f"[FIT] inertia adopted={iner.adopted}  reason={iner.reason or 'ok'}  "
                f"snr={iner.snr_I:.1f}  triangle={iner.triangle_ok}  "
                f"I_diag={np.array2string(iner.I_voigt[:3], precision=4)}",
                flush=True,
            )
        except Exception as exc:
            print(f"[FIT] inertia skipped: {exc}", flush=True)

    doc = empty_document()
    cache = read_tool_offset_cache()
    doc["payload"]["tool_id"] = cache[0] if cache else "gripper2"
    doc["payload"]["mass_kg"] = fit.mass_kg
    doc["payload"]["first_moment_kg_m"] = fit.h_L.tolist()
    doc["payload"]["inertia_kg_m2"] = iner.I_voigt.tolist() if iner is not None and iner.adopted else None
    doc["calibration_session"]["bias0"] = fit.bias0.tolist()
    doc["calibration_session"]["bias_drift_per_s"] = fit.bias_drift_per_s.tolist()
    doc["calibration_session"]["drift_enabled"] = fit.drift_enabled
    doc["tool_binding"]["active_tool_name"] = cache[0] if cache else "gripper2"
    doc["tool_binding"]["urdf_sha256"] = urdf_sha256(DEFAULT_URDF)
    doc["tool_binding"]["force_sign"] = list(contract.force_sign)
    T = np.eye(4)
    T[:3, :3] = kin.R_LT
    T[:3, 3] = kin.r_LT_L
    doc["tool_binding"]["T_link7_tcp"] = T.tolist()
    phi = phi16(fit.mass_kg, fit.h_L, fit.bias0, iner.I_voigt if iner is not None and iner.adopted else None)
    rec = phi_dict16(phi)
    if iner is None or not iner.adopted:
        for k in ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"):
            rec[k] = 0.0
        doc["validation"]["inertia_ident_failed"] = None if iner is None else (iner.reason or "not_adopted")
    doc["phi_mhb"] = {k: rec[k] for k in ("m", "mc_x", "mc_y", "mc_z", "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0")}
    doc["phi_recommended"] = rec
    doc["validation"]["force_dynamic_valid"] = bool(delay is not None and not delay_rejected(delay))
    doc["validation"]["moment_dynamic_valid"] = bool(iner is not None and iner.moment_dynamic_valid)
    if iner is not None:
        doc["validation"]["unmodeled_inertia_torque_bound_nm"] = iner.unmodeled_bound_nm
    if delay is not None:
        doc["delay"]["delay_sensor_vs_joint_s"] = delay.delay_sensor_vs_joint_s
        doc["delay"]["delay_online_effective_s"] = delay.delay_online_effective_s
        doc["delay"]["delay_ci95_s"] = delay.delay_ci95_s
        doc["delay"]["delay_hit_search_boundary"] = delay.delay_hit_search_boundary
    doc["com_recommended"] = com
    doc["static"] = {
        "rank_m0": fit.rank_m0,
        "cond_m0": fit.cond_m0,
        "n_windows": len(windows),
        "source_log": str(path),
        "rms_all": rms_all,
        "rms_force": (residuals.get("all") or {}).get("rms_force"),
        "rms_moment": (residuals.get("all") or {}).get("rms_moment"),
        "per_pose_residual": residuals,
    }
    doc["provenance"]["source_log_sha256"] = [str(path)]
    write_phi_v2(out_json, doc)
    live_on = bool((cfg.get("output") or {}).get("auto_promote_live", True))
    print_summary(
        phi_mhb,
        frame_cfg,
        rms_all=rms_all,
        per_pose=summary_poses,
        out_json=PHI_JSON if live_on else out_json,
    )
    print(f"[OK] sidecar {out_json}", flush=True)
    if live_on:
        promote_mhb_to_live(
            PHI_JSON,
            rec,
            com=com,
            rms_all=rms_all,
            per_pose=summary_poses,
        )
        print(
            f"[OK] live {PHI_JSON}  m={fit.mass_kg:.3f} kg  I=0  "
            f"(restart Window A to load)",
            flush=True,
        )
    return doc


def run_hardware_campaign(
    cfg: dict,
    log_csv: Path,
    *,
    out_json: Path | None = None,
    opts: CampaignOpts | None = None,
) -> int:
    opts = opts or CampaignOpts()
    out_json = Path(out_json) if out_json is not None else PHI_JSON_V2
    log_csv = Path(log_csv)
    minutes = estimate_campaign_s(cfg, opts) / 60.0
    dyn = "P1 only" if opts.p1_only else ("P1+P2+P-I" if inertia_ident_enabled(cfg, opts) else "P1+P2")
    print(
        f"[PLAN] payload ID V2  ~{minutes:.0f} min  "
        f"MOVEJ={'skip' if opts.skip_movej else 'mid'}  {dyn}",
        flush=True,
    )
    print("[PLAN] force off; writes m,h,b to live force_id_phi.json  I=0", flush=True)
    camp = PayloadIdCampaign(cfg, log_csv, out_json=out_json, opts=opts)
    rc = 1
    stop = True
    try:
        rc = camp.run()
        stop = rc != 0
    except KeyboardInterrupt:
        print("[STOP] interrupted — fitting whatever is in the log", flush=True)
        # Ctrl-C is recoverable when the fresh controller/arm snapshot is
        # healthy.  close() sends the stop boundary before restoring 8↔7.
        camp._restore_blocked = camp._hard_fault_active("keyboard_interrupt")
        rc = 130
    except CampaignAbort as exc:
        print(f"[ABORT] {exc}", flush=True)
        camp._restore_blocked = camp._hard_fault_active(str(exc))
        rc = 2
    finally:
        camp.close(stop=stop)
    try:
        fit_hardware_log(log_csv, cfg, out_json=out_json, kin=camp.kin, contract=camp.contract)
    except Exception as exc:
        print(f"[FIT] no sidecar: {exc}", flush=True)
        if rc == 0:
            rc = 3
    return rc
