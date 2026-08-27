"""Launch and talk to the wbc_rt process over named shared memory."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, TrackerStatus
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.wbc_rt.config_dump import dump_wbc_config
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P


_QP_STATUS_NAMES = {
    P.QP_NOT_RUN: "not_run",
    P.QP_SOLVED: "solved",
    P.QP_MAX_ITER: "max_iter",
    P.QP_PRIMAL_INFEASIBLE: "primal_infeasible",
    P.QP_DUAL_INFEASIBLE: "dual_infeasible",
    P.QP_CLOSEST_PRIMAL_FEASIBLE: "closest_primal_feasible",
    P.QP_NONFINITE: "nonfinite",
    P.QP_CERTIFICATE_FAILED: "certificate_failed",
    P.QP_OVERRUN: "overrun",
    P.QP_EXCEPTION: "exception",
}

_FAILURE_NAMES = {
    0: "none",
    1: "input_nonfinite",
    2: "box_infeasible",
    3: "qp1_status",
    4: "qp1_certificate",
    5: "qp2_status",
    6: "qp2_certificate",
    7: "solve_overrun",
    8: "final_certificate",
    9: "input_stale",
}


def classify_native_fallback(
    fallback_u: int,
    failure_code: int,
    *,
    native_failed: bool,
) -> tuple[str, str, bool]:
    """Map native SHM fallback/failure.  Only a true fail/stop latches."""
    if native_failed or int(fallback_u) == P.FALLBACK_STOP:
        return (
            "stop",
            _FAILURE_NAMES.get(int(failure_code), "native_solver_failed"),
            True,
        )
    return "none", "", False


def failed_timeout_out():
    """Explicit stop-shaped OUT record for a STEP wait timeout or torn read."""
    o = np.zeros((), dtype=P.WBC_OUT_DTYPE)
    o["status"] = P.STATUS_FAIL
    o["flags"] = P.OUT_FAILED | P.OUT_STALE
    return o


def find_wbc_rt_binary(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() and os.access(p, os.X_OK) else None
    env = os.environ.get("WBC_RT_BIN")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "native" / "wbc_rt" / "build" / "wbc_rt",
        here.parents[5] / "native" / "wbc_rt" / "build" / "wbc_rt",
        Path("/usr/local/bin/wbc_rt"),
    ]
    which = shutil.which("wbc_rt")
    if which:
        candidates.insert(0, Path(which))
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


class NativeWbcClient:
    """SHM client.  Owns the child process when it creates the segments."""

    def __init__(self, controller, *, timeout_s: float = 0.100) -> None:
        self.ctrl = controller
        self.cfg = controller.cfg
        self.timeout_s = float(timeout_s)
        prefix = str(getattr(self.cfg, "native_shm_prefix", "rm75_wbc"))
        self.in_name = f"{prefix}_in"
        self.out_name = f"{prefix}_out"
        self._shm_in = None
        self._shm_out = None
        self._in = None
        self._out = None
        self._proc: subprocess.Popen | None = None
        self._cfg_path: Path | None = None
        self._seq = 0
        self._step_count = 0
        self._started = False

    @staticmethod
    def _begin_write(rec) -> int:
        """Open an input seqlock write and return its even base generation."""
        base = int(rec["generation"]) & ((1 << 64) - 1)
        if base & 1:
            base = (base + 1) & ((1 << 64) - 1)
        rec["generation"] = np.uint64((base + 1) & ((1 << 64) - 1))
        return base

    @staticmethod
    def _end_write(rec, base: int) -> None:
        rec["generation"] = np.uint64((int(base) + 2) & ((1 << 64) - 1))

    def _stable_out_copy(self):
        """Return a torn-read-safe copy of the native output record."""
        if self._out is None:
            return None
        for _ in range(4):
            g0 = int(self._out["generation"][0])
            if g0 & 1:
                time.sleep(0.00005)
                continue
            snap = self._out[0].copy()
            g1 = int(self._out["generation"][0])
            if g0 == g1 and not (g1 & 1) and int(snap["generation"]) == g1:
                return snap
        return None

    def start(self) -> None:
        binary = find_wbc_rt_binary(getattr(self.cfg, "native_bin", None))
        if binary is None:
            raise FileNotFoundError(
                "wbc_rt binary not found; build native/wbc_rt or set WBC_RT_BIN"
            )
        env = os.environ.copy()
        cmeel = env.get(
            "CMEEL_PREFIX",
            "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix",
        )
        lib = str(Path(cmeel) / "lib")
        env["LD_LIBRARY_PATH"] = lib + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        info = subprocess.run(
            [str(binary), "--protocol-info"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        try:
            version, in_size, out_size = (int(v) for v in info.stdout.split())
        except (ValueError, TypeError):
            raise RuntimeError(f"wbc_rt protocol-info failed: {info.stdout!r}")
        if (version, in_size, out_size) != (P.WBC_VERSION, P.WBC_IN_SIZE, P.WBC_OUT_SIZE):
            raise RuntimeError(
                "wbc_rt protocol mismatch: "
                f"native=({version},{in_size},{out_size}) "
                f"client=({P.WBC_VERSION},{P.WBC_IN_SIZE},{P.WBC_OUT_SIZE})"
            )
        tmp = Path(tempfile.mkdtemp(prefix="wbc_rt_"))
        self._cfg_path = tmp / "wbc.cfg"
        dump_wbc_config(
            self.cfg,
            self._cfg_path,
            urdf_path=self.ctrl.kin.urdf_path,
            kin=self.ctrl.kin,
        )
        self._shm_in = create_named_shm(self.in_name, P.WBC_IN_SIZE)
        self._shm_out = create_named_shm(self.out_name, P.WBC_OUT_SIZE)
        self._in = P.view_in(self._shm_in.buf)
        self._out = P.view_out(self._shm_out.buf)
        self._in[0].fill(0)
        self._out[0].fill(0)
        self._in["magic"] = P.WBC_MAGIC
        self._in["version"] = P.WBC_VERSION
        self._out["magic"] = P.WBC_MAGIC
        self._out["version"] = P.WBC_VERSION
        self._proc = subprocess.Popen(
            [
                str(binary),
                "--config",
                str(self._cfg_path),
                "--in",
                self.in_name,
                "--out",
                self.out_name,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"wbc_rt exited during start (code {self._proc.returncode})"
                )
            ready = self._stable_out_copy()
            if ready is not None and int(ready["status"]) == P.STATUS_READY:
                self._started = True
                print(
                    f"[joint_ik] inner.backend=native wbc_rt pid={self._proc.pid} "
                    f"bin={binary}",
                    flush=True,
                )
                return
            time.sleep(0.01)
        self.shutdown()
        raise TimeoutError("wbc_rt did not become READY")

    def shutdown(self) -> None:
        try:
            if self._started and self._in is not None:
                self._command(P.CMD_SHUTDOWN, wait=False)
        except Exception:
            pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=1.0)
            except Exception:
                self._proc.kill()
            self._proc = None
        self._in = None
        self._out = None
        close_named_shm(self._shm_in)
        close_named_shm(self._shm_out)
        self._shm_in = None
        self._shm_out = None
        self._started = False

    def _wait_seq(self, seq: int, *, timeout_s: float | None = None) -> bool:
        limit = time.monotonic() + float(self.timeout_s if timeout_s is None else timeout_s)
        polls = 0
        while time.monotonic() < limit:
            g0 = int(self._out["generation"][0])
            if g0 & 1:
                time.sleep(0.00005)
                continue
            observed = int(self._out["seq"][0])
            g1 = int(self._out["generation"][0])
            if g0 == g1 and not (g1 & 1) and observed == int(seq):
                return True
            if self._proc is not None and self._proc.poll() is not None:
                return False
            # The native worker is a same-host real-time helper.  A long
            # fixed sleep here adds an avoidable millisecond-scale phase lag
            # at 200 Hz; yield only occasionally so the common path observes
            # the completed generation immediately while still being fair to
            # the worker and other threads.
            polls += 1
            if polls & 0x3f == 0:
                time.sleep(0)
        return False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _command(
        self,
        cmd: int,
        *,
        cmd_f=None,
        cmd_u=None,
        q_meas=None,
        wait: bool = True,
        timeout_s: float | None = None,
    ) -> bool:
        rec = self._in[0]
        base = self._begin_write(rec)
        rec["cmd"] = np.uint32(cmd)
        rec["magic"] = P.WBC_MAGIC
        rec["version"] = P.WBC_VERSION
        rec["cmd_f"][:] = 0.0
        rec["cmd_u"][:] = 0
        rec["qdot_ff"][:] = 0.0
        rec["pose_d"][:] = 0.0
        rec["vel_ff"][:] = 0.0
        rec["path_twist"][:] = 0.0
        rec["feedback_twist"][:] = 0.0
        if cmd_f is not None:
            arr = np.asarray(cmd_f, dtype=float).reshape(-1)
            n = min(arr.size, 16)
            rec["cmd_f"][:n] = arr[:n]
        if cmd_u is not None:
            arr = np.asarray(cmd_u, dtype=np.uint32).reshape(-1)
            n = min(arr.size, 8)
            rec["cmd_u"][:n] = arr[:n]
        if q_meas is not None:
            rec["q_meas"][:] = np.asarray(q_meas, dtype=float).reshape(8)
        seq = self._next_seq()
        self._step_count += 1
        rec["cmd_seq"] = np.uint64(seq)
        rec["seq"] = np.uint64(seq)
        self._end_write(rec, base)
        if not wait:
            return True
        return self._wait_seq(seq, timeout_s=timeout_s if timeout_s is not None else 0.5)

    def enable(self) -> None:
        self._command(P.CMD_ENABLE)

    def stop(self) -> None:
        self._command(P.CMD_STOP)

    def reset(self, q0) -> None:
        self._command(P.CMD_RESET, q_meas=q0, cmd_f=np.asarray(q0, dtype=float))
        self._sync_q()

    def begin_hybrid_episode(self, q_meas, qdot_applied=None) -> None:
        extra = np.zeros(16)
        if qdot_applied is not None:
            extra[:8] = np.asarray(qdot_applied, dtype=float).reshape(-1)[:8]
        self._command(P.CMD_BEGIN_HYBRID, q_meas=q_meas, cmd_f=extra)

    def set_rail_mode(self, mode, *, q_ref_m=None, locked_style=None) -> None:
        mode_u = P.RAIL_COUPLED
        if mode == RailMode.LOCKED or str(mode).split(".")[-1].lower() == "locked":
            mode_u = P.RAIL_LOCKED
        style_u = P.STYLE_HOLD
        if locked_style is not None:
            name = str(getattr(locked_style, "name", locked_style)).split(".")[-1].lower()
            if "rail" in name:
                style_u = P.STYLE_RAIL_ONLY
            elif "tcp" in name:
                style_u = P.STYLE_TCP_FIXED
        cmd_f = np.zeros(16)
        if q_ref_m is not None:
            cmd_f[0] = float(q_ref_m)
            cmd_f[1] = 1.0
        self._command(P.CMD_SET_RAIL_MODE, cmd_u=[mode_u, style_u], cmd_f=cmd_f)

    def push_flags(self) -> None:
        c = self.ctrl
        bits = 0
        if getattr(c, "_plan_drives_rail", False):
            bits |= P.FLAG_PLAN_DRIVES_RAIL
        if getattr(c, "_direct_joint_ptp", False):
            bits |= P.FLAG_DIRECT_PTP
        if getattr(c, "_arm_task_suppressed", False):
            bits |= P.FLAG_ARM_SUPPRESS
        if getattr(c, "_centering_suppressed", False):
            bits |= P.FLAG_CENTER_SUPPRESS
        if getattr(c, "_manipulability_active", False):
            bits |= P.FLAG_MANIP_ACTIVE
        if getattr(c, "_rail_ext_active", True):
            bits |= P.FLAG_RAIL_EXT_ACTIVE
        self._command(P.CMD_SET_FLAGS, cmd_u=[bits])

    def set_stroke(self, d_star: float, psi_star: float) -> None:
        self._command(P.CMD_SET_STROKE, cmd_f=[d_star, psi_star])

    def plan_scan_stroke(self, y_center_m, amplitude_m, q_rad=None) -> tuple[float, float]:
        q = self.ctrl.q_cmd if q_rad is None else q_rad
        ok = self._command(
            P.CMD_PLAN_STROKE,
            q_meas=q,
            cmd_f=[float(y_center_m), float(amplitude_m)],
            timeout_s=2.0,
        )
        out = self._stable_out_copy()
        if ok and out is not None:
            d = float(out["cmd_f"][0])
            psi = float(out["cmd_f"][1])
            if np.isfinite(d):
                return d, psi
        return float("nan"), float("nan")

    def set_rail_pose_target(self, y_rail_m) -> None:
        cmd_f = np.zeros(16)
        if y_rail_m is None:
            cmd_f[1] = 0.0
        else:
            cmd_f[0] = float(y_rail_m)
            cmd_f[1] = 1.0
        self._command(P.CMD_SET_RAIL_POSE_TARGET, cmd_f=cmd_f)

    def capture_rail_extension_ref(self) -> None:
        self._command(P.CMD_CAPTURE_RAIL_EXT_REF, q_meas=self.ctrl.q_cmd)

    def set_rail_extension_mode(self, mode: str) -> None:
        self._command(P.CMD_SET_RAIL_EXT_MODE, cmd_f=[1.0 if str(mode) == "pose_attract" else 0.0])

    def _sync_q(self, out=None) -> None:
        if self._out is None and out is None:
            return
        o = self._stable_out_copy() if out is None else out
        if o is None:
            return
        self.ctrl.q_cmd = np.asarray(o["q_cmd"], dtype=float).copy()
        self.ctrl.last_u_alloc = float(o["u_alloc"])
        self.ctrl.last_u_mid = float(o["u_mid"])
        self.ctrl.last_v_r_ref = float(o["v_r_ref"])
        self.ctrl.last_slack_norm = float(o["slack"])
        self.ctrl.last_sigma_min = float(o["sigma_min"])

    def step(self, v_cmd, stamp=None, *, q_meas=None, **kwargs) -> TrackerStatus:
        stale = False
        twist = np.asarray(v_cmd, dtype=float).reshape(-1).copy()
        if twist.size != 6:
            raise ValueError("v_cmd must be a 6-vector")
        if stamp is not None and np.isfinite(float(stamp)):
            age = time.monotonic() - float(stamp)
            if age > float(self.cfg.feedback_timeout_s):
                stale = True
                twist[:] = 0.0
        if not getattr(self.ctrl, "_enabled", True):
            stale = True
            twist[:] = 0.0
        inner = self.update(twist, q_meas=q_meas, command_stale=stale, **kwargs)
        return TrackerStatus(
            v_cmd_received=np.asarray(inner.v_cmd_received, dtype=float).copy(),
            v_cmd_feasible=np.asarray(inner.v_cmd_feasible, dtype=float).copy(),
            v_tcp_estimated=np.asarray(inner.v_tcp_estimated, dtype=float).copy(),
            task_residual=np.asarray(inner.protected_residual, dtype=float).copy(),
            slack_norm=float(inner.slack_norm),
            joint_limited=bool(inner.joint_limited),
            rail_limited=bool(inner.rail_limited),
            wall_active=bool(inner.wall_active),
            secondary_suppressed=bool(inner.secondary_suppressed),
            command_stale=bool(stale or inner.command_stale),
            step=inner,
        )

    def update(self, twist, dt=None, q_meas=None, qdot_ff=None, **kwargs) -> JointIkStep:
        if q_meas is None:
            raise ValueError("q_meas is required for every Cartesian QPIK tick")
        rec = self._in[0]
        base = self._begin_write(rec)
        rec["magic"] = P.WBC_MAGIC
        rec["version"] = P.WBC_VERSION
        rec["cmd"] = P.CMD_STEP
        rec["t_mono"] = time.monotonic()
        rec["dt_nom"] = float(self.cfg.dt if dt is None else dt)
        dt_wall = kwargs.get("dt_wall_s")
        rec["dt_wall"] = float(dt_wall) if dt_wall is not None else float(rec["dt_nom"])
        rec["v_cmd"][:] = np.asarray(twist, dtype=float).reshape(6)
        rec["q_meas"][:] = np.asarray(q_meas, dtype=float).reshape(8)
        rec["rail_q"] = float(np.asarray(q_meas, dtype=float).reshape(-1)[0])
        rec["cmd_f"][:] = 0.0
        # Clear all optional feed-forward ownership before setting this tick's
        # flags/values.  A stale producer can therefore never inherit a prior
        # qdot_ff or twist merely because the field remained in SHM.
        rec["qdot_ff"][:] = 0.0
        rec["pose_d"][:] = 0.0
        rec["vel_ff"][:] = 0.0
        rec["path_twist"][:] = 0.0
        rec["feedback_twist"][:] = 0.0
        posture_d = kwargs.get("posture_d")
        posture_psi = kwargs.get("posture_psi")
        posture_q = kwargs.get("posture_q", kwargs.get("q_star"))
        flags = 0
        if kwargs.get("contact_active"):
            flags |= P.IN_CONTACT
        if kwargs.get("command_stale"):
            flags |= P.IN_STALE
        if kwargs.get("seed_q_cmd"):
            flags |= P.IN_SEED_QCMD
        rail_v = kwargs.get("rail_exec_vel_m_s")
        if rail_v is not None and np.isfinite(float(rail_v)):
            rec["rail_v"] = float(rail_v)
            flags |= P.IN_HAS_RAIL_V
        vfz = kwargs.get("v_force_z")
        if vfz is not None and np.isfinite(float(vfz)):
            rec["v_force_z"] = float(vfz)
            flags |= P.IN_HAS_V_FORCE
        if qdot_ff is not None and not bool(kwargs.get("command_stale")):
            rec["qdot_ff"][:] = np.asarray(qdot_ff, dtype=float).reshape(-1)[:8]
            flags |= P.IN_HAS_QDOT_FF
        pose_d = kwargs.get("pose_d")
        if pose_d is not None:
            rec["pose_d"][:] = np.asarray(pose_d, dtype=float).reshape(-1)[:6]
            flags |= P.IN_HAS_POSE_D
        vel_ff = kwargs.get("vel_ff")
        if vel_ff is not None:
            rec["vel_ff"][:] = np.asarray(vel_ff, dtype=float).reshape(-1)[:6]
            flags |= P.IN_HAS_VEL_FF
        path_twist = kwargs.get("path_twist")
        if path_twist is not None:
            rec["path_twist"][:] = np.asarray(path_twist, dtype=float).reshape(6)
            flags |= P.IN_HAS_PATH_TWIST
        feedback_twist = kwargs.get("feedback_twist")
        if feedback_twist is not None:
            rec["feedback_twist"][:] = np.asarray(feedback_twist, dtype=float).reshape(6)
            flags |= P.IN_HAS_FEEDBACK_TWIST
        if posture_d is not None or posture_psi is not None:
            rec["cmd_f"][0] = float(
                posture_d if posture_d is not None else float("nan")
            )
            rec["cmd_f"][1] = float(
                posture_psi if posture_psi is not None else float("nan")
            )
            flags |= P.IN_HAS_POSTURE
        if posture_q is not None:
            q_star = np.asarray(posture_q, dtype=float).reshape(-1)
            if q_star.size != 8 or not np.all(np.isfinite(q_star)):
                raise ValueError("posture_q/q_star must be a finite 8-vector")
            rec["cmd_f"][3:11] = q_star
            flags |= P.IN_HAS_QSTAR
        rec["flags"] = np.uint32(flags)
        seq = self._next_seq()
        rec["cmd_seq"] = np.uint64(seq)
        rec["seq"] = np.uint64(seq)
        self._end_write(rec, base)
        ok = self._wait_seq(seq)
        if not ok:
            # Do not issue a second STEP while the old request may still be
            # executing.  Never classify from the last stable OUT record.
            self.ctrl.q_cmd = np.asarray(self.ctrl.q_cmd, dtype=float).copy()
            try:
                self._command(P.CMD_STOP, wait=False)
            except Exception:
                pass
            o = failed_timeout_out()
        else:
            o = self._stable_out_copy()
        if o is None:
            o = failed_timeout_out()
        if ok:
            self._sync_q(o)
        q_cmd = np.asarray(o["q_cmd"], dtype=float).copy() if ok else np.asarray(self.ctrl.q_cmd, dtype=float).copy()
        qdot = np.asarray(o["qdot"], dtype=float).copy() if ok else np.zeros(8, dtype=float)
        v_recv = np.asarray(o["v_cmd_received"], dtype=float).copy()
        v_feas = np.asarray(o["v_cmd_feasible"], dtype=float).copy()
        v_tcp = np.asarray(o["v_tcp_estimated"], dtype=float).copy()
        resid = np.asarray(o["task_residual"], dtype=float).copy()
        stale = (not ok) or bool(int(o["flags"]) & P.OUT_STALE) or bool(kwargs.get("command_stale"))
        native_failed = (
            (not ok)
            or int(o["status"]) == P.STATUS_FAIL
            or bool(int(o["flags"]) & P.OUT_FAILED)
        )
        failure_code = int(o["failure_code"])
        step = JointIkStep(
            q_send=q_cmd,
            qdot=qdot,
            twist_base=v_recv,
            sigma_min=float(o["sigma_min"]),
            manip=float("nan"),
            slack_norm=float(o["slack"]),
            n_cbf_active=0,
            follow_err_rad=0.0,
            qp_backend="native",
            qp_solver_status=(
                "timeout" if not ok else ("failed" if native_failed else "ok")
            ),
            qp_solver_solve_ms=float(o["solve_ms"]),
            qp_solver_call_count=int(self._step_count),
            failure_code=failure_code,
            u_alloc=float(o["u_alloc"]),
            u_mid=float(o["u_mid"]),
            v_r_ref=float(o["v_r_ref"]),
            d_star_m=float(o["d_star"]),
            d_pref_m=float(o["d_pref"]),
            psi_deg=float(np.degrees(o["psi"])) if np.isfinite(float(o["psi"])) else float("nan"),
            sigma_arm=float(o["sigma_arm"]),
            v_cmd_received=v_recv,
            v_cmd_feasible=v_feas,
            v_tcp_estimated=v_tcp,
            protected_residual=resid,
            e_qp=resid,
            e_qp_norm=float(o["e_qp"]),
            command_stale=bool(stale),
            joint_limited=bool(int(o["joint_limited"])),
            rail_limited=bool(int(o["rail_limited"])),
            wall_active=bool(int(o["wall_active"])),
            secondary_suppressed=bool(int(o["secondary_suppressed"])),
            controller_mode=(
                "direct_joint_ptp"
                if bool(getattr(self.ctrl, "_direct_joint_ptp", False))
                else "qpik"
            ),
            nullspace_norm=float(o["ns_norm"]),
            nullspace_centering_norm=float(o["ns_centering"]),
            nullspace_manip_norm=float(o["ns_manip"]),
            nullspace_arm_angle_norm=float(o["ns_arm_angle"]),
            nullspace_damping_norm=float(o["ns_damping"]),
            nullspace_rail_lock_norm=float(o["ns_rail_lock"]),
            sat_scale=float(o["sat_scale"]),
            sec_target_norm=float(o["sec_target_norm"]),
            homotopy_s=float(o["homotopy_s"]),
            psi_star_deg=(
                float(np.degrees(o["psi_star"]))
                if np.isfinite(float(o["psi_star"]))
                else float("nan")
            ),
            rail_motion_share=float(o["rail_motion_share"]),
            u_task_raw=float(o["u_task_raw"]),
            u_task_feasible=float(o["u_task_feasible"]),
            u_pi_raw=float(o["u_pi_raw"]),
            u_mid_cmd=float(o["u_mid_cmd"]),
            u_post_raw=float(o["u_post_raw"]),
            u_post_feasible=float(o["u_post_feasible"]),
            u_mid_applied=float(o["u_mid_applied"]),
            d_star_dot_cmd=float(o["d_star_dot_cmd"]),
            u_escape_raw=float(o["u_escape_raw"]),
            u_escape_feasible=float(o["u_escape_feasible"]),
            escape_active=float(o["escape_active"]),
            escape_dir=float(o["escape_dir"]),
            u_base=float(o["u_base"]),
            u_feasible=float(o["u_feasible"]),
            v_r_lpf=float(o["v_r_lpf"]),
            e_d=float(o["e_d"]),
            V_d_proxy=float(o["V_d_proxy"]),
            j4_design_slack=float(o["j4_design_slack"]),
            qpik_dexterity_slack=float(o["sigma_slack"]),
            rail_box_lo=float(o["rail_box_lo"]),
            rail_box_hi=float(o["rail_box_hi"]),
            rail_bind_lo=int(o["rail_bind_lo"]),
            rail_bind_hi=int(o["rail_bind_hi"]),
            rail_task_vel_used=float(o["rail_task_vel_used"]),
            # Native protocol v5 does not carry the legacy rail-extension
            # reach/FF diagnostics; expose finite neutral values rather than
            # leaking dataclass NaNs into CSV/replay consumers.
            v_reach=0.0,
            v_ff_rail=0.0,
            rail_h1=float(o["rail_h1"]),
            rail_h2=float(o["rail_h2"]),
            rail_qdot_prev=float(o["rail_qdot_prev"]),
            rail_qdot_prev2=float(o["rail_qdot_prev2"]),
        )
        fallback_u = int(o["fallback_level"])
        level, reason, latched = classify_native_fallback(
            fallback_u, failure_code, native_failed=native_failed
        )
        if level != "none":
            step.fallback_level = level
            step.fallback_reason = reason
            step.solver_fault_latched = latched
        step.qp1_status = _QP_STATUS_NAMES.get(int(o["qp1_status"]), "unknown")
        step.qp2_status = _QP_STATUS_NAMES.get(int(o["qp2_status"]), "unknown")
        step.qp2_fallback = fallback_u == P.FALLBACK_QP1
        step.posture_gate_scale = float(o["posture_gate"])
        step.posture_gate_active = step.posture_gate_scale >= 0.999
        step.qpik_hard_residual_max = float(o["final_hard_violation"])
        step.final_hard_violation = float(o["final_hard_violation"])
        step.task_lock_violation = float(o["task_lock_violation"])
        step.final_box_violation = float(o["final_box_violation"])
        step.qp_solver_overrun = bool(int(o["qp_overrun"]))
        self.ctrl.last_secondary_norm = float(step.nullspace_norm)
        self.ctrl.last_sat_scale = float(step.sat_scale)
        if hasattr(self.ctrl, "core") and self.ctrl.core is not None:
            self.ctrl.core.last_dexterity_slack = float(o["sigma_slack"])
            self.ctrl.core.last_rail_box_lo = float(o["rail_box_lo"])
            self.ctrl.core.last_rail_box_hi = float(o["rail_box_hi"])
            self.ctrl.core.last_rail_bind_lo = int(o["rail_bind_lo"])
            self.ctrl.core.last_rail_bind_hi = int(o["rail_bind_hi"])
            self.ctrl.core.last_rail_task_vel_used = float(o["rail_task_vel_used"])
            self.ctrl.core.last_rail_h1 = float(o["rail_h1"])
            self.ctrl.core.last_rail_h2 = float(o["rail_h2"])
            self.ctrl.core.last_rail_qdot_prev = float(o["rail_qdot_prev"])
            self.ctrl.core.last_rail_qdot_prev2 = float(o["rail_qdot_prev2"])
        if self.ctrl.rail_ext_task is not None and np.isfinite(float(o["d_pref"])):
            self.ctrl.rail_ext_task.d_pref_m = float(o["d_pref"])
        if self.ctrl.posture_retarget is not None:
            if np.isfinite(float(o["d_star"])):
                self.ctrl.posture_retarget.d_star_m = float(o["d_star"])
                self.ctrl.posture_retarget._d_star = float(o["d_star"])
            if np.isfinite(float(o["psi_star"])):
                self.ctrl.posture_retarget.psi_star_rad = float(o["psi_star"])
            if np.isfinite(float(o["homotopy_s"])):
                self.ctrl.posture_retarget.homotopy_s = float(o["homotopy_s"])
        return step
