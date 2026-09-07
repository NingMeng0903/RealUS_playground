"""Launch and talk to the wbc_rt process over named shared memory."""

from __future__ import annotations

import os
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import weakref
from pathlib import Path

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkStep,
    TrackerStatus,
    isolate_native_process,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.wbc_rt.config_dump import dump_wbc_config
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.build_id import (
    assert_native_matches_tree,
    combined_hash,
)
from rm75_control.control.joint_admittance_8dof.qp_cert import qp_status_name


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

    @staticmethod
    def _rail_commit_authority(output, box_lo, box_hi) -> float:
        from ..tasks.rail_command import committed_rail_contributions
        brake = float(np.clip(0.0, box_lo[0], box_hi[0]))
        return committed_rail_contributions(
            float(output["u_feasible"]), float(output["rail_base_shaped"]),
            float(output["rail_total_committed"]), brake,
        )[2]

    def __init__(self, controller, *, timeout_s: float = 0.020) -> None:
        # The controller owns this client; a strong back-reference keeps both
        # Python QP bindings alive until interpreter teardown.
        self.ctrl = weakref.proxy(controller)
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
        self._notify: socket.socket | None = None
        self._transaction_lock = threading.RLock()
        self._cfg_path: Path | None = None
        self._seq = 0
        self._started = False
        self._pending_commit_seq = 0
        self._abort_next = False
        self._fault_latched = False
        self._published_q_cmd = None
        self._published_qdot = None
        self._last_wait_s = float("nan")
        self._last_reply_seq = 0
        self._last_completed_solve_ms = float("nan")
        self._last_completed_qp1_ms = float("nan")
        self._last_completed_qp2_ms = float("nan")
        self._last_completed_assembly_ms = float("nan")
        self._last_completed_n_cbf = 0
        self._last_wait_reason = ""
        self._timeout_streak = 0
        self._inflight_seq = 0
        self._inflight_t0 = 0.0
        # One missed 20-ms wakeup can hold. Bound the age of that request,
        # not the number of subsequent 5-ms polls of the same SHM slot.
        self._inflight_limit_s = 0.050
        self._soft_miss_seq = 0
        self._coast_warn_age_s = -1.0

    def start(self) -> None:
        try:
            self._start()
        except BaseException:
            self.shutdown()
            raise

    def _start(self) -> None:
        binary = find_wbc_rt_binary(getattr(self.cfg, "native_bin", None))
        if binary is None:
            raise FileNotFoundError(
                "wbc_rt binary not found; build native/wbc_rt or set WBC_RT_BIN"
            )
        try:
            hashed = subprocess.check_output([str(binary), "--hash"], text=True).strip()
        except Exception as exc:
            raise RuntimeError(f"wbc_rt --hash failed: {exc}") from exc
        assert_native_matches_tree(hashed)
        _ = combined_hash()
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
        env = os.environ.copy()
        cmeel = env.get(
            "CMEEL_PREFIX",
            "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix",
        )
        lib = str(Path(cmeel) / "lib")
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        self._notify, child_notify = socket.socketpair()
        self._notify.setblocking(False)
        try:
            self._proc = subprocess.Popen([
                str(binary),
                "--config",
                str(self._cfg_path),
                "--in",
                self.in_name,
                "--out",
                self.out_name,
                "--notify-fd",
                str(child_notify.fileno()),
            ], env=env, pass_fds=(child_notify.fileno(),),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            child_notify.close()
        if self._proc is not None and self._proc.pid:
            isolate_native_process(
                self._proc.pid,
                cpu=getattr(self.cfg, "native_cpu", None),
                control_cpu=getattr(self.cfg, "control_cpu", None),
            )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"wbc_rt exited during start (code {self._proc.returncode})"
                )
            if int(self._out["status"][0]) == P.STATUS_READY:
                out_magic = int(self._out["magic"][0])
                out_ver = int(self._out["version"][0])
                if out_magic != P.WBC_MAGIC or out_ver != P.WBC_VERSION:
                    self.shutdown()
                    raise RuntimeError(
                        f"wbc_rt protocol mismatch: native magic/version "
                        f"{out_magic}/{out_ver} vs client "
                        f"{P.WBC_MAGIC}/{P.WBC_VERSION} "
                        f"(in={P.WBC_IN_SIZE} out={P.WBC_OUT_SIZE})"
                    )
                self._started = True
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
                self._proc.wait(timeout=1.0)
            self._proc = None
        if self._notify is not None:
            self._notify.close()
            self._notify = None
        self._in = None
        self._out = None
        close_named_shm(self._shm_in)
        close_named_shm(self._shm_out)
        self._shm_in = None
        self._shm_out = None
        self._started = False
        if self._cfg_path is not None:
            shutil.rmtree(self._cfg_path.parent, ignore_errors=True)
            self._cfg_path = None

    def abort_pending(self) -> None:
        self._abort_next = True
        self._pending_commit_seq = 0

    def _wait_seq(self, seq: int, *, timeout_s: float | None = None) -> bool:
        t0 = time.monotonic()
        budget = float(self.timeout_s if timeout_s is None else timeout_s)
        limit = t0 + budget
        target = int(seq)
        self._last_wait_reason = "deadline"
        while True:
            if int(self._out["seq"][0]) == target:
                self._last_wait_s = time.monotonic() - t0
                self._last_wait_reason = ""
                self._last_reply_seq = target
                return True
            if self._proc is not None and self._proc.poll() is not None:
                self._last_wait_s = time.monotonic() - t0
                self._last_wait_reason = "process_exit"
                return False
            remaining = limit - time.monotonic()
            if remaining <= 0.0:
                break
            notifier = getattr(self, "_notify", None)
            if notifier is not None:
                # Readiness wakes immediately; unlike sleep polling it does
                # not wait for timer slack, and unlike spinning it releases
                # the GIL and CPU while the native worker runs.
                ready, _, _ = select.select([notifier], [], [], remaining)
                if ready and not notifier.recv(4096):
                    self._last_wait_s = time.monotonic() - t0
                    self._last_wait_reason = "process_exit"
                    return False
            else:
                # SHM-only test/legacy peers. Production start always creates
                # the notification channel.
                time.sleep(min(0.0002, remaining))
        if int(self._out["seq"][0]) == target:
            self._last_wait_s = time.monotonic() - t0
            self._last_wait_reason = ""
            self._last_reply_seq = target
            return True
        self._last_wait_s = time.monotonic() - t0
        return False

    def _notify_request(self) -> None:
        if self._notify is not None:
            try:
                self._notify.send(b"\x01")
            except BlockingIOError:
                # A full socket already contains a wakeup; SHM is authoritative.
                pass

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
        # Setters can be called from the service/UI while an RT tick waits.
        # The single-slot SHM protocol permits exactly one publisher at a time.
        with self._transaction_lock:
            return self._command_locked(cmd, cmd_f=cmd_f, cmd_u=cmd_u,
                                        q_meas=q_meas, wait=wait, timeout_s=timeout_s)

    def _command_locked(self, cmd, *, cmd_f=None, cmd_u=None, q_meas=None,
                        wait=True, timeout_s=None) -> bool:
        if self._inflight_seq:
            if int(self._out["seq"][0]) != int(self._inflight_seq):
                # A setter must not overwrite the single STEP slot while native
                # is still solving.  Wait out the in-flight reply, then proceed.
                if not self._wait_seq(
                    self._inflight_seq,
                    timeout_s=0.5 if timeout_s is None else float(timeout_s),
                ):
                    self._fault_latched = True
                    return False
            self._inflight_seq = 0
        rec = self._in[0]
        rec["seq"] = np.uint64(0)
        rec["cmd"] = np.uint32(cmd)
        rec["magic"] = P.WBC_MAGIC
        rec["version"] = P.WBC_VERSION
        rec["cmd_f"][:] = 0.0
        rec["cmd_u"][:] = 0
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
        rec["cmd_seq"] = np.uint64(seq)
        rec["seq"] = np.uint64(seq)
        try:
            self._notify_request()
        except OSError:
            self._last_wait_reason = "process_exit"
            self._last_wait_s = 0.0
            self._fault_latched = True
            return False
        if not wait:
            return True
        ok = self._wait_seq(seq, timeout_s=timeout_s if timeout_s is not None else 0.5)
        if not ok:
            self._fault_latched = True
        return ok

    def enable(self) -> None:
        self._command(P.CMD_ENABLE)

    def stop(self) -> None:
        self._command(P.CMD_STOP)

    def reset(self, q0) -> None:
        if not self._command(P.CMD_RESET, q_meas=q0, cmd_f=np.asarray(q0, dtype=float)):
            raise TimeoutError("wbc_rt reset was not acknowledged")
        self._fault_latched = False
        self._timeout_streak = 0
        self._pending_commit_seq = 0
        self._abort_next = False
        self._inflight_seq = 0
        self._inflight_t0 = 0.0
        self._soft_miss_seq = 0
        self._coast_warn_age_s = -1.0
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
        if ok:
            d = float(self._out["cmd_f"][0][0])
            psi = float(self._out["cmd_f"][0][1])
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

    def _sync_q(self) -> None:
        if self._out is None:
            return
        self.ctrl.q_cmd = np.asarray(self._out["q_cmd"][0], dtype=float).copy()
        self.ctrl.last_u_alloc = float(self._out["u_alloc"][0])
        self.ctrl.last_u_mid = float(self._out["u_mid"][0])
        self.ctrl.last_v_r_ref = float(self._out["v_r_ref"][0])
        self.ctrl.last_slack_norm = float(self._out["slack"][0])
        self.ctrl.last_sigma_min = float(self._out["sigma_min"][0])

    def _hold_step(self, twist, *, fallback_level: str, fallback_reason: str,
                   latched: bool) -> JointIkStep:
        return JointIkStep(
            q_send=np.asarray(self.ctrl.q_cmd, dtype=float).copy(),
            qdot=np.zeros(8), twist_base=np.asarray(twist, dtype=float).copy(),
            sigma_min=float("nan"), manip=float("nan"), slack_norm=float("nan"),
            n_cbf_active=0, follow_err_rad=float("nan"), qp_backend="native",
            qp_solver_status="timeout", qp_solver_solve_ms=float("nan"),
            fallback_level=fallback_level, fallback_reason=fallback_reason,
            solver_fault_latched=latched, command_stale=True,
            v_cmd_received=np.asarray(twist, dtype=float).copy(),
            native_roundtrip_ms=self._last_wait_s * 1000.0,
        )

    def _timeout_step(self, twist) -> JointIkStep:
        # A deadline miss leaves the shared reply either old or being written.
        # Never present that memory as a certified command (or current timing).
        self._fault_latched = True
        self._pending_commit_seq = 0
        self._published_q_cmd = None
        self._published_qdot = None
        return self._hold_step(
            twist,
            fallback_level="stop",
            fallback_reason="native_timeout",
            latched=True,
        )

    def _coast_step(self, twist) -> JointIkStep:
        self._pending_commit_seq = 0
        self._published_q_cmd = None
        self._published_qdot = None
        return self._hold_step(
            twist,
            fallback_level="none",
            fallback_reason="native_timeout_coast",
            latched=False,
        )

    def _deadline_miss_step(self, twist) -> JointIkStep:
        if self._last_wait_reason == "process_exit":
            return self._timeout_step(twist)
        if self._proc is not None and self._proc.poll() is not None:
            self._last_wait_reason = "process_exit"
            return self._timeout_step(twist)
        age = (
            time.monotonic() - self._inflight_t0
            if self._inflight_seq
            else float(self._last_wait_s)
        )
        self._last_wait_s = age
        if age >= self._inflight_limit_s:
            self._last_wait_reason = "request_age"
            return self._timeout_step(twist)
        seq = int(self._inflight_seq)
        if seq and seq != int(self._soft_miss_seq):
            self._soft_miss_seq = seq
            self._timeout_streak = 1
        try:
            self._notify_request()
        except OSError:
            self._last_wait_reason = "process_exit"
            return self._timeout_step(twist)
        return self._coast_step(twist)

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
        with self._transaction_lock:
            return self._update_locked(twist, dt=dt, q_meas=q_meas,
                                       qdot_ff=qdot_ff, **kwargs)

    def _update_locked(self, twist, dt=None, q_meas=None, qdot_ff=None, **kwargs) -> JointIkStep:
        if q_meas is None:
            raise ValueError("q_meas is required for every Cartesian QPIK tick")
        if self._fault_latched:
            return self._timeout_step(twist)
        if self._inflight_seq:
            seq = int(self._inflight_seq)
            age = time.monotonic() - self._inflight_t0
            if age >= self._inflight_limit_s:
                self._last_wait_s = age
                self._last_wait_reason = "request_age"
                return self._timeout_step(twist)
            if int(self._out["seq"][0]) == seq:
                self._last_wait_s = time.monotonic() - self._inflight_t0
                self._last_wait_reason = ""
                self._last_reply_seq = seq
                return self._accept_ok_step(
                    twist, seq, q_meas=q_meas, qdot_ff=qdot_ff, **kwargs
                )
            self._last_wait_s = time.monotonic() - self._inflight_t0
            self._last_wait_reason = "deadline"
            return self._deadline_miss_step(twist)
        rec = self._in[0]
        rec["seq"] = np.uint64(0)
        rec["magic"] = P.WBC_MAGIC
        rec["version"] = P.WBC_VERSION
        rec["cmd"] = P.CMD_STEP
        rec["t_mono"] = time.monotonic()
        rec["dt_nom"] = float(self.cfg.dt if dt is None else dt)
        dt_wall = kwargs.get("dt_wall_s")
        rec["dt_wall"] = float(dt_wall) if dt_wall is not None else float(rec["dt_nom"])
        rec["v_cmd"][:] = np.asarray(twist, dtype=float).reshape(6)
        rec["q_meas"][:] = np.asarray(q_meas, dtype=float).reshape(8)
        rec["rail_refresh_dt"] = float(kwargs.get("rail_refresh_dt_s") or
                                             self.cfg.rail_refresh_dt_s)
        rec["rail_q"] = float(np.asarray(q_meas, dtype=float).reshape(-1)[0])
        rec["cmd_f"][:] = 0.0
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
        if qdot_ff is not None:
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
        auto_commit = bool(kwargs.get("auto_commit", True))
        commit_prev = kwargs.get("commit_prev")
        abort_prev = bool(kwargs.get("abort_prev", False)) or bool(self._abort_next)
        self._abort_next = False
        if auto_commit:
            flags |= P.IN_AUTO_COMMIT
        if abort_prev:
            flags |= P.IN_ABORT_PREV
            self._pending_commit_seq = 0
        elif commit_prev is not False:
            flags |= P.IN_COMMIT_PREV
        rec["cmd_u"][0] = np.uint32(int(commit_prev or self._pending_commit_seq or 0))
        rec["flags"] = np.uint32(flags)
        seq = self._next_seq()
        rec["cmd_seq"] = np.uint64(seq)
        rec["seq"] = np.uint64(seq)
        try:
            self._notify_request()
        except OSError:
            self._last_wait_reason = "process_exit"
            self._last_wait_s = 0.0
            return self._timeout_step(twist)
        self._inflight_seq = seq
        self._inflight_t0 = time.monotonic()
        ok = self._wait_seq(seq)
        if time.monotonic() - self._inflight_t0 >= self._inflight_limit_s:
            self._last_wait_s = time.monotonic() - self._inflight_t0
            self._last_wait_reason = "request_age"
            return self._timeout_step(twist)
        if not ok:
            return self._deadline_miss_step(twist)
        return self._accept_ok_step(
            twist, seq, q_meas=q_meas, qdot_ff=qdot_ff, **kwargs
        )

    def _accept_ok_step(self, twist, seq, *, q_meas, qdot_ff=None, **kwargs) -> JointIkStep:
        self._timeout_streak = 0
        self._inflight_seq = 0
        self._soft_miss_seq = 0
        self._coast_warn_age_s = -1.0
        o = self._out[0].copy()
        self._last_completed_solve_ms = float(o["solve_ms"])
        self._last_completed_qp1_ms = float(o["qp1_solve_ms"])
        self._last_completed_qp2_ms = float(o["qp2_solve_ms"])
        self._last_completed_assembly_ms = float(o["assembly_ms"])
        self._last_completed_n_cbf = int(o["n_cbf_active"])
        auto_commit = bool(kwargs.get("auto_commit", True))
        q_cmd = np.asarray(o["q_cmd"], dtype=float).copy()
        qdot = np.asarray(o["qdot"], dtype=float).copy()
        self._published_q_cmd = q_cmd.copy()
        self._published_qdot = qdot.copy()
        native_status_early = int(o["status"])
        if auto_commit:
            self._sync_q()
            if native_status_early == P.STATUS_OK:
                self._pending_commit_seq = 0
                record = getattr(self.ctrl, "_record_applied_qdot", None)
                if callable(record):
                    record(qdot)
        elif native_status_early == P.STATUS_OK:
            self._pending_commit_seq = seq
        v_recv = np.asarray(o["v_cmd_received"], dtype=float).copy()
        v_feas = np.asarray(o["v_cmd_feasible"], dtype=float).copy()
        v_tcp = np.asarray(o["v_tcp_estimated"], dtype=float).copy()
        resid = np.asarray(o["task_residual"], dtype=float).copy()
        stale = bool(int(o["flags"]) & P.OUT_STALE) or bool(kwargs.get("command_stale"))
        native_status = int(o["status"])
        qp1_name = qp_status_name(o["qp1_status"])
        qp2_name = qp_status_name(o["qp2_status"])
        if native_status == P.STATUS_FAIL:
            solver_status = "failed"
        elif qp2_name in ("solved", "max_iter"):
            solver_status = qp2_name
        elif qp1_name in ("solved", "max_iter", "failed"):
            solver_status = qp1_name
        elif native_status == P.STATUS_OK:
            solver_status = "ok"
        else:
            solver_status = "failed"
        box_lo = np.asarray(o["box_lo"], dtype=float).copy()
        box_hi = np.asarray(o["box_hi"], dtype=float).copy()
        qdot_prev_used = np.asarray(o["qdot_prev"], dtype=float).copy()
        qdot_prev2_used = np.asarray(o["qdot_prev2"], dtype=float).copy()
        step = JointIkStep(
            q_send=q_cmd,
            qdot=qdot,
            twist_base=v_recv,
            sigma_min=float(o["sigma_min"]),
            manip=float(o["sigma_min"]) if int(o["manip_active"]) else float("nan"),
            slack_norm=float(o["slack"]),
            n_cbf_active=int(o["n_cbf_active"]),
            follow_err_rad=float(o["follow_err_rad"]),
            qp_backend="native",
            qp_solver_status=solver_status,
            qp_solver_iterations=int(o["qp1_iter"]) + int(o["qp2_iter"]),
            qp_solver_call_count=(
                int(qp1_name != "not_run") + int(qp2_name != "not_run")
            ),
            qp_solver_solve_ms=float(o["solve_ms"]),
            qp_solver_overrun=bool(float(o["solve_ms"]) > 5.0),
            qp1_status=qp1_name,
            qp2_status=qp2_name,
            qp1_solve_ms=float(o["qp1_solve_ms"]),
            qp2_solve_ms=float(o["qp2_solve_ms"]),
            qp_assembly_ms=float(o["assembly_ms"]),
            qp_kinematics_ms=float(o["kinematics_ms"]),
            qp_collision_ms=float(o["collision_ms"]),
            qp_solve_phase_ms=float(o["qp_total_ms"]),
            native_dispatch_ms=float(o["ipc_wait_ms"]),
            native_roundtrip_ms=self._last_wait_s * 1000.0,
            native_transport_ms=max(0.0, self._last_wait_s * 1000.0 - float(o["solve_ms"])),
            qp_fallback_ms=float(o["fallback_ms"]),
            qpik_total_ms=float(o["solve_ms"]),
            qpik_hard_residual_max=float(o["hard_residual_max"]),
            qpik_equality_residual_max=float(o["equality_residual_max"]),
            qp2_fallback=qp2_name == "failed",
            qdot_raw=qdot.copy(),
            qdot_pre_commit=qdot.copy(),
            qdot_committed=qdot.copy(),
            box_degenerate=bool(int(o["box_degenerate"])),
            box_infeasible=bool(int(o["box_infeasible"])),
            box_excess_max=float(o["box_excess_max"]),
            manip_active=bool(int(o["manip_active"])),
            qdot_qp_vs_sent_max=float(o["qdot_qp_vs_sent_max"]),
            dual_cancel=float(o["dual_cancel"]),
            secondary_alpha=float(o["secondary_alpha"]),
            box_lo=box_lo,
            box_hi=box_hi,
            qdot_prev_used=qdot_prev_used,
            qdot_prev2_used=qdot_prev2_used,
            rail_exec_for_qp_m_s=float(o["rail_exec"]),
            u_alloc=float(o["u_alloc"]),
            u_mid=float(o["u_mid"]),
            v_r_ref=float(o["v_r_ref"]),
            d_star_m=float(o["d_star"]),
            d_pref_m=float(o["d_pref"]),
            psi_deg=float(np.degrees(o["psi"])) if np.isfinite(float(o["psi"])) else float("nan"),
            sigma_arm=float(o["sigma_arm"]),
            task_progress=float(o["task_progress_alpha"]),
            task_paused=bool(o["task_paused"]),
            task_pause_reason={0: "", 1: "task_infeasible", 2: "publication_infeasible"}.get(int(o["task_pause_reason"]), "native_pause"),
            fallback_level="stop" if native_status == P.STATUS_FAIL else "none",
            fallback_reason=(
                {0: "", 1: "task_infeasible", 2: "publication_infeasible"}.get(
                    int(o["task_pause_reason"]), "native_pause"
                )
            ),
            solver_fault_latched=bool(native_status == P.STATUS_FAIL),
            rail_base_shaped=float(o["rail_base_shaped"]),
            rail_base_raw=float(o["rail_base_raw"]),
            rail_base_committed=float(o["rail_total_committed"] - o["rail_post_committed"]),
            rail_commit_authority=self._rail_commit_authority(o, box_lo, box_hi),
            rail_post_committed=float(o["rail_post_committed"]),
            rail_total_committed=float(o["rail_total_committed"]),
            rail_pi_xi=float(o["rail_pi_xi"]),
            rail_d_ref=float(o["rail_d_ref"]),
            rail_ref_acceleration=float(o["rail_ref_acceleration"]),
            rail_preview_residual=float(o["rail_preview_residual"]),
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
            plan_drives_rail=bool(getattr(self.ctrl, "_plan_drives_rail", False)),
            rail_vel_pin=(
                float(np.asarray(qdot_ff, dtype=float).reshape(-1)[0])
                if qdot_ff is not None
                and (
                    bool(getattr(self.ctrl, "_plan_drives_rail", False))
                    or bool(getattr(self.ctrl, "_direct_joint_ptp", False))
                )
                else float("nan")
            ),
            rail_qdot_ff=(
                float(np.asarray(qdot_ff, dtype=float).reshape(-1)[0])
                if qdot_ff is not None
                else float("nan")
            ),
            rail_q_hat_m=(
                float(self.ctrl.rail_observer.q_hat)
                if getattr(self.ctrl.rail_observer, "_initialized", False)
                else float("nan")
            ),
            rail_goal_err_m=float(q_cmd[0]) - float(
                np.asarray(q_meas, dtype=float).reshape(-1)[0]
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
            rail_h1=float(o["rail_h1"]),
            rail_h2=float(o["rail_h2"]),
            rail_qdot_prev=float(o["rail_qdot_prev"]),
            rail_qdot_prev2=float(o["rail_qdot_prev2"]),
        )
        jac = self.ctrl.kin.jacobian(np.asarray(q_meas, dtype=float))
        rail_model = jac[:, 0] * float(o["rail_exec"])
        arm_model = np.asarray(o["v_task_actual"], dtype=float) - rail_model
        step.rail_model_twist = rail_model.copy()
        step.arm_model_twist = arm_model.copy()
        step.rail_xy_contribution = rail_model[:2].copy()
        step.arm_xy_contribution = arm_model[:2].copy()
        step.protected_target = v_recv.copy()
        step.protected_achieved = v_tcp.copy()
        step.qpik_working_slack = resid.copy()
        step.e_shape = v_recv - v_feas
        step.e_qp = v_feas - v_tcp
        step.e_exec = jac @ qdot - v_tcp
        step.e_shape_norm = float(np.linalg.norm(step.e_shape))
        step.e_qp_norm = float(np.linalg.norm(step.e_qp))
        step.e_exec_norm = float(np.linalg.norm(step.e_exec))
        self.ctrl.last_secondary_norm = float(step.nullspace_norm)
        self.ctrl.last_sat_scale = float(step.sat_scale)
        if hasattr(self.ctrl, "core") and self.ctrl.core is not None:
            self.ctrl.core.last_task_target = v_recv.copy()
            self.ctrl.core.last_task_achieved = v_tcp.copy()
            self.ctrl.core.last_task_residual = resid.copy()
            self.ctrl.core.last_rail_exec_contrib = rail_model.copy()
            self.ctrl.core.last_arm_contrib = arm_model.copy()
            self.ctrl.core.last_progress_scale = float(o["task_progress_alpha"])
            self.ctrl.core.last_preview_velocity = np.asarray(o["rail_preview_arm"], dtype=float)[1:].copy()
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
            if auto_commit:
                self.ctrl.core.qdot_prev = qdot.copy()
                self.ctrl.core.qdot_prev2 = qdot_prev_used.copy()
                self.ctrl.core._qdot_prev_seen = qdot_prev_used.copy()
            self.ctrl.core.last_lo_box = box_lo
            self.ctrl.core.last_hi_box = box_hi
            self.ctrl.core.last_qp1_status = qp1_name
            self.ctrl.core.last_qp2_status = qp2_name
            self.ctrl.core.last_qp1_solve_ms = float(o["qp1_solve_ms"])
            self.ctrl.core.last_qp2_solve_ms = float(o["qp2_solve_ms"])
            self.ctrl.core.last_qp1_iter = int(o["qp1_iter"])
            self.ctrl.core.last_qp2_iter = int(o["qp2_iter"])
            self.ctrl.core.last_qp2_fallback = qp2_name == "failed"
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
