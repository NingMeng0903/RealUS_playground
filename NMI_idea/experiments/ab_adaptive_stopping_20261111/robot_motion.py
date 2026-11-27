"""Native ProDMP/dIRD motion adapter for the closed-loop B experiment.

The adapter keeps the simulator's anatomy and image renderer out of the
motion layer.  A caller supplies a current measured contact frame and a
local ProDMP goal.  The goal is converted to a TCP pose, screened by the
audited dIRD capability evaluator, then executed by the existing deterministic
native QP replay.  Each execution returns the complete controller exit state
so the next continuation starts from the measured ``q, qdot`` history.

This module is an experiment adapter.  It opens no robot connection and does
not claim an IK endpoint or a dIRD score is a complete path certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import time
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
OBS = HERE.parent / "ab_observation_stress_20261111"
CAPABILITY_ROOT = HERE.parent / "ab_module_ladder_20261110" / "capability"
REPLAY_ROOT = HERE.parent / "ab_validation_20261110" / "repair_20261110" / "qp_cross_track_fix"
REPLAY_DIRD = HERE.parent / "ab_validation_20261110" / "repair_20261110" / "dird"
SELECTED = HERE.parent / "ab_module_ladder_20261110" / "planning" / "native_regression_20261110" / "public_prior_layout" / "SELECTED.json"
sys.path.insert(0, str(OBS))
sys.path.insert(0, str(CAPABILITY_ROOT))
sys.path.insert(0, str(REPLAY_ROOT))
sys.path.insert(0, str(REPLAY_DIRD))

from motion import MotionReference, ProDMPReference, duration_matrix, reference as prodmp_reference  # noqa: E402
from capability import CapabilityEvaluator  # noqa: E402
from scan_replay import ScanReplayBridge  # noqa: E402


class _SE3ProDMPReference:
    """Full SE(3) reference driven by one integral-basis scalar phase.

    The scalar phase is the existing rest-to-rest ProDMP ``0 -> 1``. Position
    follows the endpoint chord and rotation follows the shortest exponential
    map, so no measured contact-frame correction is projected away.
    """

    def __init__(self, start: np.ndarray, goal: np.ndarray, duration_s: float):
        self.start = _se3(start)
        self.goal = _se3(goal)
        self._duration = float(duration_s)
        if not np.isfinite(self._duration) or self._duration <= 0.:
            raise ValueError("duration_s must be finite and positive")
        self._phase = ProDMPReference([0., 0., 0.], [1., 1., 1.], np.eye(4),
                                      duration_s=self._duration)
        self._translation = self.goal[:3, 3] - self.start[:3, 3]
        self._rotation = self.start[:3, :3].T @ self.goal[:3, :3]
        self._omega = Rotation.from_matrix(self._rotation).as_rotvec()

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        del pose0, t_s

    def duration_s(self) -> float:
        return self._duration

    def sample(self, t_s: float) -> MotionReference:
        state = self._phase.local_state(float(t_s))
        u = float(state.position[0])
        u_dot = float(state.derivative[0] / self._duration)
        rotation = self.start[:3, :3] @ Rotation.from_rotvec(u * self._omega).as_matrix()
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = self.start[:3, 3] + u * self._translation
        pose = np.r_[pose_matrix[:3, 3], Rotation.from_matrix(rotation).as_euler("xyz")]
        linear = u_dot * self._translation
        angular = u_dot * (self.start[:3, :3] @ self._omega)
        return MotionReference(pose_d=pose, vel_ff=np.r_[linear, angular], t_ref=float(t_s))


class _ProDMPReplayBridge(ScanReplayBridge):
    """Replay bridge whose reference factory is the integral-basis ProDMP.

    The duration is encoded in the immutable event provenance.  QP replay
    restores complete histories by replaying those events, so decoding the
    duration from provenance preserves the exact reference during a fork.
    """

    def __init__(self, *, backend: str = "native"):
        # ScanReplayBridge's historical constructor hard-codes the native
        # backend.  Keep its profile/reference setup, then explicitly select
        # the caller's backend so the documented Python development fallback
        # is real rather than silently native.
        super().__init__()
        if backend not in ("native", "python"):
            raise ValueError("backend must be native or python")
        self.backend = backend
        self._prodmp_duration_s = None

    @staticmethod
    def _duration_from_provenance(provenance: str) -> float | None:
        marker = "prodmp_duration_s="
        if marker not in str(provenance):
            return None
        raw = str(provenance).split(marker, 1)[1].split(";", 1)[0]
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("invalid ProDMP duration in replay provenance") from exc
        if not np.isfinite(value) or value <= 0.:
            raise ValueError("ProDMP duration must be finite and positive")
        return value

    def _run_pose_path(self, *args, **kwargs):
        provenance = kwargs.get("provenance", "")
        old = self._prodmp_duration_s
        self._prodmp_duration_s = self._duration_from_provenance(provenance)
        try:
            result = super()._run_pose_path(*args, **kwargs)
            # The inherited bridge labels every path as a
            # ``WorldPoseSplineReference``.  This experiment actually
            # constructs the integral-basis ProDMP in ``make_pose_reference``.
            result["actual_reference_kind"] = "ProDMPIntegralBasisReference"
            result["reference_model"] = {
                "family": "ProDMP",
                "basis": "integral_basis",
                "space": "full_SE3",
                "duration_s": self._prodmp_duration_s,
                "source": "ab_adaptive_stopping_20261111.robot_motion._SE3ProDMPReference",
            }
            return result
        finally:
            self._prodmp_duration_s = old

    def make_pose_reference(self, tcp, speed_m_s, cfg):
        del speed_m_s
        tcp = np.asarray(tcp, dtype=float)
        if tcp.ndim != 3 or tcp.shape[0] < 2:
            raise ValueError("ProDMP replay requires at least two TCP poses")
        anchor = tcp[0].copy()
        relative = np.linalg.inv(anchor) @ tcp[-1]
        duration = self._prodmp_duration_s
        if duration is None:
            angle = float(Rotation.from_matrix(relative[:3, :3]).magnitude())
            duration = max(.1, np.linalg.norm(relative[:3, 3]) / .02, angle / .5)
        return _SE3ProDMPReference(anchor, tcp[-1], duration)


def _se3(value: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ValueError("pose must be finite with shape (4,4)")
    if not np.allclose(arr[3], [0., 0., 0., 1.], atol=1e-8):
        raise ValueError("pose has an invalid homogeneous row")
    if not np.allclose(arr[:3, :3].T @ arr[:3, :3], np.eye(3), atol=2e-7):
        raise ValueError("pose rotation is not orthonormal")
    if np.linalg.det(arr[:3, :3]) < 1. - 2e-7:
        raise ValueError("pose rotation is not proper")
    return arr.copy()


def _q8(value: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (8,) or not np.isfinite(arr).all():
        raise ValueError("robot state must be finite with shape (8,)")
    return arr.copy()


@dataclass(frozen=True)
class WindowConstraint:
    """Public geometric restriction for one synthetic acoustic window.

    ``xmin/xmax`` and ``ymin/ymax`` are evaluated in the material/contact
    frame.  ``blocked`` consists of axis-aligned forbidden rectangles in the
    same chart.  The callback is deliberately independent from hidden target
    anatomy and can therefore be used by every policy arm.
    """

    name: str
    xmin: float = -0.03
    xmax: float = 0.03
    ymin: float = -0.03
    ymax: float = 0.03
    blocked: tuple[tuple[float, float, float, float], ...] = ()

    def check_local(self, xy: Iterable[float]) -> tuple[bool, str]:
        p = np.asarray(xy, dtype=float).reshape(-1)
        if p.shape != (2,) or not np.isfinite(p).all():
            return False, "nonfinite_local_point"
        if not (self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax):
            return False, "outside_window"
        for x0, x1, y0, y1 in self.blocked:
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return False, "inside_obstacle"
        return True, "ok"

    def check_trace(self, trace: list[dict], anchor: np.ndarray) -> dict:
        """Check every actual TCP sample, including the entry sample."""
        anchor = _se3(anchor)
        inv = np.linalg.inv(anchor)
        violations = []
        checked = 0
        for index, row in enumerate(trace):
            pose = _se3(row["pose_actual"])
            local = inv @ pose
            checked += 1
            passed, reason = self.check_local(local[:2, 3])
            if not passed:
                violations.append({"index": index, "reason": reason, "local_xy": local[:2, 3].tolist()})
        return {"passed": not violations, "checked_samples": checked, "violations": violations,
                "constraint": self.name}


@dataclass(frozen=True)
class MotionState:
    """Complete native replay state required for a lawful continuation."""

    q: np.ndarray
    qdot: np.ndarray
    acceleration: np.ndarray
    snapshot: dict

    @classmethod
    def from_result(cls, result: dict) -> "MotionState":
        state = result.get("exit_state")
        if not isinstance(state, dict):
            raise ValueError("native result did not return a complete exit_state")
        return cls(_q8(state["q_exit"]), _q8(state["qdot_exit"]), _q8(state["acceleration_exit"]), state)


class NativeMotionEnvironment:
    """Connect measured contact frames, ProDMP paths and native QP replay."""

    def __init__(self, *, q_seed: np.ndarray | None = None, anchor: np.ndarray | None = None,
                 backend: str = "native"):
        payload = json.loads(SELECTED.read_text())
        self.q_seed = _q8(payload["q_seed"] if q_seed is None else q_seed)
        self.anchor = _se3(payload["T_world_body"] if anchor is None else anchor)
        if backend not in ("native", "python"):
            raise ValueError("backend must be native or python")
        self.capability = CapabilityEvaluator()
        self.backend = backend
        self.replay = _ProDMPReplayBridge(backend=backend)

    def close(self) -> None:
        close = getattr(self.replay, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "NativeMotionEnvironment":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def measured_goal_pose(self, contact_frame: np.ndarray, goal_local: Iterable[float]) -> np.ndarray:
        """Decode a local ``(x_m,y_m,yaw_rad)`` goal through the live frame."""
        frame = _se3(contact_frame)
        goal = np.asarray(goal_local, dtype=float).reshape(-1)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("goal_local must be finite (x_m,y_m,yaw_rad)")
        local = np.eye(4)
        # The third chart coordinate is signed tilt about the local x axis.
        # This is the acoustic-plane steering angle used by the fixture.
        local[:3, :3] = Rotation.from_rotvec([goal[2], 0., 0.]).as_matrix()
        local[:3, 3] = [goal[0], goal[1], 0.]
        return frame @ local

    def prodmp_path(self, start_local: Iterable[float], goal_local: Iterable[float],
                    contact_frame: np.ndarray, duration_s: float, *, samples: int = 31) -> np.ndarray:
        """Sample an integral-basis ProDMP in the current measured frame."""
        frame = _se3(contact_frame)
        start_pose = self.measured_goal_pose(frame, start_local)
        goal_pose = self.measured_goal_pose(frame, goal_local)
        ref = _SE3ProDMPReference(start_pose, goal_pose, float(duration_s))
        times = np.linspace(0., float(duration_s), max(2, int(samples)))
        return np.asarray([self._pose_from_reference(ref.sample(float(t)).pose_d) for t in times])

    @staticmethod
    def _pose_from_reference(pose6: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose6, dtype=float).reshape(-1)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("reference sample must have six finite pose values")
        out = np.eye(4)
        out[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        out[:3, 3] = pose[:3]
        return out

    def screen_dird(self, tcp_path: np.ndarray, q: np.ndarray) -> dict:
        """Return batched dIRD screening scores before execution."""
        path = np.asarray(tcp_path, dtype=float)
        if path.ndim != 3 or path.shape[1:] != (4, 4) or len(path) < 2:
            raise ValueError("tcp_path must have shape (N,4,4), N>=2")
        q = _q8(q)
        return self.capability.evaluate_path(path[[0, -1]], q, exact=False, native=False)

    def execute(self, tcp_path: np.ndarray, state: MotionState | None = None, *,
                speed_m_s: float = .02, settle_s: float = .25,
                constraint: WindowConstraint | None = None,
                constraint_frame: np.ndarray | None = None,
                prodmp_duration_s: float | None = None,
                label: str = "segment", return_trace: bool = True) -> dict:
        """Execute a real native replay from a complete prior exit state."""
        path = np.asarray(tcp_path, dtype=float)
        q = self.q_seed if state is None else _q8(state.q)
        entry_qdot = np.zeros(8) if state is None else _q8(state.qdot)
        entry_acceleration = np.zeros(8) if state is None else _q8(state.acceleration)
        started = time.perf_counter()
        if prodmp_duration_s is not None:
            duration = float(prodmp_duration_s)
            if not np.isfinite(duration) or duration <= 0.:
                raise ValueError("prodmp_duration_s must be finite and positive")
            provenance = f"closed-loop {label};prodmp_duration_s={duration:.17g}"
        else:
            provenance = f"closed-loop {label}"
        result = self.replay.query_pose_path(path, q, speed_m_s=float(speed_m_s),
            settle_s=float(settle_s), provenance=provenance, return_trace=return_trace,
            initial_state=None if state is None else state.snapshot, return_state=True)
        trace = result.get("trace", [])
        constraint_result = {"passed": True, "checked_samples": 0, "violations": []}
        if constraint is not None and trace:
            frame = self.anchor if constraint_frame is None else _se3(constraint_frame)
            constraint_result = constraint.check_trace(trace, frame)
        result["constraint"] = constraint_result
        result["entry_q"] = q.copy()
        result["entry_qdot"] = entry_qdot.copy()
        result["entry_acceleration"] = entry_acceleration.copy()
        result["history_continuation"] = bool(state is None or isinstance(state.snapshot, dict))
        result["wall_seconds_adapter"] = time.perf_counter() - started
        result["simulated_duration_s"] = result.get("integrated_duration_s", result.get("simulated_duration_s"))
        solver_clean = not result.get("failures") and int(result.get("task_paused_count", 0)) == 0
        tracking_clean = bool(result.get("motion_tracking_pass_2mm_2deg", False))
        result["whole_segment_pass"] = bool(result.get("exit_state_valid", False)
            and solver_clean and tracking_clean and constraint_result["passed"])
        # Keep a stable public label even when a backend returns an early
        # failure before the replay bridge can attach its result metadata.
        result.setdefault("actual_reference_kind", "ProDMPIntegralBasisReference")
        result.setdefault("reference_model", {
            "family": "ProDMP", "basis": "integral_basis", "space": "full_SE3",
            "duration_s": prodmp_duration_s,
        })
        result["dird_screen"] = self.screen_dird(path, q)
        return result

    def execute_local(self, start_local: Iterable[float], goal_local: Iterable[float],
                      contact_frame: np.ndarray, duration_s: float,
                      state: MotionState | None = None, *, settle_s: float = .25,
                      speed_m_s: float = .02, constraint: WindowConstraint | None = None,
                      label: str = "local_segment", return_trace: bool = True) -> dict:
        """Execute a local ProDMP from the current measured robot pose."""
        frame = _se3(contact_frame)
        q = self.q_seed if state is None else _q8(state.q)
        requested_start = self.measured_goal_pose(frame, start_local)
        target = self.measured_goal_pose(frame, goal_local)
        actual_start = self.capability.fk(q)
        requested_start_error = float(np.linalg.norm(actual_start[:3, 3] - requested_start[:3, 3]))
        path = np.stack((actual_start, target))
        result = self.execute(path, state, speed_m_s=speed_m_s, settle_s=settle_s,
            constraint=constraint, constraint_frame=frame,
            prodmp_duration_s=float(duration_s), label=label, return_trace=return_trace)
        result["prodmp"] = {
            "duration_s": float(duration_s), "requested_start_local": np.asarray(start_local, dtype=float),
            "goal_local": np.asarray(goal_local, dtype=float),
            "requested_start_pose_error_m": requested_start_error,
            "target_reference": "full measured contact-frame SE(3) pose",
            "rotation_axis": "contact-frame local x tilt",
            "reference_kind": "integral_basis_ProDMP",
            "actual_first_reference": "current measured FK pose",
        }
        return result


__all__ = ["WindowConstraint", "MotionState", "NativeMotionEnvironment"]
