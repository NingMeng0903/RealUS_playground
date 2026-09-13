"""Finite reference intervals for a publication-owned accepted clock.

This module has no controller, publisher, or wall clock. Call ``commit_time``
only after the complete command has been successfully published; rejected,
partial, or timed-out publications must retain their previous accepted time.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
from scipy.spatial.transform import Rotation

from rm75_control.control.admittance_common.reference import MotionReference


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class FiniteIntervalReference:
    """Adapt a finite pose reference without advancing its accepted clock.

    Public timestamps are in the caller's global reference clock. ``origin_s``
    anchors the bounded local interval [0, duration_s], including hot installs.
    Pose feedback is always p(s). Feedforward integrates the candidate interval
    h_ref=min(h_exec, T-s), divided by the *full* h_exec. Angular feedforward is
    the shortest finite rotation in the base/world frame, matching
    ``MotionReference`` and ``ForearmReference`` conventions (xyz Euler poses).

    ``dt_s`` is an optional default execution hold, not wall-clock elapsed time.
    A caller with variable holds supplies it to each candidate and commit.
    Reference exhaustion is geometric/timing evidence only; actual measured
    endpoint distance and mechanical stopping must be checked by the caller.
    """

    def __init__(self, reference, dt_s=None):
        self.reference = reference
        self.dt_s = None if dt_s is None else self._hold(dt_s)
        self.origin_s = 0.0
        self._source_origin_s = 0.0
        self._duration()  # Fail before control starts for unbounded references.

    def _duration(self):
        duration = _finite(self.reference.duration_s, "reference duration")
        if duration < 0.0:
            raise ValueError("reference duration must be nonnegative")
        return duration

    @property
    def duration_s(self):
        return self._duration()

    def _hold(self, dt_s=None):
        if dt_s is None:
            dt_s = self.dt_s
        if dt_s is None:
            raise ValueError("execution hold dt_s is required")
        hold = _finite(dt_s, "execution hold")
        if hold <= 0.0:
            raise ValueError("execution hold must be positive")
        return hold

    def set_origin(self, pose0, *, t_s=0.0):
        origin = 0.0 if t_s is None else _finite(t_s, "reference origin")
        _finite(origin + self.duration_s, "reference end timestamp")
        hook = getattr(self.reference, "set_origin", None)
        source_origin = 0.0
        if hook is not None:
            parameters = inspect.signature(hook).parameters
            if "t_s" in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            ):
                hook(pose0, t_s=origin)
                source_origin = origin
            else:
                hook(pose0)
        self._duration()
        self.origin_s = origin
        self._source_origin_s = source_origin

    def local_time(self, t_s):
        accepted = _finite(t_s, "accepted time")
        if accepted >= self.origin_s + self.duration_s:
            return self.duration_s
        return float(np.clip(accepted - self.origin_s, 0.0, self.duration_s))

    def _sample_local(self, local_s):
        return self.reference.sample(self._source_origin_s + local_s)

    def candidate_duration(self, t_s, dt_s=None):
        return min(self._hold(dt_s), max(0.0, self.duration_s - self.local_time(t_s)))

    def candidate(self, t_s, dt_s=None):
        """Return (current-pose MotionReference, h_ref); mutate no clock."""
        hold = self._hold(dt_s)
        start = self.local_time(t_s)
        interval = min(hold, max(0.0, self.duration_s - start))
        first = self._sample_local(start)
        last = self._sample_local(min(self.duration_s, start + interval))
        pose0 = np.asarray(first.pose_d, dtype=float).reshape(6)
        pose1 = np.asarray(last.pose_d, dtype=float).reshape(6)
        if not np.isfinite(pose0).all() or not np.isfinite(pose1).all():
            raise ValueError("reference poses must be finite")
        velocity = np.zeros(6)
        if interval > 0.0:
            velocity[:3] = (pose1[:3] - pose0[:3]) / hold
            if not np.array_equal(pose0[3:], pose1[3:]):
                first_rotation = Rotation.from_euler("xyz", pose0[3:])
                last_rotation = Rotation.from_euler("xyz", pose1[3:])
                velocity[3:] = (last_rotation * first_rotation.inv()).as_rotvec() / hold
        if not np.isfinite(velocity).all():
            raise ValueError("reference finite-interval velocity must be finite")
        return MotionReference(pose0.copy(), velocity, self.origin_s + start,
                               valid=bool(first.valid and last.valid)), interval

    def sample(self, t_s, dt_s=None):
        return self.candidate(t_s, dt_s)[0]

    def commit_time(self, t_s, alpha, dt_s=None):
        """Compute the successful-publication clock, never commit implicitly.

        Only positive accepted advancement can snap the last few clock ULPs.
        Alpha zero preserves the bounded clock even when geometry is exhausted.
        """
        accepted = float(np.clip(_finite(alpha, "accepted alpha"), 0.0, 1.0))
        start = self.local_time(t_s)
        interval = self.candidate_duration(t_s, dt_s)
        advance = accepted * interval
        if advance == 0.0:
            return self.origin_s + start
        advanced = min(self.duration_s, start + advance)
        tolerance = 8 * np.finfo(float).eps * max(self.duration_s, 1.0)
        if self.duration_s - advanced <= tolerance:
            advanced = self.duration_s
        return self.origin_s + advanced

    def exhaustion_reason(self, t_s):
        start = self.local_time(t_s)
        if start >= self.duration_s:
            return "reference_clock_end"
        ramp = _finite(getattr(self.reference, "ramp", 0.0), "reference final ramp")
        if ramp < 0.0:
            raise ValueError("reference final ramp must be nonnegative")
        if start >= self.duration_s - ramp and np.array_equal(
            self._sample_local(start).pose_d,
            self._sample_local(self.duration_s).pose_d,
        ):
            return "reference_geometry_roundoff"
        return ""
