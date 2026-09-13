"""Measured-motion registration and optional response-sign policy.

No command enters this history. Registered displacement is evidence for a
heuristic repair-request discount, never an acoustic Jacobian or causal proof.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import window_rows
from .types import ContactObservation, vector


class MotionHistory:
    def __init__(self, geometry, *, max_gap_s=.02, retention_s=2., lateral_windows=None):
        self.rows = window_rows(geometry) if lateral_windows is None else window_rows(geometry, lateral_windows)
        self.max_gap_s, self.retention_s = float(max_gap_s), float(retention_s)
        self.samples = deque()
        self.last_id = -1
        self.last_time_s = -np.inf

    def add(self, time_s, measured_twist_tool, measurement_id, *, valid=True):
        t = float(time_s)
        if measurement_id <= self.last_id:
            return False
        if not np.isfinite(t):
            self.samples.clear()
            self.last_id = measurement_id
            return False
        if t <= self.last_time_s:
            return False
        self.last_id, self.last_time_s = measurement_id, t
        if not valid:
            self.samples.clear()
            return False
        try:
            v = vector(measured_twist_tool, (6,))
        except (ValueError, TypeError):
            self.samples.clear()
            return False
        self.samples.append((t, self.rows @ v, float(np.linalg.norm(v[:2]))))
        while len(self.samples) > 2 and self.samples[1][0] < t-self.retention_s:
            self.samples.popleft()
        return True

    def reconfigure(self, geometry, lateral_windows=None):
        self.rows = window_rows(geometry) if lateral_windows is None else window_rows(geometry, lateral_windows)
        self.samples.clear()
        # A configuration change cannot resurrect a consumed sensor sample.

    def displacement(self, start_s, end_s):
        result = self.motion_evidence(start_s, end_s)
        return None if result is None else result[:2]

    def motion_evidence(self, start_s, end_s):
        """Net local motion, planar travel, and total local unloading travel.

        All three integrate only the registered exposure interval. The last
        component detects loading followed by unloading inside that interval.
        """
        if len(self.samples) < 2 or not start_s < end_s:
            return None
        ts = np.array([s[0] for s in self.samples])
        if start_s < ts[0]-1e-9 or end_s > ts[-1]+1e-9:
            return None
        overlap = (ts[:-1] < end_s) & (ts[1:] > start_s)
        if np.any(np.diff(ts)[overlap] > self.max_gap_s+1e-9):
            return None
        grid = np.r_[start_s, ts[(ts > start_s) & (ts < end_s)], end_s]
        values = np.array([np.r_[s[1], s[2]] for s in self.samples])
        interpolated = np.stack([np.interp(grid, ts, values[:, i]) for i in range(4)], axis=1)
        integral = np.sum(.5*(interpolated[:-1]+interpolated[1:])*np.diff(grid)[:, None], axis=0)
        left, right = interpolated[:-1, :3], interpolated[1:, :3]
        negative = .5*(np.maximum(-left, 0.)+np.maximum(-right, 0.))
        crossing = left*right < 0.
        # Exact negative triangle for a linearly interpolated sign crossing.
        crossing_area = .5*np.maximum(-left, -right)**2/np.maximum(abs(right-left), 1e-300)
        negative = np.where(crossing, crossing_area, negative)
        unloading = np.sum(negative*np.diff(grid)[:, None], axis=0)
        return integral[:3], float(integral[3]), unloading


@dataclass(frozen=True)
class ResponseConfig:
    displacement_min_m: float = 1e-5
    quality_change_min: float = .005
    min_observation_s: float = .08
    max_interval_s: float = .4
    max_planar_travel_m: float = .004
    update_fraction: float = .2
    gamma_floor: float = .1


    def __post_init__(self):
        values = (self.displacement_min_m, self.quality_change_min, self.min_observation_s,
                  self.max_interval_s, self.max_planar_travel_m, self.update_fraction, self.gamma_floor)
        if (not np.isfinite(values).all() or min(values) <= 0
                or self.min_observation_s > self.max_interval_s
                or self.update_fraction > 1 or self.gamma_floor > 1):
            raise ValueError("invalid response evidence bounds")


class ResponseConsistency:
    def __init__(self, config=None):
        self.config = config or ResponseConfig()
        self.previous = None
        self.gamma = np.ones(2)
        self.updates = np.zeros(2, dtype=int)
        self.reason = "empty"
        self.version = None
        self.anchors = [None, None]

    def invalidate(self):
        self.previous = None
        self.anchors = [None, None]
        self.reason = "image_unavailable"

    def update(self, obs: ContactObservation | None, history, *, now_s, max_age_s=.3):
        if obs is None or not obs.fresh(now_s, max_age_s):
            self.invalidate()
            return self.gamma.copy()
        old = self.previous
        version = (obs.source_id, obs.version)
        if self.version is not None and version != self.version:
            self.gamma[:] = 1.
            self.updates[:] = 0
            self.anchors = [None, None]
            old = None
            self.reason = "version_changed"
        self.version = version
        if old is not None and (obs.frame_seq <= old.frame_seq or obs.effective_time_s <= old.effective_time_s):
            return self.gamma.copy()
        self.previous = obs
        cfg = self.config
        for slot, window in enumerate((0, 2)):
            anchor = self.anchors[slot]
            if anchor is None:
                self.anchors[slot] = obs
                continue
            interval = obs.effective_time_s-anchor.effective_time_s
            movement = history.motion_evidence(anchor.effective_time_s, obs.effective_time_s)
            if interval > cfg.max_interval_s or movement is None or movement[1] > cfg.max_planar_travel_m:
                self.anchors[slot] = obs
                self.reason = "registration_or_planar_motion"
                continue
            local, _, unloading = movement
            if unloading[window] >= cfg.displacement_min_m:
                self.anchors[slot] = obs
                self.reason = "unloading_resets_evidence"
                continue
            if interval+1e-12 < cfg.min_observation_s or local[window] < cfg.displacement_min_m:
                self.reason = "awaiting_completed_actual_motion"
                continue
            # Only an actually completed positive local loading experiment can
            # discount repair. Texture change without such motion is ignored.
            # Near-zero response is evidence of ineffective repair, not a skip.
            dq = obs.quality[window]-anchor.quality[window]
            target = 1. if dq >= cfg.quality_change_min else cfg.gamma_floor
            self.gamma[slot] += cfg.update_fraction*(target-self.gamma[slot])
            self.updates[slot] += 1
            self.anchors[slot] = obs
            self.reason = "registered_positive_response" if target == 1. else "registered_no_or_wrong_response"
        return self.gamma.copy()
