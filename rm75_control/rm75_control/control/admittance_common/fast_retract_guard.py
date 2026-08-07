"""Low-latency veto for stale active over-force retraction.

The 6 Hz force used by the passive admittance remains untouched.  Compensated
raw force is used only to clear/hold a negative active reference after a force
drop has already been observed.  Consequently this guard can remove injected
motion but can neither command a press nor disable passive over-force escape.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FastRetractGuardConfig:
    enabled: bool = True
    cutoff_hz: float = 20.0
    stop_margin_n: float = 0.25
    stop_margin_fraction: float = 0.05
    rearm_margin_n: float = 0.45
    rearm_margin_fraction: float = 0.10
    stop_confirm_s: float = 0.015
    rearm_confirm_s: float = 0.010
    min_hold_s: float = 0.025
    max_sensor_age_s: float = 0.020

    @classmethod
    def from_dict(cls, raw: dict) -> FastRetractGuardConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("fast_retract_guard", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            cutoff_hz=float(p.get("cutoff_hz", 20.0)),
            stop_margin_n=float(p.get("stop_margin_n", 0.25)),
            stop_margin_fraction=float(p.get("stop_margin_fraction", 0.05)),
            rearm_margin_n=float(p.get("rearm_margin_n", 0.45)),
            rearm_margin_fraction=float(p.get("rearm_margin_fraction", 0.10)),
            stop_confirm_s=float(p.get("stop_confirm_s", 0.015)),
            rearm_confirm_s=float(p.get("rearm_confirm_s", 0.010)),
            min_hold_s=float(p.get("min_hold_s", 0.025)),
            max_sensor_age_s=float(p.get("max_sensor_age_s", 0.020)),
        )


class FastRetractGuard:
    def __init__(self, cfg: FastRetractGuardConfig) -> None:
        self.cfg = cfg
        self._raw_window: deque[float] = deque(maxlen=3)
        self.reset()

    def reset(self) -> None:
        self._raw_window.clear()
        self.fast_force_n = float("nan")
        self.armed = False
        self.hold = False
        self.valid = False
        self._stop_timer_s = 0.0
        self._rearm_timer_s = 0.0
        self._hold_timer_s = 0.0
        self.stop_count = 0
        self.rearm_count = 0

    def _update_fast_force(self, raw_force_n: float, dt_s: float) -> float:
        self._raw_window.append(float(raw_force_n))
        raw_median = float(np.median(np.asarray(self._raw_window, dtype=float)))
        if not np.isfinite(self.fast_force_n):
            self.fast_force_n = raw_median
            return self.fast_force_n
        fc = max(float(self.cfg.cutoff_hz), 0.0)
        alpha = (
            1.0 - math.exp(-2.0 * math.pi * fc * max(dt_s, 0.0))
            if fc > 0.0
            else 1.0
        )
        self.fast_force_n += float(np.clip(alpha, 0.0, 1.0)) * (
            raw_median - self.fast_force_n
        )
        return self.fast_force_n

    def update(
        self,
        *,
        raw_force_n: float | None,
        desired_force_n: float,
        filtered_eff_n: float,
        active_reference_m_s: float,
        dt_s: float,
        sensor_age_s: float | None,
        instability_index: float,
    ) -> bool:
        cfg = self.cfg
        dt = max(float(dt_s), 0.0)
        age_valid = (
            sensor_age_s is None
            or (
                np.isfinite(sensor_age_s)
                and float(sensor_age_s) <= max(cfg.max_sensor_age_s, 0.0)
            )
        )
        self.valid = bool(
            cfg.enabled
            and raw_force_n is not None
            and np.isfinite(raw_force_n)
            and np.isfinite(desired_force_n)
            and age_valid
        )
        if not self.valid:
            # Fail open: the established passive + active escape law remains.
            # Discard stale fast-path history as well; after a sensor dropout
            # the first fresh sample must prime a new filter episode rather
            # than blend with pre-dropout force.
            self._raw_window.clear()
            self.fast_force_n = float("nan")
            self.armed = False
            self.hold = False
            self._stop_timer_s = 0.0
            self._rearm_timer_s = 0.0
            self._hold_timer_s = 0.0
            return False

        fast_force = self._update_fast_force(float(raw_force_n), dt)
        target = abs(float(desired_force_n))
        stop_margin = max(
            float(cfg.stop_margin_n),
            float(cfg.stop_margin_fraction) * target,
        )
        rearm_margin = max(
            float(cfg.rearm_margin_n),
            float(cfg.rearm_margin_fraction) * target,
        )
        # This is a *crossing* guard, not an over-force regulator.  Arm on
        # the high side, then stop only after the fast/raw path has crossed
        # the target's low side while the delayed control force still asks
        # for retraction.  Using ``target + stop_margin`` as the stop level
        # erased the negative reference needed to follow a surface moving
        # steadily toward the probe, especially at a 1 N setpoint.
        arm_level = target + stop_margin
        stop_level = max(target - stop_margin, 0.0)
        rearm_level = target + rearm_margin
        retract_episode = (
            float(filtered_eff_n) < 0.0
            and float(active_reference_m_s) <= 0.0
        )

        if self.hold:
            self._hold_timer_s += dt
            if fast_force >= rearm_level:
                self._rearm_timer_s += dt
            else:
                self._rearm_timer_s = 0.0

            can_leave = self._hold_timer_s + 1e-12 >= max(
                cfg.min_hold_s,
                0.0,
            )
            rearm_confirm = max(
                cfg.rearm_confirm_s,
                0.015 if instability_index > 0.6 else 0.0,
            )
            if can_leave and (
                self._rearm_timer_s + 1e-12 >= rearm_confirm
                or float(filtered_eff_n) >= 0.0
            ):
                self.hold = False
                self.armed = fast_force >= arm_level
                self._rearm_timer_s = 0.0
                self._hold_timer_s = 0.0
                self.rearm_count += 1
            return self.hold

        if not retract_episode:
            self.armed = False
            self._stop_timer_s = 0.0
            return False

        if fast_force >= arm_level:
            self.armed = True
            self._stop_timer_s = 0.0
            return False

        if self.armed and fast_force <= stop_level:
            self._stop_timer_s += dt
            confirm = max(
                cfg.stop_confirm_s,
                0.020 if instability_index > 0.6 else 0.0,
            )
            if self._stop_timer_s + 1e-12 >= max(confirm, 0.0):
                self.hold = True
                self._hold_timer_s = 0.0
                self._stop_timer_s = 0.0
                self.stop_count += 1
        else:
            # Stay armed throughout the hysteresis band, but confirmation is
            # continuous-time below the low-side crossing only.
            self._stop_timer_s = 0.0
        return self.hold
