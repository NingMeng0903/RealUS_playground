"""Physical normal-contact tracking, separate from the force-task latch.

The force task must remain active while a moving surface temporarily leaves
the probe.  Environment-stiffness adaptation has a different requirement: it
must know when the probe is no longer carrying load so that a later impact can
re-arm stiff-first damping.

This tracker therefore never ends a task.  It only classifies the load-bearing
contact episode using:

* filtered force for a conservative, confirmed loss decision;
* compensated raw force for a low-latency re-acquisition decision;
* hysteresis and confirmation times so a short 4--12 Hz trough does not re-arm
  stiff-first every half-cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalContactConfig:
    enabled: bool = True
    enter_n: float = 0.80
    hard_enter_n: float = 1.50
    exit_n: float = 0.35
    enter_confirm_s: float = 0.010
    exit_confirm_s: float = 0.100

    @classmethod
    def from_dict(cls, raw: dict) -> PhysicalContactConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("physical_contact", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            enter_n=float(
                p.get(
                    "enter_n",
                    c.get("physical_contact_enter_n", c.get("contact_threshold_n", 0.8)),
                )
            ),
            hard_enter_n=float(
                p.get(
                    "hard_enter_n",
                    c.get("physical_contact_hard_enter_n", 1.5),
                )
            ),
            exit_n=float(
                p.get(
                    "exit_n",
                    c.get("physical_contact_exit_n", 0.35),
                )
            ),
            enter_confirm_s=float(
                p.get(
                    "enter_confirm_s",
                    c.get("physical_contact_enter_confirm_s", 0.010),
                )
            ),
            exit_confirm_s=float(
                p.get(
                    "exit_confirm_s",
                    c.get("physical_contact_exit_confirm_s", 0.100),
                )
            ),
        )


@dataclass(frozen=True)
class PhysicalContactUpdate:
    present: bool
    state: str
    acquired: bool = False
    reacquired: bool = False
    lost: bool = False


class PhysicalContactTracker:
    """Four-state load-bearing contact tracker.

    ``CONTACT`` and ``SUSPECT_LOSS`` both count as physically present.  A
    confirmed ``LOST`` episode is required before the next force rise can emit
    ``reacquired=True``.
    """

    FREE = "free"
    CONTACT = "contact"
    SUSPECT_LOSS = "suspect_loss"
    LOST = "lost"

    def __init__(self, cfg: PhysicalContactConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = self.FREE
        self.low_timer_s = 0.0
        self.high_timer_s = 0.0
        self.ever_acquired = False
        self.filtered_force_n = 0.0
        self.raw_force_n = 0.0

    @property
    def present(self) -> bool:
        return self.state in (self.CONTACT, self.SUSPECT_LOSS)

    def force_state(self, present: bool) -> PhysicalContactUpdate:
        """Explicit-state compatibility path used by deterministic tests."""
        was_present = self.present
        had_contact = self.ever_acquired
        self.low_timer_s = 0.0
        self.high_timer_s = 0.0
        if present:
            self.state = self.CONTACT
            self.ever_acquired = True
            acquired = not was_present
            return PhysicalContactUpdate(
                present=True,
                state=self.state,
                acquired=acquired,
                reacquired=acquired and had_contact,
            )
        self.state = self.LOST if had_contact else self.FREE
        return PhysicalContactUpdate(
            present=False,
            state=self.state,
            lost=was_present,
        )

    def update(
        self,
        filtered_force_n: float,
        raw_force_n: float,
        *,
        dt_s: float,
    ) -> PhysicalContactUpdate:
        cfg = self.cfg
        dt = max(float(dt_s), 0.0)
        self.filtered_force_n = float(filtered_force_n)
        self.raw_force_n = float(raw_force_n)

        if not cfg.enabled:
            present = max(self.filtered_force_n, self.raw_force_n) >= cfg.enter_n
            return self.force_state(present)

        finite = np.isfinite(self.filtered_force_n) and np.isfinite(
            self.raw_force_n
        )
        if not finite:
            # Missing data must never manufacture a contact transition.
            self.low_timer_s = 0.0
            self.high_timer_s = 0.0
            return PhysicalContactUpdate(self.present, self.state)

        if self.present:
            self.high_timer_s = 0.0
            if self.filtered_force_n < cfg.exit_n:
                self.low_timer_s += dt
                self.state = self.SUSPECT_LOSS
                if self.low_timer_s + 1e-12 >= max(cfg.exit_confirm_s, 0.0):
                    self.state = self.LOST
                    self.low_timer_s = 0.0
                    return PhysicalContactUpdate(
                        present=False,
                        state=self.state,
                        lost=True,
                    )
            else:
                self.low_timer_s = 0.0
                self.state = self.CONTACT
            return PhysicalContactUpdate(self.present, self.state)

        self.low_timer_s = 0.0
        hard_hit = (
            cfg.hard_enter_n > 0.0
            and self.raw_force_n >= cfg.hard_enter_n
        )
        high = (
            self.raw_force_n >= cfg.enter_n
            or self.filtered_force_n >= cfg.enter_n
        )
        if hard_hit:
            self.high_timer_s = max(cfg.enter_confirm_s, 0.0)
        elif high:
            self.high_timer_s += dt
        else:
            self.high_timer_s = 0.0

        if self.high_timer_s + 1e-12 >= max(cfg.enter_confirm_s, 0.0):
            had_contact = self.ever_acquired
            self.ever_acquired = True
            self.state = self.CONTACT
            self.high_timer_s = 0.0
            return PhysicalContactUpdate(
                present=True,
                state=self.state,
                acquired=True,
                reacquired=had_contact,
            )

        self.state = self.LOST if self.ever_acquired else self.FREE
        return PhysicalContactUpdate(False, self.state)
