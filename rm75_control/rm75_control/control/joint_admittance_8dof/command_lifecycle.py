"""Opt-in publication transaction for the handheld Cartesian controller.

Pad reuses the scan 5 ms command hold and 200 Hz runner. This adapter only
commits after both transports succeed; it does not change the force period.
"""
from __future__ import annotations

import numpy as np

from .command_power import final_command_qdot


class CommandLifecycleTransaction:
    def __init__(self, owner):
        self.owner = owner
        self.state = "idle"
        self.final_tool = None
        self.facts = {}

    def begin_candidate(self):
        if self.state != "idle":
            raise RuntimeError("unresolved command lifecycle candidate")
        self.state = "prepared"
        self.final_tool = None
        self.facts = {}

    def review(self, step, *, published_rail_m, dt_s, now_s):
        if self.state != "prepared":
            raise RuntimeError("command lifecycle review without a fresh candidate")
        final_qdot = final_command_qdot(
            step.qdot, proposed_rail_m=float(step.q_send[0]),
            published_rail_m=float(published_rail_m), dt_s=float(dt_s),
        )
        rotation = np.asarray(self.owner.pending_rotation_base_tcp, dtype=float).reshape(3, 3)
        final_base = np.asarray(step.v_tcp_estimated, dtype=float).reshape(6)
        self.final_tool = np.r_[rotation.T @ final_base[:3], rotation.T @ final_base[3:]]
        self.facts = {
            "velocity_model": "arm_plus_measured_rail_v1",
            "command_qdot": final_qdot.tolist(),
            "rail_target_m": float(published_rail_m),
            "rail_proposal_m": float(step.q_send[0]),
            "rail_model_rate_m_s": float(final_qdot[0]),
            "rail_execution_rate_m_s": float(getattr(step, "rail_exec_for_qp_m_s", 0.0)),
            "rocking_policy_tier": getattr(step, "rocking_policy_tier", ""),
            "rocking_limited": bool(getattr(step, "rocking_limited", False)),
            "rocking_lower_rad_s": float(getattr(step, "rocking_lower_rad_s", float("nan"))),
            "rocking_upper_rad_s": float(getattr(step, "rocking_upper_rad_s", float("nan"))),
        }
        if not np.all(np.isfinite(self.final_tool)):
            return False
        if not self.owner.review(self.final_tool.copy(), now_s=now_s, facts=dict(self.facts)):
            return False
        self.state = "reviewed"
        return True

    def started(self):
        if self.state != "reviewed":
            raise RuntimeError("command lifecycle dispatch without final review")
        if not self.owner.started():
            return False
        self.state = "dispatching"
        return True

    def commit(self, *, now_s, rail_enabled):
        if self.state != "dispatching":
            raise RuntimeError("command lifecycle commit without dispatch")
        facts = dict(self.facts, arm="sent", rail="sent" if rail_enabled else "disabled")
        self.owner.commit(self.final_tool.copy(), now_s=now_s, facts=facts)
        self.state = "idle"
        self.final_tool = None
        self.facts = {}

    def arm_dispatch(self, *, now_s):
        if self.state != 'dispatching':
            raise RuntimeError('arm dispatch without accepted command')
        stamp = getattr(self.owner, 'arm_dispatch_time', None)
        if callable(stamp) and not stamp(float(now_s)):
            return False
        self.facts['arm_dispatch_time_s'] = float(now_s)
        return True

    def abort(self, reason, *, definitely_not_sent=None, facts=None, latch=None):
        if self.state == "idle":
            return
        if definitely_not_sent is None:
            definitely_not_sent = self.state != "dispatching"
        # Clear local state even if the owner's cleanup fails. No candidate can
        # be reviewed or dispatched again after an abort.
        self.state = "idle"
        self.final_tool = None
        self.facts = {}
        kwargs = dict(definitely_not_sent=definitely_not_sent, facts=facts)
        if latch is not None:
            kwargs['latch'] = latch
        self.owner.abort(reason, **kwargs)
