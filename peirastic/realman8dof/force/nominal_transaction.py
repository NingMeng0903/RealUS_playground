"""Explicit command-state transaction for the existing tool-frame force law.

The controller still consumes contact, stiffness and force observations once in
``prepare``. Only the named command histories below are speculative. Legacy
TDPA/flow/shield values remain diagnostic models, never measured port work or a
passivity certificate. Active variants of those optional laws need their own
transaction adapters and are deliberately excluded here.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


# These are the integrators and command history, not a controller checkpoint.
_CONTROLLER_COMMAND_FIELDS = (
    "last_v_cmd", "v_force_z", "v_force_cmd_z", "_v_zoh_z", "v_r_z",
    "_force_point_base", "_force_point_inited", "force_point_z",
    "last_pose_d_combined", "x_adm_z", "x_d_z", "x_tilde_z",
    "_u_force_slewed", "_u_force_slew_dot", "u_sent_z",
    "tdpa_e_obs_j", "flow_tank_energy",
    "_lat_soften_hold_s",
)
_FLOW_COMMAND_FIELDS = (
    "xp", "vp", "aux_anchor", "x_aux", "x_safe", "v_aux", "alpha",
    "integral_position_error", "_initialized", "_vp_history",
    "_prev_press_request", "_accounted_press_m_s", "tank_energy",
    "energy_mismatch_j",
)
_SHIELD_COMMAND_FIELDS = ("_delay", "_v_plant", "_a_plus", "_u_prev", "_u_prev2")
# The air-bias measurement filter consumes the sample once in prepare. Only
# its command-port diagnostic work integral waits for command acceptance.
_TDPA_COMMAND_FIELDS = ("e_obs_j",)


def _capture(obj, names):
    return {name: copy(getattr(obj, name)) for name in names}


def _restore(obj, state):
    for name, value in state.items():
        setattr(obj, name, copy(value))


def finite_twist(value, name="twist"):
    out = np.asarray(value, dtype=float).reshape(6).copy()
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def check_measurement_id(value, previous):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("measurement_id must be a monotonically increasing integer")
    if int(value) <= previous:
        raise ValueError("duplicate or stale measurement_id")
    return int(value)


@dataclass
class _Pending:
    before: tuple
    proposed: tuple
    nominal: np.ndarray
    dt: float
    normal: np.ndarray
    sign: float
    force: float
    contact: bool


class NominalCommandTransaction:
    """One outstanding proposal, bounded explicit state, no controller deepcopy."""

    def __init__(self, controller):
        self.controller = controller
        self.last_measurement_id = -1
        self.last_source_sample_id=None
        self.last_source_t_s=None
        self.pending = None

    def _objects(self):
        c = self.controller
        return (c, c._proactive_ff, c._bidirectional_flow, c._safety_shield, c._tdpa)

    def _capture(self):
        names = (_CONTROLLER_COMMAND_FIELDS, ("v_r",), _FLOW_COMMAND_FIELDS,
                 _SHIELD_COMMAND_FIELDS, _TDPA_COMMAND_FIELDS)
        return tuple(_capture(obj, fields) for obj, fields in zip(self._objects(), names))

    def _restore(self, state):
        for obj, fields in zip(self._objects(), state):
            _restore(obj, fields)

    def prepare(self, measurement_id, compute, *, pose, f_des, f_ext, dt_s, dt_actual=None,
                control_step_id=None,source_sample_id=None,source_t_s=None,measurement_fresh=True):
        if self.pending is not None:
            raise RuntimeError("nominal proposal must be committed or aborted first")
        if control_step_id is not None and measurement_id is not None:
            raise ValueError('use control_step_id or legacy measurement_id, not both')
        seq = check_measurement_id(measurement_id if control_step_id is None else control_step_id,self.last_measurement_id)
        if control_step_id is not None:
            if source_sample_id is None or source_t_s is None or not np.isfinite(source_t_s) or source_t_s<0:
                raise ValueError('explicit control clock requires source identity and finite source time')
            if measurement_fresh:
                if (self.last_source_t_s is not None and
                        (source_t_s<=self.last_source_t_s or source_sample_id==self.last_source_sample_id)):
                    raise ValueError('new source must advance source identity and time')
            elif source_sample_id!=self.last_source_sample_id or source_t_s!=self.last_source_t_s:
                raise ValueError('held source must match the last consumed source')
        c = self.controller
        if c.cfg.control_frame != "tool":
            raise ValueError("nominal transactions require control_frame=tool")
        if (c.cfg.bidirectional_flow.mode == "active"
                or c.cfg.safety_shield.applies_command()
                or (c.cfg.tdpa.enabled and c.cfg.tdpa.apply)):
            raise ValueError("nominal transaction supports diagnostic optional layers only")
        pose = finite_twist(pose, "pose")
        desired = finite_twist(f_des, "desired wrench")
        measured = finite_twist(f_ext, "measured wrench")
        dt = float(dt_s if dt_actual is None else dt_actual)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("nominal transaction requires a positive finite dt")
        dt = float(np.clip(dt, 1e-4, 0.10))
        before = self._capture()
        episode_seen = bool(c._episode_seen)
        self.last_measurement_id = seq
        if control_step_id is not None and measurement_fresh:
            self.last_source_sample_id=source_sample_id
            self.last_source_t_s=float(source_t_s)
        try:
            output = compute()
            proposed = self._capture()
            # Resets caused by a consumed contact observation are not command
            # integration. Retain their zero/origin even when this proposal is
            # aborted, otherwise the one-shot event would disappear forever.
            rising = bool(c.physical_contact_acquire_event) and (
                not episode_seen or bool(c.contact_episode_rearm_event)
            )
            if rising:
                rotation = Rotation.from_euler(c.cfg.euler_order, pose[3:]).as_matrix()
                origin = pose[:3].copy()
                if c.cfg.system_delay_s > 0.0:
                    origin += rotation @ before[0]["last_v_cmd"][:3] * c.cfg.system_delay_s
                before[0]["_force_point_base"] = origin
                before[0]["_force_point_inited"] = True
                normal = rotation[:, 2]
                before[0]["force_point_z"] = float(normal @ origin)
                combined = before[0]["last_pose_d_combined"].copy()
                combined[:3] += normal * (float(normal @ origin) - float(normal @ combined[:3]))
                before[0]["last_pose_d_combined"] = combined
            if c.physical_contact_loss_event:
                for field in ("v_force_z", "_v_zoh_z", "x_adm_z", "x_d_z", "x_tilde_z"):
                    before[0][field] = 0.0
            if (rising or c.physical_contact_loss_event
                    or (not c.contact_present and c.force_task_armed)
                    or (c.overforce_escape and before[0]["v_r_z"] > 0.0)
                    or c.force_reference_fast_clear or c.force_reference_reversal_reset):
                before[0]["v_r_z"] = 0.0
                before[1]["v_r"] = 0.0
        finally:
            # Filters, measured positions/velocities, Ke and contact observations
            # are intentionally not part of this restore.
            self._restore(before)
        self.pending = _Pending(
            before, proposed, finite_twist(output.telemetry["v_cmd"]), dt,
            Rotation.from_euler(c.cfg.euler_order, pose[3:]).as_matrix()[:, 2],
            1.0 if desired[2] >= 0 else -1.0, float(measured[2]),
            bool(c.contact_present),
        )
        return output

    def commit_applied(self, applied_z, *, final_full_twist=None):
        p = self.pending
        if p is None:
            raise RuntimeError("no nominal proposal to commit")
        z = float(applied_z)
        if not np.isfinite(z):
            raise ValueError("accepted normal velocity must be finite")
        if final_full_twist is None:
            full = p.nominal.copy()
            full[2] = z
        else:
            full = finite_twist(final_full_twist)
        c = self.controller
        self._restore(p.proposed)
        if z != float(p.nominal[2]):
            delta = z - float(p.nominal[2])
            c.v_force_z = z if p.contact else 0.0
            c._v_zoh_z = z if p.contact else 0.0
            c._force_point_base += p.normal * delta * p.dt
            c.force_point_z = float(p.normal @ c._force_point_base)
            c.last_pose_d_combined[:3] += p.normal * delta * p.dt
            if p.contact:
                c.x_adm_z += delta * p.dt
                # A changed action does not earn an unexecuted desired-position
                # increment. Keep the spring reference anchored to committed state.
                c.x_d_z = p.before[0]["x_d_z"]
            c.x_tilde_z = c.x_adm_z - c.x_d_z
            c._u_force_slewed = p.sign * z
            c._u_force_slew_dot = (p.sign * z - p.before[0]["_u_force_slewed"]) / p.dt
            # Conditional integration: discard this proposal's active reference
            # and clear the residual drive after external command modification.
            c._proactive_ff.reset()
            c.v_r_z = 0.0
            flow = c._bidirectional_flow
            _restore(flow, p.before[2])
            flow_sign = 1.0 if float(flow.cfg.normal_sign) >= 0.0 else -1.0
            flow_z = flow_sign * z
            flow.xp = (flow.xp if flow._initialized else flow.xa) + flow_z * p.dt
            flow._initialized = True
            flow.vp = flow_z
            flow.integral_position_error = 0.0
            flow._vp_history.append(flow_z)
            del flow._vp_history[:-flow._VP_HISTORY_MAX]
            shield = c._safety_shield
            _restore(shield, p.before[3])
            shield._commit_sent(p.sign * z)
            _restore(c._tdpa, p.before[4])
            if c._tdpa.cfg.enabled:
                # Do not invoke commit again: it also updates the air-bias
                # filter, which has already consumed this measurement.
                c._tdpa.e_obs_j += (p.sign * p.force - c._tdpa.f_bias_n) * p.sign * z * p.dt
                if c._tdpa.e_obs_j > 0.0:
                    c._tdpa.e_obs_j *= math.exp(
                        -p.dt / max(float(c._tdpa.cfg.e_leak_pos_s), 1e-3)
                    )
        c.last_v_cmd = full
        if not np.array_equal(full[:2], p.nominal[:2]):
            c._lat_soften_hold_s = p.before[0]["_lat_soften_hold_s"]
            c._lateral_chase_scale(float(np.linalg.norm(full[:2])), dt_s=p.dt)
        c.v_force_cmd_z = z
        c.u_sent_z = p.sign * z
        c.tdpa_e_obs_j = float(c._tdpa.e_obs_j)
        c.tdpa_alpha = float(c._tdpa.alpha)
        c.flow_tank_energy = float(c._bidirectional_flow.tank_energy)
        self.pending = None

    def abort(self):
        if self.pending is None:
            raise RuntimeError("no nominal proposal to abort")
        # Already restored at prepare. Physical observation/monitor balances
        # remain advanced; this is not an energy refund or physical rollback.
        self.pending = None

    def reset(self):
        # Retain the sequence high-water mark across resets; a late sample from
        # before reset cannot be consumed again in the same adapter instance.
        self.pending = None
