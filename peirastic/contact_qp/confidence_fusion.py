"""Image-owned rocking and explicit loading/scan authorization (experimental).

The centroid is a normalized image coordinate, not an identified image Jacobian
or a physical angle. CoP minimizes load-weighted added motion for a GIVEN
rotation; it is neither a stiffness center nor a constant-force guarantee.
"""
from __future__ import annotations

import math
import numpy as np

POLICIES = ('confidence_angular_v1', 'confidence_cop_v1')
FUSION_VERSION = 'confidence_cop_components_v1'
FEATURE_VERSION = 'confidence_column_centroid_v1'
TASK_SOURCE = 'loading_scan_v1'


def cop_preference(wrench_tool, geometry, *, contact, minimum_force_n):
    """Current +Z compressive load on the checked centered tool-Z face."""
    w = np.asarray(wrench_tool, dtype=float)
    if not contact:
        return 0., 'no_contact'
    if w.shape != (6,) or not np.isfinite(w).all():
        return 0., 'invalid_wrench'
    if w[2] < minimum_force_n or w[2] <= 0.:
        return 0., 'insufficient_compressive_load'
    cp = -float(w[4]) / float(w[2])
    if abs(cp) > geometry.half_length_m:
        return 0., 'outside_face'
    return cp, 'valid'


class ConfidenceAngularTask:
    """One target per frame, M/D dynamics from successful final publications.

    Observations remain consumed on rejection; command state changes only on
    commit. Holding a frame holds a velocity *target*, never an increment.
    """
    def __init__(self, *, mass, damping, repair_speed_m_s, lever_m,
                 image_x_sign, deadband, c_min, max_velocity, max_acceleration):
        values = (mass, damping, repair_speed_m_s, lever_m, max_velocity, max_acceleration)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('positive finite angular dynamics and geometry required')
        if image_x_sign not in (-1, 1) or not 0 <= deadband < 1 or not 0 < c_min <= 1:
            raise ValueError('invalid confidence direction or threshold')
        self.tau = mass / damping
        self.scale = repair_speed_m_s / lever_m
        self.image_x_sign = image_x_sign
        self.deadband = deadband
        self.c_min = c_min
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.omega_base = np.zeros(3)
        self.identity = None
        self.evidence_target = 0.
        self.evidence_reason = 'no_image'

    def preview(self, observation, *, rotation_base_tcp, dt_s, force_gate, enabled):
        if not math.isfinite(dt_s) or dt_s <= 0 or not 0 <= force_gate <= 1:
            raise ValueError('invalid angular step or force gate')
        fresh_evidence = False
        if observation is not None:
            if observation.confidence_feature_version != FEATURE_VERSION:
                raise ValueError('confidence centroid feature version mismatch')
            identity = (observation.source_id, observation.version, observation.frame_seq)
            if identity != self.identity:
                fresh_evidence = True
                self.identity = identity
                self.evidence_target = 0.
                if not observation.confidence_centroid_valid:
                    self.evidence_reason = 'invalid_centroid'
                elif np.all(observation.quality[[0, 2]] >= self.c_min):
                    self.evidence_reason = 'lateral_quality_met'
                else:
                    x = float(observation.confidence_centroid_x)
                    deficit = max(0., abs(x) - self.deadband) / (1. - self.deadband)
                    self.evidence_target = self.image_x_sign * math.copysign(self.scale * deficit, x)
                    self.evidence_reason = 'centroid_deadband' if deficit == 0 else 'weak_side_centroid'
        requested = self.evidence_target if observation is not None and enabled else 0.
        target = float(np.clip(requested * force_gate, -self.max_velocity, self.max_velocity))
        previous = float((np.asarray(rotation_base_tcp).T @ self.omega_base)[1])
        # Exact held-target M-D response: I*w_dot + D*w = D*w_target.
        omega = previous + (target - previous) * (-math.expm1(-dt_s / self.tau))
        omega = float(np.clip(omega, previous - self.max_acceleration * dt_s,
                              previous + self.max_acceleration * dt_s))
        omega = float(np.clip(omega, -self.max_velocity, self.max_velocity))
        reason = ('image_paused' if observation is None else 'waiting_for_contact' if not enabled
                  else self.evidence_reason)
        return omega, dict(visual_policy=FUSION_VERSION, visual_evidence_updated=fresh_evidence,
            visual_centroid_x=None if observation is None else observation.confidence_centroid_x,
            visual_evidence_target_rad_s=self.evidence_target,
            visual_requested_omega_rad_s=requested, visual_gated_target_rad_s=target,
            visual_dynamic_target_rad_s=omega, visual_previous_final_omega_rad_s=previous,
            visual_reason=reason, visual_force_gate=force_gate)

    def commit(self, final_tool, rotation_base_tcp):
        velocity = np.asarray(final_tool, dtype=float)
        if velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError('finite final command model required')
        self.omega_base = np.asarray(rotation_base_tcp) @ velocity[3:]


def task_components(nominal_tool):
    """Supply authorization excludes ALL rocking and paired normal motion."""
    nominal = np.asarray(nominal_tool, dtype=float)
    if nominal.shape != (6,) or not np.isfinite(nominal).all():
        raise ValueError('finite unmixed nominal required')
    loading = np.zeros(6)
    loading[2] = nominal[2]
    scan = nominal.copy()
    scan[[2, 4]] = 0.
    return loading, scan


def accepted_loading(candidate, loading_requested, cp):
    """Anti-windup in OUTER coordinates, never rail-compensated model Z.

    Remove the full CoP preference, including persistent rocking. A mechanical
    projection may reduce loading but cannot synthesize extra loading state.
    The remaining normal command is an explicit geometric allocation component.
    """
    z = float(candidate[2]) - cp * float(candidate[4])
    return float(np.clip(z, min(0., loading_requested), max(0., loading_requested)))
