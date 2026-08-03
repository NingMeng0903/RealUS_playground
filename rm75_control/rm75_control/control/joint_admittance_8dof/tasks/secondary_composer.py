"""Priority-aware composition of nullspace secondary tasks.

Joint limit repulsion always runs; the arm-angle task fades out CONTINUOUSLY
as any joint approaches its physical limit (no on/off switch - a binary gate
at a fixed activation chattered against the limit-repulsion task when the
nullspace parked the arm right on the threshold).

The composed soft-task velocity (centering + arm-angle + viscous damping) is
magnitude-capped per joint: near a kinematic singularity the SR-damped
projector opens up (N -> I), and an uncapped centering gradient - large when
the posture is far from q_nominal, e.g. a straight arm at start-up - would
otherwise drive the whole arm at rad/s scale while the Cartesian task is soft.
The joint-plan feedforward ``qdot_ff`` is added AFTER the cap: it is the
primary content of a joint-space move and is already velocity-limited by the
plan itself and by the QP box.

Rail behaviour is decoupled from this composer: RailMode.COUPLED lets the QP
resolve rail motion normally; LOCKED + HOLD applies the RailLockTask below;
LOCKED + RAIL_ONLY / TCP_FIXED are driven by qdot_ff[0] plus the QP rail-vel
pin in constraint_mgr — the composer only forwards the arm portion of qdot_ff.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTask
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockTask


def max_limit_activation(
    q_rad: np.ndarray,
    q_mid: np.ndarray,
    half: np.ndarray,
    *,
    activation: float,
) -> float:
    """Peak limit-repulsion activation in [0, 1] (same metric as JointCenteringTask)."""
    q = np.asarray(q_rad, dtype=float)
    u_limit = (q - q_mid) / half
    span = max(1.0 - activation, 1e-6)
    over = np.clip((np.abs(u_limit) - activation) / span, 0.0, 1.0)
    return float(np.max(over))


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class SecondaryComposer:
    """Compose centering + arm-angle + feedforward with limit priority."""

    def __init__(
        self,
        centering: JointCenteringTask,
        arm_task: ArmAngleTask | None,
        *,
        manipulability: ManipulabilityTask | None = None,
        rail_lock: RailLockTask | None = None,
        arm_activation_limit: float = 0.92,
        arm_fade_band: float = 0.05,
        d_null: float = 0.0,
        adaptive_d_null_gain: float = 1.0,
        v_max: np.ndarray | None = None,
        max_qdot_frac: float = 0.2,
    ) -> None:
        self.centering = centering
        self.arm_task = arm_task
        self.manipulability = manipulability
        self.rail_lock = rail_lock
        self.arm_activation_limit = float(arm_activation_limit)
        self.arm_fade_band = float(arm_fade_band)
        self.d_null = float(d_null)
        self.adaptive_d_null_gain = float(adaptive_d_null_gain)
        self.v_max = None if v_max is None else np.asarray(v_max, dtype=float)
        self.max_qdot_frac = float(max_qdot_frac)
        self.last_limit_activation: float = 0.0
        self.last_arm_smooth: float = 1.0

    @classmethod
    def from_controller_parts(
        cls,
        centering: JointCenteringTask,
        arm_task: ArmAngleTask | None,
        nullspace_cfg: NullspaceTaskConfig,
        *,
        manipulability: ManipulabilityTask | None = None,
        rail_lock: RailLockTask | None = None,
        d_null: float = 0.0,
        adaptive_d_null_gain: float = 1.0,
        v_max: np.ndarray | None = None,
        max_qdot_frac: float = 0.2,
    ) -> "SecondaryComposer":
        return cls(
            centering,
            arm_task,
            manipulability=manipulability,
            rail_lock=rail_lock,
            arm_activation_limit=nullspace_cfg.activation + 0.07,
            d_null=d_null,
            adaptive_d_null_gain=adaptive_d_null_gain,
            v_max=v_max,
            max_qdot_frac=max_qdot_frac,
        )

    def _arm_weight(self, u_max: float) -> float:
        """Continuous arm-task weight vs peak limit activation.

        1.0 while well clear of limits, smoothstep-fading to 0.0 across
        ``[arm_activation_limit - band, arm_activation_limit + band]``.  A
        continuous function of u_max cannot chatter the way the old binary
        ``u_max < limit`` gate did.
        """
        band = max(self.arm_fade_band, 1e-6)
        return _smoothstep01((self.arm_activation_limit + band - u_max) / (2.0 * band))

    def compose(
        self,
        q_rad: np.ndarray,
        qdot_ff: np.ndarray | None,
        qdot_prev: np.ndarray | None,
        *,
        arm_suppressed: bool,
        sigma_min: float = 1.0,
        sigma_ref: float = 0.08,
        sigma_escape_ref: float = 0.0,
        centering_suppressed: bool = False,
        manipulability_active: bool = False,
        centering_sigma_fade: bool = True,
        centering_gain_scale: float = 1.0,
        max_qdot_frac_override: float | None = None,
        dt_s: float = 0.005,
    ) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        cfg = self.centering.cfg
        u_max = max_limit_activation(
            q,
            self.centering.q_mid,
            self.centering.half,
            activation=cfg.activation,
        )
        self.last_limit_activation = u_max

        qdot_soft = np.zeros_like(q)
        rail_hold = self.rail_lock is not None and self.rail_lock.active
        # Nullspace attractor (centering) and ∇μ escape must COEXIST when σ
        # dips: replacing centering with manip alone let J5 drift to the
        # flipped wrist on long hard-contact scans.  Always apply centering
        # first (unless suppressed); add manipulability on top.  When manip is
        # active in the escape band, σ-yield proximal centering (floor 0.25) so
        # ∇μ can lead without a full tug-of-war; never kill the attractor.
        # J6 (index 6) is exempt from yield: |J6|≈0 is wrist singularity and
        # yielding the 45° nominal pull let scans park there and jitter.
        if not centering_suppressed:
            qdot_center = float(max(centering_gain_scale, 0.0)) * self.centering(q)
            escape_ref = float(sigma_escape_ref) if float(sigma_escape_ref) > 1e-9 else float(sigma_ref)
            if (
                manipulability_active
                and escape_ref > 1e-9
                and float(sigma_min) < escape_ref
            ):
                yield_scale = float(
                    np.clip(float(sigma_min) / escape_ref, 0.25, 1.0)
                )
                qdot_full = qdot_center
                qdot_center = yield_scale * qdot_center
                if qdot_center.shape[0] > 6:
                    qdot_center[6] = qdot_full[6]
            qdot_soft = qdot_center
        # Rail is a base translation: ∂μ/∂q0 is analytically zero, but the FD
        # gradient in ManipulabilityTask can produce small numerical residuals
        # that get unit-normalised to k_mu.  Always exclude rail from the
        # manipulability push — its purpose is to escape ARM singularities,
        # never to be a stealth rail driver behind the primary QP's back.
        if manipulability_active and self.manipulability is not None:
            qdot_soft = qdot_soft + self.manipulability(
                q, sigma_min=sigma_min, exclude_rail=True, dt_s=float(dt_s)
            )
        if rail_hold:
            qdot_soft = qdot_soft + self.rail_lock(q)

        d_eff = self.d_null
        if self.adaptive_d_null_gain > 0.0 and u_max > 0.0:
            d_eff = d_eff * (1.0 + self.adaptive_d_null_gain * u_max)
        if d_eff > 0.0 and qdot_prev is not None:
            qdot_soft = qdot_soft - d_eff * np.asarray(qdot_prev, dtype=float)

        # Per-joint magnitude cap on the soft tasks (see module docstring).
        cap_frac = (
            self.max_qdot_frac
            if max_qdot_frac_override is None
            else float(max_qdot_frac_override)
        )
        if self.v_max is not None and cap_frac > 0.0:
            cap = cap_frac * self.v_max
            qdot_soft = np.clip(qdot_soft, -cap, cap)

        if not rail_hold:
            qdot_soft[0] = 0.0

        # Near σ≈0 attenuate centering/manip/damping — NOT arm_angle.
        # Disabled during COUPLED rail-extension scan: rail carries base
        # translation, centering keeps arm posture (Yamamoto & Yun split).
        if (
            centering_sigma_fade
            and not manipulability_active
            and sigma_min < sigma_ref
        ):
            fade = sigma_min / max(sigma_ref, 1e-6)
            qdot_soft = qdot_soft * fade

        qdot0 = qdot_soft
        if self.arm_task is not None and not arm_suppressed:
            w_arm = self._arm_weight(u_max)
            if w_arm > 0.0:
                qdot_arm = self.arm_task(q)
                self.last_arm_smooth = w_arm * float(self.arm_task.last_singularity_smooth)
                qdot0 = qdot0 + w_arm * qdot_arm
            else:
                self.last_arm_smooth = 0.0
        else:
            self.last_arm_smooth = 1.0 if self.arm_task is None else 0.0

        if qdot_ff is not None:
            qdot0 = qdot0 + np.asarray(qdot_ff, dtype=float)

        return qdot0
