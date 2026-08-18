"""S-R-S arm-angle (swivel) redundancy parametrization for the RM75-F.

The RM75 is a spherical-shoulder (J1-J3), elbow (J4), spherical-wrist (J5-J7)
arm, so its single redundant DOF has an exact geometric coordinate: the swivel
angle psi - the rotation of the elbow point E about the shoulder-wrist axis SW,
measured from a fixed reference plane (Shimizu et al. 2008; Kreutz-Delgado).

Using psi as an explicit nullspace coordinate is more deterministic than a
joint-space posture attractor: holding psi_ref pins the elbow branch exactly
(the value observed at the IK solution / teach pose), while the primary
Cartesian task and the joint-limit repulsion stay untouched.

Frames used (verified against the URDF: |S-E| = 256 mm, |E-W| = 210 mm,
joint_1/2, joint_3/4, joint_5/6 pairs are coincident):

    S = origin of joint_2  (shoulder center, fixed in base)
    E = origin of joint_4  (elbow center)
    W = origin of joint_6  (wrist center)
"""

from __future__ import annotations

import time

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance_8dof.ik_types import project_onto_task_nullspace
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    fold_psi_to_positive,
)

_SHOULDER_JOINT = "joint_2"
_ELBOW_JOINT = "joint_4"
_WRIST_JOINT = "joint_6"

# Reference direction defining psi = 0: the base -Z axis projected off the SW
# axis ("elbow hanging down" plane).  Any fixed vector not parallel to SW works;
# base Z is a good choice for a table-mounted arm.
_V_REF = np.array([0.0, 0.0, -1.0])


@dataclass
class ArmAngleTaskConfig:
    enabled: bool = False
    k_psi: float = 1.0            # swivel tracking gain (1/s)
    psi_ref_rad: float | None = None   # None -> capture at reset()
    fd_eps_rad: float = 1e-4      # central-difference step for the gradient
    safe_denom_eps: float = 1e-4  # floor on grad_psi . gN to prevent blow-up
    # exp(-gain * obs^2) attenuation near the algorithmic singularity, with
    # obs = (ne / |E-S|) * nr, both DIMENSIONLESS in [0, 1].  (An earlier
    # version used ne in meters (~0.16 max on the RM75) with gain 100, which
    # attenuated the task to ~3% in EVERY posture - psi looked "stuck".)
    obs_decay_gain: float = 400.0
    # Floor on the observability fade so a stretched arm still tracks ψ.
    obs_smooth_floor: float = 0.3
    max_qdot_frac: float = 0.15   # clip |qdot| to this fraction of v_max per joint
    # Global posture attractor for the SRS planner (pose_ik.resolve_pose_ik_srs).
    # ψ_home is the target the enumeration pulls toward on every new pose so
    # the arm stays in a consistent "posture family" (elbow always to the
    # same side, no random re-branching).  ``None`` means "capture the swivel
    # angle of the controller's very first reset()" — the arm defaults to
    # whatever posture the operator taught.
    psi_home_rad: float | None = None
    # Hard-cap the ψ swing allowed by the planner (used by resolve_pose_ik_srs).
    # An IK candidate whose ψ is more than this away from ψ_seed is dropped
    # before the goal-score ranking — this is the anti-twist guard.
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0
    # Optional absolute ψ envelope (e.g. cable-carrier protection).  None
    # disables the hard limit; if set, both ends are checked.
    psi_hard_lower_rad: float | None = None
    psi_hard_upper_rad: float | None = None
    # Smoothstep ramp after reset / set_reference so a large ψ error cannot
    # dump a full-scale nullspace kick on the first tick.
    engage_s: float = 0.0


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class ArmAngleTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) tracking psi_ref.

    The raw gradient grad_psi mostly points along directions that also move the
    TCP (psi changes when the wrist moves), so a naive gradient step nearly
    vanishes after nullspace projection.  The step is therefore normalized with
    the TASK-NULLSPACE-projected gradient gN = N(J) grad_psi:

        qdot0 = k_psi * wrap(psi_ref - psi(q)) * gN / (grad_psi . gN)

    which gives d(psi)/dt = k_psi * err exactly on the self-motion manifold
    (the QP re-projects, which is idempotent on gN).
    """

    def __init__(self, kin: RobotKinematics, cfg: ArmAngleTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ArmAngleTaskConfig()
        self.psi_ref = self.cfg.psi_ref_rad
        # Continuous (unwrapped) swivel target — arctan2 reports psi in
        # (-pi, pi]; crossing the branch fires a 2pi jump in the raw angle
        # while the arm barely moves, which makes wrap(psi_ref-psi) look like
        # a huge nullspace error and drives violent swivel chatter on hardware.
        self._psi_ref_unwrapped: float | None = None
        # NOTE: psi is analytically invariant to the rail position q[0]: S, E
        # and W all translate together with the base, so SW / SE (and hence
        # psi) are unchanged.  An earlier patch froze the rail coordinate for
        # the psi geometry ("_rail_ref_m"); it was a no-op and has been
        # removed (see tests/test_arm_angle_rail_invariance.py).
        self._model = kin.model
        self._data = self._model.createData()
        self._jids = tuple(
            self._model.getJointId(n) for n in (_SHOULDER_JOINT, _ELBOW_JOINT, _WRIST_JOINT)
        )
        self.last_singularity_smooth: float = 1.0
        self._engage_t = time.monotonic()

    def _sw_observability(self, q_rad: np.ndarray) -> tuple[float, float, float]:
        """Return (ne_norm, nr, obs) for algorithmic-singularity attenuation.

        ``ne_norm`` = elbow off-axis offset normalized by the upper-arm length
        |E-S| (sin of the shoulder-elbow angle off the SW axis) and ``nr`` =
        |V_REF x w_hat| (sin of the SW-vs-reference-vector angle) are both
        dimensionless in [0, 1]; their product is the observability measure.
        """
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        s_id, e_id, w_id = self._jids
        S = np.asarray(self._data.oMi[s_id].translation)
        E = np.asarray(self._data.oMi[e_id].translation)
        W = np.asarray(self._data.oMi[w_id].translation)
        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        se = E - S
        n_se = float(np.linalg.norm(se))
        if n_sw < 1e-9 or n_se < 1e-9:
            return 0.0, 0.0, 0.0
        w_hat = sw / n_sw
        e_perp = se - np.dot(se, w_hat) * w_hat
        r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
        ne = float(np.linalg.norm(e_perp)) / n_se
        nr = float(np.linalg.norm(r_perp))
        return ne, nr, ne * nr

    # ---- geometry ----------------------------------------------------------
    def arm_angle(self, q_rad: np.ndarray) -> float:
        """Swivel angle psi(q) in (-pi, pi] (invariant to the rail q[0])."""
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        s_id, e_id, w_id = self._jids
        S = np.asarray(self._data.oMi[s_id].translation)
        E = np.asarray(self._data.oMi[e_id].translation)
        W = np.asarray(self._data.oMi[w_id].translation)

        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        if n_sw < 1e-9:
            return 0.0
        w_hat = sw / n_sw

        # Elbow direction and reference direction, both projected off the SW axis.
        e_perp = (E - S) - np.dot(E - S, w_hat) * w_hat
        r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
        ne = float(np.linalg.norm(e_perp))
        nr = float(np.linalg.norm(r_perp))
        if ne < 1e-9 or nr < 1e-9:
            # Arm fully stretched (elbow on the SW axis) or SW parallel to the
            # reference vector: psi is undefined; report 0 (gradient ~0 too).
            return 0.0
        e_u = e_perp / ne
        r_u = r_perp / nr
        return float(np.arctan2(np.dot(np.cross(r_u, e_u), w_hat), np.dot(r_u, e_u)))

    def _psi_unwrapped(self, q_rad: np.ndarray) -> float:
        """Swivel angle continuous near the active reference (no ±pi branch flip)."""
        psi = self.arm_angle(q_rad)
        if self._psi_ref_unwrapped is None:
            return psi
        return float(self._psi_ref_unwrapped + _wrap_pi(psi - self._psi_ref_unwrapped))

    def grad_arm_angle(self, q_rad: np.ndarray) -> np.ndarray:
        """d psi / d q via central differences on arm joints only (rail excluded)."""
        q = np.asarray(q_rad, dtype=float)
        eps = self.cfg.fd_eps_rad
        g = np.zeros_like(q)
        for i in range(1, q.size):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            g[i] = (self._psi_unwrapped(qp) - self._psi_unwrapped(qm)) / (2.0 * eps)
        return g

    # ---- task interface ------------------------------------------------------
    def reset(self, q_rad: np.ndarray) -> None:
        """Capture psi_ref from the current configuration if not already set
        (by config or an explicit set_reference from the application)."""
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.psi_ref = fold_psi_to_positive(self.arm_angle(q))
        # Same SEW plane as −π; keep the tracker on the positive half so a
        # later set_reference(70°) slews 180°→70°, not −180°→−290°.
        self._psi_ref_unwrapped = fold_psi_to_positive(float(self.psi_ref))
        self._engage_t = time.monotonic()

    def set_reference(self, psi_ref_rad: float) -> None:
        psi_ref_rad = float(psi_ref_rad)
        self._engage_t = time.monotonic()
        if self._psi_ref_unwrapped is not None:
            self._psi_ref_unwrapped = float(
                self._psi_ref_unwrapped + _wrap_pi(psi_ref_rad - self._psi_ref_unwrapped)
            )
        else:
            self._psi_ref_unwrapped = psi_ref_rad
        self.psi_ref = psi_ref_rad

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.reset(q)
        psi = self._psi_unwrapped(q)
        g = self.grad_arm_angle(q)
        if float(np.dot(g, g)) < 1e-10:
            return np.zeros_like(q)
        J = self.kin.jacobian(q)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        # Kinematic nullspace only — must match qp.use_dyn_nullspace (off on
        # RM75) so d(psi)/dt ~= k_psi * err in the executed QP solution.
        gN = project_onto_task_nullspace(J, g, sigma_min=sigma_min)
        denom = float(np.dot(g, gN))
        err = float(self._psi_ref_unwrapped) - psi
        _, _, obs = self._sw_observability(q)
        smooth = 1.0 - np.exp(-self.cfg.obs_decay_gain * obs * obs)
        floor = float(np.clip(self.cfg.obs_smooth_floor, 0.0, 1.0))
        smooth = max(float(smooth), floor)
        self.last_singularity_smooth = float(smooth)
        safe_denom = max(denom, 0.0) + self.cfg.safe_denom_eps
        engage_s = max(float(getattr(self.cfg, "engage_s", 0.0)), 0.0)
        if engage_s > 1.0e-9:
            u = float(np.clip((time.monotonic() - self._engage_t) / engage_s, 0.0, 1.0))
            ramp = u * u * (3.0 - 2.0 * u)
        else:
            ramp = 1.0
        qdot = ramp * smooth * self.cfg.k_psi * err * gN / safe_denom
        v_cap = self.cfg.max_qdot_frac * np.asarray(self.kin.v_max, dtype=float)
        return np.clip(qdot, -v_cap, v_cap)
