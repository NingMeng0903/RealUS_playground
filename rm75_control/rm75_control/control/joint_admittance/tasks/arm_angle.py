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

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance.ik_types import project_onto_task_nullspace
from rm75_control.control.joint_admittance.model import RobotKinematics

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
    max_qdot_frac: float = 0.15   # clip |qdot| to this fraction of v_max per joint


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
        self._model = kin.model
        self._data = self._model.createData()
        self._jids = tuple(
            self._model.getJointId(n) for n in (_SHOULDER_JOINT, _ELBOW_JOINT, _WRIST_JOINT)
        )
        self.last_singularity_smooth: float = 1.0

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
        """Swivel angle psi(q) in (-pi, pi]."""
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

    def grad_arm_angle(self, q_rad: np.ndarray) -> np.ndarray:
        """d psi / d q via central differences (7 dof, ~us-scale FK each)."""
        q = np.asarray(q_rad, dtype=float)
        eps = self.cfg.fd_eps_rad
        g = np.zeros_like(q)
        for i in range(q.size):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            g[i] = _wrap_pi(self.arm_angle(qp) - self.arm_angle(qm)) / (2.0 * eps)
        return g

    # ---- task interface ------------------------------------------------------
    def reset(self, q_rad: np.ndarray) -> None:
        """Capture psi_ref from the current configuration if not already set
        (by config or an explicit set_reference from the application)."""
        if self.psi_ref is None:
            self.psi_ref = self.arm_angle(q_rad)

    def set_reference(self, psi_ref_rad: float) -> None:
        self.psi_ref = float(psi_ref_rad)

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.reset(q)
        psi = self.arm_angle(q)
        g = self.grad_arm_angle(q)
        if float(np.dot(g, g)) < 1e-10:
            return np.zeros_like(q)
        J = self.kin.jacobian(q)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        M = self.kin.mass_matrix(q)
        # Direction: ORTHOGONAL kinematic projection of the gradient - safe by
        # construction (g . N_kin g = ||N_kin g||^2 >= 0).  Do NOT take the
        # direction from the oblique N_dyn: with the RM75's tiny wrist
        # inertias (~1e-4 kg m^2) M^{-1} amplifies the damped projection's
        # out-of-nullspace residue and the "dynamically consistent" direction
        # can point the WRONG way along psi (observed: psi diverging then
        # parking at the qdot clip - on hardware, the nullspace twist-then-
        # oscillate failure).
        gN = project_onto_task_nullspace(J, g, sigma_min=sigma_min)
        # Normalization: against the direction the QP will actually execute
        # (it re-projects the composed secondary through the inertia-floored
        # N_dyn), so d(psi)/dt ~= k_psi * err in EXECUTED motion, not just in
        # the commanded vector.  Falls back to the kinematic denom if the
        # executed-metric one degenerates.
        gN_exec = project_onto_task_nullspace(
            J, gN, sigma_min=sigma_min, M=M, use_dyn=True
        )
        denom = float(np.dot(g, gN_exec))
        if denom < 1e-6:
            denom = float(np.dot(g, gN))
        if denom < 1e-10:
            # psi not controllable within the task nullspace at this q
            return np.zeros_like(q)
        err = _wrap_pi(float(self.psi_ref) - psi)
        _, _, obs = self._sw_observability(q)
        smooth = 1.0 - np.exp(-self.cfg.obs_decay_gain * obs * obs)
        self.last_singularity_smooth = float(smooth)
        safe_denom = denom + self.cfg.safe_denom_eps
        qdot = smooth * self.cfg.k_psi * err * gN / safe_denom
        v_cap = self.cfg.max_qdot_frac * np.asarray(self.kin.v_max, dtype=float)
        return np.clip(qdot, -v_cap, v_cap)
