"""WBC velocity-IK core: slack-variable QP + CBF self-collision constraints.

Formulation (Escande et al. 2014 slack task + Faverjon velocity damper / Khazoom CBF):

    x = [qdot; w]  in R^{nv+6}

    min  0.5 (qdot - qdot_nom)^T W_reg (qdot - qdot_nom) + 0.5 w^T W_task w
    s.t. J_tcp qdot - w = v_cmd                     (equality)
         l_box <= qdot <= u_box                     (joint boxes)
         J_col qdot >= v_safe                       (CBF, optional)

H is block-diagonal (no J^T J).  ProxQP warm-started each tick.

This layer consumes a *given* task twist ``v_cmd`` verbatim (Escande et al. 2014
Sec. III): the position-feedback loop that produces the twist lives exactly once
in the caller (outer loop / pose_ik), never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.control.joint_admittance.collision_model import (
    CollisionConfig,
    CollisionModel,
)
from rm75_control.control.joint_admittance.ik_types import (
    IkStepResult,
    SrDampingConfig,
    project_onto_task_nullspace,
    sr_damping_lambda,
)
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.solver.cbf_constraints import build_cbf_rows
from rm75_control.control.joint_admittance.solver.constraint_mgr import (
    VelocityBoxConstraints,
    build_wbc_inequalities,
)
from rm75_control.control.joint_admittance.utils.safety import SafetyLimits

N_SLACK = 6


@dataclass
class QpConfig:
    task_weight: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)
    )
    reg: np.ndarray = field(default_factory=lambda: np.full(7, 1.0e-2))
    backend: str = "proxqp"
    eps_abs: float = 1e-6
    max_iter: int = 200
    euler_order: str = "xyz"
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping: SrDampingConfig = field(default_factory=SrDampingConfig)
    # Escande slack task weight vs σ_min: when J is ill-conditioned, holding
    # W_task constant makes the solver chase mm-level Cartesian error by
    # slamming qdot into the velocity box (limit-cycle chatter).  Scale
    # W_task ∝ max((σ/σ_ref)², task_weight_min_frac) so slack w absorbs
    # infeasible twist instead of joint-limit ping-pong.  SR projection handles
    # nullspace; this handles the PRIMARY equality — complementary roles.
    task_weight_min_frac: float = 0.01
    # Weight QP reg by diag(M(q)) for dynamics-consistent nullspace resolution.
    use_mass_weighted_reg: bool = True
    # Floor on diag(M) in the mass-weighted reg: wrist inertias are ~1e-3,
    # which drove the effective reg to ~1e-6 x task_weight and ill-conditioned
    # the QP (occasional ProxQP failures = one-tick freezes).
    mass_reg_floor: float = 0.05
    # Use Khatib N_dyn instead of kinematic N in secondary projection.
    use_dyn_nullspace: bool = True
    # Faverjon/Tournassoud joint-limit velocity damper band (rad): allowed
    # speed toward a limit ramps to 0 across this zone before the margin.
    limit_damper_band_rad: float = 0.15


class _ProxQpWbcBackend:
    def __init__(self, nv: int, max_cbf: int, cfg: QpConfig) -> None:
        import proxsuite

        self._px = proxsuite
        self.nv = nv
        self.n_slack = N_SLACK
        self.n_var = nv + self.n_slack
        self.n_eq = N_SLACK
        self.n_in = nv + max_cbf
        self.qp = proxsuite.proxqp.dense.QP(self.n_var, self.n_eq, self.n_in)
        self._eps_tight = float(cfg.eps_abs)
        self._eps_loose = max(self._eps_tight * 100.0, 1.0e-4)
        self.qp.settings.eps_abs = self._eps_tight
        self.qp.settings.max_iter = cfg.max_iter
        self.qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        self._initialized = False
        self.fail_count = 0
        self._warn_every = 25
        self._warn_seen = 0

    def _status(self):
        return self.qp.results.info.status

    def _solved(self) -> bool:
        return self._status() == self._px.proxqp.QPSolverOutput.PROXQP_SOLVED

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        if not self._initialized:
            self.qp.init(H, g, A, b, C, lo, hi)
            self._initialized = True
        else:
            # Warm-start fuse: reusing multipliers from a failed tick poisons the
            # next solve (MAX_ITER death spiral from tick 1 onward).  Cold-start
            # only while recovering; restore warm-start after a clean solve.
            if self.fail_count > 0:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                )
            else:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
            self.qp.settings.eps_abs = self._eps_tight
            self.qp.update(H=H, g=g, A=A, b=b, C=C, l=lo, u=hi)

        self.qp.solve()

        if not self._solved():
            self.qp.settings.initial_guess = (
                self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
            )
            self.qp.settings.eps_abs = self._eps_loose
            self.qp.solve()

        if not self._solved():
            self.fail_count += 1
            self._warn_seen += 1
            if self._warn_seen % self._warn_every == 1:
                print(
                    f"[WBC WARN] ProxQP {self._status()} "
                    f"(fail_count={self.fail_count}, "
                    f"suppressing next {self._warn_every - 1})",
                    flush=True,
                )
            return None

        self.fail_count = 0
        self._warn_seen = 0
        return np.asarray(self.qp.results.x, dtype=float)


class _OsqpWbcBackend:
    """Fallback when ProxQP unavailable (no warm equality+ineq resize)."""

    def __init__(self, nv: int, max_cbf: int, cfg: QpConfig) -> None:
        import osqp
        import scipy.sparse as sp

        self._osqp = osqp
        self._sp = sp
        self.nv = nv
        self.n_slack = N_SLACK
        self.n_var = nv + self.n_slack
        self.n_in = nv + max_cbf
        self.cfg = cfg
        self.prob = None

    def solve(self, H, g, A, b, C, lo, hi):
        sp = self._sp
        A_full = np.vstack([C, A])
        l_full = np.concatenate([lo, b])
        u_full = np.concatenate([hi, b])
        P = sp.csc_matrix(np.triu(H))
        A_csc = sp.csc_matrix(A_full)
        if self.prob is None:
            self.prob = self._osqp.OSQP()
            self.prob.setup(
                P, g, A_csc, l_full, u_full,
                verbose=False, warm_start=True,
                eps_abs=self.cfg.eps_abs, eps_rel=self.cfg.eps_abs,
                max_iter=self.cfg.max_iter,
            )
        else:
            self.prob.update(Px=P.data, q=g, Ax=A_csc.data, l=l_full, u=u_full)
        res = self.prob.solve()
        if res.x is None or np.any(np.isnan(res.x)):
            return None
        return np.asarray(res.x, dtype=float)


class QpIkController:
    """Slack-variable WBC velocity-IK core: (q, v_cmd) -> qdot."""

    def __init__(
        self,
        kin: RobotKinematics,
        limits: SafetyLimits,
        cfg: QpConfig | None = None,
        collision: CollisionModel | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or QpConfig()
        self.constraints = VelocityBoxConstraints(
            limits, damper_band_rad=self.cfg.limit_damper_band_rad
        )
        self.collision_cfg = self.cfg.collision
        self._max_cbf = max(1, int(self.collision_cfg.max_pairs))
        self.collision = collision
        if self.collision_cfg.enabled and self.collision is None:
            self.collision = CollisionModel(kin.model)
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self.backend = self._make_backend(kin.nv)

        w_reg = np.asarray(self.cfg.reg, dtype=float)
        if w_reg.ndim == 0 or w_reg.size == 1:
            w_reg = np.full(kin.nv, float(w_reg))
        self._w_reg = w_reg
        self._w_task = np.asarray(self.cfg.task_weight, dtype=float)

    def _make_backend(self, nv: int):
        want = self.cfg.backend.lower()
        if want == "proxqp":
            try:
                return _ProxQpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception:
                pass
        if want in ("osqp", "proxqp"):
            try:
                return _OsqpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception as exc:
                raise RuntimeError(
                    "No QP backend available (install proxsuite or osqp)"
                ) from exc
        raise ValueError(f"unknown QP backend {self.cfg.backend!r}")

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__.replace("_", "").replace("Backend", "").lower()

    def reset(self, q0_rad: np.ndarray) -> None:
        del q0_rad
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)

    def set_collision_enabled(self, enabled: bool) -> None:
        self.collision_cfg.enabled = bool(enabled)

    def step(
        self,
        q_prev: np.ndarray,
        twist_ref: np.ndarray,
        dt: float,
        secondary_qdot: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float = 0.0,
    ) -> IkStepResult:
        q_prev = np.asarray(q_prev, dtype=float)
        v_cmd = np.asarray(twist_ref, dtype=float)

        J = self.kin.jacobian(q_prev)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())

        nv = self.kin.nv
        ns = N_SLACK
        n_var = nv + ns

        # Chiaverini SR projection: λ(σ) grows as σ→0 so N→I and secondary
        # tasks / qdot_ff keep control of singular directions.
        proj_damping = sr_damping_lambda(sigma_min, self.cfg.sr_damping)
        M = self.kin.mass_matrix(q_prev) if self.cfg.use_dyn_nullspace or self.cfg.use_mass_weighted_reg else None
        qdot_nom = (
            project_onto_task_nullspace(
                J,
                secondary_qdot,
                damping=proj_damping,
                sigma_min=sigma_min,
                sr_cfg=self.cfg.sr_damping,
                M=M,
                use_dyn=self.cfg.use_dyn_nullspace and M is not None,
            )
            if secondary_qdot is not None
            else np.zeros(nv, dtype=float)
        )

        w_reg = self._w_reg.copy()
        w_task = self._w_task.copy()
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        if sigma_ref > 1e-9 and sigma_min < sigma_ref:
            frac = sigma_min / sigma_ref
            task_scale = max(frac * frac, self.cfg.task_weight_min_frac)
            w_task *= task_scale

        H = np.zeros((n_var, n_var), dtype=float)
        if self.cfg.use_mass_weighted_reg and M is not None:
            m_diag = np.maximum(np.diag(M), self.cfg.mass_reg_floor)
            H[:nv, :nv] = np.diag(w_reg * m_diag)
        else:
            H[:nv, :nv] = np.diag(w_reg)
        H[nv:, nv:] = np.diag(w_task)
        g = np.zeros(n_var, dtype=float)
        g[:nv] = -np.diag(H[:nv, :nv]) * qdot_nom if self.cfg.use_mass_weighted_reg and M is not None else -w_reg * qdot_nom

        A = np.zeros((ns, n_var), dtype=float)
        A[:, :nv] = J
        A[:, nv:] = -np.eye(ns)
        b = v_cmd

        lo_box, hi_box = self.constraints.bounds(
            q_prev, dt, self.qdot_prev, q_meas=q_meas, resync_err=resync_err
        )
        if self.collision is not None and self.collision_cfg.enabled:
            cbf = build_cbf_rows(self.collision, self.kin, q_prev, self.collision_cfg)
        else:
            from rm75_control.control.joint_admittance.solver.cbf_constraints import CbfRows

            cbf = CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

        C, lo, hi = build_wbc_inequalities(
            nv, ns, lo_box, hi_box, cbf, self._max_cbf
        )

        x = self.backend.solve(
            np.ascontiguousarray(H),
            np.ascontiguousarray(g),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(C),
            np.ascontiguousarray(lo),
            np.ascontiguousarray(hi),
        )
        if x is None:
            # Solver failure: decay the previous velocity instead of a hard
            # qdot=0 (which was a one-tick full stop mid-motion - a jerk the
            # drivers see as a discontinuity).  The decayed command stays
            # inside the previous tick's feasible box by construction.
            qdot = 0.5 * self.qdot_prev
            slack = np.zeros(ns, dtype=float)
        else:
            qdot = x[:nv]
            slack = x[nv:]
        self.qdot_prev = qdot
        q_next = q_prev + qdot * dt
        return IkStepResult(
            q_next=q_next,
            qdot=qdot,
            sigma_min=sigma_min,
            manip=self.kin.manipulability(J),
            slack_norm=float(np.linalg.norm(slack)),
            n_cbf_active=int(cbf.jacobian.shape[0]),
        )
