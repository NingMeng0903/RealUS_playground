"""WBC velocity-IK core: strict two-level QP + CBF self-collision constraints.

Formulation (Escande et al. 2014 slack task + Faverjon velocity damper / Khazoom CBF):

    x = [qdot; w]  in R^{nv+6}

    QP1: min 0.5 wᵀ W_task w
         J_task qdot - w = v_cmd                   (protected equality)
         l_box <= qdot <= u_box, J_col qdot >= v_safe

    QP2: keep QP1's achieved Cartesian velocity as a hard equality on the
         *full* Jacobian (including the next rail command).  Attractors may
         only move in the realizable TCP nullspace; a rail preference cannot
         buy task slack.

H is block-diagonal (no J^T J).  ProxQP warm-started each tick.

This layer consumes a *given* task twist ``v_cmd`` verbatim (Escande et al. 2014
Sec. III): the position-feedback loop that produces the twist lives exactly once
in the caller (outer loop / pose_ik), never here.  If ``rail_exec_vel_m_s`` is
provided, its measured TCP contribution is subtracted from the current task;
the rail command remains a next-sample secondary decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
)
from rm75_control.control.joint_admittance_8dof.ik_types import (
    IkStepResult,
    SrDampingConfig,
    project_onto_task_nullspace,
    sr_damping_lambda,
)
from rm75_control.control.joint_admittance_8dof.solver import cpp_kernel
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.qp_cert import (
    measure_qdot_box,
    qp_status_name,
)
from rm75_control.control.joint_admittance_8dof.solver.branch_barrier import (
    BranchBarrierBuilder,
    BranchBarrierConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
    J4DesignComfortBuilder,
    J4DesignComfortConfig,
    JointComfortBuilder,
    JointComfortConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    CbfRows,
    CbfSlotTracker,
    build_cbf_rows,
)
from rm75_control.control.joint_admittance_8dof.filters import (
    first_order_lpf,
    first_order_lpf_vec,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    RAIL_BIND_BRANCH,
    RAIL_BIND_COLLAPSE,
    RAIL_BIND_NONE,
    VelocityBoxConstraints,
    build_wbc_inequalities,
    collapse_interval,
    note_rail_bind,
)
from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    PrefInequalityRows,
    SigmaSetBasedConfig,
    SigmaSetBasedTracker,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

N_TASK_SLACK = 6
N_PREF_SLACK = 9  # [sigma, branch, J1..J7 comfort]
MAX_PREF_ROWS = 16  # 1 sigma + 7 branch + 7 comfort
# Backward-compatible alias used by older call sites / tests.
N_SLACK = N_TASK_SLACK


@dataclass
class QpConfig:
    task_weight: np.ndarray = field(
        default_factory=lambda: np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0], dtype=float)
    )
    # Effort allocation for ultrasound scanning on a 7-DOF arm + rail:
    #
    #   idx 0   rail (prismatic, m)      1.0e-2  — same as shoulder; primary
    #                                              task recruits rail for base-Y
    #                                              when sigma dips. Secondary
    #                                              rail drive is zeroed in qp;
    #                                              patient limits are v_max /
    #                                              a_max_rail, not a 5x reg tax.
    #   idx 1-4 shoulder/elbow           1.0e-2  — base motion is fine for
    #                                              gross pose adjustments.
    #   idx 5-7 wrist 1/2/3              5.0e-3  — cheapest: fine-scale
    #                                              orientation (probe tilt)
    #                                              is exactly what a scan
    #                                              wants to do with the
    #                                              wrist, not the shoulder.
    #
    # With ``use_mass_weighted_reg=True`` these baseline weights are further
    # multiplied by ``max(diag(M(q)), mass_reg_floor)`` — heavier joints
    # (shoulder) become naturally more expensive than the wrist even inside
    # the arm cluster.  Mass weighting keeps shoulder dearer than wrist; rail
    # joins the primary equality when the arm Jacobian is ill-conditioned.
    reg: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.2e-2, 1.2e-2, 1.2e-2],
            dtype=float,
        )
    )
    backend: str = "proxqp"
    use_cpp_kernel: bool = True
    eps_abs: float = 1e-6
    max_iter: int = 400
    # Clamp applied in ProxQP backend so a yaml typo (e.g. 3000) cannot freeze
    # the 200 Hz loop for seconds near singularities / CBF.
    max_iter_cap: int = 400
    euler_order: str = "xyz"
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping: SrDampingConfig = field(default_factory=SrDampingConfig)
    # σ-adaptive primary-task weight (Chiaverini-style): as σ_min ↘, scale
    # W_task toward task_weight_min_frac so the slack absorbs infeasible
    # v_cmd instead of saturating qdot with near-zero TCP motion.  LPF on the
    # scale avoids the bang-bang chatter that motivated the (over-broad) Bug 1
    # removal — only the primary cost softens; rail_extension / reg stay put.
    task_weight_min_frac: float = 0.05
    task_weight_lpf_tau_s: float = 0.25
    # Chiaverini 1994 numerical filtering: only the degenerate left
    # singular directions of W^{1/2} J lose task weight.  Off falls
    # back to the isotropic (σ_min / σ_ref)² scale.
    aniso_task_damping: bool = True
    # Weight QP reg by diag(M(q)) for dynamics-consistent nullspace resolution.
    use_mass_weighted_reg: bool = True
    # Floor on diag(M) in the mass-weighted reg: wrist inertias are ~1e-3,
    # which drove the effective reg to ~1e-6 x task_weight and ill-conditioned
    # the QP (occasional ProxQP failures = one-tick freezes).
    mass_reg_floor: float = 0.05
    # Exempt the rail (joint 0) from mass weighting.  diag(M)[0] is the full
    # carriage + arm mass (~9.8 kg on the RM75 rig), which priced rail motion
    # 30-400x above the arm joints: the QP stretched the arm to near-straight
    # (sigma_arm ~ 0.03) before rail motion became marginally cheaper.  With
    # the exemption the rail's effective reg is exactly ``reg[0]`` — an
    # absolute, yaml-tunable cost, sized against the arm's mass-weighted regs.
    mass_weight_exempt_rail: bool = True
    # LPF time constant (s) on the mass-weighted reg diagonal.  diag(M(q))
    # re-evaluated every tick makes H change tick-to-tick, degrading ProxQP
    # warm starts (a vibration input near singular poses where iteration
    # counts already spike).  0 disables (legacy per-tick behaviour).
    mass_reg_lpf_tau_s: float = 0.2
    # Faverjon/Tournassoud joint-limit velocity damper band: allowed speed
    # toward a limit ramps to 0 across this zone before the margin.  Units are
    # PER JOINT: rad for the arm, metres for the prismatic rail.  The old
    # scalar band applied 0.15 "rad" = 0.15 m to the rail — the damper started
    # throttling rail velocity from |y| > 6.5 cm (60% of the ±0.25 m travel),
    # exactly where the rail is needed most to rescue arm singularities.
    limit_damper_band_rad: float = 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: float = 0.01   # rail joint 0 (metres)
    # Rail stopping-envelope look-ahead.  0 uses the control period only.
    limit_damper_rail_reaction_s: float = 0.06
    warn_on_fail: bool = True
    # Unused yaml keys are accepted by config.py and ignored here.
    twist_sigma_floor: float = 0.02
    sigma_setbased: SigmaSetBasedConfig = field(default_factory=SigmaSetBasedConfig)
    branch_barrier: BranchBarrierConfig = field(default_factory=BranchBarrierConfig)
    joint_comfort: JointComfortConfig = field(default_factory=JointComfortConfig)
    j4_design_comfort: J4DesignComfortConfig = field(default_factory=J4DesignComfortConfig)
    # Arm joints this close to a stop count as physically saturated (rad).
    near_arm_margin_rad: float = 0.08
    # Soft velocity continuity: ½ w_s ‖q̇ − q̇_prev‖² added to the QP cost
    # (no extra decision variable).  0 disables.
    # May be a scalar or one value per joint.  A vector lets the rail use no
    # velocity-continuity preference while the arm keeps its tuned value.
    smoothness_weight: float | np.ndarray = 0.15
    # Third-order box on |a_k - a_{k-1}|.  The velocity and acceleration boxes
    # alone let the commanded acceleration flip sign every tick; this bounds
    # how fast it may turn.  0 disables either axis.
    j_max_arm_rad_s3: float = 300.0
    j_max_rail_m_s3: float = 120.0;


class _ProxQpWbcBackend:
    def __init__(
        self,
        nv: int,
        max_cbf: int,
        cfg: QpConfig,
        *,
        n_eq: int = N_TASK_SLACK,
        allow_retry: bool = True,
    ) -> None:
        import proxsuite

        self._px = proxsuite
        self.nv = nv
        self.n_task_slack = N_TASK_SLACK
        self.n_pref_slack = N_PREF_SLACK
        self.n_slack = N_TASK_SLACK  # task equality slacks only
        self.n_var = nv + N_TASK_SLACK + N_PREF_SLACK
        self.n_eq = int(n_eq)
        self.n_in = nv + max_cbf + MAX_PREF_ROWS + N_PREF_SLACK
        self.qp = proxsuite.proxqp.dense.QP(self.n_var, self.n_eq, self.n_in)
        self._eps_tight = float(cfg.eps_abs)
        # Retry tolerance near singularities: ProxQP hits MAX_ITER when the
        # equality Jqdot=w+v_cmd is nearly rank-deficient (σ→0).  A ~100x
        # looser eps on the retry lets the solver accept "good enough" without
        # a full-stop fallback; typical converged residuals are already
        # 1e-5..1e-4 in this regime.
        self._eps_loose = max(self._eps_tight * 100.0, 1.0e-4)
        # Store max_iter locally — do NOT keep self.cfg (retry must not touch it).
        # Cap for realtime: yaml historically had 3000 and a single failed tick
        # could hold the GIL for >10 s (looks like mid-MoveJ freeze, no fault).
        cap = int(getattr(cfg, "max_iter_cap", 400) or 400)
        self._max_iter = int(min(max(int(cfg.max_iter), 1), max(cap, 1)))
        self.qp.settings.eps_abs = self._eps_tight
        self.qp.settings.max_iter = self._max_iter
        self.qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        self._initialized = False
        self.fail_count = 0
        self._warn_on_fail = bool(cfg.warn_on_fail)
        # Rate-limit MAX_ITER warnings: at 200 Hz a singular pose can spam
        # thousands of identical lines and itself starve the control loop.
        self._warn_every = 25
        self._warn_seen = 0
        self._tick_s = 5.0e-3
        # Strict HQP uses one solve per level.  Keep the legacy retry for the
        # old constructor/API, but never retry a strict level: a retry here
        # would silently turn the fixed two-solve budget into an SNS loop.
        self._allow_retry = bool(allow_retry)
        self.last_solve_ms = 0.0
        self.last_status = "not_run"
        self.last_iter = 0

    def _status(self):
        return self.qp.results.info.status

    def _status_name(self) -> str:
        return qp_status_name(self._status())

    def _solved(self) -> bool:
        s = self._status()
        return s in (
            self._px.proxqp.QPSolverOutput.PROXQP_SOLVED,
            self._px.proxqp.QPSolverOutput.PROXQP_MAX_ITER_REACHED,
        )

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        *,
        warm_start_x: np.ndarray | None = None,
    ) -> np.ndarray:
        import time as _time

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
            self.qp.settings.max_iter = self._max_iter
            self.qp.update(H=H, g=g, A=A, b=b, C=C, l=lo, u=hi)

        t0 = _time.perf_counter()
        if warm_start_x is not None:
            seed = np.asarray(warm_start_x, dtype=float).reshape(self.n_var)
            self.qp.settings.initial_guess = self._px.proxqp.InitialGuess.WARM_START
            self.qp.solve(seed, None, None)
        else:
            self.qp.solve()
        elapsed = _time.perf_counter() - t0
        self.last_solve_ms = elapsed * 1000.0
        self.last_status = self._status_name()
        self.last_iter = int(getattr(self.qp.results.info, "iter", 0) or 0)

        if not self._solved() and self._allow_retry:
            remaining = self._tick_s - elapsed
            if remaining > 1.0e-3:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                )
                self.qp.settings.eps_abs = self._eps_loose
                retry_iters = int(
                    min(max(int(self._max_iter), 1), 200, max(int(remaining / 0.00005), 20))
                )
                self.qp.settings.max_iter = retry_iters
                self.qp.solve()
                self.qp.settings.max_iter = int(self._max_iter)
                self.last_solve_ms = (
                    _time.perf_counter() - t0
                ) * 1000.0
                self.last_status = self._status_name()
                self.last_iter = int(getattr(self.qp.results.info, "iter", 0) or 0)

        if not self._solved():
            self.fail_count += 1
            self._warn_seen += 1
            if self._warn_on_fail and self._warn_seen % self._warn_every == 1:
                print(
                    f"[WBC WARN] ProxQP {self._status()} "
                    f"(fail_count={self.fail_count}, "
                    f"suppressing next {self._warn_every - 1})",
                    flush=True,
                )
            return None

        self.fail_count = 0
        self._warn_seen = 0
        self.last_status = self._status_name()
        return np.asarray(self.qp.results.x, dtype=float)


class _OsqpWbcBackend:
    """Fallback when ProxQP unavailable (no warm equality+ineq resize)."""

    def __init__(
        self,
        nv: int,
        max_cbf: int,
        cfg: QpConfig,
        *,
        allow_retry: bool = False,
    ) -> None:
        import osqp
        import scipy.sparse as sp

        self._osqp = osqp
        self._sp = sp
        self.nv = nv
        self.n_task_slack = N_TASK_SLACK
        self.n_pref_slack = N_PREF_SLACK
        self.n_slack = N_TASK_SLACK
        self.n_var = nv + N_TASK_SLACK + N_PREF_SLACK
        self.n_in = nv + max_cbf + MAX_PREF_ROWS + N_PREF_SLACK
        self.cfg = cfg
        self.prob = None
        self.last_solve_ms = 0.0
        self.last_status = "not_run"
        self.last_iter = 0
        self._allow_retry = bool(allow_retry)

    def solve(self, H, g, A, b, C, lo, hi, *, warm_start_x=None):
        import time as _time

        sp = self._sp
        t0 = _time.perf_counter()
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
        if warm_start_x is not None:
            self.prob.warm_start(x=np.asarray(warm_start_x, dtype=float))
        res = self.prob.solve()
        self.last_solve_ms = (_time.perf_counter() - t0) * 1000.0
        if res.x is None or np.any(np.isnan(res.x)):
            self.last_status = "failed"
            return None
        self.last_status = "solved"
        self.last_iter = int(getattr(res, "iter", 0) or 0)
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
        task_weight = np.asarray(self.cfg.task_weight, dtype=float).reshape(-1)
        if (
            task_weight.size != N_TASK_SLACK
            or not np.all(np.isfinite(task_weight))
            or np.any(task_weight <= 0.0)
        ):
            raise ValueError(
                "task_weight must contain six finite, strictly positive values"
            )
        reg_weight = np.asarray(self.cfg.reg, dtype=float).reshape(-1)
        if reg_weight.size not in (1, int(kin.nv)):
            raise ValueError(
                f"reg must be scalar or contain {int(kin.nv)} values"
            )
        if not np.all(np.isfinite(reg_weight)) or np.any(reg_weight < 0.0):
            raise ValueError("reg must contain finite, non-negative values")
        # Per-joint damper band: arm in rad, prismatic rail (joint 0) in m.
        damper_band = np.full(kin.nv, float(self.cfg.limit_damper_band_rad))
        damper_band[0] = float(self.cfg.limit_damper_band_rail_m)
        jd = self.cfg.j4_design_comfort
        self.constraints = VelocityBoxConstraints(
            limits,
            damper_band_rad=damper_band,
            rail_reaction_s=float(self.cfg.limit_damper_rail_reaction_s),
            j4_design_enabled=bool(jd.enabled),
            j4_design_lo=float(jd.lower_rad),
            j4_design_hi=float(jd.upper_rad),
            j4_design_gamma=float(jd.gamma),
        )
        self.collision_cfg = self.cfg.collision
        self._max_cbf = max(1, int(self.collision_cfg.max_pairs))
        self.collision = collision
        if self.collision_cfg.enabled and self.collision is None:
            self.collision = CollisionModel(kin.model)
        self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        self.sigma_setbased = SigmaSetBasedTracker(self.cfg.sigma_setbased)
        self.branch_barrier = BranchBarrierBuilder(self.cfg.branch_barrier)
        self.joint_comfort = JointComfortBuilder(self.cfg.joint_comfort)
        self.j4_design_comfort = J4DesignComfortBuilder(self.cfg.j4_design_comfort)
        self.last_j4_design_slack = 0.0
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self.qdot_prev2 = np.zeros(kin.nv, dtype=float)
        self._qdot_prev_seen = np.zeros(kin.nv, dtype=float)
        j_max = np.full(kin.nv, float(self.cfg.j_max_arm_rad_s3), dtype=float)
        j_max[0] = float(self.cfg.j_max_rail_m_s3)
        self._j_max = j_max if np.all(j_max > 0.0) else None
        self._m_diag_lpf: np.ndarray | None = None
        self._task_scale_lpf: float = 1.0
        self._task_weight_state_init: bool = False
        self._s_lpf: np.ndarray | None = None
        self._U_prev: np.ndarray | None = None
        self.last_task_weight_mat: np.ndarray = np.diag(
            np.asarray(self.cfg.task_weight, dtype=float)
        )
        self.last_s_sigma: np.ndarray = np.ones(N_TASK_SLACK, dtype=float)
        self.solve_count = 0
        self.last_status = "not_run"
        self.last_failed = False
        self.last_dexterity_slack = 0.0
        self.last_branch_slack = 0.0
        self.last_comfort_slack = np.zeros(7, dtype=float)
        self.last_sns_scale = 1.0
        self.last_cbf_min_dist = float("nan")
        self.last_cbf_pair = ""
        self.last_cbf_active_names: tuple[str, ...] = ()
        self.last_comp_projected_frac = 0.0
        self.last_wln_scale = np.ones(kin.nv, dtype=float)
        self._wln_scale_prev = np.ones(kin.nv, dtype=float)
        self.q_star: np.ndarray | None = None
        self.q_star_signs: np.ndarray | None = None
        self.backend = self._make_backend(kin.nv, slot="qp1")
        # Both levels have six fixed equality rows.  QP1 uses
        # ``J qdot - residual = target``; QP2 directly locks
        # ``J qdot = achieved_qp1``.  The direct form avoids a redundant
        # 12-row [task; residual==0] system that ProxQP could misclassify as
        # infeasible near rank loss even though the QP1 point was feasible.
        self._backend_qp2 = self._make_backend(kin.nv, n_eq=N_TASK_SLACK, slot="qp2")

        # Strict-HQP telemetry.  These are controller attributes rather than
        # IkStepResult fields for backwards compatibility with existing loop
        # and CSV consumers; callers that need them can read them immediately
        # after ``step``.
        self.last_qp1_status = "not_run"
        self.last_qp2_status = "not_run"
        self.last_qp1_iter = 0
        self.last_qp2_iter = 0
        self.last_qp1_solve_ms = 0.0
        self.last_qp2_solve_ms = 0.0
        self.last_qp_total_ms = 0.0
        self.last_fallback_ms = 0.0
        self.last_task_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_residual_norm = 0.0
        self.last_qp1_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp1_residual_norm = 0.0
        self.last_qp2_residual_norm = 0.0
        self.last_task_target = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_achieved = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_exec_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_cmd_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_arm_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_fallback = False
        self.last_zero_slack_feasible = False
        self.last_hard_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qdot_qp1 = np.zeros(kin.nv, dtype=float)
        self.last_qp1_hard_violation = 0.0
        self.last_final_hard_violation = 0.0
        self.last_lo_box = np.full(kin.nv, -np.inf, dtype=float)
        self.last_hi_box = np.full(kin.nv, np.inf, dtype=float)
        self.last_rail_box_lo = 0.0
        self.last_rail_box_hi = 0.0
        self.last_rail_bind_lo = RAIL_BIND_NONE
        self.last_rail_bind_hi = RAIL_BIND_NONE
        self.last_rail_task_vel_used = 0.0
        self.last_rail_h1 = 0.0
        self.last_rail_h2 = 0.0
        self.last_rail_qdot_prev = 0.0
        self.last_rail_qdot_prev2 = 0.0
        self.last_qp2_seed_violation = 0.0
        self.last_qp2_seed_equality = 0.0
        # Final-publication certificate.  Retain the qdot-only hard set and
        # QP1 task value needed to verify the command that will actually be
        # sent.  Preference slack rows are deliberately excluded here.
        self.last_hard_cbf_jacobian = np.zeros((0, kin.nv), dtype=float)
        self.last_hard_cbf_lower = np.zeros(0, dtype=float)
        self.last_qp1_task_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_jacobian = np.zeros((N_TASK_SLACK, kin.nv), dtype=float)
        self.last_lock_jacobian = np.zeros((N_TASK_SLACK, kin.nv), dtype=float)
        self.last_lock_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_final_task_lock_violation = 0.0
        self.last_a_mirror_frac = float("nan")
        self.last_j_mirror_frac = float("nan")
        self.last_qp_overrun = False
        self._rail_exec_prev: float | None = None
        self._rail_a_prev: float | None = None

        w_reg = np.asarray(self.cfg.reg, dtype=float)
        if w_reg.ndim == 0 or w_reg.size == 1:
            w_reg = np.full(kin.nv, float(w_reg))
        self._w_reg = w_reg
        self._w_task = task_weight

    def _make_backend(
        self,
        nv: int,
        *,
        n_eq: int = N_TASK_SLACK,
        slot: str = "qp1",
    ):
        want = self.cfg.backend.lower()
        key = (want, int(nv), int(n_eq), int(self._max_cbf), str(slot))
        cache = getattr(self.kin, "_qp_backend_cache", None)
        if cache is None:
            cache = {}
            setattr(self.kin, "_qp_backend_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            return cached
        backend = None
        if want == "proxqp":
            try:
                backend = _ProxQpWbcBackend(
                    nv,
                    self._max_cbf,
                    self.cfg,
                    n_eq=n_eq,
                    allow_retry=False,
                )
            except Exception:
                backend = None
        if backend is None and want in ("osqp", "proxqp"):
            try:
                backend = _OsqpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception as exc:
                raise RuntimeError(
                    "No QP backend available (install proxsuite or osqp)"
                ) from exc
        if backend is None:
            raise ValueError(f"unknown QP backend {self.cfg.backend!r}")
        cache[key] = backend
        return backend

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__.replace("_", "").replace("Backend", "").lower()

    def reset(self, q0_rad: np.ndarray | None = None) -> None:
        del q0_rad  # QP state is velocity history / LPF only
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)
        self.qdot_prev2 = np.zeros(self.kin.nv, dtype=float)
        self._qdot_prev_seen = np.zeros(self.kin.nv, dtype=float)
        self._m_diag_lpf = None
        self._task_scale_lpf = 1.0
        self._task_weight_state_init = False
        self._s_lpf = None
        self._U_prev = None
        self.last_task_weight_mat = np.diag(np.asarray(self.cfg.task_weight, dtype=float))
        self.last_s_sigma = np.ones(N_TASK_SLACK, dtype=float)
        self.solve_count = 0
        self.last_status = "not_run"
        self.last_failed = False
        self.last_dexterity_slack = 0.0
        self.last_branch_slack = 0.0
        self.last_comfort_slack = np.zeros(7, dtype=float)
        self.last_sns_scale = 1.0
        self.last_cbf_min_dist = float("nan")
        self.last_cbf_pair = ""
        self.last_cbf_active_names = ()
        self.last_wln_scale = np.ones(self.kin.nv, dtype=float)
        self._wln_scale_prev = np.ones(self.kin.nv, dtype=float)
        self.last_qp1_status = "not_run"
        self.last_qp2_status = "not_run"
        self.last_qp1_iter = 0
        self.last_qp2_iter = 0
        self.last_qp1_solve_ms = 0.0
        self.last_qp2_solve_ms = 0.0
        self.last_qp_total_ms = 0.0
        self.last_fallback_ms = 0.0
        self.last_task_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_residual_norm = 0.0
        self.last_qp1_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp1_residual_norm = 0.0
        self.last_qp2_residual_norm = 0.0
        self.last_task_target = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_achieved = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_exec_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_cmd_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_arm_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_fallback = False
        self.last_hard_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qdot_qp1 = np.zeros(self.kin.nv, dtype=float)
        self.last_qp1_hard_violation = 0.0
        self.last_final_hard_violation = 0.0
        self.last_lo_box = np.full(self.kin.nv, -np.inf, dtype=float)
        self.last_hi_box = np.full(self.kin.nv, np.inf, dtype=float)
        self.last_rail_box_lo = 0.0
        self.last_rail_box_hi = 0.0
        self.last_rail_bind_lo = RAIL_BIND_NONE
        self.last_rail_bind_hi = RAIL_BIND_NONE
        self.last_rail_task_vel_used = 0.0
        self.last_rail_h1 = 0.0
        self.last_rail_h2 = 0.0
        self.last_rail_qdot_prev = 0.0
        self.last_rail_qdot_prev2 = 0.0
        self.last_qp2_seed_violation = 0.0
        self.last_qp2_seed_equality = 0.0
        self.last_hard_cbf_jacobian = np.zeros((0, self.kin.nv), dtype=float)
        self.last_hard_cbf_lower = np.zeros(0, dtype=float)
        self.last_qp1_task_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_jacobian = np.zeros(
            (N_TASK_SLACK, self.kin.nv), dtype=float
        )
        self.last_lock_jacobian = np.zeros(
            (N_TASK_SLACK, self.kin.nv), dtype=float
        )
        self.last_lock_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_final_task_lock_violation = 0.0
        self.last_a_mirror_frac = float("nan")
        self.last_j_mirror_frac = float("nan")
        self.last_qp_overrun = False
        self._rail_exec_prev = None
        self._rail_a_prev = None
        self.sigma_setbased.reset()
        self.branch_barrier.reset()
        self.joint_comfort.reset()
        self.j4_design_comfort.reset()
        self.last_j4_design_slack = 0.0

    def set_q_star(self, q_star: np.ndarray | None) -> None:
        """Homotopy / centering attractor (not necessarily yaml signs)."""
        if q_star is None:
            self.q_star = None
        else:
            self.q_star = np.asarray(q_star, dtype=float).reshape(-1).copy()

    def set_q_star_signs(self, q_star: np.ndarray | None) -> None:
        """Yaml-family signs for the near-zero branch barrier."""
        if q_star is None:
            self.q_star_signs = None
        else:
            self.q_star_signs = np.asarray(q_star, dtype=float).reshape(-1).copy()

    def sync_applied(self, qdot: np.ndarray) -> None:
        """Seed velocity history from an already-applied command."""
        self.qdot_prev = np.asarray(qdot, dtype=float).reshape(-1).copy()
        # An episode boundary is not a jerk event: start the third-order
        # history flat so the first tick is not boxed against a stale value.
        self.qdot_prev2 = self.qdot_prev.copy()
        self._qdot_prev_seen = self.qdot_prev.copy()

    def validate_final_qdot(self, qdot: np.ndarray) -> tuple[float, float]:
        """Certify a post-QP command against P0 and the QP1 task lock.

        Returns ``(hard_violation, task_lock_violation)`` as infinity norms.
        This is intentionally independent of QP2 preference slacks: only the
        velocity box, measured-rail CBF rows and the protected task value can
        make a hardware command unsafe or violate the hierarchy.
        """

        qdot_arr = np.asarray(qdot, dtype=float).reshape(-1)
        if qdot_arr.size != self.kin.nv or not np.all(np.isfinite(qdot_arr)):
            return float("inf"), float("inf")
        hard = max(
            float(np.max(np.maximum(self.last_lo_box - qdot_arr, 0.0), initial=0.0)),
            float(np.max(np.maximum(qdot_arr - self.last_hi_box, 0.0), initial=0.0)),
        )
        if self.last_hard_cbf_jacobian.size:
            cbf_value = self.last_hard_cbf_jacobian @ qdot_arr
            hard = max(
                hard,
                float(
                    np.max(
                        np.maximum(self.last_hard_cbf_lower - cbf_value, 0.0),
                        initial=0.0,
                    )
                ),
            )
        lock_jac = (
            self.last_lock_jacobian
            if self.last_lock_jacobian.size
            else self.last_task_jacobian
        )
        lock_vel = (
            self.last_lock_velocity
            if self.last_lock_velocity.size
            else self.last_qp1_task_velocity
        )
        if lock_jac.shape[1] != qdot_arr.size:
            return hard, float("inf")
        task_value = lock_jac @ qdot_arr
        task_lock = float(
            np.max(np.abs(task_value - lock_vel), initial=0.0)
        )
        return hard, task_lock

    def _task_scale_sigma(self, sigma_min: float, dt: float) -> float:
        """LPF-smoothed W_task scale in [min_frac, 1] from σ_min."""
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        raw = 1.0
        if sigma_ref > 1e-9 and sigma_min < sigma_ref:
            frac = float(sigma_min) / sigma_ref
            raw = max(frac * frac, float(self.cfg.task_weight_min_frac))
        tau = float(self.cfg.task_weight_lpf_tau_s)
        if not self._task_weight_state_init:
            self._task_scale_lpf = raw
            self._task_weight_state_init = True
            return float(self._task_scale_lpf)
        self._task_scale_lpf = first_order_lpf(
            self._task_scale_lpf, raw, dt, tau
        )
        return float(self._task_scale_lpf)

    def _task_weight_matrix(
        self,
        J: np.ndarray,
        dt: float,
        *,
        keep_task_weight: bool,
    ) -> np.ndarray:
        """Task slack Hessian block.  Aniso: only degenerate directions fade."""
        w = np.asarray(self._w_task, dtype=float).reshape(-1)
        ns = int(w.size)
        if keep_task_weight or not bool(getattr(self.cfg, "aniso_task_damping", True)):
            scale = 1.0 if keep_task_weight else self._task_scale_sigma(
                float(np.linalg.svd(J, compute_uv=False).min()), dt
            )
            mat = np.diag(w * scale)
            self.last_task_weight_mat = mat
            self.last_s_sigma = np.full(ns, scale, dtype=float)
            return mat
        w_sqrt = np.sqrt(np.maximum(w, 1.0e-12))
        jw = w_sqrt[:, None] * np.asarray(J, dtype=float)
        u, s_j, _vt = np.linalg.svd(jw, full_matrices=False)
        if u.shape[1] < ns:
            u_full = np.eye(ns, dtype=float)
            u_full[:, : u.shape[1]] = u
            u = u_full
            s_pad = np.zeros(ns, dtype=float)
            s_pad[: s_j.size] = s_j
            s_j = s_pad
        if self._U_prev is not None and self._U_prev.shape == u.shape:
            for i in range(u.shape[1]):
                if float(np.dot(u[:, i], self._U_prev[:, i])) < 0.0:
                    u[:, i] *= -1.0
        self._U_prev = u.copy()
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        min_frac = float(self.cfg.task_weight_min_frac)
        s_raw = np.ones(ns, dtype=float)
        for i, si in enumerate(s_j[:ns]):
            if sigma_ref > 1.0e-9 and float(si) < sigma_ref:
                s_raw[i] = max((float(si) / sigma_ref) ** 2, min_frac)
        tau = float(self.cfg.task_weight_lpf_tau_s)
        if (not self._task_weight_state_init) or self._s_lpf is None or self._s_lpf.size != ns:
            self._s_lpf = s_raw.copy()
            self._task_weight_state_init = True
        elif tau > 1.0e-9 and dt > 1.0e-9:
            self._s_lpf = first_order_lpf_vec(self._s_lpf, s_raw, dt, tau)
        else:
            self._s_lpf = s_raw.copy()
        self.last_s_sigma = np.asarray(self._s_lpf, dtype=float).copy()
        usu = u @ np.diag(self.last_s_sigma) @ u.T
        mat = (w_sqrt[:, None] * usu) * w_sqrt[None, :]
        self.last_task_weight_mat = mat
        return mat

    def _update_mirror_telemetry(
        self,
        J: np.ndarray,
        *,
        rail_exec: float | None,
        h1: float,
    ) -> None:
        """Fraction of the arm a/j boxes spent cancelling measured rail motion."""
        self.last_a_mirror_frac = float("nan")
        self.last_j_mirror_frac = float("nan")
        if rail_exec is None or not np.isfinite(float(rail_exec)):
            self._rail_exec_prev = None
            self._rail_a_prev = None
            return
        period = float(h1)
        if not np.isfinite(period) or period <= 1.0e-9:
            self._rail_exec_prev = float(rail_exec)
            return
        a_rail = 0.0
        if self._rail_exec_prev is not None:
            a_rail = (float(rail_exec) - float(self._rail_exec_prev)) / period
        self._rail_exec_prev = float(rail_exec)
        j_rail = 0.0
        if self._rail_a_prev is not None:
            j_rail = (a_rail - float(self._rail_a_prev)) / period
        self._rail_a_prev = float(a_rail)
        ja = np.asarray(J[:, 1:], dtype=float)
        jr = np.asarray(J[:, 0], dtype=float)
        if ja.size == 0:
            return
        # Telemetry only.  Skip the 6×7 pinv when the rail is not accelerating.
        if abs(a_rail) < 1.0e-9 and abs(j_rail) < 1.0e-9:
            self.last_a_mirror_frac = 0.0
            self.last_j_mirror_frac = 0.0
            return
        try:
            qa_dir, *_ = np.linalg.lstsq(ja, jr, rcond=None)
        except np.linalg.LinAlgError:
            return
        a_max = self.constraints.lim.a_max
        if a_max is not None:
            a_arm = np.asarray(a_max, dtype=float).reshape(-1)[1:]
            if a_arm.size:
                qa = qa_dir * a_rail
                den = np.maximum(np.abs(a_arm), 1.0e-9)
                self.last_a_mirror_frac = float(np.max(np.abs(qa) / den))
        if self._j_max is not None:
            j_arm = np.asarray(self._j_max, dtype=float).reshape(-1)[1:]
            if j_arm.size:
                qj = qa_dir * j_rail
                den = np.maximum(np.abs(j_arm), 1.0e-9)
                self.last_j_mirror_frac = float(np.max(np.abs(qj) / den))

    def set_collision_enabled(self, enabled: bool) -> None:
        self.collision_cfg.enabled = bool(enabled)

    def _merge_pref_rows(
        self, *parts: PrefInequalityRows
    ) -> PrefInequalityRows:
        jac_list = [p.jacobian for p in parts if p.active and p.jacobian.size]
        if not jac_list:
            nv = self.kin.nv
            return PrefInequalityRows(
                jacobian=np.zeros((0, nv)),
                slack_col=np.zeros(0, dtype=int),
                lower=np.zeros(0),
                active=False,
            )
        jac = np.vstack(jac_list)
        scol = np.concatenate([p.slack_col for p in parts if p.active and p.jacobian.size])
        lo = np.concatenate([p.lower for p in parts if p.active and p.jacobian.size])
        if jac.shape[0] > MAX_PREF_ROWS:
            jac = jac[:MAX_PREF_ROWS]
            scol = scol[:MAX_PREF_ROWS]
            lo = lo[:MAX_PREF_ROWS]
        return PrefInequalityRows(
            jacobian=jac, slack_col=scol.astype(int), lower=lo, active=True
        )

    def _solve_qp(self, backend, H, g, A, b, C, lo, hi, *, warm_start_x=None):
        if bool(getattr(self.cfg, "use_cpp_kernel", True)) and cpp_kernel.available():
            packed = cpp_kernel.solve_dense_qp(
                H,
                g,
                A,
                b,
                C,
                lo,
                hi,
                warm_x=warm_start_x,
                max_iter=int(
                    min(max(int(self.cfg.max_iter), 1), int(self.cfg.max_iter_cap))
                ),
                eps_abs=float(self.cfg.eps_abs),
            )
            if packed is not None:
                x, ms, status = packed
                backend.last_solve_ms = float(ms)
                backend.last_status = str(status)
                return x
        return backend.solve(H, g, A, b, C, lo, hi, warm_start_x=warm_start_x)

    def step(
        self,
        q_prev: np.ndarray,
        twist_ref: np.ndarray,
        dt: float,
        secondary_qdot: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_reg_scale: float = 1.0,
        rail_reg_scale: float = 1.0,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        zero_secondary_rail: bool = False,
        rail_task_vel_m_s: float | None = None,
        rail_task_weight: float = 0.0,
        box_dt: float | None = None,
        box_h1: float | None = None,
        box_h2: float | None = None,
        keep_task_weight: bool = False,
        pref_slack_scale: float = 1.0,
        rail_exec_vel_m_s: float | None = None,
        jacobian: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        mass_matrix: np.ndarray | None = None,
        kinematics_ready: bool = False,
        rail_open_travel: bool = False,
        arm_qdot_pref: np.ndarray | None = None,
    ) -> IkStepResult:
        t_total = time.perf_counter()
        q_prev = np.asarray(q_prev, dtype=float).reshape(-1)
        nv = self.kin.nv
        if q_prev.size != nv:
            raise ValueError(f"q_prev must have {nv} joints, got {q_prev.size}")
        # ``qdot_prev`` is whatever the loop actually applied last tick (it may
        # rewrite it after clamping), so shift the third-order history here.
        self.qdot_prev2 = self._qdot_prev_seen
        self._qdot_prev_seen = np.asarray(self.qdot_prev, dtype=float).copy()
        v_cmd0 = np.asarray(twist_ref, dtype=float).reshape(N_TASK_SLACK)
        self.solve_count += 1
        self.last_qp2_fallback = False
        self.last_fallback_ms = 0.0

        # The measured state is authoritative for the kinematic snapshot.  A
        # precomputed snapshot may be supplied by the caller to avoid doing
        # FK/J/SVD/M twice in a 200 Hz loop.
        q_geom = (
            np.asarray(q_meas, dtype=float).reshape(-1)
            if q_meas is not None
            else q_prev
        )
        if q_geom.size != nv:
            raise ValueError(f"q_meas must have {nv} joints, got {q_geom.size}")
        J = (
            np.asarray(jacobian, dtype=float)
            if jacobian is not None
            else self.kin.jacobian(q_geom)
        )
        if J.shape != (N_TASK_SLACK, nv) or not np.all(np.isfinite(J)):
            raise ValueError(f"jacobian must have shape {(N_TASK_SLACK, nv)}")
        sigma_arr = (
            np.asarray(sigma, dtype=float).reshape(-1)
            if sigma is not None
            else self.kin.singular_values(J)
        )
        sigma_min = float(np.min(sigma_arr)) if sigma_arr.size else 0.0

        # When available, the rail feedback represents the motion that has
        # actually happened during this sample.  The rail command remains a
        # decision variable for the next sample, but is excluded from the
        # current task map so the arm solves the measured residual directly.
        rail_exec = None
        rail_exec_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        J_task = np.asarray(J, dtype=float).copy()
        if rail_exec_vel_m_s is not None and np.isfinite(float(rail_exec_vel_m_s)):
            rail_exec = float(rail_exec_vel_m_s)
            rail_exec_contrib = J[:, 0] * rail_exec
            J_task[:, 0] = 0.0
        b_task = v_cmd0 - rail_exec_contrib
        self.last_comp_projected_frac = 0.0
        # Public telemetry is expressed in the caller's original Cartesian
        # coordinates.  ``b_task`` is the internal arm-only target after
        # subtracting the measured rail contribution.
        self.last_task_target = v_cmd0.copy()
        self.last_rail_exec_contrib = rail_exec_contrib.copy()

        # Chiaverini SR projection is a secondary preference only.  It is
        # never present in QP1, so a posture preference cannot purchase task
        # slack there.
        proj_damping = sr_damping_lambda(sigma_min, self.cfg.sr_damping)
        M = (
            np.asarray(mass_matrix, dtype=float)
            if mass_matrix is not None
            else (
                self.kin.mass_matrix(q_geom)
                if self.cfg.use_mass_weighted_reg
                else None
            )
        )
        if M is not None and M.shape != (nv, nv):
            raise ValueError(f"mass_matrix must have shape {(nv, nv)}")
        qdot_nom = (
            (
                cpp_kernel.project_nullspace(
                    J_task,
                    secondary_qdot,
                    damping=proj_damping,
                    M=M,
                    use_dyn=False,
                )
                if bool(getattr(self.cfg, "use_cpp_kernel", True))
                else project_onto_task_nullspace(
                    J_task,
                    secondary_qdot,
                    damping=proj_damping,
                    sigma_min=sigma_min,
                    sr_cfg=self.cfg.sr_damping,
                    M=M,
                    use_dyn=False,
                )
            )
            if secondary_qdot is not None
            else np.zeros(nv, dtype=float)
        )
        if zero_secondary_rail and qdot_nom.size:
            qdot_nom[0] = 0.0
        if arm_qdot_pref is not None:
            pref = np.asarray(arm_qdot_pref, dtype=float).reshape(-1)
            n = min(pref.size, qdot_nom.size)
            qdot_nom[1:n] = pref[1:n]

        # Limit avoidance and the velocity box use the same measured geometry.
        w_reg = self._w_reg.copy()
        self.last_wln_scale = np.ones(self.kin.nv, dtype=float)
        if rail_locked and rail_lock_reg_scale > 1.0:
            w_reg[0] *= float(rail_lock_reg_scale)
        if (not rail_locked) and float(rail_reg_scale) > 1.0:
            w_reg[0] *= float(rail_reg_scale)
        w_task_mat = self._task_weight_matrix(
            J_task, dt, keep_task_weight=keep_task_weight
        )
        q_star_box = (
            self.q_star_signs
            if self.q_star_signs is not None
            else (self.q_star if self.q_star is not None else q_geom)
        )
        self.branch_barrier._update_dwell(q_geom, dt, q_star=q_star_box)
        rail_w_eff = float(rail_task_weight)
        pref_w = max(float(pref_slack_scale), 1.0e-6)
        n_task = N_TASK_SLACK
        n_pref = N_PREF_SLACK
        n_var = nv + n_task + n_pref

        # Shared hard constraints (P0) are built once and fed unchanged to
        # both levels.  Preference rows are added only to QP2 below.
        lo_box, hi_box = self.constraints.bounds(
            q_geom,
            dt,
            self.qdot_prev,
            q_meas=q_meas,
            q_cmd=q_prev,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            qdot_prev2=self.qdot_prev2,
            j_max=self._j_max,
            box_dt=box_dt,
            box_h1=box_h1,
            box_h2=box_h2,
            rail_lead_exempt=(
                abs(float(q_prev[0]) - float(q_geom[0]))
                > float(np.asarray(resync_err, dtype=float).reshape(-1)[0])
                if np.size(np.asarray(resync_err))
                else False
            ),
        )
        bind_lo = int(self.constraints.last_rail_bind_lo)
        bind_hi = int(self.constraints.last_rail_bind_hi)
        olo, ohi = float(lo_box[0]), float(hi_box[0])
        lo_box, hi_box = self.branch_barrier.tighten_box(
            lo_box,
            hi_box,
            q_geom,
            q_star_box,
            self.constraints.lim.v_max,
            rail_open_travel=bool(rail_open_travel),
            q_lower=self.constraints.lim.q_lower,
            q_upper=self.constraints.lim.q_upper,
        )
        bind_lo, bind_hi = note_rail_bind(
            bind_lo, bind_hi, olo, ohi, lo_box[0], hi_box[0], RAIL_BIND_BRANCH
        )
        olo, ohi = float(lo_box[0]), float(hi_box[0])
        lo_box, hi_box = collapse_interval(
            lo_box,
            hi_box,
            qdot_prev=self.qdot_prev,
            a_max=self.constraints.lim.a_max,
            dt=dt,
        )
        bind_lo, bind_hi = note_rail_bind(
            bind_lo, bind_hi, olo, ohi, lo_box[0], hi_box[0], RAIL_BIND_COLLAPSE
        )
        self.last_lo_box = np.asarray(lo_box, dtype=float).copy()
        self.last_hi_box = np.asarray(hi_box, dtype=float).copy()
        self.last_rail_box_lo = float(lo_box[0])
        self.last_rail_box_hi = float(hi_box[0])
        self.last_rail_bind_lo = int(bind_lo)
        self.last_rail_bind_hi = int(bind_hi)
        if rail_task_vel_m_s is not None and np.isfinite(float(rail_task_vel_m_s)):
            self.last_rail_task_vel_used = float(rail_task_vel_m_s)
        else:
            self.last_rail_task_vel_used = 0.0
        self.last_rail_h1 = float(
            box_h1 if box_h1 is not None else (box_dt if box_dt is not None else dt)
        )
        self.last_rail_h2 = (
            float(box_h2) if box_h2 is not None and np.isfinite(float(box_h2)) else float("nan")
        )
        self.last_rail_qdot_prev = float(self.qdot_prev[0])
        self.last_rail_qdot_prev2 = float(self.qdot_prev2[0])
        if self.collision is not None and self.collision_cfg.enabled:
            cbf = build_cbf_rows(
                self.collision,
                self.kin,
                q_geom,
                self.collision_cfg,
                tracker=self._cbf_slots,
                kinematics_ready=bool(kinematics_ready),
            )
        else:
            cbf = CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
            self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        if rail_exec is not None and cbf.jacobian.size:
            # CBF is a constraint on actual instantaneous motion just like the
            # protected TCP task.  Do not let a lagging rail command masquerade
            # as the rail velocity that is really changing collision distance.
            cbf_jac = np.asarray(cbf.jacobian, dtype=float).copy()
            cbf_lower = np.asarray(cbf.lower, dtype=float).copy()
            cbf_lower -= cbf_jac[:, 0] * rail_exec
            cbf_jac[:, 0] = 0.0
            cbf = CbfRows(
                jacobian=cbf_jac,
                lower=cbf_lower,
                slot_index=(
                    None
                    if cbf.slot_index is None
                    else np.asarray(cbf.slot_index, dtype=int).copy()
                ),
                names=tuple(cbf.names),
            )
        # Retain exactly the measured-rail affine CBF used by both QP levels
        # so the command publication path can certify any downstream rewrite.
        self.last_hard_cbf_jacobian = np.asarray(
            cbf.jacobian, dtype=float
        ).copy()
        self.last_hard_cbf_lower = np.asarray(cbf.lower, dtype=float).copy()
        self.last_task_jacobian = np.asarray(J_task, dtype=float).copy()
        self.last_lock_jacobian = np.asarray(J, dtype=float).copy()
        self.last_lock_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_cbf_min_dist = float("nan")
        self.last_cbf_pair = ""
        if self.collision is not None and self.collision_cfg.enabled:
            closest = self.collision.closest_pair()
            if closest is not None:
                self.last_cbf_min_dist = float(closest.distance)
                self.last_cbf_pair = f"{closest.name_a}:{closest.name_b}"
        _assemble = (
            cpp_kernel.build_wbc_inequalities
            if bool(getattr(self.cfg, "use_cpp_kernel", True))
            else build_wbc_inequalities
        )
        C_hard, lo, hi = _assemble(
            nv,
            n_task,
            lo_box,
            hi_box,
            cbf,
            self._max_cbf,
            n_pref_slack=n_pref,
            max_pref_rows=MAX_PREF_ROWS,
        )

        # QP1: only the protected task residual is optimized.  In particular,
        # qdot and preference-slack variables have exactly zero cost here:
        # even a tiny qdot regularizer would mathematically permit trading an
        # otherwise-zero Cartesian residual for less joint motion.  ProxQP's
        # own proximal terms handle the positive-semidefinite Hessian.
        if bool(getattr(self.cfg, "use_cpp_kernel", True)):
            H1, g1, A1 = cpp_kernel.setup_qp1(
                nv, n_task, n_pref, w_task_mat, J_task, use_native=True
            )
        else:
            H1 = np.zeros((n_var, n_var), dtype=float)
            H1[nv : nv + n_task, nv : nv + n_task] = w_task_mat
            g1 = np.zeros(n_var, dtype=float)
            A1 = np.zeros((n_task, n_var), dtype=float)
            A1[:, :nv] = J_task
            A1[:, nv : nv + n_task] = -np.eye(n_task)

        # ProxQP may return a point a few nanometres outside an inequality
        # while still satisfying ``eps_abs``.  QP2 then locks J*qdot from
        # that point and can incorrectly classify the hierarchy as primal
        # infeasible.  Solve QP1 against a conservatively inset hard set so
        # its achieved task is reproducibly feasible in QP2.  Exact pins
        # (lo==hi) are deliberately left untouched.
        feasibility_inset = max(2.0 * float(self.cfg.eps_abs), 1.0e-8)
        lo1 = np.asarray(lo, dtype=float).copy()
        hi1 = np.asarray(hi, dtype=float).copy()
        finite_lo = np.isfinite(lo1)
        finite_hi = np.isfinite(hi1)
        room = hi1 - lo1
        inset_both = finite_lo & finite_hi & (room > 2.0 * feasibility_inset)
        inset_lo_only = finite_lo & ~finite_hi
        inset_hi_only = ~finite_lo & finite_hi
        lo1[inset_both | inset_lo_only] += feasibility_inset
        hi1[inset_both | inset_hi_only] -= feasibility_inset

        x1 = self._solve_qp(
            self.backend,
            np.ascontiguousarray(H1),
            np.ascontiguousarray(g1),
            np.ascontiguousarray(A1),
            np.ascontiguousarray(b_task),
            np.ascontiguousarray(C_hard),
            np.ascontiguousarray(lo1),
            np.ascontiguousarray(hi1),
        )
        self.last_qp1_solve_ms = float(
            getattr(self.backend, "last_solve_ms", 0.0)
        )
        self.last_qp1_status = qp_status_name(
            getattr(self.backend, "last_status", "failed" if x1 is None else "solved")
        )
        self.last_qp1_iter = int(getattr(self.backend, "last_iter", 0) or 0)
        self.last_zero_slack_feasible = False
        if x1 is None:
            t_fallback = time.perf_counter()
            # Fail closed.  A scaled previous command is not certified against
            # this tick's acceleration/jerk/CBF set and must never leak out of
            # the low-level API as if it were a valid QP result.  Window A will
            # additionally invoke its rail+arm fault stop before publication.
            qdot = np.zeros_like(self.qdot_prev)
            residual = b_task - J_task @ qdot
            self.last_qp1_residual = residual.copy()
            self.last_qp1_residual_norm = float(np.linalg.norm(residual))
            self.last_qp2_residual = residual.copy()
            self.last_qp2_residual_norm = self.last_qp1_residual_norm
            self.last_qp2_status = "not_run"
            self.last_qp2_solve_ms = 0.0
            self.last_failed = True
            self.last_status = "failed"
            self.last_sns_scale = 1.0
            self.last_qp2_fallback = False
            dex_s = br_s = 0.0
            comfort = np.zeros(7, dtype=float)
            self.last_qdot_qp1 = np.asarray(qdot, dtype=float).copy()
            self.last_qp1_task_velocity = np.zeros(
                N_TASK_SLACK, dtype=float
            )
            self.last_qp1_hard_violation = float("nan")
            self.last_final_hard_violation = float("nan")
            self.last_final_task_lock_violation = float("nan")
            self.last_fallback_ms = (time.perf_counter() - t_fallback) * 1000.0
        else:
            qdot1 = np.asarray(x1[:nv], dtype=float).copy()
            if rail_exec is not None:
                # Rail is not in QP1's task map.  Seed the next command at the
                # allocator preference (clipped to this tick's box) so QP2 and
                # QP2-fallback send a defined qdot[0].  Do not lock the full
                # Jacobian to that seed: the arm must keep compensating the
                # measured rail, not a command the drive has not executed.
                seed = float(rail_exec)
                if (
                    rail_task_vel_m_s is not None
                    and np.isfinite(float(rail_task_vel_m_s))
                    and not rail_locked
                ):
                    seed = float(rail_task_vel_m_s)
                qdot1[0] = float(np.clip(seed, lo_box[0], hi_box[0]))
                x1 = np.asarray(x1, dtype=float).copy()
                x1[0] = qdot1[0]
            self.last_qdot_qp1 = qdot1.copy()
            excess, deg, inf, subst1 = measure_qdot_box(qdot1, lo_box, hi_box)
            self.last_box_excess_max = float(excess)
            self.last_box_degenerate = bool(deg)
            self.last_box_infeasible = bool(inf)
            if subst1:
                t_fallback = time.perf_counter()
                qdot = np.zeros_like(self.qdot_prev)
                residual = b_task - J_task @ qdot
                self.last_qp1_residual = residual.copy()
                self.last_qp1_residual_norm = float(np.linalg.norm(residual))
                self.last_qp2_residual = residual.copy()
                self.last_qp2_residual_norm = self.last_qp1_residual_norm
                self.last_qp2_status = "not_run"
                self.last_qp2_solve_ms = 0.0
                self.last_failed = True
                self.last_status = "failed"
                self.last_sns_scale = 1.0
                self.last_qp2_fallback = False
                dex_s = br_s = 0.0
                comfort = np.zeros(7, dtype=float)
                self.last_qp1_hard_violation = float(excess)
                self.last_final_hard_violation = float(excess)
                self.last_fallback_ms = (time.perf_counter() - t_fallback) * 1000.0
            if not subst1:
                hard_lo_violation = np.maximum(lo - C_hard @ x1, 0.0)
                hard_hi_violation = np.maximum(C_hard @ x1 - hi, 0.0)
                self.last_qp1_hard_violation = float(
                    max(
                        np.max(hard_lo_violation, initial=0.0),
                        np.max(hard_hi_violation, initial=0.0),
                    )
                )
                t1 = J_task @ qdot1
                self.last_qp1_task_velocity = np.asarray(t1, dtype=float).copy()
                if rail_exec is not None:
                    lock_jac = np.asarray(J_task, dtype=float).copy()
                    lock_vel = np.asarray(t1, dtype=float).copy()
                else:
                    lock_jac = np.asarray(J, dtype=float).copy()
                    lock_vel = np.asarray(t1, dtype=float).copy()
                self.last_lock_jacobian = lock_jac
                self.last_lock_velocity = lock_vel
                residual1 = b_task - t1
                self.last_qp1_residual = residual1.copy()
                self.last_qp1_residual_norm = float(np.linalg.norm(residual1))

                # Build QP2's existing weighted secondary objective.  Its task
                # equality is augmented with w_task=0, locking QP1's achieved
                # task exactly while allowing all lower-priority preferences.
                if self.cfg.use_mass_weighted_reg and M is not None:
                    m_diag = np.maximum(np.diag(M), self.cfg.mass_reg_floor)
                    if self.cfg.mass_weight_exempt_rail:
                        m_diag[0] = 1.0
                    tau = float(self.cfg.mass_reg_lpf_tau_s)
                    if tau > 1.0e-9 and dt > 1.0e-9:
                        if self._m_diag_lpf is None:
                            self._m_diag_lpf = m_diag.copy()
                        else:
                            self._m_diag_lpf = first_order_lpf_vec(
                                self._m_diag_lpf, m_diag, dt, tau
                            )
                        m_diag = self._m_diag_lpf
                    h_reg = w_reg * m_diag
                else:
                    h_reg = w_reg
                slack_w = np.zeros(n_pref, dtype=float)
                slack_w[0] = float(self.cfg.sigma_setbased.slack_weight)
                slack_w[1] = (
                    float(self.cfg.branch_barrier.slack_weight)
                    * pref_w
                    * float(self.branch_barrier.last_dwell_scale)
                )
                comfort_w = float(self.cfg.joint_comfort.slack_weight) * pref_w
                if n_pref > 2:
                    slack_w[2:] = comfort_w
                rail_w_qp2 = 0.0
                rail_vel_qp2 = 0.0
                if (
                    rail_task_vel_m_s is not None
                    and rail_w_eff > 0.0
                    and not rail_locked
                    and rail_vel_pin_m_s is None
                ):
                    rail_w_qp2 = float(rail_w_eff)
                    rail_vel_qp2 = float(rail_task_vel_m_s)
                smooth_raw = np.asarray(
                    getattr(self.cfg, "smoothness_weight", 0.0), dtype=float
                ).reshape(-1)
                if smooth_raw.size == 1:
                    smooth = np.full(nv, float(smooth_raw[0]), dtype=float)
                elif smooth_raw.size == nv:
                    smooth = smooth_raw.copy()
                else:
                    raise ValueError(
                        f"smoothness_weight must be scalar or length {nv}, got {smooth_raw.size}"
                    )
                smooth = np.maximum(smooth, 0.0)
                H2, g2 = cpp_kernel.setup_qp2_costs(
                    nv,
                    n_task,
                    n_pref,
                    h_reg,
                    qdot_nom,
                    slack_w,
                    rail_w=rail_w_qp2,
                    rail_vel=rail_vel_qp2,
                    smooth=smooth,
                    qdot_prev=self.qdot_prev,
                    use_native=bool(getattr(self.cfg, "use_cpp_kernel", True)),
                )

                sigma_rows = self.sigma_setbased.build_row(self.kin, q_geom)
                q_star = (
                    self.q_star_signs
                    if self.q_star_signs is not None
                    else (self.q_star if self.q_star is not None else q_geom)
                )
                # Soft branch rows never bound (max slack 2e-6).  Keep the hard
                # Faverjon damper in tighten_box.
                branch_rows = PrefInequalityRows(
                    jacobian=np.zeros((0, nv)),
                    slack_col=np.zeros(0, dtype=int),
                    lower=np.zeros(0),
                    active=False,
                )
                comfort_rows = self.joint_comfort.build_rows(
                    q_geom, self.constraints.lim.q_lower, self.constraints.lim.q_upper
                )
                design_rows = self.j4_design_comfort.build_rows(q_geom)
                pref = self._merge_pref_rows(
                    sigma_rows, branch_rows, comfort_rows, design_rows
                )
                C2, lo2, hi2 = _assemble(
                    nv,
                    n_task,
                    lo_box,
                    hi_box,
                    cbf,
                    self._max_cbf,
                    n_pref_slack=n_pref,
                    max_pref_rows=MAX_PREF_ROWS,
                    pref_jacobian=pref.jacobian,
                    pref_slack_col=pref.slack_col,
                    pref_lower=pref.lower,
                )
                A2 = np.zeros((n_task, n_var), dtype=float)
                A2[:, :nv] = lock_jac
                b2 = lock_vel
                # Same-tick feasible hot start: qdot1 already satisfies all hard
                # constraints and exactly produces b2.  Fill only the one-sided
                # preference slacks needed by the added QP2 rows.  Seeding from
                # the previous tick here caused false PRIMAL_INFEASIBLE statuses
                # when the acceleration box moved between samples.
                x2_seed = np.zeros(n_var, dtype=float)
                x2_seed[:nv] = qdot1
                for k in range(n_pref):
                    col = nv + n_task + k
                    rows = C2[:, col] > 0.5
                    finite_rows = rows & np.isfinite(lo2)
                    if np.any(finite_rows):
                        base = C2[finite_rows, :nv] @ qdot1
                        need = float(np.max(lo2[finite_rows] - base, initial=0.0))
                        x2_seed[col] = max(need, 0.0) + feasibility_inset
                seed_c = C2 @ x2_seed
                self.last_qp2_seed_violation = float(
                    max(
                        np.max(np.maximum(lo2 - seed_c, 0.0), initial=0.0),
                        np.max(np.maximum(seed_c - hi2, 0.0), initial=0.0),
                    )
                )
                self.last_qp2_seed_equality = float(
                    np.max(np.abs(A2 @ x2_seed - b2), initial=0.0)
                )
                qp2_exception_status = ""
                try:
                    x2 = self._solve_qp(
                        self._backend_qp2,
                        np.ascontiguousarray(H2),
                        np.ascontiguousarray(g2),
                        np.ascontiguousarray(A2),
                        np.ascontiguousarray(b2),
                        np.ascontiguousarray(C2),
                        np.ascontiguousarray(lo2),
                        np.ascontiguousarray(hi2),
                        warm_start_x=np.ascontiguousarray(x2_seed),
                    )
                except Exception as exc:
                    # QP1 is already a valid protected solution.  A secondary
                    # backend exception must not turn into a stale-velocity send.
                    x2 = None
                    qp2_exception_status = f"exception:{type(exc).__name__}"
                self.last_qp2_solve_ms = float(
                    getattr(self._backend_qp2, "last_solve_ms", 0.0)
                )
                self.last_qp2_iter = int(
                    getattr(self._backend_qp2, "last_iter", 0) or 0
                )
                self.last_qp2_status = qp2_exception_status or qp_status_name(
                    getattr(
                        self._backend_qp2,
                        "last_status",
                        "failed" if x2 is None else "solved",
                    )
                )
                if x2 is not None:
                    qdot2 = np.asarray(x2[:nv], dtype=float)
                    _e, _d, _i, subst2 = measure_qdot_box(qdot2, lo_box, hi_box)
                    if subst2:
                        x2 = None
                        self.last_qp2_status = "failed"
                if x2 is None:
                    t_fallback = time.perf_counter()
                    qdot = qdot1
                    x = x1
                    C_final, lo_final, hi_final = C_hard, lo, hi
                    self.last_qp2_fallback = True
                    self.last_fallback_ms = (
                        time.perf_counter() - t_fallback
                    ) * 1000.0
                else:
                    qdot = np.asarray(x2[:nv], dtype=float)
                    x = x2
                    C_final, lo_final, hi_final = C2, lo2, hi2
                _e2, _d2, _i2, subst_pub = measure_qdot_box(qdot, lo_box, hi_box)
                if subst_pub:
                    t_fallback = time.perf_counter()
                    qdot = np.zeros_like(self.qdot_prev)
                    residual = b_task - J_task @ qdot
                    self.last_failed = True
                    self.last_status = "failed"
                    self.last_qp2_fallback = True
                    self.last_fallback_ms = (
                        time.perf_counter() - t_fallback
                    ) * 1000.0
                    dex_s = br_s = 0.0
                    comfort = np.zeros(7, dtype=float)
                else:
                    self.last_j4_design_slack = 0.0
                    if x is not None and int(np.asarray(x).size) > nv + n_task + 2:
                        self.last_j4_design_slack = float(
                            max(0.0, float(np.asarray(x).reshape(-1)[nv + n_task + 2]))
                        )
                    final_c = C_final @ x
                    self.last_final_hard_violation = float(
                        max(
                            np.max(np.maximum(lo_final - final_c, 0.0), initial=0.0),
                            np.max(np.maximum(final_c - hi_final, 0.0), initial=0.0),
                        )
                    )
                    self.last_final_task_lock_violation = float(
                        np.max(
                            np.abs(lock_jac @ np.asarray(qdot, dtype=float) - lock_vel),
                            initial=0.0,
                        )
                    )
                    residual = b_task - J_task @ qdot
                    self.last_qp2_residual = residual.copy()
                    self.last_qp2_residual_norm = float(np.linalg.norm(residual))
                    dex_s = float(max(0.0, x[nv + n_task]))
                    br_s = float(max(0.0, x[nv + n_task + 1]))
                    comfort = np.maximum(
                        0.0, np.asarray(x[nv + n_task + 2 : nv + n_task + 9], dtype=float)
                    )
                    if comfort.size < 7:
                        comfort = np.pad(comfort, (0, 7 - int(comfort.size)))
                    comfort = comfort[:7]
                    self.last_failed = False
                    self.last_status = self.last_qp2_status or self.last_qp1_status
                    self.last_sns_scale = 1.0

        self.last_qp_total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_qp_overrun = bool(self.last_qp_total_ms > 5.0)
        # Preserve the legacy loop's ``core.backend.last_solve_ms`` telemetry,
        # but make it represent the complete two-level controller budget.
        self.backend.last_solve_ms = float(self.last_qp_total_ms)
        self._update_mirror_telemetry(
            J,
            rail_exec=rail_exec,
            h1=float(box_h1 if box_h1 is not None else (box_dt if box_dt is not None else dt)),
        )
        self.last_task_residual = np.asarray(residual, dtype=float).copy()
        # Legacy array retained for compatibility, but hard feasibility now
        # has its own scalar telemetry instead of aliasing Cartesian slack.
        self.last_hard_residual = np.full(
            N_TASK_SLACK, float(self.last_final_hard_violation), dtype=float
        )
        self.last_task_residual_norm = float(np.linalg.norm(residual))
        self.last_rail_cmd_contrib = J[:, 0] * float(qdot[0])
        self.last_arm_contrib = J[:, 1:] @ np.asarray(qdot[1:], dtype=float)
        # For a measured-rail tick, the actual contribution is measured rail
        # plus arm motion; without feedback the command is the best available
        # rail contribution and preserves legacy semantics.
        rail_actual = (
            rail_exec_contrib
            if rail_exec_vel_m_s is not None and np.isfinite(float(rail_exec_vel_m_s))
            else self.last_rail_cmd_contrib
        )
        self.last_task_achieved = rail_actual + self.last_arm_contrib
        if cbf.jacobian.size:
            cbf_value = np.asarray(cbf.jacobian, dtype=float) @ np.asarray(
                qdot, dtype=float
            )
            cbf_tol = max(2.0 * float(self.cfg.eps_abs), 1.0e-7)
            active_mask = np.abs(cbf_value - np.asarray(cbf.lower, dtype=float)) <= cbf_tol
            self.last_cbf_active_names = tuple(
                name
                for name, active in zip(tuple(cbf.names), active_mask)
                if bool(active)
            )
        else:
            self.last_cbf_active_names = ()
        self.last_dexterity_slack = dex_s
        self.last_branch_slack = br_s
        self.last_comfort_slack = np.asarray(comfort, dtype=float).reshape(7)
        self.sigma_setbased.last_slack = dex_s
        self.branch_barrier.last_slack = br_s
        # A failed QP1 has no certified command.  Preserve the applied-history
        # state until the outer safety stop/reset path explicitly synchronizes
        # it; do not seed a future jerk box from the diagnostic zero result.
        if not self.last_failed:
            self.qdot_prev = np.asarray(qdot, dtype=float).copy()
        q_next = q_prev + qdot * dt
        return IkStepResult(
            q_next=q_next,
            qdot=qdot,
            sigma_min=sigma_min,
            manip=self.kin.manipulability(J),
            slack_norm=float(self.last_task_residual_norm),
            n_cbf_active=int(cbf.jacobian.shape[0]),
            dexterity_slack=dex_s,
            branch_slack=br_s,
            sns_scale=1.0,
        )
