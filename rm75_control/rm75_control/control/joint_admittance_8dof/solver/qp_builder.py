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
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    CbfSlotTracker,
    build_cbf_rows,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
    build_wbc_inequalities,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

N_SLACK = 6


@dataclass
class QpConfig:
    task_weight: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)
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
            [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3],
            dtype=float,
        )
    )
    backend: str = "proxqp"
    eps_abs: float = 1e-6
    max_iter: int = 200
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
    # Use Khatib N_dyn instead of kinematic N in secondary projection.
    use_dyn_nullspace: bool = True
    # Faverjon/Tournassoud joint-limit velocity damper band: allowed speed
    # toward a limit ramps to 0 across this zone before the margin.  Units are
    # PER JOINT: rad for the arm, metres for the prismatic rail.  The old
    # scalar band applied 0.15 "rad" = 0.15 m to the rail — the damper started
    # throttling rail velocity from |y| > 6.5 cm (60% of the ±0.25 m travel),
    # exactly where the rail is needed most to rescue arm singularities.
    limit_damper_band_rad: float = 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: float = 0.05   # rail joint 0 (metres)
    warn_on_fail: bool = True
    # On ProxQP failure: qdot ← fail_qdot_decay * qdot_prev (not a hard 0.5
    # chop — that was a one-tick jerk when the solver hiccupped).
    fail_qdot_decay: float = 0.85
    # Hard wall-clock budget for one ProxQP attempt+retry (ms).  Exceeding
    # this skips the retry and returns fail — prevents GIL freezes of
    # multiple seconds near σ→0 that starve the rail Modbus loop (PANIC).
    max_solve_ms: float = 8.0
    # Below this σ_min, Cartesian twist (incl. force) is scaled down so
    # nullspace escape / rail recruitment can win over force-driven collapse.
    twist_sigma_floor: float = 0.08
    # Avoidance onset = sigma_ref * scale.  Must lead the twist brake (>1) so
    # rail/∇μ can accelerate before Cartesian is clamped, but stay below the
    # healthy-D band (σ≈0.11–0.13) — 2.0 kept D permanently escaping.
    sigma_escape_ref_scale: float = 1.25


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
        self._max_solve_s = max(1.0e-3, float(getattr(cfg, "max_solve_ms", 8.0)) * 1.0e-3)

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
        self.qp.solve()
        elapsed = _time.perf_counter() - t0

        if not self._solved():
            # First retry: cold-start + loose eps + fewer iters.  Skip the
            # retry if the first attempt already burned the wall budget —
            # near σ→0 a second full solve can hold the GIL for seconds
            # (rail Modbus starves → encoder freeze → PANIC; Ctrl+C feels dead).
            remaining = self._max_solve_s - elapsed
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
        # Per-joint damper band: arm in rad, prismatic rail (joint 0) in m.
        damper_band = np.full(kin.nv, float(self.cfg.limit_damper_band_rad))
        damper_band[0] = float(self.cfg.limit_damper_band_rail_m)
        self.constraints = VelocityBoxConstraints(
            limits, damper_band_rad=damper_band
        )
        self.collision_cfg = self.cfg.collision
        self._max_cbf = max(1, int(self.collision_cfg.max_pairs))
        self.collision = collision
        if self.collision_cfg.enabled and self.collision is None:
            self.collision = CollisionModel(kin.model)
        self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self._m_diag_lpf: np.ndarray | None = None
        self._task_scale_lpf: float = 1.0
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

    def reset(self, q0_rad: np.ndarray | None = None) -> None:
        del q0_rad  # QP state is velocity history / LPF only
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)
        self._m_diag_lpf = None
        self._task_scale_lpf = 1.0

    def _task_scale_sigma(self, sigma_min: float, dt: float) -> float:
        """LPF-smoothed W_task scale in [min_frac, 1] from σ_min."""
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        raw = 1.0
        if sigma_ref > 1e-9 and sigma_min < sigma_ref:
            frac = float(sigma_min) / sigma_ref
            raw = max(frac * frac, float(self.cfg.task_weight_min_frac))
        tau = float(self.cfg.task_weight_lpf_tau_s)
        if tau > 1e-9 and dt > 1e-9:
            alpha = min(1.0, dt / tau)
            self._task_scale_lpf += alpha * (raw - self._task_scale_lpf)
            return float(self._task_scale_lpf)
        self._task_scale_lpf = float(raw)
        return float(raw)

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
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_reg_scale: float = 1.0,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        zero_secondary_rail: bool = False,
        rail_task_vel_m_s: float | None = None,
        rail_task_weight: float = 0.0,
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
        # Rail bleed guard: the SR-damped nullspace basis N couples all joints,
        # so even a rail-clean secondary_qdot (composer zeroes [0]) is smeared
        # by the projection into a nonzero qdot_nom[0].  In COUPLED / RAIL_ONLY
        # / TCP_FIXED we do not want secondary tasks (centering / manip /
        # arm_task / damping) to drive rail via this projection back-door —
        # rail motion is recruited only by the primary Cartesian equality
        # Jqdot = v_cmd and by the EXPLICIT preferred-extension rail task
        # (rail_task_vel_m_s / rail_task_weight below), never by projected
        # nullspace velocities.  Zero the rail bias here.
        if zero_secondary_rail and qdot_nom.shape[0] > 0:
            qdot_nom[0] = 0.0

        w_reg = self._w_reg.copy()
        w_task = self._w_task.copy()
        if rail_locked and rail_lock_reg_scale > 1.0:
            w_reg[0] *= float(rail_lock_reg_scale)
        w_task *= self._task_scale_sigma(sigma_min, dt)

        # rail_extension hint stays at its full weight (the task itself scales
        # by σ_scale via Bug 2 — do NOT double-schedule here).
        rail_w_eff = float(rail_task_weight)

        H = np.zeros((n_var, n_var), dtype=float)
        if self.cfg.use_mass_weighted_reg and M is not None:
            m_diag = np.maximum(np.diag(M), self.cfg.mass_reg_floor)
            if self.cfg.mass_weight_exempt_rail:
                # Rail cost is reg[0] verbatim: diag(M)[0] is the ~10 kg
                # carriage+arm mass, which over-priced rail motion 30-400x
                # vs the arm and starved rail recruitment (arm stretched to
                # near-straight before the rail moved).
                m_diag[0] = 1.0
            tau = float(self.cfg.mass_reg_lpf_tau_s)
            if tau > 1e-9 and dt > 1e-9:
                if self._m_diag_lpf is None:
                    self._m_diag_lpf = m_diag.copy()
                else:
                    alpha = min(1.0, dt / tau)
                    self._m_diag_lpf += alpha * (m_diag - self._m_diag_lpf)
                m_diag = self._m_diag_lpf
            H[:nv, :nv] = np.diag(w_reg * m_diag)
        else:
            H[:nv, :nv] = np.diag(w_reg)
        H[nv:, nv:] = np.diag(w_task)
        g = np.zeros(n_var, dtype=float)
        g[:nv] = -np.diag(H[:nv, :nv]) * qdot_nom if self.cfg.use_mass_weighted_reg and M is not None else -w_reg * qdot_nom

        # Preferred-extension rail task (Yamamoto & Yun 1994 base-arm
        # coordination): a soft scalar task w/2*(qdot[0] - v_rail)^2 added
        # directly to the cost.  The Cartesian equality rows (much heavier)
        # keep the TCP on the reference while the arm absorbs the rail motion,
        # so tracking is NOT sacrificed — unlike a nullspace-projected rail
        # drive, which the SR-damped projector smears near singularities
        # (Dietrich et al. 2015).  Weight is scheduled continuously by the
        # caller (0 in the extension dead zone: the rail does not wander).
        if (
            rail_task_vel_m_s is not None
            and rail_w_eff > 0.0
            and not rail_locked
            and rail_vel_pin_m_s is None
        ):
            H[0, 0] += rail_w_eff
            g[0] -= rail_w_eff * float(rail_task_vel_m_s)

        A = np.zeros((ns, n_var), dtype=float)
        A[:, :nv] = J
        A[:, nv:] = -np.eye(ns)
        b = v_cmd

        lo_box, hi_box = self.constraints.bounds(
            q_prev,
            dt,
            self.qdot_prev,
            q_meas=q_meas,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
        )
        if self.collision is not None and self.collision_cfg.enabled:
            cbf = build_cbf_rows(
                self.collision,
                self.kin,
                q_prev,
                self.collision_cfg,
                tracker=self._cbf_slots,
            )
        else:
            from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows

            cbf = CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
            self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)

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
            # Solver failure: exponential decay of previous velocity.  Near
            # σ→0 decay harder — keeping a large qdot_prev is what drove the
            # elbow straight in force-hybrid retract before the next tick
            # burned seconds in ProxQP.
            decay = float(self.cfg.fail_qdot_decay)
            sigma_ref = float(self.cfg.sr_damping.sigma_ref)
            if sigma_ref > 1e-9 and sigma_min < sigma_ref:
                decay = min(decay, 0.4)
            qdot = decay * self.qdot_prev
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
