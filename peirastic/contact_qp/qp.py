"""Normalized Cartesian contact QP; all acoustic rows are declared policies.

``nominal_twist`` is the FULL clamped baseline, including ``path_twist``.
Ordinary candidates have V = H [vn, omega, alpha], at the original TCP.
Normal recovery outside that subspace must take the adapter's mechanical route.
This module has no robot, transport, UI, or command-integrator dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from types import MappingProxyType
import time
from typing import Mapping

import numpy as np

from .geometry import aperture_rows, motion_basis, window_rows
from .repair_policy import DifferentialRepairConfig
from .port_constraint import PortEnergyConstraint
from .types import (SCHEMA_VERSION, ContactObservation, ContactStatus, ProbeGeometry,
                    TwistConstraints, positive, vector)


def _immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _immutable(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(v) for v in value)
    return value


@dataclass(frozen=True)
class QpConfig:
    schema_version: int = SCHEMA_VERSION
    allocation_policy: str = "legacy_v7"
    differential_repair: DifferentialRepairConfig = field(default_factory=DifferentialRepairConfig)
    force_target_n: float = 4.0
    force_sign_band_n: float = 0.1
    aperture_budget_m_s: float = 0.00075
    c_min: float = 0.5
    quality_policy_version: str = "synthetic_quality_policy_v1_unverified_hardware"
    repair_speed_m_s: float = 0.002
    keep_speed_m_s: float = 0.004
    max_image_age_s: float = 0.30
    max_step_s: float = 0.01
    certificate_horizon_s: float = 0.01
    normal_scale_m_s: float = 0.002
    angular_scale_rad_s: float = 0.1
    slack_scale_m_s: float = 0.002
    normal_weight: float = 1.0
    angular_weight: float = 1.0
    progress_weight: float = 1.0
    slack_weight: float = 10.0
    aperture_cost_weight: float = 1.0
    max_velocity: np.ndarray = field(default_factory=lambda: np.array([.04, .04, .01, .6, .28, .6]))
    max_acceleration: np.ndarray = field(default_factory=lambda: np.array([1., 1., .8, 2., 3., 2.]))
    angle_limit_rad: float = math.radians(150.)
    lateral_windows: tuple = ((.04, .34), (.34, .66), (.66, .96))
    enable_visual: bool = True
    enable_aperture: bool = True
    enable_force_priority: bool = True
    enable_consistency: bool = False
    enable_progress_loss: bool = True
    compute_interval_diagnostics: bool = False
    subspace_tolerance: np.ndarray = field(default_factory=lambda: np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]))
    solver_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-8
    max_iterations: int = 200
    inner_iterations: int = 100
    solver_preconditioning: bool = False
    solver_policy: str = "legacy_v1"

    def __post_init__(self):
        if self.solver_policy not in ("legacy_v1", "bounded_retry_v1"):
            raise ValueError("unknown contact QP solver policy")
        if self.allocation_policy not in ('legacy_v7','differential_repair_v8'):
            raise ValueError('unknown allocation policy')
        policy=self.differential_repair
        if isinstance(policy,Mapping):policy=DifferentialRepairConfig(**policy)
        if not isinstance(policy,DifferentialRepairConfig):raise ValueError('invalid differential repair configuration')
        object.__setattr__(self,'differential_repair',policy)
        if self.schema_version != SCHEMA_VERSION or float(self.force_target_n) != 4.0:
            raise ValueError("contact QP v1 requires the fixed 4 N target")
        for name in ("force_sign_band_n", "aperture_budget_m_s", "repair_speed_m_s", "keep_speed_m_s",
                     "aperture_cost_weight"):
            object.__setattr__(self, name, positive(getattr(self, name), name, zero=True))
        for name in ("max_image_age_s", "max_step_s", "certificate_horizon_s", "normal_scale_m_s",
                     "angular_scale_rad_s", "slack_scale_m_s", "normal_weight", "angular_weight",
                     "progress_weight", "slack_weight", "angle_limit_rad", "solver_tolerance",
                     "feasibility_tolerance"):
            object.__setattr__(self, name, positive(getattr(self, name), name))
        if not math.isfinite(self.c_min) or not 0 < self.c_min < 1 or not self.quality_policy_version:
            raise ValueError("explicit policy version and c_min in (0,1) required")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("positive integer max_iterations required")
        if isinstance(self.inner_iterations, bool) or not isinstance(self.inner_iterations, int) or self.inner_iterations < 1:
            raise ValueError("positive integer inner_iterations required")
        for name in ("enable_visual", "enable_aperture", "enable_force_priority", "enable_consistency",
                     "enable_progress_loss", "compute_interval_diagnostics", "solver_preconditioning"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        for name in ("max_velocity", "max_acceleration", "subspace_tolerance"):
            value = vector(getattr(self, name), (6,), name=name)
            if np.any(value <= 0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        windows = vector(self.lateral_windows, (3, 2), name="lateral_windows")
        if np.any(windows[:, 0] >= windows[:, 1]) or np.any(windows < 0) or np.any(windows > 1):
            raise ValueError("invalid window fractions")
        object.__setattr__(self, "lateral_windows", tuple(map(tuple, windows)))


@dataclass(frozen=True)
class QpInput:
    geometry: ProbeGeometry
    nominal_twist: np.ndarray
    path_twist: np.ndarray
    force_n: float
    dt_s: float
    now_s: float
    observation: ContactObservation | None = None
    mechanical: TwistConstraints = field(default_factory=TwistConstraints)
    previous_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    measured_angle: float = 0.0
    gamma: np.ndarray = field(default_factory=lambda: np.ones(2))
    center_interval_m: tuple[float, float] | None = None
    schema_version: int = SCHEMA_VERSION
    energy: PortEnergyConstraint | None = None
    # Command-slew elapsed time; distinct from the future hold/energy interval.
    acceleration_dt_s: float | None = None
    repair_execution_enabled: bool = True
    repair_angle_reference_reset: bool = False

    def __post_init__(self):
        if type(self.repair_angle_reference_reset) is not bool:raise ValueError("repair_angle_reference_reset must be bool")
        if type(self.repair_execution_enabled) is not bool:raise ValueError("repair_execution_enabled must be bool")
        if self.schema_version != SCHEMA_VERSION or not isinstance(self.geometry, ProbeGeometry):
            raise ValueError("invalid input schema/geometry")
        if not isinstance(self.mechanical, TwistConstraints) or self.mechanical.frame != "tcp_tool":
            raise ValueError("QP mechanical rows must be expressed at TCP in tool components")
        if self.energy is not None and (not isinstance(self.energy, PortEnergyConstraint)
                or self.energy.hold_s < float(self.dt_s)):
            raise ValueError("energy hold must cover the complete execution interval")
        if self.observation is not None and not isinstance(self.observation, ContactObservation):
            raise ValueError("invalid contact observation")
        for name in ("nominal_twist", "path_twist", "previous_twist"):
            object.__setattr__(self, name, vector(getattr(self, name), (6,), name=name))
        motion_basis(self.path_twist)
        force_n = float(self.force_n)
        if not math.isfinite(force_n):
            raise ValueError("compression-positive force measurement must be finite")
        object.__setattr__(self, "force_n", force_n)
        object.__setattr__(self, "dt_s", positive(self.dt_s, "dt_s"))
        object.__setattr__(self, "acceleration_dt_s", positive(
            self.dt_s if self.acceleration_dt_s is None else self.acceleration_dt_s, "acceleration_dt_s"))
        object.__setattr__(self, "now_s", positive(self.now_s, "now_s", zero=True))
        if not math.isfinite(self.measured_angle):
            raise ValueError("measured_angle must be finite")
        raw_gamma = np.asarray(self.gamma, dtype=float)
        gamma = vector(np.full(2, float(raw_gamma)) if raw_gamma.ndim == 0 else raw_gamma, (2,), name="gamma")
        if np.any((gamma < 0) | (gamma > 1)):
            raise ValueError("gamma must be in [0,1]")
        object.__setattr__(self, "gamma", gamma)
        if self.center_interval_m is not None:
            aperture_rows(self.geometry, self.center_interval_m)
            object.__setattr__(self, "center_interval_m", tuple(map(float, self.center_interval_m)))


@dataclass(frozen=True)
class QpResult:
    qp_twist: np.ndarray | None
    alpha: float
    slack: np.ndarray
    hard_constraints: TwistConstraints
    status: ContactStatus
    diagnostics: Mapping = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    energy_certificate: PortEnergyConstraint | None = None
    created_time_s: float = 0.

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION or not isinstance(self.hard_constraints, TwistConstraints):
            raise ValueError("invalid QP result schema/constraints")
        if not math.isfinite(self.created_time_s) or self.created_time_s < 0:
            raise ValueError("finite nonnegative result creation time required")
        if self.energy_certificate is not None and not isinstance(self.energy_certificate, PortEnergyConstraint):
            raise ValueError("invalid energy certificate")
        if self.qp_twist is not None:
            object.__setattr__(self, "qp_twist", vector(self.qp_twist, (6,), name="qp_twist"))
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("invalid result alpha")
        slack = vector(self.slack, (2,), name="slack")
        if np.any(slack < 0):
            raise ValueError("negative visual slack")
        object.__setattr__(self, "slack", slack)
        object.__setattr__(self, "status", ContactStatus(self.status))
        object.__setattr__(self, "diagnostics", _immutable(self.diagnostics))

    def final_velocity_admissible(self, velocity, *, now_s=None, tolerance=1e-8):
        try:
            now=self.created_time_s if now_s is None else float(now_s)
            if (not math.isfinite(now) or now < self.created_time_s
                    or now >= self.hard_constraints.valid_until_s):
                return False
            energy=(None if self.energy_certificate is None else
                    self.energy_certificate.aged(now-self.created_time_s))
            return bool(self.success and self.hard_constraints.violation(velocity) <= tolerance
                        and (energy is None or energy.admissible(velocity, tolerance_w=tolerance,
                                                                velocity_tolerance=tolerance)))
        except (TypeError, ValueError, OverflowError):
            return False

    @property
    def success(self):
        return self.qp_twist is not None and self.status in (
            ContactStatus.NOMINAL, ContactStatus.REPAIR, ContactStatus.IMAGE_UNAVAILABLE)


def _normalize_rows(a, lo, hi):
    scale = np.maximum(np.max(np.abs(a), axis=1), 1e-12)
    return a / scale[:, None], lo / scale, hi / scale


def _violation(a, lo, hi, value):
    values = a @ value
    if not np.isfinite(values).all():
        return math.inf
    return float(max(0., np.max(lo - values, initial=-math.inf),
                     np.max(values - hi, initial=-math.inf)))


def _omega_interval(a, lo, hi, *, tolerance=1e-9):
    """Exact projection of bounded 3D linear inequalities onto coordinate 1.

    Fourier-Motzkin eliminates alpha then normal speed; only algebraic rows
    are introduced. A large external row set uses a bounded LP fallback.
    Inputs are dimensionless so the elimination tolerance is well scaled.
    """
    rows = np.concatenate((a[np.isfinite(hi)], -a[np.isfinite(lo)]))
    bounds = np.concatenate((hi[np.isfinite(hi)], -lo[np.isfinite(lo)]))
    remaining = [0, 1, 2]
    for original_axis in (2, 0):
        index = remaining.index(original_axis)
        coeff = rows[:, index]
        plus, minus = coeff > 1e-12, coeff < -1e-12
        zero = ~(plus | minus)
        if int(plus.sum()) * int(minus.sum()) + int(zero.sum()) > 4096:
            from scipy.optimize import linprog
            values = []
            for sign in (1., -1.):
                objective = np.array([0., sign, 0.])
                result = linprog(objective, A_ub=np.concatenate((a[np.isfinite(hi)], -a[np.isfinite(lo)])),
                                 b_ub=np.concatenate((hi[np.isfinite(hi)], -lo[np.isfinite(lo)])),
                                 bounds=[(None, None)] * 3, method="highs")
                if result.status == 2:
                    return None
                if not result.success:
                    raise RuntimeError("omega projection LP failed")
                values.append(float(result.x[1]))
            return tuple(values)
        keep_rows = np.delete(rows[zero], index, axis=1)
        keep_bounds = bounds[zero]
        if plus.any() and minus.any():
            upper = rows[plus] / coeff[plus, None]
            lower = rows[minus] / coeff[minus, None]
            combined = upper[:, None, :] - lower[None, :, :]
            combined_bounds = bounds[plus] / coeff[plus]
            combined_bounds = combined_bounds[:, None] - (bounds[minus] / coeff[minus])[None, :]
            keep_rows = np.concatenate((keep_rows, np.delete(combined.reshape(-1, len(remaining)), index, axis=1)))
            keep_bounds = np.concatenate((keep_bounds, combined_bounds.ravel()))
        rows, bounds = keep_rows, keep_bounds
        remaining.remove(original_axis)
    coeff = rows[:, 0]
    zero = np.abs(coeff) <= 1e-12
    if np.any(bounds[zero] < -tolerance):
        return None
    lower = np.max(bounds[coeff < -1e-12] / coeff[coeff < -1e-12], initial=-math.inf)
    upper = np.min(bounds[coeff > 1e-12] / coeff[coeff > 1e-12], initial=math.inf)
    return None if lower > upper + tolerance else (float(lower), float(upper))


def _linear_feasible(a, lo, hi):
    from scipy.optimize import linprog
    a,lo,hi=_normalize_rows(np.asarray(a),np.asarray(lo),np.asarray(hi))
    upper=np.isfinite(hi);lower=np.isfinite(lo)
    matrix=np.concatenate((a[upper],-a[lower]));bound=np.r_[hi[upper],-lo[lower]]
    result=linprog(np.zeros(a.shape[1]),A_ub=matrix,b_ub=bound,
                   bounds=[(None,None)]*a.shape[1],method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.status == 2:
        return False
    if result.status != 0 or _violation(a,lo,hi,result.x) > 1e-8:
        raise RuntimeError('feasibility_diagnostic_solver_failed')
    return True


def _energy_rows(energy, scaled_basis, variables):
    """Absolute-value epigraphs with zero objective weight; one shared row."""
    uncertain=np.flatnonzero(energy.uncertainty>0.)
    damped=np.flatnonzero(energy.damping_coefficient>0.)
    maps=np.concatenate((np.eye(6)[uncertain],energy.contact_map[damped]))
    coefficients=np.r_[energy.uncertainty[uncertain],energy.damping_coefficient[damped]]
    count=len(maps);total=variables+count
    mapping=maps @ scaled_basis
    scales=np.maximum(np.max(abs(mapping),axis=1),1e-6) if count else np.empty(0)
    rows=[]
    for sign in (1.,-1.):
        block=np.zeros((count,total));block[:,:3]=sign*mapping
        if count:block[:,variables:]=-np.diag(scales)
        rows.extend(block)
    shared=np.zeros(total);shared[:3]=energy.wrench_environment @ scaled_basis
    shared[variables:]=-coefficients*scales
    rows.append(shared)
    lo=np.r_[np.full(2*count,-math.inf),energy.tracking_cost_w-energy.beta*energy.available_j/energy.hold_s-energy.task_power_w]
    hi=np.r_[np.zeros(2*count),math.inf]
    a,l,u=_normalize_rows(np.asarray(rows),lo,hi)
    return a,l,u,count


class ContactQp:
    """Stateless physical law; cached ProxQP workspace is numerical state only."""

    def __init__(self, config: QpConfig | None = None):
        self.config = config or QpConfig()
        self.repair_episode=None
        if self.config.allocation_policy=='differential_repair_v8':
            from .repair_episode import RepairEpisode
            policy=self.config.differential_repair
            self.repair_episode=RepairEpisode(max_permission_s=policy.max_permission_s,
                max_angle_travel_rad=policy.max_angle_travel_rad,healthy_frames=policy.healthy_frames,
                permission_mode=policy.permission_mode)
        self._solver = None
        self._solver_shape = None
        self._numeric_attempts = []
        self._numeric_deferred_reason = None

    @staticmethod
    def _deadline_expired(deadline_s):
        if deadline_s is None:
            return False
        now = time.monotonic()
        return not math.isfinite(now) or now >= deadline_s

    def _solve_numeric(self, hessian, gradient, c, lo, hi, *, energy_active=False,
                       deadline_s=None):
        self._numeric_attempts = []
        self._numeric_deferred_reason = None
        if self.config.solver_policy == "bounded_retry_v1":
            return self._solve_numeric_bounded(hessian, gradient, c, lo, hi,
                                               energy_active=energy_active, deadline_s=deadline_s)
        if self._deadline_expired(deadline_s):
            self._numeric_deferred_reason = "solver_deadline_exceeded"
            return np.full(len(gradient), np.nan), "not_started", False, 0
        result = self._solve_numeric_legacy(hessian, gradient, c, lo, hi,
                                            energy_active=energy_active)
        if self._deadline_expired(deadline_s):
            self._numeric_deferred_reason = "solver_deadline_exceeded"
            return result[0], result[1], False, result[3]
        return result

    def _solve_numeric_bounded(self, hessian, gradient, c, lo, hi, *, energy_active,
                               deadline_s):
        """Two finite numerical attempts; no physical row or tolerance changes.

        Iteration caps bound native work, not OS scheduling. Absolute deadline
        checks fence late results; publication must independently check its age.
        """
        import proxsuite
        shape = (len(gradient), 0, len(c))
        primary_precondition = self.config.solver_preconditioning or energy_active
        specs = (("cached_primary", 30, primary_precondition, 1e-5),
                 ("fresh_ruiz_retry", 200, True, 1e-3))
        last = (np.full(len(gradient), np.nan), "not_started", False, 0)
        total_iterations = 0
        for name, outer_cap, precondition, rho in specs:
            if self._deadline_expired(deadline_s):
                self._numeric_deferred_reason = "solver_deadline_exceeded"
                break
            started = time.perf_counter()
            workspace_key = (shape, precondition, "bounded_retry_v1")
            fresh = name != "cached_primary" or self._solver_shape != workspace_key
            solver = self._solver
            try:
                if fresh:
                    solver = proxsuite.proxqp.dense.QP(*shape)
                    settings = solver.settings
                    settings.eps_abs = self.config.solver_tolerance
                    settings.eps_rel = 0.
                    settings.max_iter = outer_cap
                    settings.max_iter_in = 10
                    settings.eps_primal_inf = 1e-10
                    settings.eps_dual_inf = 1e-10
                    settings.check_duality_gap = True
                    settings.eps_duality_gap_abs = self.config.solver_tolerance
                    settings.eps_duality_gap_rel = 0.
                    solver.init(hessian, gradient, np.empty((0, len(gradient))), np.empty(0),
                                c, lo, hi, compute_preconditioner=precondition, rho=rho)
                    if name == "cached_primary":
                        self._solver, self._solver_shape = solver, workspace_key
                else:
                    solver.update(H=hessian, g=gradient, C=c, l=lo, u=hi)
                solver.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
                # Initialisation/equilibration also consumes the same deadline.
                if self._deadline_expired(deadline_s):
                    self._numeric_deferred_reason = "solver_deadline_exceeded"
                    self._numeric_attempts.append(dict(attempt=name, fresh_workspace=fresh,
                        status="deadline_before_solve", accepted=False,
                        elapsed_s=time.perf_counter()-started, max_outer_iterations=outer_cap,
                        max_inner_iterations=10, preconditioning=precondition, rho=rho))
                    break
                solver.solve()
                info = solver.results.info
                x = np.array(solver.results.x, copy=True)
                finite = bool(np.isfinite(x).all() and np.isfinite(solver.results.z).all())
                primal, dual, gap = float(info.pri_res), float(info.dua_res), float(info.duality_gap)
                residual = _violation(c, lo, hi, x)
                solved = info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
                accepted = bool(solved and finite and np.isfinite([primal, dual, gap]).all()
                    and 0. <= primal <= self.config.solver_tolerance
                    and 0. <= dual <= self.config.solver_tolerance
                    and abs(gap) <= self.config.solver_tolerance
                    and residual <= self.config.feasibility_tolerance)
                expired = self._deadline_expired(deadline_s)
                total_iterations += int(info.iter)
                self._numeric_attempts.append(dict(attempt=name, fresh_workspace=fresh,
                    status=str(info.status), accepted=accepted and not expired,
                    finite=finite, primal_residual=primal, dual_residual=dual, duality_gap=gap,
                    constraint_violation=residual, iterations=int(info.iter),
                    outer_iterations=int(info.iter_ext), max_outer_iterations=outer_cap,
                    max_inner_iterations=10, preconditioning=precondition, rho=rho,
                    elapsed_s=time.perf_counter()-started, deadline_exceeded=expired))
                last = x, str(info.status), accepted and not expired, total_iterations
                if expired:
                    self._numeric_deferred_reason = "solver_deadline_exceeded"
                    break
                if accepted:
                    if name != "cached_primary":
                        self._solver, self._solver_shape = None, None
                    return last
            except (ValueError, RuntimeError) as exc:
                self._numeric_attempts.append(dict(attempt=name, fresh_workspace=fresh,
                    status="numeric_exception", exception=str(exc), accepted=False,
                    elapsed_s=time.perf_counter()-started, max_outer_iterations=outer_cap,
                    max_inner_iterations=10, preconditioning=precondition, rho=rho))
                last = np.full(len(gradient), np.nan), "numeric_exception", False, total_iterations
            # Failed numerical state is never the next proposal's starting state.
            self._solver, self._solver_shape = None, None
        self._solver, self._solver_shape = None, None
        self._numeric_deferred_reason = self._numeric_deferred_reason or "solver_attempts_exhausted"
        return last[0], last[1], False, total_iterations

    def _solve_numeric_legacy(self, hessian, gradient, c, lo, hi, *, energy_active=False):
        import proxsuite
        shape = (len(gradient), 0, len(c))
        precondition = self.config.solver_preconditioning or energy_active
        workspace_key = (shape, precondition)
        if self._solver_shape != workspace_key:
            self._solver = proxsuite.proxqp.dense.QP(*shape)
            self._solver_shape = workspace_key
            self._solver.settings.eps_abs = self.config.solver_tolerance
            self._solver.settings.eps_rel = 0.
            self._solver.settings.max_iter = self.config.max_iterations
            self._solver.settings.max_iter_in = self.config.inner_iterations
            self._solver.settings.eps_primal_inf = 1e-10
            self._solver.settings.eps_dual_inf = 1e-10
            # The shared energy row adds a different scale even when slack.
            # The recorded uncalibrated/003 failure needs Ruiz equilibration (9
            # iterations vs MAX_ITER_REACHED). Keep legacy no-energy behavior.
            # Energy rows, objective, tolerances and final review are unchanged.
            self._solver.init(hessian, gradient, np.empty((0, len(gradient))), np.empty(0), c, lo, hi,
                              compute_preconditioner=precondition)
        else:
            self._solver.update(H=hessian, g=gradient, C=c, l=lo, u=hi)
        # No reuse of a failed dual iterate or command history across proposals.
        self._solver.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        self._solver.settings.check_duality_gap = energy_active
        self._solver.settings.eps_duality_gap_abs = self.config.solver_tolerance
        self._solver.settings.eps_duality_gap_rel = 0.
        self._solver.solve()
        status = self._solver.results.info.status
        return (np.array(self._solver.results.x, copy=True), str(status),
                status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED,
                int(self._solver.results.info.iter))

    def solve(self, data: QpInput, *, interval_diagnostics: bool | None = None,
              deadline_s: float | None = None, online: bool = False) -> QpResult:
        if not isinstance(data, QpInput):
            raise TypeError("ContactQp.solve requires QpInput")
        start = time.perf_counter()
        if type(online) is not bool:
            raise ValueError("online must be boolean")
        if deadline_s is not None:
            if isinstance(deadline_s, (bool, np.bool_)) or not math.isfinite(deadline_s) or deadline_s < 0:
                raise ValueError("deadline_s must be a finite nonnegative monotonic time")
            deadline_s = float(deadline_s)
        self._numeric_attempts = []
        self._numeric_deferred_reason = None
        cfg, mech = self.config, data.mechanical
        v8=cfg.allocation_policy=="differential_repair_v8"
        balance_policy=v8 and cfg.differential_repair.revision=='v8r3_confidence_balance'
        force_gate=cfg.differential_repair.force_gate(data.force_n) if v8 else 1.
        compute_intervals = cfg.compute_interval_diagnostics if interval_diagnostics is None else bool(interval_diagnostics)
        compute_intervals = compute_intervals and not online
        diagnostics = {"quality_policy_version": cfg.quality_policy_version,
                       "acoustic_derivative_certified": False, "port_verified": False,
                       "command_slew_dt_s": data.acceleration_dt_s, "command_hold_s": data.dt_s,
                       "allocation_policy":cfg.allocation_policy,"repair_force_gate":force_gate,
                       "differential_repair_revision":cfg.differential_repair.revision,
                       "balance_deadband":cfg.differential_repair.balance_deadband if balance_policy else None,
                       "solver_policy": cfg.solver_policy, "solver_deadline_s": deadline_s,
                       "online_failure_diagnostics": not online}

        def failure(status, reason):
            diagnostics.update(reason=reason, total_time_s=time.perf_counter() - start)
            if status == ContactStatus.DEFERRED:
                diagnostics.update(retryable=True, qp_input=asdict(data),
                                   qp_config=asdict(cfg), numeric_attempts=self._numeric_attempts)
            return QpResult(None, 0., np.zeros(2), TwistConstraints(valid_until_s=data.now_s), status, diagnostics)

        if self._deadline_expired(deadline_s):
            return failure(ContactStatus.DEFERRED, "solver_deadline_exceeded")

        if (data.dt_s > cfg.max_step_s or (len(mech.A) and
                (not math.isfinite(mech.valid_until_s) or data.now_s >= mech.valid_until_s))):
            return failure(ContactStatus.CERTIFICATE_INVALID, "expired_or_unbounded_mechanical_certificate_or_step")
        basis = motion_basis(data.path_twist)
        nominal_y = np.array([data.nominal_twist[2], data.nominal_twist[4], 1.])
        nominal_full = basis @ nominal_y
        if np.any(np.abs(nominal_full - data.nominal_twist) > cfg.subspace_tolerance):
            return failure(ContactStatus.TASK_INFEASIBLE, "nominal_outside_ordinary_motion_basis")
        zero_path = np.linalg.norm(data.path_twist) <= 1e-14
        scales = np.array([cfg.normal_scale_m_s, cfg.angular_scale_rad_s, 1.,
                           cfg.slack_scale_m_s, cfg.slack_scale_m_s])
        scaled_basis = basis * scales[:3]
        identity = np.eye(6)
        rows = [*identity, *identity, identity[4]]
        lower = [*(-cfg.max_velocity), *(data.previous_twist - cfg.max_acceleration * data.acceleration_dt_s),
                 (-cfg.angle_limit_rad - data.measured_angle) / data.dt_s]
        upper = [*cfg.max_velocity, *(data.previous_twist + cfg.max_acceleration * data.acceleration_dt_s),
                 (cfg.angle_limit_rad - data.measured_angle) / data.dt_s]
        labels = [*(f"velocity_{i}" for i in range(6)), *(f"acceleration_{i}" for i in range(6)), "angle_limit"]
        if len(mech.A):
            rows.extend(mech.A)
            lower.extend(mech.lower)
            upper.extend(mech.upper)
            labels.extend(mech.labels or tuple(f"mechanical_{i}" for i in range(len(mech.A))))
        true_mechanical_a=np.asarray(rows).copy()
        true_mechanical_lo=np.asarray(lower).copy()
        true_mechanical_hi=np.asarray(upper).copy()
        def subspace_failure():
            if not _linear_feasible(true_mechanical_a,true_mechanical_lo,true_mechanical_hi):
                return failure(ContactStatus.MECHANICAL_INFEASIBLE,"mechanical_rows_infeasible")
            return failure(ContactStatus.TASK_INFEASIBLE,"motion_subspace_conflict")

        # Alpha is independent of normal/rocking because b relinquishes axes2/4.
        progress_row = data.path_twist / max(float(data.path_twist @ data.path_twist), 1e-28)
        if not zero_path:
            rows.append(progress_row)
            lower.append(0.)
            upper.append(1.)
            labels.append("progress_range")
        mechanical_count = len(rows)
        endpoint_full = aperture_rows(data.geometry, data.center_interval_m)
        endpoint = endpoint_full.copy()
        endpoint[:, [0, 1, 3, 5]] = 0.
        endpoint_nominal = endpoint @ data.nominal_twist
        reliable = abs(data.force_n - cfg.force_target_n) > cfg.force_sign_band_n + 1e-12
        if cfg.enable_force_priority and reliable and not v8:
            sign = math.copysign(1., data.force_n - cfg.force_target_n)
            rows.extend(sign * endpoint)
            lower.extend([-math.inf] * 2)
            upper.extend(sign * endpoint_nominal)
            labels.extend(("force_priority_left_endpoint", "force_priority_right_endpoint"))
        force_count = len(rows)
        if cfg.enable_aperture:
            rows.extend(endpoint)
            lower.extend(endpoint_nominal - cfg.aperture_budget_m_s)
            upper.extend(endpoint_nominal + cfg.aperture_budget_m_s)
            labels.extend(("aperture_left_endpoint", "aperture_right_endpoint"))
        if data.energy is not None:
            energy_a,energy_lo,energy_hi=data.energy.velocity_rows()
            if np.any(energy_lo>energy_hi):
                return failure(ContactStatus.TASK_INFEASIBLE,"energy_tracking_bound_infeasible")
            rows.extend(energy_a);lower.extend(energy_lo);upper.extend(energy_hi)
            labels.extend(f"energy_contact_speed_{i}" for i in range(len(energy_a)))
        hard_a, hard_lo, hard_hi = np.asarray(rows), np.asarray(lower), np.asarray(upper)
        cn, ln, un = _normalize_rows(hard_a @ scaled_basis, hard_lo, hard_hi)
        if not np.isfinite(cn).all() or np.isnan(ln).any() or np.isnan(un).any():
            return failure(ContactStatus.CERTIFICATE_INVALID, "nonfinite_transformed_hard_rows")
        # Zero b has no Cartesian alpha row; fix its unobservable coefficient.
        alpha_row = np.array([[0., 0., 1.]])
        alpha_hi = 0. if zero_path else 1.
        def interval(count):
            projected = _omega_interval(np.concatenate((cn[:count], alpha_row)),
                                        np.r_[ln[:count], 0.], np.r_[un[:count], alpha_hi],
                                        tolerance=cfg.feasibility_tolerance)
            return None if projected is None else tuple(v * scales[1] for v in projected)
        intervals = (None, None, None)
        if compute_intervals:
            try:
                im = interval(mechanical_count)
                iff = im if force_count == mechanical_count else interval(force_count)
                ia = iff if force_count == len(rows) else interval(len(rows))
                intervals = im, iff, ia
            except RuntimeError as exc:
                return failure(ContactStatus.SOLVER_FAILED, str(exc))
        diagnostics.update(omega_interval_mechanical=intervals[0], omega_interval_force_priority=intervals[1],
                           omega_interval_complete=intervals[2], force_sign_reliable=reliable,
                           interval_diagnostics_computed=compute_intervals,
                           nominal_subspace_residual=float(np.max(np.abs(nominal_full - data.nominal_twist))))
        if compute_intervals and intervals[0] is None:
            try:
                return subspace_failure()
            except RuntimeError as exc:
                return failure(ContactStatus.SOLVER_FAILED,str(exc))
        if compute_intervals and intervals[2] is None:
            return failure(ContactStatus.TASK_INFEASIBLE, "relative_baseline_rows_conflict_with_mechanical_admission")
        observation = data.observation
        image_valid = bool(observation is not None and observation.fresh(data.now_s, cfg.max_image_age_s))
        quality = observation.quality[[0, 2]] if image_valid else np.zeros(2)
        deficit = np.maximum(cfg.c_min - quality, 0.) / cfg.c_min
        margin = np.maximum(quality - cfg.c_min, 0.) / (1. - cfg.c_min)
        loss = float(np.max(deficit)) if image_valid else 1.
        alpha_preferred = 1. - .75 * loss if cfg.enable_progress_loss else 1.
        gamma = data.gamma if cfg.enable_consistency else np.ones(2)
        episode=None;repair_allowed=True
        if v8:
            try:
                episode=self.repair_episode.update(now_s=data.now_s,measured_angle=data.measured_angle,
                    observation=observation,image_valid=image_valid,c_min=cfg.c_min,force_gate=force_gate,
                    execution_enabled=data.repair_execution_enabled,angle_reference_reset=data.repair_angle_reference_reset,
                    balance_deadband=cfg.differential_repair.balance_deadband if balance_policy else None)
            except ValueError as exc:
                return failure(ContactStatus.CERTIFICATE_INVALID,str(exc))
            diagnostics['repair_episode']=episode
            diagnostics['repair_permission_mode']=cfg.differential_repair.permission_mode
            repair_allowed=episode['repair_allowed']
        requests = (force_gate if repair_allowed else 0.) * gamma * cfg.repair_speed_m_s * deficit - cfg.keep_speed_m_s * margin
        visual = window_rows(data.geometry, cfg.lateral_windows)[[0, 2]]
        visual_enabled = image_valid and cfg.enable_visual and (not v8 or force_gate>0.)
        visual_window_active=np.full(2,visual_enabled,dtype=bool)
        if v8 and not repair_allowed:visual_window_active &= quality>=cfg.c_min
        visual_enabled=bool(visual_window_active.any())
        requests=np.where(visual_window_active,requests,0.)
        differential_enabled = v8 and visual_enabled and repair_allowed
        differential_deficit = gamma*deficit if differential_enabled else np.zeros(2)
        differential_imbalance=float(differential_deficit[0]-differential_deficit[1])
        if balance_policy:
            differential_imbalance=cfg.differential_repair.confidence_imbalance(quality,gamma) if differential_enabled else 0.
        differential_requested = cfg.repair_speed_m_s*force_gate*abs(differential_imbalance)
        differential_row=(visual[0]-visual[1]).copy();differential_row[[0,1,3,5]]=0.
        differential_sign=float(np.sign(differential_imbalance))
        differential_nominal=differential_sign*float(differential_row @ data.nominal_twist)
        diagnostics.update(image_valid=image_valid, acquisition_loss=loss, alpha_preferred=alpha_preferred,
                           visual_rows_active=visual_enabled,visual_window_active=visual_window_active, visual_requests_m_s=requests if visual_enabled else np.zeros(2),
                           gamma_effective=gamma, quality=quality)
        x_nominal = np.r_[nominal_y / scales[:3], 0., 0.]
        # At this point every quadratic term has its global minimum at nominal.
        # Returning the input array copy avoids numerical drift from solving it.
        transparent = bool(not zero_path and alpha_preferred == 1. and
                           np.array_equal(nominal_full, data.nominal_twist) and
                           _violation(cn, ln, un, x_nominal[:3]) <= cfg.solver_tolerance and
                           (not visual_enabled or np.all((visual @ data.nominal_twist)[visual_window_active] >= requests[visual_window_active]))
                           and (not v8 or differential_nominal>=differential_requested)
                           and (data.energy is None or data.energy.admissible(data.nominal_twist,
                                tolerance_w=cfg.solver_tolerance,velocity_tolerance=cfg.solver_tolerance)))
        solver_time = 0.
        if transparent:
            twist, alpha, slack = data.nominal_twist.copy(), 1., np.zeros(2)
            diagnostics.update(transparent=True, iterations=0, solver_status="exact_nominal_optimum")
        else:
            target = x_nominal.copy()
            target[2] = alpha_preferred
            weights = np.array([cfg.normal_weight, cfg.angular_weight, cfg.progress_weight,
                                cfg.slack_weight, cfg.slack_weight])
            hessian = np.diag(weights)
            gradient = -weights * target
            if cfg.enable_aperture and cfg.aperture_cost_weight > 0:
                penalty = np.zeros((2, 5))
                penalty[:, :3] = (endpoint @ scaled_basis) / cfg.normal_scale_m_s
                reference = endpoint_nominal / cfg.normal_scale_m_s
                hessian += cfg.aperture_cost_weight * penalty.T @ penalty
                gradient -= cfg.aperture_cost_weight * penalty.T @ reference
            constraints = np.pad(cn, ((0, 0), (0, 2)))
            constraints = np.concatenate((constraints, np.eye(5)[2:]))
            lo = np.r_[ln, 0., 0., 0.]
            hi = np.r_[un, alpha_hi, math.inf, math.inf]
            if visual_enabled:
                visual_c = np.zeros((2, 5))
                visual_c[:, :3] = visual @ scaled_basis
                visual_c[:, 3:] = np.eye(2) * cfg.slack_scale_m_s
                vc, vl, vu = _normalize_rows(visual_c[visual_window_active], requests[visual_window_active], np.full(int(visual_window_active.sum()), math.inf))
                constraints = np.concatenate((constraints, vc))
                lo, hi = np.r_[lo, vl], np.r_[hi, vu]
            if v8:
                hessian=np.pad(hessian,((0,3),(0,3)));gradient=np.pad(gradient,(0,3))
                constraints=np.pad(constraints,((0,0),(0,3)))
                ph,pg,pr,pu,pfacts=cfg.differential_repair.terms(
                    scaled_basis=scaled_basis,visual_rows=visual,endpoint_rows=endpoint,
                    nominal_twist=data.nominal_twist,deficits=differential_deficit,
                    repair_speed=cfg.repair_speed_m_s,normal_scale=cfg.normal_scale_m_s,
                    force_n=data.force_n,alpha_preferred=alpha_preferred,progress_weight=cfg.progress_weight,
                    differential_imbalance=differential_imbalance)
                hessian+=ph;gradient+=pg
                constraints=np.concatenate((constraints,pr,np.eye(8)[5:]))
                lo=np.r_[lo,np.full(len(pr),-math.inf),np.zeros(3)]
                hi=np.r_[hi,pu,np.full(3,math.inf)]
                diagnostics.update(pfacts)
            if data.energy is not None:
                ea,el,eu,auxiliary_count=_energy_rows(data.energy,scaled_basis,len(gradient))
                hessian=np.pad(hessian,((0,auxiliary_count),(0,auxiliary_count)))
                gradient=np.pad(gradient,(0,auxiliary_count))
                constraints=np.pad(constraints,((0,0),(0,auxiliary_count)))
                constraints=np.concatenate((constraints,ea));lo=np.r_[lo,el];hi=np.r_[hi,eu]
                diagnostics['energy_auxiliary_variables']=auxiliary_count
            solver_start = time.perf_counter()
            try:
                numeric_kwargs = dict(energy_active=data.energy is not None)
                if deadline_s is not None:
                    numeric_kwargs['deadline_s'] = deadline_s
                x, solver_status, solved, iterations = self._solve_numeric(hessian, gradient, constraints, lo, hi, **numeric_kwargs)
            except (ValueError, RuntimeError) as exc:
                return failure(ContactStatus.SOLVER_FAILED, str(exc))
            solver_time = time.perf_counter() - solver_start
            diagnostics.update(transparent=False, iterations=iterations, solver_status=solver_status)
            if cfg.solver_policy == "bounded_retry_v1":
                diagnostics['numeric_attempts'] = self._numeric_attempts
            if not solved or not np.isfinite(x).all() or _violation(constraints, lo, hi, x) > cfg.feasibility_tolerance:
                if online or self._numeric_deferred_reason is not None:
                    diagnostics['numeric_problem'] = dict(H=hessian, g=gradient, C=constraints, l=lo, u=hi)
                    return failure(ContactStatus.DEFERRED,
                        self._numeric_deferred_reason or "solver_attempts_exhausted")
                # Diagnose failures separately from the successful per-tick
                # path. No visual slack or relative row may soften mechanics.
                try:
                    if interval(mechanical_count) is None:
                        return subspace_failure()
                    if interval(len(rows)) is None:
                        return failure(ContactStatus.TASK_INFEASIBLE, "relative_baseline_rows_conflict_with_mechanical_admission")
                except RuntimeError as exc:
                    return failure(ContactStatus.SOLVER_FAILED, str(exc))
                if data.energy is not None:
                    try:
                        if not _linear_feasible(constraints,lo,hi):
                            return failure(ContactStatus.TASK_INFEASIBLE,"energy_budget_infeasible")
                    except RuntimeError as exc:
                        return failure(ContactStatus.SOLVER_FAILED,str(exc))
                return failure(ContactStatus.SOLVER_FAILED, "solver_status_or_residual")
            y = x[:3] * scales[:3]
            alpha = float(np.clip(y[2], 0., alpha_hi))
            y[2] = alpha
            twist = basis @ y
            slack = np.where(visual_window_active,np.maximum(requests - visual @ twist,0.),0.)
        if _violation(hard_a, hard_lo, hard_hi, twist) > cfg.feasibility_tolerance:
            return failure(ContactStatus.SOLVER_FAILED, "final_hard_constraint_residual")
        if data.energy is not None:
            diagnostics.update(energy_assurance=data.energy.assurance,
                energy_bounds_version=data.energy.bounds_version,
                energy_lower_power_w=data.energy.lower_power_w(twist),
                energy_margin_power_w=data.energy.margin_power_w(twist),
                energy_margin_work_j=data.energy.margin_work_j(twist))
            if not data.energy.admissible(twist,tolerance_w=cfg.feasibility_tolerance,
                                          velocity_tolerance=cfg.feasibility_tolerance):
                return failure(ContactStatus.SOLVER_FAILED,"final_energy_constraint_residual")
        # Export only mechanical/subspace/progress/relative hard rows. The
        # softened image policy rows have no physical-derivative certificate.
        export_hi = hard_hi.copy()
        if not zero_path:
            export_hi[mechanical_count - 1] = alpha
        projector = np.eye(6) - basis @ np.linalg.pinv(basis)
        export_a = np.concatenate((hard_a, projector))
        export_lo = np.r_[hard_lo, -cfg.subspace_tolerance]
        export_hi = np.r_[export_hi, cfg.subspace_tolerance]
        labels.extend(f"motion_subspace_{i}" for i in range(6))
        expiry = data.now_s + cfg.certificate_horizon_s
        if len(mech.A):
            expiry = min(expiry, mech.valid_until_s)
        exported = TwistConstraints(export_a, export_lo, export_hi, "tcp_tool", expiry,
                                    mech.sequence, mech.stop_epoch, tuple(labels))
        values = hard_a @ twist
        active = np.minimum(np.abs(values - hard_lo), np.abs(values - hard_hi)) <= cfg.feasibility_tolerance
        diagnostics.update(active_hard_rows=tuple(label for label, on in zip(labels[:len(hard_a)], active) if on),
                           max_hard_violation=_violation(hard_a, hard_lo, hard_hi, twist),
                           visual_residual_m_s=np.where(visual_window_active,visual @ twist + slack - requests,0.),
                           aperture_added_velocity_m_s=endpoint @ (twist - data.nominal_twist),
                           solver_time_s=solver_time, total_time_s=time.perf_counter() - start)
        if v8:
            total_achieved=differential_sign*float(differential_row @ twist)
            incremental_achieved=total_achieved-differential_nominal
            achieved=total_achieved
            diagnostics.update(differential_request_m_s=differential_requested,
                differential_reference='total_velocity',
                differential_nominal_achieved_m_s=differential_nominal,
                differential_total_achieved_m_s=total_achieved,
                differential_increment_achieved_m_s=incremental_achieved,
                qp_delta_omega_y_rad_s=float(twist[4]-data.nominal_twist[4]),
                differential_achieved_m_s=achieved,
                differential_shortfall_m_s=max(0.,differential_requested-achieved),
                differential_sign=differential_sign,
                legacy_force_priority_replaced=True,
                force_limit_assurance='measured_policy_not_prediction',
                differential_auxiliary_shortfall=0. if transparent else float(x[5]))
        status = ContactStatus.IMAGE_UNAVAILABLE if not image_valid else (
            ContactStatus.NOMINAL if transparent else ContactStatus.REPAIR)
        if self._deadline_expired(deadline_s):
            return failure(ContactStatus.DEFERRED, "solver_deadline_exceeded")
        return QpResult(twist, alpha, slack, exported, status, diagnostics,
                        energy_certificate=data.energy,created_time_s=data.now_s)
