"""Online environment stiffness estimation + critical-damping admittance.

Coupled contact model on the normal (tool-Z) admittance axis:

    m_d · ẍ + b_d · ẋ + K_e · x = F_ext

Damping ratio ζ = b_d / (2√(m_d K_e)). Holding ζ fixed while K_e changes
requires (Keemink et al. 2018 §III.C):

    b_d(t) = 2 ζ √(m_d · K̂_e(t))

Learning rule (Duan, Gan, Chen & Dai, RAS 102 (2018) eq. 14, asymmetric
EWMA on |ΔF/Δx| — the 27c1689 shape that hardware confirmed keeps hard
surfaces stable):

    if in_contact and gates_pass and |Δx| >= dx_threshold:
        ke_inst = |ΔF/Δx|
        λ       = ke_forgetting_inc  if ke_inst > K̂_e   (fast track up)
                  ke_forgetting      otherwise           (slow forget down)
        K̂_e   ← λ · K̂_e + (1 − λ) · ke_inst

Stiff-first impact initialisation: on a contact rising edge we jump K̂_e up to
``ke_impact_initial`` (b_d follows immediately, no slew). Underdamped first
few ticks on a hard surface is exactly what starts a bounce cascade; jumping
to overdamped and then learning DOWN on soft surfaces avoids that.

Idle / detach soft decay: neither hold-last (previous refactor: K̂_e climbs
monotonically) nor hard reset (older refactor: b_d drops to ~16 N·s/m on
every re-impact and re-starts the bounce) is safe. Both branches decay
K̂_e toward ``ke_initial`` with a time constant (τ_idle in steady contact
with small |f_err|, τ_detach out of contact). A 50 ms bounce flight keeps
almost all of the stiffness the estimator just learned about the surface
it will re-hit; a long steady press on soft tissue eventually relaxes
K̂_e so b_d drops back and the press regains bandwidth to chase a receding
surface.

Direction-agnostic tangential gate: ``gate_lateral_velocity`` acts on the
**magnitude** of the tangential (tool-XY) commanded velocity — any spatial
trajectory (Y sweep, X, arc, spline, teleop) gates learning the same way.

``reset()`` seeds K̂_e = ke_initial (new-session semantics only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


@dataclass
class AdaptiveKeConfig:
    enabled: bool = False
    zeta: float = 1.0
    ke_initial: float = 80.0
    # Asymmetric EWMA: fast track up when the surface reads stiffer than we
    # believe (impact-safe), slow forget down when it reads softer (avoid
    # over-reacting to a single quiet tick). Two λ's, not one.
    ke_forgetting: float = 0.995      # slow forget (surface softens)
    ke_forgetting_inc: float = 0.88   # fast track  (surface stiffens)
    ke_min: float = 40.0
    ke_max: float = 2500.0
    dx_threshold_m: float = 8e-5
    contact_force_n: float = 0.8
    # Stiff-first impact initialisation. On a contact rising edge K̂_e jumps
    # UP to this value (b_d follows, no slew). Underdamped-at-impact starts
    # bounce cascades; overdamped-at-impact is safe and learns down on soft
    # surfaces. 0 disables. 27c1689: 1500.
    ke_impact_initial: float = 1500.0
    # Soft-decay time constants toward ke_initial (see module docstring).
    # ke_detach_decay_s: out of contact. 1.0 s keeps ~95 % of learned K̂_e
    # through a 50 ms bounce flight and returns to seed over ~5 s.
    ke_detach_decay_s: float = 1.0
    # ke_idle_decay_s: in steady contact with no learning update AND
    # |f_err|_envelope inside the gate (steady tracking, not over-force).
    # 2.0 s keeps enough stiffness for chase while letting soft tissue relax.
    ke_idle_decay_s: float = 2.0
    # Soft-tissue idle-decay floor: decay target is max(ke_initial, ke_soft_floor)
    # instead of ke_initial alone. Impact stiff-first (ke_impact_initial) is
    # unchanged; only the downward chase decay is prevented from reaching the
    # ~16 N·s/m underdamped band (Ke=80) on a compliant surface — Phase B1.
    # 0 disables (legacy: decay all the way to ke_initial).
    ke_soft_floor: float = 300.0
    bd_max: float = 200.0
    bd_min: float = 25.0
    bd_slew_max: float = 400.0
    ke_slew_max: float = 1200.0
    displacement_source: str = "admittance"
    # Trajectory-agnostic tangential-speed gate (magnitude of tool-XY vel).
    gate_lateral_velocity: bool = True
    lateral_vel_gate_m_s: float = 0.02
    # |ΔF| spike gate: a single-tick jump above df_spike_n N is likely a
    # sensor spike or geometric coupling, not a real stiffness sample.
    gate_df_spike: bool = True
    df_spike_n: float = 4.0
    # |f_err| gate: during an over-force transient the instantaneous
    # ΔF/Δx is dominated by the loop response, not the environment.
    # Effective gate = max(f_err_gate_n, f_err_gate_frac * |f_des_z|):
    # f_err_gate_n is the small-setpoint noise floor; the relative term keeps
    # the "steady vs transient" judgement self-similar at any setpoint. A
    # fixed 1.2 N gate at a 5 N hold froze K̂_e at ke_impact_initial forever
    # (normal hand-interaction ripple > 1.2 N) — b_d stayed ~70+ N·s/m and
    # the retract felt heavily damped.
    f_err_gate_n: float = 1.2
    f_err_gate_frac: float = 0.35
    # Freeze the steady-contact downward decay while the raw Dimeas index
    # indicates a developing contact resonance.
    idle_decay_is_gate: float = 0.15
    # Hold K̂_e (no learning) this many ticks after contact acquisition so
    # the first-impact transient doesn't dominate the estimator.
    settle_ticks: int = 10

    @classmethod
    def from_dict(cls, raw: dict, parent: dict) -> AdaptiveKeConfig:
        a = raw.get("adaptive_ke", parent.get("adaptive_ke", {}))
        if not isinstance(a, dict):
            a = {}
        return cls(
            enabled=bool(a.get("enabled", parent.get("adaptive_ke_enabled", False))),
            zeta=float(a.get("zeta", parent.get("adaptive_zeta", 1.0))),
            ke_initial=float(a.get("ke_initial", parent.get("ke_initial", 80.0))),
            ke_forgetting=float(a.get("ke_forgetting", parent.get("ke_forgetting", 0.995))),
            ke_forgetting_inc=float(
                a.get("ke_forgetting_inc", parent.get("ke_forgetting_inc", 0.88))
            ),
            ke_min=float(a.get("ke_min", parent.get("ke_min", 40.0))),
            ke_max=float(a.get("ke_max", parent.get("ke_max", 2500.0))),
            dx_threshold_m=float(a.get("dx_threshold_m", parent.get("ke_dx_threshold_m", 8e-5))),
            contact_force_n=float(
                a.get("contact_force_n", parent.get("adaptive_contact_force_n", 0.8))
            ),
            ke_impact_initial=float(a.get("ke_impact_initial", 1500.0)),
            ke_detach_decay_s=float(a.get("ke_detach_decay_s", 1.0)),
            ke_idle_decay_s=float(a.get("ke_idle_decay_s", 2.0)),
            ke_soft_floor=float(a.get("ke_soft_floor", 300.0)),
            bd_max=float(a.get("bd_max", parent.get("adaptive_bd_max", 200.0))),
            bd_min=float(a.get("bd_min", parent.get("adaptive_bd_min", 25.0))),
            bd_slew_max=float(a.get("bd_slew_max", parent.get("adaptive_bd_slew_max", 400.0))),
            ke_slew_max=float(a.get("ke_slew_max", parent.get("ke_slew_max", 1200.0))),
            displacement_source=str(
                a.get("displacement_source", parent.get("ke_displacement_source", "admittance"))
            ).lower(),
            gate_lateral_velocity=bool(a.get("gate_lateral_velocity", True)),
            lateral_vel_gate_m_s=float(
                a.get("lateral_vel_gate_m_s", a.get("scan_vel_gate_m_s", 0.02))
            ),
            gate_df_spike=bool(a.get("gate_df_spike", True)),
            df_spike_n=float(a.get("df_spike_n", 4.0)),
            f_err_gate_n=float(a.get("f_err_gate_n", 1.2)),
            f_err_gate_frac=float(a.get("f_err_gate_frac", 0.35)),
            idle_decay_is_gate=float(a.get("idle_decay_is_gate", 0.15)),
            settle_ticks=int(a.get("settle_ticks", 10)),
        )


class EnvironmentStiffnessEstimator:
    """Asymmetric-λ EWMA of |ΔF/Δx| on the normal admittance axis with
    stiff-first impact + soft idle/detach decays (see module docstring).

    Outputs (K̂_e, b_d) with b_d = 2ζ√(m_d K̂_e) (Keemink 2018 critical-damping),
    slewed at ``bd_slew_max`` per second so the send path never sees a step.
    """

    def __init__(self, cfg: AdaptiveKeConfig, *, dt: float, mass_z: float = 3.0) -> None:
        self.cfg = cfg
        self.dt = max(dt, 1e-6)
        self._mass_z = max(mass_z, 1e-3)
        self.ke_est = float(cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose: np.ndarray | None = None
        self._in_contact = False
        self._update_gated = False
        self._contact_ticks = 0
        self.last_dx_m = 0.0
        self.last_df_n = 0.0
        self.update_count = 0
        # |f_err| envelope (peak-hold with ~0.3 s release) gating the idle
        # decay: an oscillation crosses f_err=0 twice per cycle, so the
        # instantaneous |f_err| under-reports over-force by ~100 %.
        self._f_err_env = 0.0

    def reset(self) -> None:
        self.ke_est = float(self.cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose = None
        self._in_contact = False
        self._update_gated = False
        self._contact_ticks = 0
        self.last_dx_m = 0.0
        self.last_df_n = 0.0
        self.update_count = 0
        self._f_err_env = 0.0

    def _critical_bd(self, mass_z: float) -> float:
        ke = max(self.ke_est, self.cfg.ke_min)
        bd = 2.0 * self.cfg.zeta * math.sqrt(max(mass_z, 1e-3) * ke)
        lo = self.cfg.bd_min if self.cfg.bd_min > 0.0 else 0.0
        return float(np.clip(bd, lo, self.cfg.bd_max))

    def _slew_ke(self, ke_target: float) -> float:
        max_dke = self.cfg.ke_slew_max * self.dt
        delta = float(np.clip(ke_target - self.ke_est, -max_dke, max_dke))
        return self.ke_est + delta

    def _slew_damping(self, bd_target: float) -> float:
        max_dbd = self.cfg.bd_slew_max * self.dt
        delta = float(np.clip(bd_target - self.bd, -max_dbd, max_dbd))
        return self.bd + delta

    @staticmethod
    def tool_z_displacement_m(
        pose: np.ndarray,
        ref_pose: np.ndarray,
        *,
        euler_order: str = "xyz",
    ) -> float:
        pose = np.asarray(pose, dtype=float)
        ref = np.asarray(ref_pose, dtype=float)
        d_base = pose[:3] - ref[:3]
        r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
        return float((r_mat.T @ d_base)[2])

    def _normal_displacement_m(
        self,
        pose: np.ndarray,
        *,
        v_force_z: float,
        euler_order: str = "xyz",
    ) -> float:
        if self.cfg.displacement_source == "pose" and self._contact_ref_pose is not None:
            return self.tool_z_displacement_m(pose, self._contact_ref_pose, euler_order=euler_order)
        self._x_adm += float(v_force_z) * self.dt
        return self._x_adm

    def _f_err_gate_eff_n(self, f_des_z: float) -> float:
        """Setpoint-relative |f_err| gate with a small-force noise floor.

        max(f_err_gate_n, f_err_gate_frac·|f_des_z|) keeps the "steady vs
        transient" judgement self-similar at any desired force instead of
        freezing K̂_e whenever the setpoint outgrows a fixed absolute gate.
        """
        cfg = self.cfg
        return max(float(cfg.f_err_gate_n), float(cfg.f_err_gate_frac) * abs(f_des_z))

    def _should_update_ke(
        self,
        f_ext_z: float,
        f_err_z: float,
        v_lateral_m_s: float,
        df: float,
        f_err_gate_n: float,
    ) -> bool:
        cfg = self.cfg
        if abs(f_ext_z) < cfg.contact_force_n:
            return False
        if abs(f_err_z) > f_err_gate_n:
            return False
        if cfg.gate_lateral_velocity and abs(v_lateral_m_s) > cfg.lateral_vel_gate_m_s:
            return False
        if cfg.gate_df_spike and abs(df) > cfg.df_spike_n:
            return False
        return True

    def update(
        self,
        f_ext_z: float,
        pose: np.ndarray,
        *,
        in_contact: bool,
        mass_z: float,
        v_force_z: float = 0.0,
        v_lateral_m_s: float = 0.0,
        f_err_z: float = 0.0,
        f_des_z: float = 0.0,
        instability_index: float = 0.0,
        euler_order: str = "xyz",
        allow_impact_init: bool = True,
        allow_idle_decay: bool = True,
    ) -> tuple[float, float]:
        """Return (ke_est, bd) after one tick.

        ``v_lateral_m_s`` is the magnitude (>=0) of the tangential (tool-XY)
        speed, direction-agnostic — see module docstring.
        ``f_des_z`` is the (ramped) tool-Z force setpoint; it sizes the
        relative |f_err| gate (see ``_f_err_gate_eff_n``).
        ``instability_index`` is the raw Dimeas Iₛ contact-resonance index.
        ``allow_idle_decay`` lets callers veto the quiet-contact decay when
        their physical-contact state is uncertain.
        ``allow_impact_init``: caller sets this False on a contact rising
        edge that follows only a brief flicker (turnaround dip), so the
        stiff-first K̂_e jump fires on genuine impacts only.
        """
        cfg = self.cfg
        self._mass_z = max(mass_z, 1e-3)
        if not cfg.enabled:
            return self.ke_est, self.bd

        # Peak-hold envelope of |f_err| (~0.3 s release).
        self._f_err_env = max(abs(f_err_z), self._f_err_env * (1.0 - self.dt / 0.3))

        # Contact rising edge: stiff-first init (safe overdamped at impact).
        if in_contact and not self._in_contact:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()
            self._x_adm = 0.0
            self._have_prev = False
            self._contact_ticks = 0
            if (
                allow_impact_init
                and cfg.ke_impact_initial > 0.0
                and self.ke_est < cfg.ke_impact_initial
            ):
                self.ke_est = min(float(cfg.ke_impact_initial), cfg.ke_max)
                # b_d jumps with K̂_e immediately: an underdamped first few
                # ticks on a hard surface is what starts a bounce cascade.
                self.bd = self._critical_bd(self._mass_z)

        if not in_contact:
            self._in_contact = False
            self._contact_ref_pose = None
            self._x_adm = 0.0
            self._have_prev = False
            self._update_gated = False
            self._contact_ticks = 0
            tau = max(float(cfg.ke_detach_decay_s), 1e-3)
            self.ke_est += (self.dt / tau) * (float(cfg.ke_initial) - self.ke_est)
            self.ke_est = float(np.clip(self.ke_est, cfg.ke_min, cfg.ke_max))
            bd_target = self._critical_bd(mass_z)
            self.bd = self._slew_damping(bd_target)
            return self.ke_est, self.bd

        self._in_contact = True
        self._contact_ticks += 1
        if self._contact_ref_pose is None:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()

        x = self._normal_displacement_m(pose, v_force_z=v_force_z, euler_order=euler_order)

        f_err_gate_n = self._f_err_gate_eff_n(f_des_z)

        gated = True
        learned = False
        if self._contact_ticks <= max(cfg.settle_ticks, 0):
            gated = True
        elif self._have_prev:
            df = f_ext_z - self._last_f_z
            dx = x - self._last_x
            self.last_df_n = float(df)
            self.last_dx_m = float(dx)
            gated = not self._should_update_ke(
                f_ext_z, f_err_z, v_lateral_m_s, df, f_err_gate_n
            )
            if not gated and abs(dx) >= cfg.dx_threshold_m:
                ke_inst = abs(df / dx)
                ke_inst = float(np.clip(ke_inst, cfg.ke_min, cfg.ke_max))
                lam = (
                    cfg.ke_forgetting_inc if ke_inst > self.ke_est else cfg.ke_forgetting
                )
                ke_target = lam * self.ke_est + (1.0 - lam) * ke_inst
                self.ke_est = self._slew_ke(ke_target)
                learned = True
                self.update_count += 1

        # Stiff-first closure (idle decay): steady tracking with no ΔF/Δx
        # update this tick lets the impact-initialised K̂_e relax toward
        # ke_initial so the press regains bandwidth to chase a receding
        # surface. Gated by |f_err| envelope (over-force transient) AND faded
        # by the Dimeas Iₛ: a building contact resonance must freeze the
        # decay even while its force ripple is still inside the (setpoint-
        # relative) |f_err| gate, otherwise b_d releases mid-bounce on a
        # hard surface.
        if (
            not learned
            and allow_idle_decay
            and cfg.ke_idle_decay_s > 1e-6
            and self._contact_ticks > max(cfg.settle_ticks, 0)
            and self._f_err_env <= f_err_gate_n
            and float(instability_index) <= float(cfg.idle_decay_is_gate)
        ):
            self.ke_est += (self.dt / cfg.ke_idle_decay_s) * (
                max(float(cfg.ke_initial), float(cfg.ke_soft_floor)) - self.ke_est
            )
            self.ke_est = float(np.clip(self.ke_est, cfg.ke_min, cfg.ke_max))

        self._update_gated = gated
        self._last_f_z = f_ext_z
        self._last_x = x
        self._have_prev = True

        bd_target = self._critical_bd(mass_z)
        self.bd = self._slew_damping(bd_target)
        return self.ke_est, self.bd

    @property
    def zeta_eff(self) -> float:
        denom = 2.0 * math.sqrt(max(self._mass_z, 1e-3) * max(self.ke_est, self.cfg.ke_min))
        if denom < 1e-9:
            return 0.0
        return self.bd / denom

    @property
    def update_gated(self) -> bool:
        return self._update_gated
