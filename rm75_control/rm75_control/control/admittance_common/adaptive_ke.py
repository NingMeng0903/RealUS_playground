"""Online environment stiffness estimation (Keemink critical damping).

Asymmetric EWMA on |ΔF/Δx|; stiff-first jump on contact rising edge;
idle/detach soft decay toward ke_initial (floored by ke_soft_floor).
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
    ke_forgetting: float = 0.995      # slow forget
    ke_forgetting_inc: float = 0.88   # fast stiffen
    ke_min: float = 40.0
    ke_max: float = 2500.0
    dx_threshold_m: float = 8e-5
    contact_force_n: float = 0.8
    ke_impact_initial: float = 1500.0  # rising-edge jump; 0 disables
    ke_detach_decay_s: float = 1.0
    ke_idle_decay_s: float = 2.0
    ke_soft_floor: float = 300.0      # idle decay floor; 0 → ke_initial
    bd_max: float = 200.0
    bd_min: float = 25.0
    bd_slew_max: float = 400.0
    ke_slew_max: float = 1200.0
    displacement_source: str = "admittance"
    gate_lateral_velocity: bool = True
    lateral_vel_gate_m_s: float = 0.02
    gate_df_spike: bool = True
    df_spike_n: float = 4.0
    f_err_gate_n: float = 1.2
    f_err_gate_frac: float = 0.35
    f_err_gate_floor_n: float = 3.0
    idle_decay_is_gate: float = 0.15
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
            f_err_gate_floor_n=float(a.get("f_err_gate_floor_n", 3.0)),
            idle_decay_is_gate=float(a.get("idle_decay_is_gate", 0.15)),
            settle_ticks=int(a.get("settle_ticks", 10)),
        )


class EnvironmentStiffnessEstimator:
    """EWMA |ΔF/Δx| estimator; outputs K̂e and critical-damping b_d."""

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
        """max(gate_n, frac·|f_des|, floor_n) — floor prevents desired→0 freeze."""
        cfg = self.cfg
        return max(
            float(cfg.f_err_gate_n),
            float(cfg.f_err_gate_frac) * abs(f_des_z),
            float(cfg.f_err_gate_floor_n),
        )

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
    ) -> tuple[float, float]:
        """Return ``(ke_est, bd)``. ``allow_impact_init`` gates rising-edge jump."""
        cfg = self.cfg
        self._mass_z = max(mass_z, 1e-3)
        if not cfg.enabled:
            return self.ke_est, self.bd

        self._f_err_env = max(abs(f_err_z), self._f_err_env * (1.0 - self.dt / 0.3))

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

        # Idle decay toward soft floor when quiet (Iₛ-gated, not |f_err|).
        if (
            not learned
            and cfg.ke_idle_decay_s > 1e-6
            and self._contact_ticks > max(cfg.settle_ticks, 0)
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
