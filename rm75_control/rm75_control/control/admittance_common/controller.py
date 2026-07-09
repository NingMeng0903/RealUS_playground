"""Tool-frame force/motion decoupling + trajectory tracking (De Schutter 1988).

Admittance controller with a single active toggle (``adaptive_ke.enabled``)
and a single tool-Z velocity cap. All axis discrimination is **tool-frame**,
so the same code serves any spatial trajectory (Y sweep, arc, spline, teleop
path) as long as the force axis is nominally normal to the surface.

Tool-Z force axis — discrete 2nd-order velocity admittance:

    M(t) · v̇ + D(t) · (v − v_r) = F_des − F_ext

with:

* fz-only **enter-only contact latch** (latched once |fz|≥threshold until
  ``reset()`` — no Schmitt unlatch during a scan, avoids latch/unlatch
  flicker and repeated stiff-first K̂_e jumps);
* Environment-stiffness-driven critical damping b_d(t) = 2ζ√(M · K̂_e)
  (Keemink et al. 2018 §III.C), K̂_e from ``adaptive_ke.py`` (asymmetric-λ
  EWMA of |ΔF/Δx| per Duan et al. 2018 eq. 14, stiff-first impact +
  soft idle/detach decays — see adaptive_ke docstring);
* **Dimeas & Aspragathos 2016 variable-INERTIA channel** on tool-Z:
  M(t) = M₀ + m_u · Iₛ(t), capped at ``var_damping_m_max`` (a hardware
  safety limit we impose — the paper itself does not cap md). Iₛ is the
  paper's Eq. (5) leaky accumulator (NOT bounded to [0,1]; it "increases
  exponentially in proportion to the magnitude of the oscillation" per
  Sec. 4.1) fed by an HP-filtered raw-force energy ratio (Eqs. 4/6, real-
  time proxy for the paper's FFT band-power ratio). Table 2 showed inertia
  adaptation outperforms damping-only by ~5× on operator effort;
* Leaky bidirectional ∫F_err proactive reference (``proactive_force_ff``) in
  the v_r slot: v̇_r = γ·F_err with Dimeas press-side Iₛ gate + retract ungated.
* Contact-time engagement ramp on the tool-Z setpoint (smooth force start).

Not in scope (see restore-adaptive-bounce-fix plan §Not in scope):
  - Human-takeover state machine (never fires in scripted scans)
  - Press-delay Smith predictor / chase-vs-approach vz split / impact chop /
    detach coast / genuine_impact bounce-vs-chase discrimination
  - ``approach_vz_tool_m_s`` (single unified tool-Z cap only)
  - Relay-switched press/release damping
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)
from rm75_control.control.admittance_common.pose_math import pose_error, wrap_pi


def smooth_deadband_eff(f_err: float, deadband_n: float, width_n: float) -> float:
    """
    C1 smooth deadband: zero inside |f|<=db, ramps to f-sign*db outside transition.
    Reduces PI limit cycles at the deadband edge (Z inertia ripple).
    """
    if width_n <= 0.0:
        if abs(f_err) <= deadband_n:
            return 0.0
        return f_err - math.copysign(deadband_n, f_err)
    af = abs(f_err)
    if af <= deadband_n:
        return 0.0
    if af >= deadband_n + width_n:
        return f_err - math.copysign(deadband_n + 0.5 * width_n, f_err)
    t = (af - deadband_n) / width_n
    gain = t * t * (3.0 - 2.0 * t)
    return math.copysign(gain * (af - deadband_n), f_err)


@dataclass
class AdmittanceConfig:
    """
    force_axes: tool-frame mask for admittance (typ. [0,0,1,0,0,0] = TCP normal).
    track_axes: tool-frame mask for PBAC (typ. [1,1,0,1,1,1] = tangent + orient).
    Trajectory pose_d / vel_ff are base-frame 6D. Fusion is tool-frame sleeve
    decoupling only (no world-XY lstsq lock) so any spatial path is supported.

    Sign convention: v_force_z > 0 presses toward the surface (tool +Z),
    F_ext grows positive with contact force, so eff = F_des − F_ext > 0 ⇒ press.
    """

    euler_order: str = "xyz"
    force_axes: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    control_frame: str = "tool"
    kp_pos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    track_axes: np.ndarray = field(default_factory=lambda: np.ones(6))
    system_delay_s: float = 0.015
    contact_threshold_n: float = 0.5
    # Lateral scan: only |fz| enters contact, not ‖f_xy‖ shear.
    contact_use_fz_only: bool = True
    deadband_n: float = 0.3
    deadband_width_n: float = 0.2
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5])
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
    )
    # Single tool-Z velocity cap on the admittance axis, applied identically
    # in and out of contact. The previous free-space split
    # (``approach_vz_tool_m_s`` ≪ ``max_vz_tool_m_s``) produced a discrete
    # 5× jump in press speed on every contact latch — the "两个下压速度挡"
    # jitter on /tmp/scan_v5.csv. One cap = no switch to jitter on.
    #
    # The unified cap is min(max_vz_tool_m_s, max_velocity[2]); the same value
    # clips both the ``v_force_z`` integrator state AND the send-path clip so
    # the state can never wind up past what physics receives (§2a of the plan).
    max_vz_tool_m_s: float = 0.05
    open_loop: bool = False
    # Engagement ramp: the tool-Z force setpoint ramps from ~contact_threshold
    # to the full value over this many seconds of latched contact (0 = step).
    desired_force_ramp_s: float = 1.0
    # --- 2nd-order admittance on the tool-Z (force) axis ---
    admittance_mass_z: float = 3.0      # virtual mass M [kg]
    admittance_damping_z: float = 60.0  # static D [N·s/m] (adaptive_ke overrides on contact)
    # Leaky ∫F_err proactive v_r (see ``proactive_force_ff``).
    proactive_ff: ProactiveFfConfig = field(default_factory=ProactiveFfConfig)
    pos_err_deadband_m: float = 0.0
    pos_correction_max_m_s: float = 0.0
    # Online K_e → b_d = 2ζ√(m_d K̂_e) for fixed ζ (critical when ζ=1).
    adaptive_ke: AdaptiveKeConfig = field(default_factory=AdaptiveKeConfig)
    # --- Dimeas & Aspragathos 2016 variable-INERTIA channel (tool-Z) ---
    # M(t) = admittance_mass_z + var_damping_m_u · Iₛ(t), capped at m_max.
    # A small residual D bump d_u·Iₛ is also added on top of the adaptive
    # b_d (docstring value: "small residual D at Iₛ=1") — the primary
    # channel is inertia, per paper Table 2 which showed damping-only
    # adaptation is 534 % worse on operator effort than inertia adaptation.
    var_damping_enabled: bool = True
    var_damping_omega_c_hz: float = 3.5     # HP cutoff — captures 4–20 Hz contact resonance
    # Iₛ final-stage EWMA smoothing. Dimeas & Aspragathos 2016 tune λ=0.99 at
    # their 1 kHz loop rate → τ=-Ts/ln(λ)≈0.0995 s. At our 200 Hz rate the
    # literal λ=0.998 the previous refactor carried over gives τ≈2.5 s — 25×
    # slower than the paper. Bounce log (bounce_2n.csv) 6.5-7 Hz episodes
    # showed Iₛ eventually crossing the gate but only after fz had already
    # built up to 5-6 N; 0.951 reproduces the paper's ~0.1 s equivalent
    # bandwidth at 200 Hz: exp(-dt/0.1) = exp(-0.005/0.1) ≈ 0.951.
    var_damping_lambda: float = 0.951
    var_damping_f_max_n: float = 7.0        # RMS scale for the magnitude term
    var_damping_d_u: float = 2.0            # small residual D at Iₛ=1  [N·s/m]
    var_damping_m_u: float = 4.0            # additive M at Iₛ=1        [kg]
    # Dimeas Eq. (8): md = m_min + m_u·Iₛ. The paper's Iₛ is UNBOUNDED (its
    # own Sec. 4.1: "not bounded by 1 ... increases exponentially in
    # proportion to the magnitude of the oscillation"), so md is likewise
    # unbounded in the paper's own control law — there is no "natural
    # bound" to derive it from. This cap is our own hardware safety limit
    # (actuator/virtual-mass authority), not a paper quantity.
    var_damping_m_max: float = 7.0
    
    var_damping_dc_alpha: float = 0.02      # slow EWMA splitting DC bias from AC

    @classmethod
    def from_dict(cls, raw: dict) -> AdmittanceConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        frames = raw.get("frames", {})
        traj = raw.get("trajectory_demo", raw.get("trajectory", {}))
        fa = np.asarray(c.get("force_axes", [0, 0, 1, 0, 0, 0]), dtype=float)
        open_loop = bool(c.get("open_loop", c.get("open_loop_scan", traj.get("open_loop", False))))
        return cls(
            euler_order=str(frames.get("euler_order", "xyz")),
            control_frame=str(frames.get("control_frame", c.get("control_frame", "tool"))),
            force_axes=fa,
            kp_pos=np.asarray(c.get("kp_pos", [0, 0, 0, 0, 0, 0]), dtype=float),
            track_axes=np.asarray(c.get("track_axes", [1, 1, 1, 1, 1, 1]), dtype=float),
            system_delay_s=float(c.get("system_delay_s", 0.015)),
            contact_threshold_n=float(c.get("contact_threshold_n", 0.5)),
            contact_use_fz_only=bool(c.get("contact_use_fz_only", True)),
            deadband_n=float(c.get("deadband_n", 0.3)),
            deadband_width_n=float(c.get("deadband_width_n", 0.2)),
            max_velocity=np.asarray(
                c.get("max_velocity", [0.2, 0.2, 0.10, 0.5, 0.5, 0.5]), dtype=float
            ),
            max_acceleration=np.asarray(
                c.get("max_acceleration", [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]), dtype=float
            ),
            max_vz_tool_m_s=float(c.get("max_vz_tool_m_s", 0.05)),
            open_loop=open_loop,
            desired_force_ramp_s=float(c.get("desired_force_ramp_s", 1.0)),
            admittance_mass_z=float(c.get("admittance_mass_z", 3.0)),
            admittance_damping_z=float(c.get("admittance_damping_z", 60.0)),
            proactive_ff=ProactiveFfConfig.from_dict(c),
            pos_err_deadband_m=float(c.get("pos_err_deadband_m", 0.0)),
            pos_correction_max_m_s=float(c.get("pos_correction_max_m_s", 0.0)),
            adaptive_ke=AdaptiveKeConfig.from_dict(raw, c),
            var_damping_enabled=bool(c.get("var_damping_enabled", True)),
            var_damping_omega_c_hz=float(c.get("var_damping_omega_c_hz", 3.5)),
            var_damping_lambda=float(c.get("var_damping_lambda", 0.951)),
            var_damping_f_max_n=float(c.get("var_damping_f_max_n", 7.0)),
            var_damping_d_u=float(c.get("var_damping_d_u", 2.0)),
            var_damping_m_u=float(c.get("var_damping_m_u", 4.0)),
            var_damping_m_max=float(c.get("var_damping_m_max", 7.0)),
            var_damping_dc_alpha=float(c.get("var_damping_dc_alpha", 0.02)),
        )


class AdmittanceController:
    """
    Pipeline (base trajectory → sleeve fusion → joint stream):
      1. v_pos_base = vel_ff + kp * (pose_d - pose)        # PBAC on track axes
      2. fuse_tool_sleeve: Tool-X/Y ← PBAC,  Tool-Z ← force admittance
      3. output v_cmd_tool (control_frame=tool) or v_cmd_base

    Sign convention on the force axis: v_force_z > 0 presses toward the
    surface (tool +Z), F_ext grows positive with contact force, so
    eff = F_des − F_ext > 0 ⇒ press, eff < 0 ⇒ retract.
    """

    def __init__(self, dt: float, config: AdmittanceConfig | None = None) -> None:
        self.dt = dt
        self.cfg = config or AdmittanceConfig()
        self.last_v_cmd = np.zeros(6)
        self._in_contact_latched = False
        # Reference-clock governor scale (0..1) from the outer orchestration.
        self.time_scale = 1.0
        # 2nd-order admittance state (tool-Z velocity carried across ticks).
        self.v_force_z = 0.0
        # Proactive reference v_r (leaky ∫F_err).
        self.v_r_z = 0.0
        self._proactive_ff = ProactiveForceIntegrator(self.cfg.proactive_ff)
        # Engagement-ramp clock (seconds of latched contact this episode).
        self._contact_time_s = 0.0
        # Blended damping (smooth adaptive-bd handoff).
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        # Adaptive Ke.
        self._ke_estimator = EnvironmentStiffnessEstimator(
            self.cfg.adaptive_ke,
            dt=dt,
            mass_z=self.cfg.admittance_mass_z,
        )
        self.ke_est = float(self.cfg.adaptive_ke.ke_initial)
        self.adaptive_bd = float(self.cfg.admittance_damping_z)
        self.zeta_eff = float(self.cfg.adaptive_ke.zeta)
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        # Dimeas variable-inertia channel state.
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self._f_dc = 0.0          # slow EWMA → DC (setpoint) component
        self._p_hi = 0.0          # EWMA of high-pass energy
        self._p_ac = 0.0          # EWMA of AC energy
        self._init_hp_filter()

    def _init_hp_filter(self) -> None:
        """2nd-order Butterworth high-pass (persistent biquad) for Iₛ."""
        fs = 1.0 / self.dt if self.dt > 0 else 100.0
        wn = min(max(self.cfg.var_damping_omega_c_hz / (0.5 * fs), 1e-3), 0.99)
        b, a = butter(2, wn, btype="high")
        self._hp_b = np.asarray(b, dtype=np.float64)
        self._hp_a = np.asarray(a, dtype=np.float64)
        self._hp_zi = np.zeros(max(len(self._hp_a), len(self._hp_b)) - 1, dtype=np.float64)
        # Fixed ~0.2 s memory regardless of control rate.
        self._is_energy_alpha = float(min(1.0, self.dt / 0.2)) if self.dt > 0 else 0.05

    def set_time_scale(self, scale: float) -> None:
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def reset(self, *, clear_velocity: bool = False) -> None:
        self._in_contact_latched = False
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._hp_zi.fill(0.0)
        self._ke_estimator.reset()
        self.ke_est = self._ke_estimator.ke_est
        self.adaptive_bd = self._ke_estimator.bd
        self.zeta_eff = self._ke_estimator.zeta_eff
        if clear_velocity:
            self.last_v_cmd.fill(0.0)

    # ------------------------------------------------------------------
    # Unified tool-Z velocity cap (single authority, kills the wind-up
    # described in the scan-jitter-fix §2a).
    # ------------------------------------------------------------------
    def _v_z_cap(self) -> float:
        """Unified tool-Z velocity limit (press and retract, in/out of contact)."""
        cap = float(self.cfg.max_vz_tool_m_s)
        mv2 = float(self.cfg.max_velocity[2]) if self.cfg.max_velocity.size >= 3 else cap
        if mv2 > 0.0:
            cap = min(cap, mv2)
        return max(cap, 0.0)

    def _contact_signal_n(self, f_ext: np.ndarray) -> float:
        """Normal-axis contact metric; fz-only ignores lateral scan shear."""
        f = np.asarray(f_ext[:3], dtype=float)
        if self.cfg.contact_use_fz_only:
            return abs(float(f[2]))
        return float(np.linalg.norm(f))

    def _update_contact_latched(self, f_ext: np.ndarray) -> bool:
        """Enter-only contact latch: |fz|≥threshold latches until ``reset()``."""
        if self._in_contact_latched:
            return True
        if self._contact_signal_n(f_ext) >= float(self.cfg.contact_threshold_n):
            self._in_contact_latched = True
        return self._in_contact_latched

    def _update_proactive_v_r(
        self,
        eff: float,
        in_contact: bool,
        dt_eff: float,
        *,
        rising_edge: bool,
    ) -> float:
        if rising_edge:
            self._proactive_ff.clear_press_on_rising_edge()
        self.v_r_z = self._proactive_ff.update(
            eff,
            in_contact=in_contact,
            dt_eff=dt_eff,
            instability_index=self.instability_index,
            v_force_z=self.v_force_z,
            v_z_cap=self._v_z_cap(),
        )
        return self.v_r_z

    @staticmethod
    def fuse_tool_sleeve(
        v_pos_base: np.ndarray,
        v_force_tool: np.ndarray,
        r_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Tool-frame orthogonal decoupling (sleeve / slider):
          Tool-X/Y ← trajectory / visual servo feedforward
          Tool-Z   ← force admittance only — no lateral compensation for Z motion.
        """
        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ np.asarray(v_pos_base[:3], dtype=float)
        v_pos_tool[3:6] = r_mat.T @ np.asarray(v_pos_base[3:6], dtype=float)

        v_cmd_tool = v_pos_tool.copy()
        v_cmd_tool[2] = float(v_force_tool[2])

        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]
        return v_cmd_tool, v_cmd_base

    def compute_velocity_command(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        desired_vel_ff: np.ndarray,
        f_ext: np.ndarray,
        desired_force: np.ndarray,
        *,
        in_contact: bool | None = None,
        enable_pbac: bool | None = None,
        f_ext_raw: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        r_mat = Rsc.from_euler(
            cfg.euler_order, current_pose[3:6], degrees=False
        ).as_matrix()

        pose_predicted = np.asarray(current_pose, dtype=float).copy()
        if cfg.system_delay_s > 0.0:
            if cfg.control_frame == "tool":
                pose_predicted[:3] += r_mat @ self.last_v_cmd[:3] * cfg.system_delay_s
            else:
                pose_predicted[:3] += self.last_v_cmd[:3] * cfg.system_delay_s

        err_pose = pose_error(desired_pose, pose_predicted, cfg.euler_order)
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        use_pbac = (not cfg.open_loop) if enable_pbac is None else bool(enable_pbac)
        if not use_pbac:
            err_pose[:] = 0.0
        # --- Translation PBAC in the TOOL frame (task-frame formalism) ---
        # Force- and velocity-controlled directions must be orthogonal IN THE
        # TASK FRAME (De Schutter & Van Brussel 1988 / Bruyninckx 1996):
        # compute the correction in the tool frame and drop the tool-Z (force)
        # component before applying gains.
        err_tool = r_mat.T @ err_pose[:3]
        err_tool[2] = 0.0
        if cfg.pos_err_deadband_m > 0.0:
            for i in (0, 1):
                if abs(err_tool[i]) <= cfg.pos_err_deadband_m:
                    err_tool[i] = 0.0
        kp_xy = np.array([
            cfg.kp_pos[0] * cfg.track_axes[0],
            cfg.kp_pos[1] * cfg.track_axes[1],
            0.0,
        ])
        v_corr_tool = kp_xy * err_tool
        if cfg.pos_correction_max_m_s > 0.0:
            v_corr_tool[:2] = np.clip(
                v_corr_tool[:2], -cfg.pos_correction_max_m_s, cfg.pos_correction_max_m_s
            )
        v_corr = np.zeros(6, dtype=float)
        v_corr[:3] = r_mat @ v_corr_tool
        # Rotational PBAC in the tool/task frame too (mirror the translation path).
        err_rot_tool = r_mat.T @ err_pose[3:6]
        kp_rot = cfg.kp_pos[3:6] * cfg.track_axes[3:6]
        v_corr[3:6] = r_mat @ (kp_rot * err_rot_tool)
        v_pos_base = vel_ff + v_corr

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        f_ext_z = float(f_ext[2])

        was_latched = self._in_contact_latched
        if in_contact is None:
            in_contact = self._update_contact_latched(f_ext)
        else:
            in_contact = bool(in_contact)
            self._in_contact_latched = in_contact

        dt_eff = self.dt * self.time_scale

        # Engagement-ramp clock: monotonically grows while in contact and
        # HOLDS across brief contact losses (a 100 ms bounce flight must not
        # pull ``f_des_z_eff`` back down to the ramp start — that was another
        # jitter source on hard-surface scans). The clock only resets on an
        # explicit ``reset()`` call (new session).
        if in_contact:
            self._contact_time_s += dt_eff

        # Contact rising edge — the estimator's stiff-first K̂_e jump fires
        # ONLY here (allow_impact_init=True). Steady-state gate updates in
        # ``compute_velocity_command`` never re-trigger the impact jump.
        rising_edge = bool(in_contact) and not was_latched

        # Dimeas & Aspragathos 2016 online instability index — fed with the
        # RAW (pre-LPF) normal-force sample so the 5–20 Hz contact-resonance
        # band is visible. Falls back to f_ext_z (filtered) if the caller
        # didn't provide a raw sample.
        f_z_for_index = float(f_ext_raw[2]) if f_ext_raw is not None else f_ext_z
        self._update_instability_index(f_z_for_index)

        # Virtual mass with Dimeas inertia inflation, capped at m_max so a
        # persistent Iₛ doesn't make the retract feel arbitrarily heavy.
        m_z = cfg.admittance_mass_z + cfg.var_damping_m_u * self.instability_index
        if cfg.var_damping_m_max > 0.0:
            m_z = min(m_z, cfg.var_damping_m_max)
        self._m_z_now = max(m_z, 1e-3)

        # Effective tool-Z setpoint (engagement ramp).
        f_des_z = self._effective_desired_z(float(f_des[2]))
        f_err_z = f_des_z - f_ext_z

        # Tangential (tool-XY) speed magnitude, direction-agnostic — any sweep
        # direction (Y, arc, spline, teleop) gates K̂_e learning the same way.
        v_lateral_m_s = float(np.linalg.norm((r_mat.T @ v_pos_base[:3])[:2]))
        if cfg.adaptive_ke.enabled:
            self.ke_est, self.adaptive_bd = self._ke_estimator.update(
                f_ext_z,
                current_pose,
                in_contact=bool(in_contact),
                mass_z=self._m_z_now,
                v_force_z=self.v_force_z,
                v_lateral_m_s=v_lateral_m_s,
                f_err_z=f_err_z,
                f_des_z=f_des_z,
                instability_index=self.instability_index,
                euler_order=cfg.euler_order,
                allow_impact_init=rising_edge,
            )
            self.zeta_eff = self._ke_estimator.zeta_eff

        v_force_tool = np.zeros(6, dtype=float)
        v_force_tool[2] = self._admittance_z(
            f_err_z,
            in_contact,
            dt_eff=dt_eff,
            rising_edge=rising_edge,
        )

        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(v_pos_base, v_force_tool, r_mat)
        v_z_cap = self._v_z_cap()
        if v_z_cap > 0.0:
            v_cmd_tool[2] = float(np.clip(v_cmd_tool[2], -v_z_cap, v_z_cap))
            if cfg.control_frame == "base":
                v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
                v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]

        v_out = v_cmd_tool if cfg.control_frame == "tool" else v_cmd_base
        v_clamp = np.clip(v_out, -cfg.max_velocity, cfg.max_velocity)
        dv_max = cfg.max_acceleration * self.dt
        v_final = np.asarray(v_clamp, dtype=float).copy()
        for i in range(6):
            if cfg.force_axes[i] > 0.5:
                # The 2nd-order admittance (M·v̇ + D·v = F_err) already bounds
                # the force-axis acceleration (=F_err/M) and is smooth by
                # construction — no per-tick Δv gate on the force axis.
                continue
            dvf = dv_max[i]
            v_final[i] = float(np.clip(
                v_final[i],
                self.last_v_cmd[i] - dvf,
                self.last_v_cmd[i] + dvf,
            ))
        self.last_v_cmd = v_final.copy()
        return v_final

    def _effective_desired_z(self, f_des_z: float) -> float:
        """Engagement-ramped tool-Z force setpoint.

        The ramp start (also the free-space approach setpoint) sits ABOVE
        latch threshold + full deadband, otherwise the deadbanded error goes
        to zero just below the contact latch and the probe deadlocks hovering
        on a soft surface. (This bit is physics, not a patch.)
        """
        cfg = self.cfg
        if cfg.desired_force_ramp_s > 1e-6 and f_des_z > 0.0:
            ramp = float(np.clip(self._contact_time_s / cfg.desired_force_ramp_s, 0.0, 1.0))
            f_start = min(
                f_des_z,
                max(
                    cfg.contact_threshold_n + cfg.deadband_n + cfg.deadband_width_n + 0.2,
                    0.35 * f_des_z,
                ),
            )
            f_eff = f_start + (f_des_z - f_start) * ramp
        else:
            f_eff = f_des_z
        self.f_des_z_eff = float(f_eff)
        return float(f_eff)

    def _update_instability_index(self, f_z: float) -> None:
        """
        Dimeas & Aspragathos 2016 online instability index, Eqs. (4)-(6),
        with Iω's FFT band-power ratio replaced by an equivalent real-time
        energy ratio (no per-tick FFT budget), fed with the RAW normal-force
        signal so the 5–20 Hz contact-resonance band is visible:

            hp   = highpass(f_z, ω_c)         # oscillation band only
            f_ac = f_z − dc(f_z)              # strip the force setpoint bias
            Iω   = E[hp²] / E[f_ac²]          # HF / AC energy ratio ∈[0,1]  (Eq. 4)
            Iᵣₘₛ = rms(f_ac) / f_max          # bounded magnitude term ∈[0,1] (Eq. 6)

        Eq. (5) is a LEAKY ACCUMULATOR, not an EWMA blend:

            Iₛ[kT] = Iω[kT]·Iᵣₘₛ[kT] + λ·Iₛ[(k−1)T]

        The paper states explicitly that Iₛ is *not* bounded by 1 (unlike
        Iω) and "increases exponentially in proportion to the magnitude of
        the [sustained] oscillation" (Sec. 4.1) — λ sets the growth/decay
        time constant, not a convex-combination weight. Iₛ feeds Eq. (8)
        (md = m_min + m_u·Iₛ) directly; there is no normalization step in
        the paper's control law (the [0,1] rescale in Fig. 6c/7c is for
        plotting the comparison against Iω only).
        """
        cfg = self.cfg
        if not cfg.var_damping_enabled:
            self.instability_index = 0.0
            return

        y, self._hp_zi = lfilter(
            self._hp_b,
            self._hp_a,
            np.asarray([f_z], dtype=np.float64),
            zi=self._hp_zi,
        )
        hp = float(y[0])

        self._f_dc += cfg.var_damping_dc_alpha * (f_z - self._f_dc)
        f_ac = f_z - self._f_dc

        a_e = self._is_energy_alpha
        self._p_hi += a_e * (hp * hp - self._p_hi)
        self._p_ac += a_e * (f_ac * f_ac - self._p_ac)

        i_omega = min(max(self._p_hi / (self._p_ac + 1e-6), 0.0), 1.0)
        i_rms = min(
            math.sqrt(max(self._p_ac, 0.0)) / max(cfg.var_damping_f_max_n, 1e-6),
            1.0,
        )
        lam = cfg.var_damping_lambda
        self.instability_index = i_omega * i_rms + lam * self.instability_index

    def _admittance_z(
        self,
        f_err: float,
        in_contact: bool,
        *,
        dt_eff: float,
        rising_edge: bool,
    ) -> float:
        """
        Discrete 2nd-order velocity admittance on tool-Z:

            M · v̇ + D · (v − v_r) = F_err

        D = b_d(t) = 2ζ√(M · K̂_e) on contact
        """
        cfg = self.cfg
        eff = smooth_deadband_eff(f_err, cfg.deadband_n, cfg.deadband_width_n)
        m = max(float(self._m_z_now), 1e-3)
        if cfg.adaptive_ke.enabled and in_contact:
            d_target = float(self.adaptive_bd)
        else:
            d_target = float(cfg.admittance_damping_z)
        if cfg.var_damping_enabled:
            d_target += cfg.var_damping_d_u * self.instability_index
            if cfg.adaptive_ke.bd_max > 0.0:
                d_target = min(d_target, float(cfg.adaptive_ke.bd_max))
        if dt_eff > 0.0:
            tau_d = 0.025 if self.instability_index > 0.5 else 0.10
            a = min(1.0, dt_eff / tau_d)
            self._d_z_smooth += a * (d_target - self._d_z_smooth)
        else:
            self._d_z_smooth = d_target
        d = self._d_z_smooth
        self.damping_z_eff = float(d)

        v_z_cap = self._v_z_cap()
        v_r = self._update_proactive_v_r(
            eff,
            in_contact,
            dt_eff,
            rising_edge=rising_edge,
        )

        v = self.v_force_z + (dt_eff / m) * (eff - d * (self.v_force_z - v_r))

        if v_z_cap > 0.0:
            v = float(np.clip(v, -v_z_cap, v_z_cap))
        self.v_force_z = v
        return v


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
