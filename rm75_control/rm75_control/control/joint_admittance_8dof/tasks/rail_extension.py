"""Preferred arm-extension / pose-attract rail task: proactive base-arm coordination.

Two operating modes (selected by the phase preset):

* ``reach`` (scan / track) — Yamamoto & Yun 1994 preferred arm extension
  ``e = (y_tcp - y_rail) - d_pref`` plus scan feedforward; σ-escape boosts
  authority when the arm nears singularity.
* ``pose_attract`` (move→D) — soft position attractor to the *target pose's*
  rail coordinate ``y_rail_target = q_target[0]``.  Monotonic, settles and
  *stops* (no hunting).  σ_min is a *guardrail only*: with dead-zone + rate
  limit it temporarily pushes along ∂σ/∂y_rail when σ drops below a
  threshold, then hands control back to the pose attractor.  Continuous
  gradient climbing is intentionally *not* used (that caused limit cycles).

Macro-micro (Khatib/Seraji): the desired rail velocity is low-pass filtered
so the rail only absorbs the slow large-displacement component; the arm
nullspace eats the fast residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, RAIL_INDEX
from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
    RailGoodness,
    SigmaMinGoodness,
)


RailExtMode = Literal["reach", "pose_attract"]


def rail_vel_ff_from_reference(
    vel_ff: np.ndarray,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    *,
    k_ff: float = 1.0,
) -> float:
    """Scalar rail speed from any reference ``vel_ff`` (base-frame linear vel).

    Projects the reference linear velocity onto the rail Jacobian column —
    works for sin, spline, hold-to-move, or any ``MotionReference`` that
    populates ``vel_ff[:3]`` in the base frame (as all current sources do).
    """
    v_lin = np.asarray(vel_ff[:3], dtype=float)
    j_rail = kin.jacobian(q_rad)[:3, RAIL_INDEX]
    denom = float(np.dot(j_rail, j_rail))
    if denom < 1e-12:
        return 0.0
    return float(k_ff) * float(np.dot(j_rail, v_lin) / denom)


@dataclass
class RailExtensionConfig:
    enabled: bool = True
    k_ext: float = 1.0
    # Base-frame reference linear velocity feedforward (Yamamoto & Yun 1996):
    # callers pass ``MotionReference.vel_ff``; the rail column projection is
    # trajectory-agnostic (sin, spline, segment, ...).
    k_ff: float = 1.0
    v_ff_thr_m_s: float = 0.01
    v_ff_span_m_s: float = 0.03
    e0_m: float = 0.05
    e1_m: float = 0.15
    w_max: float = 1.5
    # Canonical host soft command band.  The mechanical URDF travel remains
    # available to the model, but rail coordination must stop at this band.
    soft_min_m: float = 0.01
    soft_max_m: float = 0.78
    v_max_m_s: float = 0.08
    # Fade the task to zero within this distance (m) of a rail travel limit
    # when the desired velocity points into the limit.
    limit_margin_m: float = 0.08
    # Bug 2: σ-escape.  When σ_min ↘ the rail should BOOST authority (not
    # cut it — the old ``w *= sigma_scale`` was backwards) and add a
    # non-reaching velocity component along the TCP-preserving σ-ascent
    # direction so the rail acts even inside the reach dead zone.
    #
    # The QP consumes this 4d schedule verbatim.  The ordinary Cartesian
    # slack weight, rail regularisation, and hard boxes still determine the
    # realised allocation; there is no episode-specific weight cap.
    k_sigma_boost: float = 2.0
    # k_esc [m/s per unit σ]: scales the σ-escape velocity component.
    # sigma_grad_rail has units 1/m, so k_esc·(1-sig)·grad has units of m/s.
    k_esc: float = 0.5
    # Baseline w that lets the rail act even when the reach error is inside
    # the dead zone (|e| < e0), provided σ is depressed.  Fades with σ.
    w_sigma_floor: float = 1.0
    # --- move→D pose attractor (primary during preset="move") ---
    k_pose: float = 2.0          # 1/s soft P on (y_target - y_rail)
    pose_e0_m: float = 0.005     # settle dead-zone (m); stops hunting at target
    pose_e1_m: float = 0.04      # full pose-attract weight by this error
    pose_w_max: float = 4.0      # ≪ W_task=100
    # σ guardrail (pose_attract): only engages below enter, clears above exit.
    sigma_guard_enter: float = 0.45
    sigma_guard_exit: float = 0.70
    # Cap on guardrail velocity so it cannot yank the rail off the pose path.
    v_guard_max_m_s: float = 0.04
    # Macro-micro LPF on the *desired* rail velocity (seconds).
    v_lpf_tau_s: float = 0.12


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class RailExtensionTask:
    """Callable: q (rad/m) -> (v_rail_des m/s, w_ext) for the WBC QP."""

    def __init__(
        self,
        kin: RobotKinematics,
        cfg: RailExtensionConfig | None = None,
        *,
        goodness: RailGoodness | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or RailExtensionConfig()
        self.goodness: RailGoodness = goodness or SigmaMinGoodness(kin)
        self.d_pref_m: float | None = None
        self.y_rail_target_m: float | None = None
        self.mode: RailExtMode = "reach"
        self.last_err_m: float = 0.0
        self.last_weight: float = 0.0
        self.last_weight_raw: float = 0.0
        self.last_weight_capped: float = 0.0
        self.last_limit_saturated: bool = False
        self._guard_active: bool = False
        self._v_lpf: float = 0.0
        self._v_lpf_initialized: bool = False

    def reset_dynamics(self) -> None:
        """Clear filter/guard state without changing either rail reference."""
        self.last_err_m = 0.0
        self.last_weight = 0.0
        self.last_weight_raw = 0.0
        self.last_weight_capped = 0.0
        self.last_limit_saturated = False
        self._guard_active = False
        self._v_lpf = 0.0
        self._v_lpf_initialized = False

    def set_mode(self, mode: RailExtMode) -> None:
        mode_s = str(mode).strip().lower()
        if mode_s not in ("reach", "pose_attract"):
            raise ValueError(f"unknown rail extension mode {mode!r}")
        if mode_s != self.mode:
            # Reset LPF on mode switch so a scan FF residue does not kick move.
            self.reset_dynamics()
        self.mode = mode_s  # type: ignore[assignment]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Set / clear the move→D soft attractor target (metres)."""
        if y_rail_m is None:
            self.y_rail_target_m = None
            return
        lo = float(self.kin.q_lower[RAIL_INDEX])
        hi = float(self.kin.q_upper[RAIL_INDEX])
        lo = max(lo, float(self.cfg.soft_min_m))
        hi = min(hi, float(self.cfg.soft_max_m))
        if hi <= lo:
            lo = float(self.kin.q_lower[RAIL_INDEX])
            hi = float(self.kin.q_upper[RAIL_INDEX])
        self.y_rail_target_m = float(np.clip(float(y_rail_m), lo, hi))

    def extension(self, q_rad: np.ndarray) -> float:
        """Arm Y-extension: base-frame TCP y minus rail position (m)."""
        q = np.asarray(q_rad, dtype=float)
        y_tcp = float(self.kin.fk_placement(q).translation[1])
        return y_tcp - float(q[RAIL_INDEX])

    def capture_reference(self, q_rad: np.ndarray) -> None:
        self.d_pref_m = self.extension(q_rad)

    def reset(self, q_rad: np.ndarray) -> None:
        self.capture_reference(q_rad)
        self.reset_dynamics()

    def _limit_saturation(self, q_rail: float, v: float) -> float:
        """Return 0..1 scale; C¹ smoothstep fade before a directional hard stop.

        Fades only when moving *into* a usable soft bound so reversing away
        from a pinned rail recovers authority immediately.  At the soft stop
        the scale is 0; with a wide enough ``limit_margin_m`` the fade
        completes before pin.
        """
        lo = max(
            float(self.kin.q_lower[RAIL_INDEX]), float(self.cfg.soft_min_m)
        )
        hi = min(
            float(self.kin.q_upper[RAIL_INDEX]), float(self.cfg.soft_max_m)
        )
        if hi <= lo:
            lo = float(self.kin.q_lower[RAIL_INDEX])
            hi = float(self.kin.q_upper[RAIL_INDEX])

        margin = float(self.cfg.limit_margin_m)
        if margin <= 1e-6:
            outward = (v > 1e-9 and q_rail >= hi) or (
                v < -1e-9 and q_rail <= lo
            )
            self.last_limit_saturated = bool(outward)
            return 0.0 if outward else 1.0

        if v > 1e-9:
            if q_rail >= hi:
                self.last_limit_saturated = True
                return 0.0
            if q_rail > hi - margin:
                u = float(np.clip((hi - q_rail) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return _smoothstep01(u)

        elif v < -1e-9:
            if q_rail <= lo:
                self.last_limit_saturated = True
                return 0.0
            if q_rail < lo + margin:
                u = float(np.clip((q_rail - lo) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return _smoothstep01(u)

        self.last_limit_saturated = False
        return 1.0

    def _macro_lpf(self, v: float, *, dt_s: float | None) -> float:
        """First-order LPF so the rail only takes the slow (macro) component."""
        tau = float(self.cfg.v_lpf_tau_s)
        if tau <= 1e-6 or dt_s is None or dt_s <= 0.0:
            self._v_lpf = float(v)
            self._v_lpf_initialized = True
            return float(v)
        if not self._v_lpf_initialized:
            self._v_lpf = float(v)
            self._v_lpf_initialized = True
            return float(v)
        alpha = float(dt_s) / (tau + float(dt_s))
        self._v_lpf = (1.0 - alpha) * self._v_lpf + alpha * float(v)
        return float(self._v_lpf)

    def _sigma_guard_velocity(
        self,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        v_primary: float,
    ) -> float:
        """Dead-zoned σ guardrail: engage only when σ is unhealthy.

        Hysteresis (enter/exit) prevents chatter.  Never fights a strong
        primary attractor (same anti-oppose rule as the old σ-escape).
        """
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        enter = float(self.cfg.sigma_guard_enter)
        exit_ = float(self.cfg.sigma_guard_exit)
        if self._guard_active:
            if sig >= exit_:
                self._guard_active = False
        else:
            if sig < enter:
                self._guard_active = True
        if not self._guard_active:
            return 0.0
        v_g = float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
        v_g = float(np.clip(v_g, -self.cfg.v_guard_max_m_s, self.cfg.v_guard_max_m_s))
        if v_g * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            return 0.0
        return v_g

    def _call_pose_attract(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        dt_s: float | None,
    ) -> tuple[float, float]:
        if self.y_rail_target_m is None:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_weight_raw = 0.0
            self.last_weight_capped = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        y = float(q[RAIL_INDEX])
        err = float(self.y_rail_target_m) - y  # +err → move rail toward target
        self.last_err_m = err
        e0 = float(self.cfg.pose_e0_m)
        e1 = max(float(self.cfg.pose_e1_m), e0 + 1e-6)
        span = e1 - e0
        w_pose = float(self.cfg.pose_w_max) * _smoothstep01((abs(err) - e0) / span)
        v_pose = float(
            np.clip(self.cfg.k_pose * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        # Inside settle dead-zone: primary is exactly zero (stop hunting).
        if abs(err) <= e0:
            v_pose = 0.0
        v_guard = self._sigma_guard_velocity(
            sigma_scale=sigma_scale,
            sigma_grad_rail=sigma_grad_rail,
            v_primary=v_pose,
        )
        v_total = v_pose + v_guard
        v_total = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        v_total = self._macro_lpf(v_total, dt_s=dt_s)
        lim = self._limit_saturation(y, v_total)
        self.last_limit_saturated = lim < 1e-6
        v_total *= lim
        # Guardrail alone still needs a floor weight so the QP can act when
        # the pose error is already inside the dead-zone but σ is bad.
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        w_guard = float(self.cfg.w_sigma_floor) * (1.0 - sig) if self._guard_active else 0.0
        w = (w_pose + w_guard) * lim
        self.last_weight = w
        self.last_weight_raw = float(w)
        self.last_weight_capped = float(w)
        return v_total, w

    def _call_reach(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        vel_ff: np.ndarray | None,
        dt_s: float | None,
    ) -> tuple[float, float]:
        if self.d_pref_m is None:
            self.capture_reference(q)
        err = self.extension(q) - float(self.d_pref_m)
        span = max(float(self.cfg.e1_m) - float(self.cfg.e0_m), 1e-6)
        # Reach term (unchanged Yamamoto-Yun coordination).
        w_reach = float(self.cfg.w_max) * _smoothstep01(
            (abs(err) - float(self.cfg.e0_m)) / span
        )
        v_reach = float(
            np.clip(self.cfg.k_ext * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        v_ff = (
            rail_vel_ff_from_reference(vel_ff, self.kin, q, k_ff=self.cfg.k_ff)
            if vel_ff is not None
            else 0.0
        )
        v_ff *= sig
        # σ-escape: extra rail velocity along the TCP-preserving σ-ascent
        # direction; kicks in even when |err| < e0 (dead zone) if σ drops.
        # In reach/scan mode this is a soft preference (not a hard guardrail),
        # but still anti-opposes the primary so it cannot hunt against FF.
        v_escape = float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
        v_primary = v_ff + v_reach
        if v_escape * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            v_escape = 0.0
        v_total = v_primary + v_escape
        v = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        v = self._macro_lpf(v, dt_s=dt_s)
        # Rail-limit fade (applies to the combined velocity).
        lim = self._limit_saturation(float(q[RAIL_INDEX]), v)
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        thr = float(self.cfg.v_ff_thr_m_s)
        span_ff = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * _smoothstep01((abs(v_ff) - thr) / span_ff) * sig
        # Weight: reach + scan feedforward + σ-baseline floor, then σ-boost.
        w = (w_reach + w_ff + float(self.cfg.w_sigma_floor) * (1.0 - sig)) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig)
        w *= sig_boost
        self.last_err_m = float(err)
        self.last_weight_raw = float(w)
        self.last_weight_capped = float(w)
        self.last_weight = w
        return v, w

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
        dt_s: float | None = None,
    ) -> tuple[float, float]:
        """Return ``(v_rail_des, w_ext)`` for the QP.

        Args
        ----
        q_rad : current command joint vector.
        sigma_scale : 1.0 when σ_min is healthy (``≥ sigma_ref``), 0.0 at
            deep singularity.  This is the σ-health scalar computed by the
            loop, NOT the raw σ_min.  σ-escape and w-boost fade in as this
            drops.
        sigma_grad_rail : ``d σ_min / d y_rail`` under TCP-preservation
            (:mod:`rm75_control.control.joint_admittance_8dof.solver.sigma_grad`).
            Sign tells us which rail direction escapes the singularity.
            Prefer sourcing this from a :class:`RailGoodness` implementation
            (default: :class:`SigmaMinGoodness`).
        dt_s : optional control period for the macro-micro LPF.
        """
        if not self.cfg.enabled:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_weight_raw = 0.0
            self.last_weight_capped = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        q = np.asarray(q_rad, dtype=float)
        if self.mode == "pose_attract":
            return self._call_pose_attract(
                q,
                sigma_scale=sigma_scale,
                sigma_grad_rail=sigma_grad_rail,
                dt_s=dt_s,
            )
        return self._call_reach(
            q,
            sigma_scale=sigma_scale,
            sigma_grad_rail=sigma_grad_rail,
            vel_ff=vel_ff,
            dt_s=dt_s,
        )
