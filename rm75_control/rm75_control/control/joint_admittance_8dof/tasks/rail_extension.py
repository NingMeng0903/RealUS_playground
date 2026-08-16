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
    v_max_m_s: float = 0.08
    # Fade the task to zero within this distance (m) of a rail travel limit
    # when the desired velocity points into the limit.
    limit_margin_m: float = 0.15
    # Hard pin / +q0 end-flip only this close to a soft stop (not the fade).
    pin_margin_m: float = 0.008
    # Stop driving +q0 this far from soft_max so escape cannot dump the carriage
    # onto the +stop (174417 sat at 774 mm for 52 s, Y error 340 mm).
    escape_leave_m: float = 0.04
    # Host soft travel (not URDF 0/0.8). Fade and end-flip use these.
    soft_min_m: float = 0.025
    soft_max_m: float = 0.78
    # Reach may oppose MotionReference FF, but only this much (m/s) so the
    # rail can still re-extend the elbow without re-triggering LW100 Er-01.
    v_reach_cap_m_s: float = 0.02
    # Dead-zone around d_center.  Inside the band v_reach is exactly 0 so
    # the carriage is free to ride the stroke; only stretch beyond the
    # band is corrected (and then drift makes the rail expensive in Y).
    d_band_m: float = 0.08
    # Bug 2: σ-escape.  When σ_min ↘ the rail should BOOST authority (not
    # cut it — the old ``w *= sigma_scale`` was backwards) and add a
    # non-reaching velocity component along the TCP-preserving σ-ascent
    # direction so the rail acts even inside the reach dead zone.
    #
    # Invariant kept by callers: ``w_max * (1 + k_sigma_boost) ≪ W_task``
    # (default 1.5 * 3 = 4.5 vs W_task = 100 in yaml → 22:1 ratio).  This is
    # what keeps the QP preference order  ``slack > rail > free-arm``
    # untouched even during σ dips (§3 test 1 & 2 in the plan pin this).
    k_sigma_boost: float = 2.0
    # k_esc [m/s per unit σ]: scales the σ-escape velocity component.
    # sigma_grad_rail has units 1/m, so k_esc·(1-sig)·grad has units of m/s.
    # Healthy path uses continuous soft bias (dbb/4d); latch uses same gain.
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
    # Faster LPF while escape is latched (commit without hunting).
    v_lpf_tau_escape_s: float = 0.08
    # Narrow latch: only deep σ (scale) or truly near joint soft limits.
    sigma_escape_enter: float = 0.55
    sigma_escape_exit: float = 0.80
    margin_escape_enter: float = 0.12
    margin_escape_exit: float = 0.25
    # Latch when raw arm σ falls faster than this (1/s); 0 disables.
    sigma_drop_rate: float = 0.0
    # Require sustained want_enter before latching (blocks turnaround flashes).
    escape_enter_dwell_s: float = 0.05
    # Extra weight multiplier while escape latched (still capped by w_ext_cap).
    k_escape_boost: float = 1.2
    # Floor |grad| when latched without a usable grad; 0 = never invent |grad|.
    escape_grad_floor: float = 0.0
    # Boost rail soft weight when any arm joint is near its soft limit [0,1].
    k_margin_boost: float = 4.0
    w_ext_cap: float = 24.0  # still ≪ W_task=100
    # When |err_band| exceeds err0, raise rail Cartesian reg and fade k_ff
    # so the arm takes Y instead of stretching the split further.
    d_star_err0_m: float = 0.01
    d_star_err1_m: float = 0.04
    d_star_w_mult: float = 6.0
    d_star_reg_mult: float = 20.0
    # Press-stall lateral escape: keep σ-escape / d* nudge alive when Z is
    # still demanding and the carriage still has travel.  Y error is not a
    # gate — mid-stroke stalls often track Y to < 5 mm.
    press_v_force_min_m_s: float = 0.02
    press_dz_max_m: float = 0.002
    press_y_err_m: float = 0.005
    press_stall_s: float = 0.5
    d_star_nudge_m: float = 0.01
    open_travel_min_m: float = 0.01
    # One-sided lateral escape.  ``minus`` drives −q0 until the minus pin.
    escape_sign_policy: str = "minus"


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
        self.last_limit_saturated: bool = False
        self.last_in_limit_band: bool = False
        self._guard_active: bool = False
        self._escape_active: bool = False
        self._escape_sign: float = 0.0
        self._escape_flipped_at_end: bool = False
        self._escape_enter_timer_s: float = 0.0
        self._sigma_raw_prev: float | None = None
        self._v_lpf: float = 0.0
        self._v_lpf_initialized: bool = False
        self.last_v_ff: float = 0.0
        self.last_v_escape: float = 0.0
        self.last_v_reach: float = 0.0
        self.last_rail_ff_m: float = float("nan")
        self.last_track_err_m: float = 0.0
        self.last_d_star_reg_scale: float = 1.0
        self.last_k_ff_scale: float = 1.0

    def set_mode(self, mode: RailExtMode) -> None:
        mode_s = str(mode).strip().lower()
        if mode_s not in ("reach", "pose_attract"):
            raise ValueError(f"unknown rail extension mode {mode!r}")
        if mode_s != self.mode:
            # Reset LPF on mode switch so a scan FF residue does not kick move.
            self._v_lpf = 0.0
            self._v_lpf_initialized = False
            self._guard_active = False
            self._escape_active = False
            self._escape_sign = 0.0
            self._escape_flipped_at_end = False
            self._sigma_raw_prev = None
        self.mode = mode_s  # type: ignore[assignment]

    def _soft_travel(self) -> tuple[float, float]:
        """Usable rail band: host soft limits ∩ URDF, never the raw URDF stop."""
        urdf_lo = float(self.kin.q_lower[RAIL_INDEX])
        urdf_hi = float(self.kin.q_upper[RAIL_INDEX])
        lo = max(urdf_lo, float(self.cfg.soft_min_m))
        hi = min(urdf_hi, float(self.cfg.soft_max_m))
        if not (lo < hi):
            return urdf_lo, urdf_hi
        return lo, hi

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Set / clear the move→D soft attractor target (metres)."""
        if y_rail_m is None:
            self.y_rail_target_m = None
            return
        lo, hi = self._soft_travel()
        self.y_rail_target_m = float(np.clip(float(y_rail_m), lo, hi))

    def set_d_pref(self, d_pref_m: float) -> None:
        """Update the preferred arm-extension offset (metres)."""
        self.d_pref_m = float(d_pref_m)

    def extension(self, q_rad: np.ndarray) -> float:
        """Arm Y-extension: base-frame TCP y minus rail position (m)."""
        q = np.asarray(q_rad, dtype=float)
        y_tcp = float(self.kin.fk_placement(q).translation[1])
        return y_tcp - float(q[RAIL_INDEX])

    def capture_reference(self, q_rad: np.ndarray) -> None:
        self.d_pref_m = self.extension(q_rad)

    def reset(self, q_rad: np.ndarray) -> None:
        self.capture_reference(q_rad)
        self.last_err_m = 0.0
        self.last_weight = 0.0
        self.last_limit_saturated = False
        self.last_in_limit_band = False
        self._guard_active = False
        self._escape_active = False
        self._escape_sign = 0.0
        self._escape_flipped_at_end = False
        self._sigma_raw_prev = None
        self._v_lpf = 0.0
        self._v_lpf_initialized = False

    def _rail_in_limit_band(self, q_rail: float) -> bool:
        """True while the carriage sits inside either soft-limit fade band."""
        margin = float(self.cfg.limit_margin_m)
        if margin <= 1.0e-9:
            return False
        lo, hi = self._soft_travel()
        return bool(q_rail <= lo + margin or q_rail >= hi - margin)

    def _rail_in_pin_band(self, q_rail: float) -> bool:
        """True only a few millimetres from a soft stop (not the 150 mm fade)."""
        margin = float(self.cfg.pin_margin_m)
        if margin <= 1.0e-9:
            return False
        lo, hi = self._soft_travel()
        return bool(q_rail <= lo + margin or q_rail >= hi - margin)

    def _open_side_sign(self, q_rail: float) -> float:
        """+1 toward soft_max, −1 toward soft_min (the side with more travel)."""
        lo, hi = self._soft_travel()
        return 1.0 if (hi - q_rail) >= (q_rail - lo) else -1.0

    def _open_side_travel_m(self, q_rail: float) -> float:
        lo, hi = self._soft_travel()
        return float(max(q_rail - lo, hi - q_rail))

    def _plus_travel_m(self, q_rail: float) -> float:
        _lo, hi = self._soft_travel()
        return float(hi - q_rail)

    def _leave_margin_m(self) -> float:
        return max(float(self.cfg.escape_leave_m), float(self.cfg.pin_margin_m))

    def _policy_escape_sign(self) -> float:
        raw = str(getattr(self.cfg, "escape_sign_policy", "minus")).strip().lower()
        if raw in ("minus", "-", "neg", "negative"):
            return -1.0
        if raw in ("plus", "+", "pos", "positive"):
            return 1.0
        raise ValueError(f"unknown rail_extension.escape_sign_policy: {raw!r}")

    def _in_leave_band(self, q_rail: float, sign: float = 0.0) -> bool:
        lo, hi = self._soft_travel()
        leave = self._leave_margin_m()
        s = float(sign)
        if abs(s) < 1.0e-12:
            s = self._policy_escape_sign()
        if s > 0.0:
            return bool(q_rail >= hi - leave)
        if s < 0.0:
            return bool(q_rail <= lo + leave)
        return False

    def _in_plus_leave(self, q_rail: float) -> bool:
        return self._in_leave_band(q_rail, +1.0)

    def _preferred_escape_sign(self, q_rail: float, *, backoff: bool = False) -> float:
        """Policy-side escape; 0 in that leave band; reverse on the policy pin."""
        sign = self._policy_escape_sign()
        lo, hi = self._soft_travel()
        pin = float(self.cfg.pin_margin_m)
        leave = self._leave_margin_m()
        if sign < 0.0:
            if pin > 1.0e-9 and q_rail <= lo + pin:
                return 1.0
            if q_rail <= lo + leave:
                return 1.0 if backoff else 0.0
            return -1.0
        if pin > 1.0e-9 and q_rail >= hi - pin:
            return -1.0
        if q_rail >= hi - leave:
            return -1.0 if backoff else 0.0
        return 1.0

    def _rail_has_open_travel(self, q_rail: float) -> bool:
        return self._open_side_travel_m(q_rail) > float(self.cfg.open_travel_min_m)

    def _rail_end_blocks(self, q_rail: float, sign: float) -> bool:
        """True if moving with ``sign`` (+1/−1) points into the pin band."""
        margin = float(self.cfg.pin_margin_m)
        lo, hi = self._soft_travel()
        if margin <= 1e-9:
            return False
        if sign > 0.0 and q_rail >= hi - margin:
            return True
        if sign < 0.0 and q_rail <= lo + margin:
            return True
        return False

    def _maybe_flip_escape_at_rail_end(self, q_rail: float) -> None:
        """If latched into a dead end, flip sign once (still monotonic)."""
        if not self._escape_active or abs(self._escape_sign) < 1e-9:
            return
        if not self._rail_end_blocks(q_rail, self._escape_sign):
            return
        alt = -self._escape_sign
        if self._rail_end_blocks(q_rail, alt):
            # Both ends blocked — drop escape; L0 box + softσ handle the rest.
            self._escape_active = False
            self._escape_sign = 0.0
            return
        if not self._escape_flipped_at_end:
            self._escape_sign = alt
            self._escape_flipped_at_end = True

    def _clear_escape_latch(self) -> None:
        self._escape_active = False
        self._escape_sign = 0.0
        self._escape_flipped_at_end = False
        self._escape_enter_timer_s = 0.0

    def _escape_latched(
        self,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        joint_margin_frac: float,
        sigma_raw: float | None,
        dt_s: float | None,
        q_rail: float,
        trajectory_owns: bool = False,
    ) -> float:
        """Narrow hysteresis latch: deep σ ∪ true near-limit (optional dσ/dt).

        While the MotionReference owns the rail (``|v_ff|>thr``), never enter or
        keep the latch — sticky escape fighting the path caused scan stutter and
        LW100 Er-01 (overspeed) on run_20260813_151334.
        """
        if trajectory_owns:
            self._clear_escape_latch()
            if sigma_raw is not None:
                self._sigma_raw_prev = float(sigma_raw)
            return 0.0

        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        mfrac = float(np.clip(joint_margin_frac, 0.0, 1.0))
        enter = float(self.cfg.sigma_escape_enter)
        exit_ = max(float(self.cfg.sigma_escape_exit), enter)
        m_enter = float(self.cfg.margin_escape_enter)
        m_exit = max(float(self.cfg.margin_escape_exit), m_enter)

        dropping = False
        if (
            sigma_raw is not None
            and dt_s is not None
            and float(dt_s) > 1e-9
            and float(self.cfg.sigma_drop_rate) > 0.0
            and self._sigma_raw_prev is not None
        ):
            dsigma = (float(sigma_raw) - float(self._sigma_raw_prev)) / float(dt_s)
            dropping = dsigma < -float(self.cfg.sigma_drop_rate)
        if sigma_raw is not None:
            self._sigma_raw_prev = float(sigma_raw)

        want_enter = (sig < enter) or (mfrac < m_enter) or dropping
        healthy_exit = (sig >= exit_) and (mfrac >= m_exit)
        dt = float(dt_s) if dt_s is not None and float(dt_s) > 0.0 else 0.0
        dwell = max(float(self.cfg.escape_enter_dwell_s), 0.0)

        if self._escape_active:
            if healthy_exit:
                self._clear_escape_latch()
            else:
                pref = self._preferred_escape_sign(q_rail)
                if abs(pref) < 1.0e-12:
                    self._clear_escape_latch()
                elif pref * self._escape_sign < 0.0:
                    self._escape_sign = pref
        else:
            if want_enter:
                self._escape_enter_timer_s += dt
                if self._escape_enter_timer_s + 1.0e-12 >= dwell:
                    self._escape_active = True
                    self._escape_flipped_at_end = False
                    self._escape_enter_timer_s = 0.0
                    self._escape_sign = self._preferred_escape_sign(q_rail)
                    if abs(self._escape_sign) < 1.0e-12:
                        self._clear_escape_latch()
                        if sigma_raw is not None:
                            self._sigma_raw_prev = float(sigma_raw)
                        return 0.0
            else:
                self._escape_enter_timer_s = 0.0

        if not self._escape_active:
            return 0.0
        self._maybe_flip_escape_at_rail_end(q_rail)
        if not self._escape_active:
            return 0.0
        floor = float(self.cfg.escape_grad_floor)
        mag = abs(float(sigma_grad_rail))
        if floor > 0.0:
            mag = max(mag, floor)
        if mag < 1.0e-12:
            return 0.0
        return self._escape_sign * mag

    def _limit_saturation(self, q_rail: float, v: float) -> float:
        """Return 0..1 scale; C¹ smoothstep fade before a directional hard stop.

        Fades only when moving *into* a limit so reversing away from a pinned
        rail recovers authority immediately.  At the physical stop the scale is
        0; with a wide enough ``limit_margin_m`` the fade completes before pin.
        """
        margin = float(self.cfg.limit_margin_m)
        if margin <= 1e-6:
            self.last_limit_saturated = False
            return 1.0

        lo, hi = self._soft_travel()

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

    def _macro_lpf(
        self, v: float, *, dt_s: float | None, tau_s: float | None = None
    ) -> float:
        """First-order LPF so the rail only takes the slow (macro) component."""
        tau = float(self.cfg.v_lpf_tau_s if tau_s is None else tau_s)
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
        return v_total, w

    def _call_reach(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        vel_ff: np.ndarray | None,
        dt_s: float | None,
        joint_margin_frac: float = 1.0,
        sigma_raw: float | None = None,
        y_tcp_d: float | None = None,
        press_stalled: bool = False,
        tool_y_err_m: float = 0.0,
        stroke_limiters: bool = True,
    ) -> tuple[float, float]:
        if self.d_pref_m is None:
            self.capture_reference(q)
        d_star = float(self.d_pref_m)
        y = float(q[RAIL_INDEX])
        if y_tcp_d is not None and np.isfinite(float(y_tcp_d)):
            y_des = float(y_tcp_d)
        else:
            y_des = float(self.kin.fk_placement(q).translation[1])
        rail_ff = y_des - d_star
        err_raw = rail_ff - y
        band = max(float(getattr(self.cfg, "d_band_m", 0.0)), 0.0)
        if stroke_limiters:
            band = 0.0
        err = float(err_raw - np.clip(err_raw, -band, band))
        self.last_rail_ff_m = float(rail_ff)
        self.last_track_err_m = float(err_raw)
        span = max(float(self.cfg.e1_m) - float(self.cfg.e0_m), 1e-6)
        w_reach = float(self.cfg.w_max) * _smoothstep01(
            (abs(err) - float(self.cfg.e0_m)) / span
        )
        v_reach = float(
            np.clip(self.cfg.k_ext * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        err_abs = abs(err)
        e0 = max(float(self.cfg.d_star_err0_m), 0.0)
        e1 = max(float(self.cfg.d_star_err1_m), e0 + 1.0e-6)
        drift = _smoothstep01((err_abs - e0) / (e1 - e0)) if e0 > 0.0 else 0.0
        self.last_d_star_reg_scale = 1.0 + drift * max(
            float(self.cfg.d_star_reg_mult) - 1.0, 0.0
        )
        self.last_k_ff_scale = 1.0 - drift
        v_ff = (
            rail_vel_ff_from_reference(
                vel_ff, self.kin, q, k_ff=self.cfg.k_ff * self.last_k_ff_scale
            )
            if vel_ff is not None
            else 0.0
        )
        # Do not attenuate FF by σ: healthy scan must keep rail tracking any
        # MotionReference (dbb/4d). Soft σ bias is a separate term below.
        thr = float(self.cfg.v_ff_thr_m_s)
        ff_owns = abs(v_ff) > thr
        # Trajectory owns rail direction: clear sticky latch (not merely mute v).
        grad_latched = self._escape_latched(
            sigma_scale=sig,
            sigma_grad_rail=sigma_grad_rail,
            joint_margin_frac=joint_margin_frac,
            sigma_raw=sigma_raw,
            dt_s=dt_s,
            q_rail=y,
            trajectory_owns=ff_owns,
        )
        cap = max(float(self.cfg.v_reach_cap_m_s), 0.0)
        if cap > 0.0:
            cap = cap + drift * 0.06
            v_reach = float(np.clip(v_reach, -cap, cap))
        # Demoted: healthy σ (raw ≥ 0.08) never lets escape drive the rail
        # unless a press stall still needs a lateral Y offset.
        healthy_sigma = sigma_raw is not None and float(sigma_raw) >= 0.08
        use_limiters = bool(stroke_limiters)
        in_band = self._rail_in_limit_band(y) if use_limiters else False
        self.last_in_limit_band = bool(in_band)
        y_thr = max(float(self.cfg.press_y_err_m), 0.0)
        policy_sign = self._policy_escape_sign()
        backoff = bool(
            use_limiters
            and self._in_leave_band(y, policy_sign)
            and abs(float(tool_y_err_m)) >= y_thr
        )
        allow_press_escape = bool(
            (press_stalled or backoff) and self._rail_has_open_travel(y)
        )
        # Inside the fade band escape only fights the wall — unless Z is
        # still demanding and the open side still has travel.
        if in_band and not allow_press_escape:
            self._clear_escape_latch()
            v_escape = 0.0
        elif healthy_sigma and not allow_press_escape:
            self._escape_active = False
            v_escape = 0.0
        elif self._escape_active:
            v_escape = 0.25 * float(self.cfg.k_esc) * float(grad_latched)
            if (
                not allow_press_escape
                and abs(v_escape) > 1e-9
                and v_reach * v_escape < 0.0
            ):
                v_escape = 0.0
        else:
            v_escape = (
                0.25 * float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
            )
            if allow_press_escape:
                pref = self._preferred_escape_sign(y, backoff=backoff)
                v_escape = (
                    0.25
                    * float(self.cfg.k_esc)
                    * pref
                    * max(abs(float(sigma_grad_rail)), 1.0)
                )
                if abs(v_escape) > 1.0e-12:
                    self._escape_active = True
                    self._escape_sign = pref
            elif ff_owns and v_escape * v_ff < 0.0:
                v_escape = 0.0
            else:
                v_primary_ff = v_ff + v_reach
                if v_escape * v_primary_ff < 0.0 and abs(v_primary_ff) > 1.0e-4:
                    v_escape = 0.0
        v_primary = v_ff + v_reach
        pref_now = self._preferred_escape_sign(y, backoff=backoff)
        if use_limiters:
            # Park primary at either leave band so a planned +Y stroke still
            # stops short of +soft_max, while minus-policy escape parks at −.
            in_plus = self._in_plus_leave(y)
            in_policy = self._in_leave_band(y, policy_sign)
            if in_plus and (not allow_press_escape or v_primary > 0.0):
                v_primary = 0.0
            if in_policy and (not allow_press_escape or v_primary * policy_sign > 0.0):
                v_primary = 0.0
        if allow_press_escape:
            esc_sign = (
                float(self._escape_sign)
                if self._escape_active and abs(self._escape_sign) > 1.0e-12
                else pref_now
            )
            if esc_sign != 0.0 and v_primary * esc_sign < 0.0:
                v_primary = 0.0
        v_total = v_primary + v_escape
        v = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        tau = (
            float(self.cfg.v_lpf_tau_escape_s)
            if self._escape_active
            else float(self.cfg.v_lpf_tau_s)
        )
        v = self._macro_lpf(v, dt_s=dt_s, tau_s=tau)
        if use_limiters:
            lim = self._limit_saturation(y, v)
        else:
            lim = 1.0
            self.last_limit_saturated = False
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        span_ff = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * _smoothstep01((abs(v_ff) - thr) / span_ff)
        w_sigma = float(self.cfg.w_sigma_floor) * (1.0 - sig)
        w = (w_reach + w_ff + w_sigma) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig)
        w *= sig_boost
        mfrac = float(np.clip(joint_margin_frac, 0.0, 1.0))
        w *= 1.0 + float(self.cfg.k_margin_boost) * (1.0 - mfrac)
        if self._escape_active:
            w *= float(self.cfg.k_escape_boost)
        w *= 1.0 + drift * max(float(self.cfg.d_star_w_mult) - 1.0, 0.0)
        w = min(w, float(self.cfg.w_ext_cap))
        self.last_err_m = float(err)
        self.last_weight = w
        self.last_v_ff = float(v_ff)
        self.last_v_escape = float(v_escape)
        self.last_v_reach = float(v_reach)
        return v, w

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
        dt_s: float | None = None,
        joint_margin_frac: float = 1.0,
        sigma_raw: float | None = None,
        y_tcp_d: float | None = None,
        press_stalled: bool = False,
        tool_y_err_m: float = 0.0,
        stroke_limiters: bool = True,
    ) -> tuple[float, float]:
        """Return ``(v_rail_des, w_ext)`` for the QP."""
        if not self.cfg.enabled:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            self.last_in_limit_band = False
            self.last_d_star_reg_scale = 1.0
            self.last_k_ff_scale = 1.0
            return 0.0, 0.0
        q = np.asarray(q_rad, dtype=float)
        self.last_in_limit_band = self._rail_in_limit_band(float(q[RAIL_INDEX]))
        if self.mode == "pose_attract":
            self.last_d_star_reg_scale = 1.0
            self.last_k_ff_scale = 1.0
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
            y_tcp_d=y_tcp_d,
            joint_margin_frac=joint_margin_frac,
            sigma_raw=sigma_raw,
            press_stalled=press_stalled,
            tool_y_err_m=tool_y_err_m,
            stroke_limiters=stroke_limiters,
        )
