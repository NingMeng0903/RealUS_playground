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
RAIL_TASK_WEIGHT_HARD_MAX = 4.5


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
    # Healthy relocation cap.  The QP applies the final tighter deep-escape
    # boundary when the latched escape is active/signed/stopped.
    weight_hard_max: float = 4.5
    # The preferred rail task is subordinate to the effective Cartesian task.
    task_weight_max_frac: float = 0.80
    v_max_m_s: float = 0.08
    # A singular escape is deliberately slower and bounded in displacement;
    # after this macro move the rail is stopped and arm nullspace recovery owns
    # the remaining escape rather than sweeping hundreds of millimetres.
    escape_v_min_m_s: float = 0.010
    escape_v_max_m_s: float = 0.020
    escape_max_travel_m: float = 0.080
    # Fade the task to zero within this distance (m) of a rail travel limit
    # when the desired velocity points into the limit.
    limit_margin_m: float = 0.08
    # Bug 2: σ-escape.  When σ_min ↘ the rail should BOOST authority (not
    # cut it — the old ``w *= sigma_scale`` was backwards) and add a
    # non-reaching velocity component along the TCP-preserving σ-ascent
    # direction so the rail acts even inside the reach dead zone.
    #
    # Raw sigma boosting is exposed in telemetry; the healthy task cap above is
    # applied here, while the QP enforces the final deep-escape hierarchy.
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
        # Escape direction latch: once an episode starts, hold one rail sign
        # until σ clears the explicit exit threshold.  This prevents a noisy
        # finite-difference gradient from hunting left/right.
        self._sigma_escape_latched: bool = False
        self._sigma_escape_sign: float = 0.0
        self._escape_start_rail_m: float | None = None
        self._escape_travel_m: float = 0.0
        self._escape_stopped: bool = False

    @property
    def escape_active(self) -> bool:
        return bool(self._sigma_escape_latched)

    @property
    def escape_sign(self) -> float:
        return float(self._sigma_escape_sign)

    @property
    def escape_travel_m(self) -> float:
        return float(self._escape_travel_m)

    @property
    def escape_stopped(self) -> bool:
        return bool(self._escape_stopped)

    @property
    def escape_position_limit_m(self) -> float | None:
        """Host-position boundary for the current episode, if latched."""
        if (
            not self._sigma_escape_latched
            or self._escape_start_rail_m is None
            or abs(self._sigma_escape_sign) < 0.5
        ):
            return None
        span = max(float(self.cfg.escape_max_travel_m), 0.0)
        if span <= 0.0:
            return None
        start = float(self._escape_start_rail_m)
        sign = float(self._sigma_escape_sign)
        # A controller may attach while the encoder is just outside the
        # canonical soft band.  If the latched gradient points farther out,
        # stop at the current position; clipping the episode endpoint to the
        # opposite soft boundary would teleport the host target by the whole
        # gap in one tick.
        if sign > 0.0 and start >= float(self.cfg.soft_max_m):
            return start
        if sign < 0.0 and start <= float(self.cfg.soft_min_m):
            return start
        raw = start + sign * span
        return float(np.clip(raw, self.cfg.soft_min_m, self.cfg.soft_max_m))

    def stop_escape(self) -> None:
        """Stop rail motion but retain the episode sign until sigma exits."""
        if self._sigma_escape_latched:
            self._escape_stopped = True
            self._v_lpf = 0.0
            self._v_lpf_initialized = False

    def observe_escape_position(self, q_rail_m: float) -> None:
        """Update episode travel from the final host command for this tick."""
        self._update_escape_progress(float(q_rail_m))

    def reset_escape(self) -> None:
        """Clear only the singular-escape episode, preserving reach targets."""
        self._sigma_escape_latched = False
        self._sigma_escape_sign = 0.0
        self._escape_start_rail_m = None
        self._escape_travel_m = 0.0
        self._escape_stopped = False
        self._guard_active = False
        self._v_lpf = 0.0
        self._v_lpf_initialized = False

    def set_mode(self, mode: RailExtMode) -> None:
        mode_s = str(mode).strip().lower()
        if mode_s not in ("reach", "pose_attract"):
            raise ValueError(f"unknown rail extension mode {mode!r}")
        if mode_s != self.mode:
            # Reset LPF on mode switch so a scan FF residue does not kick move.
            self._v_lpf = 0.0
            self._v_lpf_initialized = False
            self._guard_active = False
            # A direction belongs to exactly one mode/episode.  Carrying the
            # reach sign into pose_attract made the next phase commit to a
            # stale side before its gradient had even been refreshed.
            self.reset_escape()
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
        self.last_err_m = 0.0
        self.last_weight = 0.0
        self.last_weight_raw = 0.0
        self.last_weight_capped = 0.0
        self.last_limit_saturated = False
        self._guard_active = False
        self._v_lpf = 0.0
        self._v_lpf_initialized = False
        self.reset_escape()

    def _latched_sigma_grad(
        self,
        sigma_grad_rail: float,
        *,
        sigma_min: float,
        sigma_escape_enter: float,
        sigma_escape_exit: float,
        q_rail_m: float | None = None,
    ) -> float:
        """Return a gradient with one sign per escape episode.

        Enter is strict (σ < 0.10 by default), exit is inclusive (σ ≥ 0.12),
        so the interval between them is genuine hysteresis.  A zero gradient
        does not create an episode; the first finite non-zero sample does.
        """
        g = float(sigma_grad_rail)
        s = float(sigma_min)
        enter = float(sigma_escape_enter)
        exit_ = max(float(sigma_escape_exit), enter)
        if not np.isfinite(s) or enter <= 0.0:
            return g
        if self._sigma_escape_latched:
            if s >= exit_:
                self.reset_escape()
                return g
            self._update_escape_progress(q_rail_m)
            sign = float(self._sigma_escape_sign)
            if abs(sign) < 1e-12:
                sign = float(np.sign(g))
                if abs(sign) > 1e-12:
                    self._sigma_escape_sign = sign
            return abs(g) * sign if abs(sign) > 1e-12 else 0.0
        if s < enter:
            sign = float(np.sign(g))
            if abs(sign) > 1e-12:
                self._sigma_escape_latched = True
                self._sigma_escape_sign = sign
                self._escape_start_rail_m = (
                    None if q_rail_m is None else float(q_rail_m)
                )
                self._escape_travel_m = 0.0
                self._escape_stopped = False
                # Never carry a pre-episode LPF sample across the sign latch:
                # the first escape tick must not emit the previous direction.
                self._v_lpf = 0.0
                self._v_lpf_initialized = False
                return abs(g) * sign
        return g

    def _update_escape_progress(self, q_rail_m: float | None) -> None:
        if not self._sigma_escape_latched or q_rail_m is None:
            return
        q_rail = float(q_rail_m)
        if self._escape_start_rail_m is None:
            self._escape_start_rail_m = q_rail
        self._escape_travel_m = abs(q_rail - float(self._escape_start_rail_m))
        max_travel = max(float(self.cfg.escape_max_travel_m), 0.0)
        if max_travel > 0.0 and self._escape_travel_m >= max_travel:
            self._escape_stopped = True

        sign = float(self._sigma_escape_sign)
        if sign > 0.0 and q_rail >= float(self.cfg.soft_max_m):
            self._escape_stopped = True
        elif sign < 0.0 and q_rail <= float(self.cfg.soft_min_m):
            self._escape_stopped = True

    def _project_escape_direction(self, v: float) -> float:
        """Keep rail velocity on the latched escape sign for this episode."""
        if not self._sigma_escape_latched:
            return float(v)
        sign = float(self._sigma_escape_sign)
        if abs(sign) <= 1e-12:
            return 0.0
        return sign * max(float(v) * sign, 0.0)

    def _capped_weight(self, w: float) -> float:
        cap = min(RAIL_TASK_WEIGHT_HARD_MAX, max(float(self.cfg.weight_hard_max), 0.0))
        return float(np.clip(float(w), 0.0, cap))

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

    def _sigma_guard_hold(self, sigma_scale: float) -> bool:
        """Update and return the σ-guard latch (enter/exit hysteresis).

        True means "σ is unhealthy enough that escape outranks the primary".
        Shared by both modes so ``reach`` and ``pose_attract`` switch over at
        the same σ instead of each inventing their own rule.
        """
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        if self._guard_active:
            if sig >= float(self.cfg.sigma_guard_exit):
                self._guard_active = False
        elif sig < float(self.cfg.sigma_guard_enter):
            self._guard_active = True
        return self._guard_active

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
        if not self._sigma_guard_hold(sig):
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
        sigma_escape_scale: float,
        sigma_grad_rail: float,
        dt_s: float | None,
        sigma_min: float,
        sigma_escape_enter: float,
        sigma_escape_exit: float,
    ) -> tuple[float, float]:
        sigma_scale = sigma_escape_scale
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
        grad_used = self._latched_sigma_grad(
            sigma_grad_rail,
            sigma_min=sigma_min,
            sigma_escape_enter=sigma_escape_enter,
            sigma_escape_exit=sigma_escape_exit,
            q_rail_m=y,
        )
        if self._escape_stopped:
            self.last_weight = 0.0
            self.last_weight_raw = 0.0
            self.last_weight_capped = 0.0
            self.last_limit_saturated = True
            return 0.0, 0.0
        if self._sigma_escape_latched:
            # Once an episode commits a rail sign, an opposing pose attractor
            # may not reverse it; it is clipped to zero instead.
            v_pose = self._project_escape_direction(v_pose)
        if self._sigma_escape_latched:
            # The absolute escape latch is the guard.  The legacy normalized
            # guard threshold (0.45 -> raw sigma 0.045) started far too late
            # and could report escape_active while emitting exactly zero.
            self._guard_active = True
            v_guard = float(self.cfg.k_esc) * max(1.0 - sigma_scale, 0.0) * grad_used
            v_guard = float(
                np.clip(
                    v_guard,
                    -float(self.cfg.escape_v_max_m_s),
                    float(self.cfg.escape_v_max_m_s),
                )
            )
        else:
            v_guard = self._sigma_guard_velocity(
                sigma_scale=sigma_scale,
                sigma_grad_rail=grad_used,
                v_primary=v_pose,
            )
        v_total = v_pose + v_guard
        v_total = self._project_escape_direction(v_total)
        v_cap = (
            float(self.cfg.escape_v_max_m_s)
            if self._sigma_escape_latched
            else float(self.cfg.v_max_m_s)
        )
        v_total = float(np.clip(v_total, -v_cap, v_cap))
        v_total = self._macro_lpf(v_total, dt_s=dt_s)
        v_total = self._project_escape_direction(v_total)
        lim = self._limit_saturation(y, v_total)
        self.last_limit_saturated = lim < 1e-6
        v_total *= lim
        if self._sigma_escape_latched and lim < 1e-6:
            self._escape_stopped = True
            v_total = 0.0
        # Guardrail alone still needs a floor weight so the QP can act when
        # the pose error is already inside the dead-zone but σ is bad.
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        w_guard = float(self.cfg.w_sigma_floor) * (1.0 - sig) if self._guard_active else 0.0
        w = (w_pose + w_guard) * lim
        self.last_weight = w
        self.last_weight_raw = float(w)
        self.last_weight_capped = self._capped_weight(w)
        self.last_weight = self.last_weight_capped
        return v_total, self.last_weight_capped

    def _call_reach(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_escape_scale: float,
        sigma_grad_rail: float,
        vel_ff: np.ndarray | None,
        dt_s: float | None,
        sigma_min: float,
        sigma_escape_enter: float,
        sigma_escape_exit: float,
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
        # Mirror pose_attract: inside the dead zone the primary is exactly
        # zero.  A residual v_reach there is never large enough to move the
        # rail, but it is large enough to veto the σ-escape below.
        if abs(err) <= float(self.cfg.e0_m):
            v_reach = 0.0
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        sig_esc = float(np.clip(sigma_escape_scale, 0.0, 1.0))
        v_ff = (
            rail_vel_ff_from_reference(vel_ff, self.kin, q, k_ff=self.cfg.k_ff)
            if vel_ff is not None
            else 0.0
        )
        v_ff *= sig
        # σ-escape: extra rail velocity along the TCP-preserving σ-ascent
        # direction; kicks in even when |err| < e0 (dead zone) if σ drops.
        grad_used = self._latched_sigma_grad(
            sigma_grad_rail,
            sigma_min=sigma_min,
            sigma_escape_enter=sigma_escape_enter,
            sigma_escape_exit=sigma_escape_exit,
            q_rail_m=float(q[RAIL_INDEX]),
        )
        if self._escape_stopped:
            self.last_err_m = float(err)
            self.last_weight = 0.0
            self.last_weight_raw = 0.0
            self.last_weight_capped = 0.0
            self.last_limit_saturated = True
            return 0.0, 0.0
        v_escape = float(self.cfg.k_esc) * (1.0 - sig_esc) * float(grad_used)
        if self._sigma_escape_latched:
            v_escape = float(
                np.clip(
                    v_escape,
                    -float(self.cfg.escape_v_max_m_s),
                    float(self.cfg.escape_v_max_m_s),
                )
            )
        v_primary = v_ff + v_reach
        # Anti-oppose, σ-gated exactly like the pose_attract guardrail: while
        # σ is healthy the escape is a soft preference and must not hunt
        # against reach/FF, but once σ drops past the guard threshold escape
        # is the whole point and has to be allowed to win.  This used to be
        # unconditional, so at hybrid@D — where the force axis keeps a small
        # reach error alive — the escape was vetoed on every tick no matter
        # how far σ fell.
        if self._sigma_guard_hold(sig_esc):
            pass
        elif (
            not self._sigma_escape_latched
            and v_escape * v_primary < 0.0
            and abs(v_primary) > 1.0e-4
        ):
            v_escape = 0.0
        if self._sigma_escape_latched:
            # During a committed escape episode, discard the opposing part of
            # the reach/feedforward primary rather than allowing it to reverse
            # the latched gradient direction.
            v_primary = self._project_escape_direction(v_primary)
        v_total = v_primary + v_escape
        v_total = self._project_escape_direction(v_total)
        v_cap = (
            float(self.cfg.escape_v_max_m_s)
            if self._sigma_escape_latched
            else float(self.cfg.v_max_m_s)
        )
        v = float(np.clip(v_total, -v_cap, v_cap))
        v = self._macro_lpf(v, dt_s=dt_s)
        v = self._project_escape_direction(v)
        # Rail-limit fade (applies to the combined velocity).
        lim = self._limit_saturation(float(q[RAIL_INDEX]), v)
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        if self._sigma_escape_latched and lim < 1e-6:
            self._escape_stopped = True
            v = 0.0
        thr = float(self.cfg.v_ff_thr_m_s)
        span_ff = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * _smoothstep01((abs(v_ff) - thr) / span_ff) * sig
        # Weight: reach + scan feedforward + σ-baseline floor, then σ-boost.
        w = (w_reach + w_ff + float(self.cfg.w_sigma_floor) * (1.0 - sig_esc)) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig_esc)
        w_raw = w * sig_boost
        w = self._capped_weight(w_raw)
        self.last_err_m = float(err)
        self.last_weight_raw = float(w_raw)
        self.last_weight_capped = float(w)
        self.last_weight = w
        return v, w

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_escape_scale: float | None = None,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
        dt_s: float | None = None,
        sigma_min: float = float("nan"),
        sigma_escape_enter: float = 0.10,
        sigma_escape_exit: float = 0.12,
    ) -> tuple[float, float]:
        """Return ``(v_rail_des, w_ext)`` for the QP.

        Args
        ----
        q_rad : current command joint vector.
        sigma_scale : 1.0 when σ_min is healthy (``≥ sigma_ref``), 0.0 at
            deep singularity.  This is the σ-health scalar computed by the
            loop, NOT the raw σ_min.  Fades the *scan feedforward*: as the arm
            gets ill-conditioned the rail stops chasing the scan reference.
        sigma_escape_scale : the same kind of scalar but measured against the
            explicit absolute escape-enter threshold (normally 0.10); drives
            σ-escape velocity, the ``w_sigma_floor`` baseline, the w-boost and
            the normalized guard latch.  Avoidance has to start before the
            loop's twist brake does — the rail is accel-limited and cannot
            respond within the tick the brake fires.  Defaults to
            ``sigma_scale``.
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
        sig_esc = (
            float(sigma_scale)
            if sigma_escape_scale is None
            else float(sigma_escape_scale)
        )
        if self.mode == "pose_attract":
            return self._call_pose_attract(
                q,
                sigma_escape_scale=sig_esc,
                sigma_grad_rail=sigma_grad_rail,
                dt_s=dt_s,
                sigma_min=float(sigma_min),
                sigma_escape_enter=float(sigma_escape_enter),
                sigma_escape_exit=float(sigma_escape_exit),
            )
        return self._call_reach(
            q,
            sigma_scale=sigma_scale,
            sigma_escape_scale=sig_esc,
            sigma_grad_rail=sigma_grad_rail,
            vel_ff=vel_ff,
            dt_s=dt_s,
            sigma_min=float(sigma_min),
            sigma_escape_enter=float(sigma_escape_enter),
            sigma_escape_exit=float(sigma_escape_exit),
        )
