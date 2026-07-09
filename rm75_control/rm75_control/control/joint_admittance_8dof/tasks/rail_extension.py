"""Preferred arm-extension rail task: proactive base-arm coordination.

Mobile-manipulator coordination (Yamamoto & Yun 1994, IEEE TAC): the arm
serves the Cartesian task, while the base (here: the Y rail) tracks a
*preferred arm extension* so the arm stays near a well-conditioned posture.

The task is a scalar desired rail velocity plus a *continuously scheduled*
weight (Chan & Dubey 1995 weighted-least-norm style):

    e      = (y_tcp - y_rail) - d_pref          # arm extension error (m)
    v_rail = clip(k_ext * e, +-v_max)
    w_ext  = w_max * smoothstep((|e| - e0) / (e1 - e0))

Inside the dead zone (|e| < e0) the weight is exactly 0: the rail does not
wander when the scan fits the arm's reach.

**Rail travel limit:** no hard weight switching.  Over the last
``limit_margin_m`` before a physical stop (motion direction only), authority
fades with a C¹ smoothstep so the QP can hand Y velocity to the arm before
the rail pins.  Hardware logs with a 2 cm linear clip showed w_ext collapsing
in ~0.4 s at 5 cm/s scan — arm jerk and σ dips.  Pinned-at-limit still yields
zero desired rail velocity, but the fade window should finish first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, RAIL_INDEX


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
    limit_margin_m: float = 0.08
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
    k_esc: float = 0.5
    # Baseline w that lets the rail act even when the reach error is inside
    # the dead zone (|e| < e0), provided σ is depressed.  Fades with σ.
    w_sigma_floor: float = 1.0


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class RailExtensionTask:
    """Callable: q (rad/m) -> (v_rail_des m/s, w_ext) for the WBC QP."""

    def __init__(self, kin: RobotKinematics, cfg: RailExtensionConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or RailExtensionConfig()
        self.d_pref_m: float | None = None
        self.last_err_m: float = 0.0
        self.last_weight: float = 0.0
        self.last_limit_saturated: bool = False

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

        lo = float(self.kin.q_lower[RAIL_INDEX])
        hi = float(self.kin.q_upper[RAIL_INDEX])

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

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
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
        """
        if not self.cfg.enabled:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        if self.d_pref_m is None:
            self.capture_reference(q_rad)
        q = np.asarray(q_rad, dtype=float)
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
        v_escape = float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
        v_primary = v_ff + v_reach
        if v_escape * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            v_escape = 0.0
        v_total = v_primary + v_escape
        v = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        # Rail-limit fade (applies to the combined velocity).
        lim = self._limit_saturation(float(q[RAIL_INDEX]), v)
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        thr = float(self.cfg.v_ff_thr_m_s)
        span = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * _smoothstep01((abs(v_ff) - thr) / span) * sig
        # Weight: reach + scan feedforward + σ-baseline floor, then σ-boost.
        w = (w_reach + w_ff + float(self.cfg.w_sigma_floor) * (1.0 - sig)) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig)
        w *= sig_boost
        self.last_err_m = float(err)
        self.last_weight = w
        return v, w
