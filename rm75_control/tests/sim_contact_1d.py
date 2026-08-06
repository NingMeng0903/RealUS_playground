"""1-D delayed contact plant for bounce / barrier verification.

Plant:
    f = max(Ke · x, 0)           # unilateral spring, x>0 into surface
    v_tcp(t) = v_cmd(t − T_dead) # pure transport delay
Admittance (implicit Euler, same form as controller):
    (M/dt + D) v+ = M/dt · v + D · v_r + (f_des − f)

Baseline (no barrier, D=25, free-flight press to v_cap) reproduces the
2.6–3.0 Hz bounce limit cycle seen in run_20260804_152810.  With the
stiffness-scheduled barrier the cycle collapses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)


@dataclass
class SimConfig:
    dt: float = 0.005
    t_end_s: float = 4.0
    t_dead_s: float = 0.040
    ke: float = 9000.0
    f_des: float = 2.0
    mass: float = 1.0
    damping: float = 25.0
    v_cap: float = 0.08
    x0: float = -0.005  # start 5 mm above surface
    use_barrier: bool = False
    # Ideal 1-D plant: two-sided still kills bounce.  Real robot hunted
    # (run_161942) → hardware YAML uses enabled=false / press_only.
    press_only: bool = False
    kappa_delay: float = 0.8
    d_impact_hold_s: float = 0.15
    d_impact_release_s: float = 0.10
    bd_max: float = 600.0


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray
    f: np.ndarray
    v_cmd: np.ndarray
    v_tcp: np.ndarray
    ke_barrier: np.ndarray
    damping: np.ndarray
    peak_f: float
    frac_low: float
    band_2_5_frac: float
    n_loss: int


def _band_energy_frac(f: np.ndarray, dt: float, lo: float, hi: float) -> float:
    x = np.asarray(f, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 32:
        return 0.0
    win = np.hanning(n)
    P = np.abs(np.fft.rfft(x * win)) ** 2
    freq = np.fft.rfftfreq(n, dt)
    tot = float(P[(freq >= 0.2) & (freq < 40)].sum())
    if tot <= 0.0:
        return 0.0
    band = float(P[(freq >= lo) & (freq < hi)].sum())
    return band / tot


def run_sim(cfg: SimConfig) -> SimResult:
    dt = cfg.dt
    n = int(cfg.t_end_s / dt)
    delay_n = max(1, int(round(cfg.t_dead_s / dt)))
    v_buf: deque[float] = deque([0.0] * delay_n, maxlen=delay_n)

    barrier = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=cfg.use_barrier,
            press_only=cfg.press_only,
            t_dead_s=cfg.t_dead_s,
            t_pred_s=0.030,
            budget_min_n=1.5,
            budget_frac=0.75,
            f_keep_n=0.3,
            v_floor_press_m_s=0.008,
            ke_seek_default=300.0,
            ke_impact_seed=3000.0,
            ke_free_hold_s=1.0,
            cap_lpf_tau_s=0.0,
            # Soften re-approach after bounce lift-off (ideal plant only).
            limit_free_seek=True,
            yield_overforce_n=50.0,  # don't open retract on bounce overshoot
        )
    )

    t = np.zeros(n)
    x = np.zeros(n)
    f = np.zeros(n)
    v_cmd = np.zeros(n)
    v_tcp = np.zeros(n)
    ke_b = np.zeros(n)
    damp = np.zeros(n)

    pos = float(cfg.x0)
    vel_cmd = 0.0
    vel_tcp = 0.0
    in_contact = False
    contact_conf = 0.0
    d_impact = 0.0
    impact_timer = 0.0
    was_contact = False
    n_loss = 0

    for i in range(n):
        # Plant: delayed velocity → position → unilateral spring force.
        vel_tcp = float(v_buf[0])
        pos += vel_tcp * dt
        force = max(cfg.ke * pos, 0.0) if pos > 0.0 else 0.0
        in_contact = force >= 0.35

        if in_contact and not was_contact:
            impact_timer = cfg.d_impact_hold_s + cfg.d_impact_release_s
        if was_contact and not in_contact:
            n_loss += 1
        was_contact = in_contact

        # Barrier update.
        barrier.update_fdot(force, dt)
        barrier.note_contact_edge(in_contact)
        barrier.update_ke(
            f_z=force,
            v_tcp_z=vel_tcp,
            in_contact=in_contact,
            dt_eff=dt,
            f_des_z=cfg.f_des,
        )
        cap_p, cap_r = barrier.caps(
            f_z=force,
            f_des_z=cfg.f_des,
            in_contact=in_contact,
            v_z_cap=cfg.v_cap,
            seek_vz_m_s=cfg.v_cap,
            retract_fast_hold=False,
        )

        # Impact-only delay damping (steady contact stays at D0).
        if in_contact:
            contact_conf = min(1.0, contact_conf + dt / 0.08)
        else:
            contact_conf = max(0.0, contact_conf - dt / 0.05)
        if cfg.use_barrier:
            d_mag = (
                cfg.kappa_delay
                * barrier.ke_barrier
                * cfg.t_dead_s
                * contact_conf
            )
            if impact_timer > cfg.d_impact_release_s:
                d_impact = d_mag
            elif impact_timer > 0.0:
                blend = impact_timer / max(cfg.d_impact_release_s, 1e-6)
                d_impact = d_mag * blend
            else:
                d_impact = 0.0
            impact_timer = max(0.0, impact_timer - dt)
            damping = min(cfg.damping + d_impact, cfg.bd_max)
        else:
            damping = cfg.damping

        # Admittance (implicit Euler), free-space press fills to v_cap.
        f_err = cfg.f_des - force
        if not in_contact:
            # Match logged behaviour: under-force drives to v_cap.
            vel_cmd = cfg.v_cap
        else:
            denom = cfg.mass / dt + max(damping, 0.0)
            vel_cmd = (
                (cfg.mass / dt) * vel_cmd + f_err
            ) / max(denom, 1e-6)
        # Symmetric clip then barrier.
        vel_cmd = float(np.clip(vel_cmd, -cfg.v_cap, cfg.v_cap))
        if cfg.use_barrier:
            if vel_cmd >= 0.0:
                vel_cmd = min(vel_cmd, cap_p)
            else:
                vel_cmd = max(vel_cmd, -cap_r)

        v_buf.append(vel_cmd)

        t[i] = i * dt
        x[i] = pos
        f[i] = force
        v_cmd_arr = vel_cmd
        v_cmd[i] = v_cmd_arr
        v_tcp[i] = vel_tcp
        ke_b[i] = barrier.ke_barrier
        damp[i] = damping

    # Analyse steady contact window (last 2 s).
    mask = t >= max(0.0, cfg.t_end_s - 2.0)
    f_win = f[mask]
    peak_f = float(np.max(f_win)) if f_win.size else 0.0
    frac_low = float(np.mean(f_win < 0.35)) if f_win.size else 1.0
    band = _band_energy_frac(f_win, dt, 2.0, 5.0)
    return SimResult(
        t=t,
        x=x,
        f=f,
        v_cmd=v_cmd,
        v_tcp=v_tcp,
        ke_barrier=ke_b,
        damping=damp,
        peak_f=peak_f,
        frac_low=frac_low,
        band_2_5_frac=band,
        n_loss=n_loss,
    )


def main() -> None:
    for ke in (400.0, 3000.0, 9000.0):
        base = run_sim(SimConfig(ke=ke, use_barrier=False))
        fixed = run_sim(SimConfig(ke=ke, use_barrier=True))
        print(
            f"Ke={ke:5.0f}  "
            f"BASE peak={base.peak_f:5.2f}N low={100*base.frac_low:5.1f}% "
            f"2-5Hz={100*base.band_2_5_frac:5.1f}% loss={base.n_loss}  |  "
            f"BARR peak={fixed.peak_f:5.2f}N low={100*fixed.frac_low:5.1f}% "
            f"2-5Hz={100*fixed.band_2_5_frac:5.1f}% loss={fixed.n_loss}"
        )


if __name__ == "__main__":
    main()
