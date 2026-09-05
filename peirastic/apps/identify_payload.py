"""Payload ID V2 Window B driver: static m,h,b → dynamic check → optional I.

Plans poses and Fourier about link_7 / armtip. After fit, m,h,b are written
to live ``force_id_phi.json`` (I stays 0). P-I scan/fit stays off unless
``--inertia``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from rm75_control.force.compensation.identification import com_report, print_summary
from rm75_control.force.compensation.paths import CONFIG_FORCE, CONFIG_ID_V2, LOG_DIR, PHI_JSON_V2
from rm75_control.force.compensation.regressor import FrameConfig
from rm75_control.force.compensation.v2.fit_staged import (
    StaticWindow,
    delay_rejected,
    fft_lines,
    fit_delay_on_lines,
    fit_inertia_moments,
    fit_static_windows,
    static_residual_report,
)
from rm75_control.force.compensation.v2.fourier import FourierSpec, axis_twist_L, full_trajectory_closure, measure_mask
from rm75_control.force.compensation.v2.frames import gravity_force_link7
from rm75_control.force.compensation.v2.regressor_v2 import payload_wrench_mhb
from rm75_control.force.compensation.v2.schema import empty_document, phi16, phi_dict16, write_phi_v2
from rm75_control.force.compensation.v2.static_select import build_default_set


def campaign_phases(cfg: dict) -> list[dict]:
    """Deterministic phase list: rail lock, static holds, D-T, D-R, D-M, P-I."""
    st = cfg.get("static") or {}
    dyn = cfg.get("dynamic") or {}
    phases = [
        {"id": "movej_mid", "mode": "MOVEJ", "rail_m": float((cfg.get("rail") or {}).get("target_m", 0.4))},
        {"id": "rail_lock", "mode": "SERVO_TWIST", "secondary": "payload_id", "twist": [0.0] * 6},
        {"id": "static_holds", "mode": "SERVO_TWIST_HOLD", "n_train": int(st.get("n_train", 14)), "n_holdout": int(st.get("n_holdout", 4)), "secondary": "payload_id"},
        {"id": "dt_x", "mode": "SERVO_TWIST", "axis": 0, "rotational": False, "secondary": "payload_id"},
        {"id": "dt_y", "mode": "SERVO_TWIST", "axis": 1, "rotational": False, "secondary": "payload_id"},
        {"id": "dt_z", "mode": "SERVO_TWIST", "axis": 2, "rotational": False, "secondary": "payload_id"},
        {"id": "dr_x", "mode": "SERVO_TWIST", "axis": 3, "rotational": True, "secondary": "payload_id"},
        {"id": "dr_y", "mode": "SERVO_TWIST", "axis": 4, "rotational": True, "secondary": "payload_id"},
        {"id": "dr_z", "mode": "SERVO_TWIST", "axis": 5, "rotational": True, "secondary": "payload_id"},
        {"id": "dm_holdout", "mode": "SERVO_TWIST", "holdout": True, "secondary": "payload_id"},
        {"id": "inertia", "mode": "SERVO_TWIST", "high_alpha": True, "enabled": bool((cfg.get("inertia") or {}).get("enabled", False)), "secondary": "payload_id"},
    ]
    _ = dyn
    return phases


def load_v2_yaml(path: Path | None = None) -> dict:
    p = Path(path) if path is not None else CONFIG_ID_V2
    return yaml.safe_load(p.read_text()) or {}


def synthetic_static_windows(poses, *, mass, h, bias, sigma=0.02, seed=0) -> list[StaticWindow]:
    rng = np.random.default_rng(seed)
    out = []
    t0 = 0.0
    for i, g in enumerate(poses.train_g):
        f = gravity_force_link7(mass, g, bias[:3])
        tau = np.cross(g, h) + bias[3:6]
        w = np.concatenate([f, tau]) + rng.normal(scale=sigma, size=6)
        out.append(
            StaticWindow(
                g_L=g,
                wrench_L=w,
                t_s=t0,
                is_train=True,
                is_anchor=(i % 4 == 0),
                block_id=i // 4,
                name=f"train_{i}",
            )
        )
        t0 += 8.0
    for i, g in enumerate(poses.holdout_g):
        f = gravity_force_link7(mass, g, bias[:3])
        tau = np.cross(g, h) + bias[3:6]
        w = np.concatenate([f, tau]) + rng.normal(scale=sigma, size=6)
        out.append(StaticWindow(g_L=g, wrench_L=w, t_s=t0, is_train=False, block_id=99, name=f"holdout_{i}"))
        t0 += 8.0
    return out


def run_dry_fit(*, out_json: Path, cfg: dict) -> dict:
    poses = build_default_set(n_train=int(cfg.get("static", {}).get("n_train", 14)))
    mass, h = 0.5338, np.array([-0.0054, -0.0061, -0.0261])
    bias = np.array([0.12, -0.08, 0.05, 0.002, -0.001, 0.0])
    windows = synthetic_static_windows(poses, mass=mass, h=h, bias=bias)
    Sigma = np.diag([0.02, 0.02, 0.02, 0.004, 0.004, 0.004]) ** 2
    fit = fit_static_windows(windows, Sigma=Sigma, r_max_m=float(cfg.get("static", {}).get("r_max_m", 0.12)))

    spec = FourierSpec(f0_hz=0.2, n_measure=8)
    t, tw, _ = axis_twist_L(spec, 2, peak=0.025, rotational=False)
    dp, dR = full_trajectory_closure(t, tw, spec=spec)
    mask = measure_mask(spec, t)
    a = np.gradient(np.gradient(np.cumsum(tw[:, 2]) * spec.dt, spec.dt), spec.dt)
    # synthetic delay
    w_pay = np.zeros((t.size, 6))
    for i in range(t.size):
        w_pay[i] = payload_wrench_mhb(
            mass_kg=fit.mass_kg,
            h_L=fit.h_L,
            a_L=np.array([0.0, 0.0, a[i]]),
            g_L=np.array([0.0, 0.0, -9.80665]),
            omega_L=np.zeros(3),
            alpha_L=np.zeros(3),
            bias=fit.bias0,
        )
    delay_samp = 6
    w_meas = np.roll(w_pay, delay_samp, axis=0)
    freqs = np.asarray(spec.harmonics, dtype=float) * spec.f0_hz
    Wm = fft_lines(t[mask], w_meas[mask], freqs)
    Wp = fft_lines(t[mask], w_pay[mask], freqs)
    delay = fit_delay_on_lines(Wm, Wp, freqs)
    delay.delay_online_effective_s = delay.delay_sensor_vs_joint_s

    iner = None
    if bool((cfg.get("inertia") or {}).get("enabled", False)):
        I_true = np.array([0.002, 0.0025, 0.0018, 0.0, 0.0, 0.0])
        al = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0], [3.0, 0, 0]])
        om = np.zeros_like(al)
        tau = []
        from rm75_control.force.compensation.regressor import inertia_op

        for a_i, w_i in zip(al, om, strict=True):
            tau.append((inertia_op(a_i) + np.zeros((3, 6))) @ I_true)
        iner = fit_inertia_moments(
            al,
            om,
            tau,
            mass_kg=fit.mass_kg,
            r_max_m=0.12,
            sigma_M=0.002,
            holdout_tau=np.array(tau[-1]),
            holdout_pred_mhb=np.zeros(3),
            holdout_pred_I=np.array(tau[-1]),
            snr_min=float(cfg.get("inertia", {}).get("snr_min", 3.0)),
        )

    doc = empty_document()
    doc["payload"]["tool_id"] = "dry_run"
    doc["payload"]["mass_kg"] = fit.mass_kg
    doc["payload"]["first_moment_kg_m"] = fit.h_L.tolist()
    doc["payload"]["inertia_kg_m2"] = iner.I_voigt.tolist() if iner is not None and iner.adopted else None
    doc["calibration_session"]["bias0"] = fit.bias0.tolist()
    doc["calibration_session"]["bias_drift_per_s"] = fit.bias_drift_per_s.tolist()
    doc["calibration_session"]["drift_enabled"] = fit.drift_enabled
    phi = phi16(fit.mass_kg, fit.h_L, fit.bias0, iner.I_voigt if iner is not None and iner.adopted else None)
    doc["phi_mhb"] = {k: float(phi[i]) for i, k in enumerate(
        ["m", "mc_x", "mc_y", "mc_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz", "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0"]
    ) if i < 4 or i >= 10}
    rec = phi_dict16(phi)
    doc["phi_recommended"] = rec
    if iner is None or not iner.adopted:
        for k in ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"):
            rec[k] = 0.0
        doc["validation"]["inertia_ident_failed"] = None if iner is None else (iner.reason or "not_adopted")
    doc["validation"]["force_dynamic_valid"] = not delay_rejected(delay)
    doc["validation"]["moment_dynamic_valid"] = bool(iner is not None and iner.moment_dynamic_valid)
    doc["validation"]["unmodeled_inertia_torque_bound_nm"] = None if iner is None else iner.unmodeled_bound_nm
    doc["delay"]["delay_sensor_vs_joint_s"] = delay.delay_sensor_vs_joint_s
    doc["delay"]["delay_online_effective_s"] = delay.delay_online_effective_s
    doc["delay"]["delay_ci95_s"] = delay.delay_ci95_s
    doc["delay"]["delay_hit_search_boundary"] = delay.delay_hit_search_boundary
    doc["static"] = {
        "rank_m0": fit.rank_m0,
        "cond_m0": fit.cond_m0,
        "drift_enabled": fit.drift_enabled,
        "se3_dp_m": dp,
        "se3_dR_rad": dR,
    }
    residuals = static_residual_report(windows, fit)
    frame_cfg = FrameConfig.from_yaml(CONFIG_FORCE)
    phi_mhb = phi16(fit.mass_kg, fit.h_L, fit.bias0, None)
    doc["com_recommended"] = com_report(phi_mhb, frame_cfg)
    doc["static"]["rms_all"] = (residuals.get("all") or {}).get("rms_all")
    doc["static"]["rms_force"] = (residuals.get("all") or {}).get("rms_force")
    doc["static"]["rms_moment"] = (residuals.get("all") or {}).get("rms_moment")
    write_phi_v2(out_json, doc)
    print_summary(
        phi_mhb,
        frame_cfg,
        rms_all=float((residuals.get("all") or {}).get("rms_all", 0.0)),
        per_pose={k: residuals[k] for k in ("all", "train", "holdout") if k in residuals},
        out_json=out_json,
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Payload ID V2")
    p.add_argument("--config", type=Path, default=CONFIG_ID_V2)
    p.add_argument("--out", type=Path, default=PHI_JSON_V2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-csv", type=Path, default=None)
    p.add_argument("--skip-movej", action="store_true")
    p.add_argument("--p1-only", action="store_true", help="static holds only; skip Fourier")
    p.add_argument("--skip-inertia", action="store_true")
    p.add_argument("--inertia", action="store_true", help="run P-I scan and I fit (off by default)")
    p.add_argument("--no-promote", action="store_true", help="do not write live force_id_phi.json")
    p.add_argument("--fit-only", type=Path, default=None, help="fit an existing campaign CSV; no motion")
    p.add_argument("--movej-v", type=float, default=0.6)
    p.add_argument("--settle-s", type=float, default=0.0)
    args = p.parse_args(argv)
    cfg = load_v2_yaml(args.config)
    if args.inertia:
        cfg.setdefault("inertia", {})["enabled"] = True
    if args.skip_inertia:
        cfg.setdefault("inertia", {})["enabled"] = False
    if args.no_promote:
        cfg.setdefault("output", {})["auto_promote_live"] = False
    if args.dry_run:
        doc = run_dry_fit(out_json=args.out, cfg=cfg)
        print(json.dumps({"ok": True, "mass_kg": doc["payload"]["mass_kg"], "out": str(args.out)}))
        return 0
    from rm75_control.force.compensation.v2.campaign import (
        CampaignOpts,
        fit_hardware_log,
        run_hardware_campaign,
    )

    if args.fit_only is not None:
        fit_hardware_log(args.fit_only, cfg, out_json=args.out)
        return 0
    log = args.log_csv or (LOG_DIR / "payload_id_v2.csv")
    try:
        return run_hardware_campaign(
            cfg,
            log,
            out_json=args.out,
            opts=CampaignOpts(
                skip_movej=bool(args.skip_movej),
                p1_only=bool(args.p1_only),
                skip_inertia=bool(args.skip_inertia),
                movej_v=float(args.movej_v),
                settle_s=float(args.settle_s),
            ),
        )
    except FileNotFoundError:
        print("[ERR] no peirastic SHM — start Window A first:", flush=True)
        print("      python -m peirastic.apps.run_controller", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
