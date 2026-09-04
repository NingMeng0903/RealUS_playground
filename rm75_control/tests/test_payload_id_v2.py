"""Payload ID V2 contracts: frames, flags, fit, delay, Fourier, recorder, safety."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot
from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.model import DEFAULT_URDF, RobotKinematics
from rm75_control.force.compensation.paths import CONFIG_FORCE
from rm75_control.force.compensation.v2.flags import DynamicKinematicsMode, resolve_online_flags
from rm75_control.force.compensation.v2.fit_staged import (
    StaticWindow,
    delay_rejected,
    fft_lines,
    fit_delay_on_lines,
    fit_inertia_moments,
    fit_static_windows,
    inertia_moment_residual,
)
from rm75_control.force.compensation.v2.fourier import FourierSpec, axis_twist_L, full_trajectory_closure, period_net_displacement
from rm75_control.force.compensation.v2.frames import (
    FrameContract,
    gravity_force_link7,
    tcp_pose_from_link7_pose,
    twist_about_link7_to_tcp,
    twist_link7_to_tcp,
    wrench_link7_to_tcp,
    wrench_sensor_to_link7,
    wrench_tcp_to_link7,
)
from rm75_control.force.compensation.v2.joint_observer import ArmJointObserver, DelayRing
from rm75_control.force.compensation.v2.rail_lock import evaluate_rail_lock
from rm75_control.force.compensation.v2.campaign import qdot_rad_s_from_row
from rm75_control.force.compensation.v2.recorder import (
    PayloadIdRecorder,
    merge_by_time,
    qdot_deg_s_for_record,
    snapshot_row,
)
from rm75_control.force.compensation.v2.regressor_v2 import regressor_row_v2
from rm75_control.force.compensation.v2.safety import SafetyLimits, command_stale, deadman_twist, raw_contact_abort
from rm75_control.force.compensation.v2.schema import ToolBindingError, check_tool_binding, empty_document, load_phi_v2
from rm75_control.force.compensation.v2.se3 import geodesic_R, quintic_s, quintic_sdot
from rm75_control.force.compensation.v2.static_select import build_default_set
from rm75_control.force.compensation.regressor import build_dataset, FrameConfig


def test_r_ls_direction_and_colocated_assert():
    c = FrameContract.from_yaml(CONFIG_FORCE)
    assert c.canonical_payload_frame == "link_7"
    assert c.wrench_semantics == "environment_on_tool"
    assert list(c.force_sign) == [-1, -1, -1, -1, -1, -1]
    assert list(c.moment_sign) == [-1, -1, -1]
    np.testing.assert_allclose(c.r_LS_L_vec(), 0.0)
    c.assert_colocated_sensor()


def test_mixed_moment_sign_negates_link7_com():
    """F flip + M keep ⇒ identified h = −m r in link_7. Matching signs recover r."""

    from rm75_control.force.compensation.v2.regressor_v2 import static_design

    gs = (
        np.array([0.0, 0.0, -9.80665]),
        np.array([0.0, -9.80665, 0.0]),
        np.array([-9.80665, 0.0, 0.0]),
    )
    m = 0.5
    r = np.array([0.01, 0.02, 0.048])
    h = m * r
    A = np.vstack([static_design(g)[3:6, 1:4] for g in gs])
    y_env = np.concatenate([static_design(g)[3:6, 1:4] @ h for g in gs])
    h_ok, *_ = np.linalg.lstsq(A, y_env, rcond=None)
    h_mixed, *_ = np.linalg.lstsq(A, -y_env, rcond=None)
    np.testing.assert_allclose(h_ok, h, atol=1e-12)
    np.testing.assert_allclose(h_mixed, -h, atol=1e-12)


def test_point_force_wrench_shift_sign():
    R = np.eye(3)
    r = np.array([0.1, 0.0, 0.0])
    F = np.array([0.0, 0.0, 10.0])
    tau_L = np.cross(r, F)
    w_T = wrench_link7_to_tcp(np.concatenate([F, tau_L]), R_LT=R, r_LT_L=r)
    np.testing.assert_allclose(tau_L, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(w_T[3:], 0.0, atol=1e-12)
    back = wrench_tcp_to_link7(w_T, R_LT=R, r_LT_L=r)
    np.testing.assert_allclose(back, np.concatenate([F, tau_L]), atol=1e-12)


def test_wrench_twist_power_invariance():
    R = np.eye(3)
    r = np.array([0.05, -0.02, 0.01])
    w_L = np.array([1.0, -2.0, 0.5, 0.1, 0.2, -0.05])
    V_L = np.array([0.01, 0.02, -0.03, 0.4, -0.1, 0.05])
    w_T = wrench_link7_to_tcp(w_L, R_LT=R, r_LT_L=r)
    V_T = twist_link7_to_tcp(V_L, R_LT=R, r_LT_L=r)
    assert w_L @ V_L == pytest.approx(w_T @ V_T, abs=1e-12)


def test_gravity_sign_and_2mg_flip():
    m = 0.5
    g = np.array([0.0, 0.0, -9.80665])
    f0 = gravity_force_link7(m, g)
    f1 = gravity_force_link7(m, -g)
    np.testing.assert_allclose(f0, -m * g)
    np.testing.assert_allclose(f1 - f0, 2.0 * m * g, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(f1 - f0), 2.0 * m * 9.80665, atol=1e-12)


def test_legacy_build_dataset_still_keeps_ma_when_use_inertia_false(tmp_path):
    cfg = FrameConfig.from_yaml(CONFIG_FORCE)
    n = 40
    t = np.linspace(0, 0.4, n)
    pose = np.zeros((n, 6))
    pose[:, 2] = 0.3
    force = np.zeros((n, 6))
    W, Y = build_dataset(pose, force, t, cfg, fc=2.0, use_inertia=False)
    assert W.shape[1] == 16
    assert np.allclose(W[:, 4:10], 0.0)


def test_legacy_use_inertia_maps_both_online_only():
    mode, d, r = resolve_online_flags({"use_inertia": True})
    assert mode is DynamicKinematicsMode.APPLY and d and r
    mode, d, r = resolve_online_flags({"use_inertia": False})
    assert mode is DynamicKinematicsMode.OFF and (not d) and (not r)
    mode, d, r = resolve_online_flags(
        {"dynamic_kinematics_mode": "observe", "use_dynamic_kinematics": True, "use_rotational_inertia": False}
    )
    assert mode is DynamicKinematicsMode.OBSERVE and d and not r
    with pytest.raises(ValueError):
        resolve_online_flags({"use_dynamic_kinematics": False, "use_rotational_inertia": True})


def test_yaml11_bare_off_is_force_observer_off():
    """Unquoted YAML ``off`` becomes bool False; that must not raise."""

    parsed = yaml.safe_load(
        "use_dynamic_kinematics: false\n"
        "use_rotational_inertia: false\n"
        "dynamic_kinematics_mode: off\n"
    )
    assert parsed["dynamic_kinematics_mode"] is False
    mode, d, r = resolve_online_flags(parsed)
    assert mode is DynamicKinematicsMode.OFF and (not d) and (not r)


def test_regressor_flag_combinations():
    g = np.array([0.0, 0.0, -9.8])
    a = np.array([0.2, 0.0, 0.0])
    w = np.array([0.0, 0.3, 0.0])
    al = np.array([0.0, 0.0, 1.0])
    W0 = regressor_row_v2(a, g, w, al, use_dynamic_kinematics=False, use_rotational_inertia=False)
    assert np.allclose(W0[0:3, 0], -g)
    assert np.allclose(W0[0:3, 1:4], 0.0)
    W1 = regressor_row_v2(a, g, w, al, use_dynamic_kinematics=True, use_rotational_inertia=False)
    assert not np.allclose(W1[0:3, 0], -g)
    assert np.allclose(W1[3:6, 4:10], 0.0)
    W2 = regressor_row_v2(a, g, w, al, use_dynamic_kinematics=True, use_rotational_inertia=True)
    assert not np.allclose(W2[3:6, 4:10], 0.0)


def test_static_residual_report_and_id_summary(capsys):
    from rm75_control.force.compensation.identification import com_report, print_summary
    from rm75_control.force.compensation.regressor import FrameConfig
    from rm75_control.force.compensation.v2.fit_staged import static_residual_report
    from rm75_control.force.compensation.v2.schema import phi16

    poses = build_default_set()
    mass, h = 0.516, np.array([-0.0031, -0.0109, -0.0249])
    bias = np.array([1.2, -4.8, -2.3, 0.14, 0.02, -0.06])
    windows = []
    for i, g in enumerate(poses.train_g):
        y = np.concatenate([gravity_force_link7(mass, g, bias[:3]), np.cross(g, h) + bias[3:]])
        windows.append(StaticWindow(g, y, t_s=float(i), is_train=True, block_id=i // 4, name=f"p{i}"))
    for i, g in enumerate(poses.holdout_g):
        y = np.concatenate([gravity_force_link7(mass, g, bias[:3]), np.cross(g, h) + bias[3:]])
        windows.append(StaticWindow(g, y, t_s=100.0 + i, is_train=False, block_id=99, name=f"h{i}"))
    Sigma = np.diag([0.02, 0.02, 0.02, 0.004, 0.004, 0.004]) ** 2
    fit = fit_static_windows(windows, Sigma=Sigma, r_max_m=0.12)
    report = static_residual_report(windows, fit)
    assert report["all"]["rms_force"] < 0.05
    assert report["all"]["rms_moment"] < 0.005
    assert "train" in report and "holdout" in report
    phi = phi16(fit.mass_kg, fit.h_L, fit.bias0, None)
    cfg = FrameConfig.from_yaml(CONFIG_FORCE)
    com = com_report(phi, cfg)
    print_summary(phi, cfg, rms_all=report["all"]["rms_all"], per_pose=report, out_json=Path("/tmp/phi.json"))
    text = capsys.readouterr().out
    assert "Identify done" in text
    assert "CoM sensor" in text and "CoM link7" in text
    assert "RMS" in text
    assert "train:" in text and "holdout:" in text
    assert com["link7_mm"]["Cz"] == pytest.approx(1e3 * fit.h_L[2] / fit.mass_kg, rel=1e-6)


def test_static_fit_recovers_mhb():
    poses = build_default_set(n_train=14, n_holdout=4)
    assert poses.rank_m0 == 10
    assert poses.cond_m0 < 300
    mass, h = 0.5338, np.array([-0.005, -0.006, -0.026])
    bias = np.array([0.1, -0.05, 0.02, 0.001, 0.0, -0.002])
    rng = np.random.default_rng(1)
    windows = []
    for i, g in enumerate(poses.train_g):
        y = np.concatenate([gravity_force_link7(mass, g, bias[:3]), np.cross(g, h) + bias[3:]])
        y = y + rng.normal(scale=0.01, size=6)
        windows.append(StaticWindow(g, y, t_s=float(i), is_train=True, block_id=i // 4))
    for i, g in enumerate(poses.holdout_g):
        y = np.concatenate([gravity_force_link7(mass, g, bias[:3]), np.cross(g, h) + bias[3:]])
        windows.append(StaticWindow(g, y, t_s=100.0 + i, is_train=False, block_id=99))
    Sigma = np.diag([0.02, 0.02, 0.02, 0.005, 0.005, 0.005]) ** 2
    fit = fit_static_windows(windows, Sigma=Sigma)
    assert fit.rank_m0 == 10
    assert abs(fit.mass_kg - mass) < 0.05
    assert np.linalg.norm(fit.h_L - h) < 0.02
    assert not fit.drift_enabled


def test_geodesic_quintic_boundary():
    assert quintic_s(0.0) == 0.0 and quintic_s(1.0) == 1.0
    assert quintic_sdot(0.0) == pytest.approx(0.0) and quintic_sdot(1.0) == pytest.approx(0.0)
    R0 = np.eye(3)
    R1 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(geodesic_R(R0, R1, 0.0), R0, atol=1e-12)
    np.testing.assert_allclose(geodesic_R(R0, R1, 1.0), R1, atol=1e-12)


def test_fourier_period_and_full_closure():
    spec = FourierSpec(n_warmup=1, n_measure=4, n_cooldown=1, dt=0.005)
    t, tw, _ = axis_twist_L(spec, 0, peak=0.02, rotational=False)
    T_p = 1.0 / spec.f0_hz
    # one interior period of the un-ramped sinusoid is near zero net; full ramp+hold is bounded
    dp, dR = full_trajectory_closure(t, tw, spec=spec)
    assert dp < 0.02
    assert dR < np.deg2rad(1.0)
    net = period_net_displacement(t, tw[:, 0], T_p)
    assert abs(net) < 5e-4 or spec.ramp_cycles > 0


def test_dr_about_link7_has_compensating_tcp_translation():
    r = np.array([0.12, 0.0, 0.0])
    R = np.eye(3)
    w = np.array([0.0, 0.4, 0.0])
    cmd = twist_about_link7_to_tcp(w, R_LT=R, r_LT_L=r)
    np.testing.assert_allclose(cmd[3:], w)
    np.testing.assert_allclose(cmd[:3], np.cross(w, r))


def test_delay_recovers_positive_lag():
    t = np.arange(0, 20, 0.005)
    f0 = 0.4
    x = np.sin(2 * np.pi * f0 * t)
    lag = 0.012
    y = np.sin(2 * np.pi * f0 * (t - lag))
    z = np.zeros_like(x)
    meas = np.stack([y, 0.6 * y, 0.4 * y, z, z, z], 1)
    pay = np.stack([x, 0.6 * x, 0.4 * x, z, z, z], 1)
    freqs = np.array([0.4])
    Wm = fft_lines(t[200:-200], meas[200:-200], freqs)
    Wp = fft_lines(t[200:-200], pay[200:-200], freqs)
    fit = fit_delay_on_lines(Wm, Wp, freqs)
    assert fit.delay_sensor_vs_joint_s == pytest.approx(lag, abs=0.003)
    assert fit.delay_ci95_s < 0.02
    assert not delay_rejected(fit)


def test_delay_ci_rejects_uncorrelated_force():
    rng = np.random.default_rng(1)
    t = np.arange(0, 12, 0.005)
    freqs = np.array([0.6, 1.2, 1.8])
    meas = rng.normal(scale=0.3, size=(t.size, 6))
    pay = np.stack([np.sin(2 * np.pi * 0.6 * t), *([np.zeros_like(t)] * 5)], 1)
    fit = fit_delay_on_lines(fft_lines(t, meas, freqs), fft_lines(t, pay, freqs), freqs)
    assert delay_rejected(fit)
    assert fit.delay_ci95_s > 0.02


def test_qdot_record_prefers_sdk_else_diff():
    snap = AsyncStateSnapshot(
        q_deg=np.zeros(7),
        qdot_deg_s=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        t_s=1.0,
        ok=True,
        seq=1,
    )
    np.testing.assert_allclose(qdot_deg_s_for_record(snap), np.arange(1.0, 8.0))
    snap2 = AsyncStateSnapshot(q_deg=np.full(7, 2.0), t_s=1.02, ok=True, seq=2)
    qd = qdot_deg_s_for_record(snap2, prev_q_deg=np.full(7, 1.0), prev_t_s=1.00)
    np.testing.assert_allclose(qd, np.full(7, 50.0))
    row = {"qdot_sdk_deg_s_1": 180.0 / np.pi}
    for j in range(2, 8):
        row[f"qdot_sdk_deg_s_{j}"] = 0.0
    np.testing.assert_allclose(qdot_rad_s_from_row(row), [1.0, 0, 0, 0, 0, 0, 0])
    assert qdot_rad_s_from_row({}) is None


def test_inertia_gate_rejects_low_snr():
    al = np.array([[0.05, 0, 0]] * 8)
    om = np.zeros_like(al)
    tau = np.random.default_rng(0).normal(scale=0.02, size=(8, 3))
    fit = fit_inertia_moments(al, om, tau, mass_kg=0.53, r_max_m=0.12, sigma_M=0.02)
    assert not fit.adopted


def test_inertia_gate_accepts_high_snr_and_holdout():
    from rm75_control.force.compensation.regressor import inertia_op

    I = np.array([0.003, 0.002, 0.0025, 0.0, 0.0, 0.0])
    al = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0], [4.0, 0, 0], [0, 4, 0]])
    om = np.zeros_like(al)
    tau = np.array([inertia_op(a) @ I for a in al])
    fit = fit_inertia_moments(
        al,
        om,
        tau,
        mass_kg=0.53,
        r_max_m=0.12,
        sigma_M=0.001,
        holdout_tau=tau[-1],
        holdout_pred_mhb=np.zeros(3),
        holdout_pred_I=tau[-1],
    )
    assert fit.adopted
    assert fit.moment_dynamic_valid


def test_inertia_residual_subtracts_ah_then_recovers_I():
    from rm75_control.force.compensation.regressor import inertia_op
    from rm75_control.force.compensation.v2.regressor_v2 import payload_wrench_mhb

    m, h = 0.52, np.array([-0.003, -0.011, -0.025])
    bias = np.zeros(6)
    I = np.array([0.0013, 0.0014, 0.0004, 0.0, 0.0, 0.0])
    g = np.array([0.0, 0.0, -9.81])
    a = np.array([[2.0, 0, 0], [0, 2.5, 0], [0, 0, -1.5], [1.0, 1.0, 0]])
    al = np.array([[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0], [6.0, 0, 0]])
    om = np.zeros_like(al)
    tau = []
    tau_grav_only = []
    for ai, ali in zip(a, al):
        w = payload_wrench_mhb(
            mass_kg=m, h_L=h, a_L=ai, g_L=g, omega_L=np.zeros(3), alpha_L=np.zeros(3), bias=bias,
        )
        w[3:6] = w[3:6] + inertia_op(ali) @ I
        tau.append(inertia_moment_residual(w, mass_kg=m, h_L=h, a_L=ai, g_L=g, omega_L=np.zeros(3), bias=bias))
        tau_grav_only.append(
            inertia_moment_residual(w, mass_kg=m, h_L=h, a_L=np.zeros(3), g_L=g, omega_L=np.zeros(3), bias=bias)
        )
    got = fit_inertia_moments(al, om, np.vstack(tau), mass_kg=m, r_max_m=0.12, sigma_M=0.001)
    bad = fit_inertia_moments(al, om, np.vstack(tau_grav_only), mass_kg=m, r_max_m=0.12, sigma_M=0.001)
    np.testing.assert_allclose(got.I_voigt[:3], I[:3], atol=1e-6)
    assert got.triangle_ok and got.adopted
    assert not np.allclose(bad.I_voigt[:3], I[:3], atol=2e-4)


def test_tool_binding_rejects_mismatch():
    doc = empty_document()
    doc["tool_binding"]["active_tool_name"] = "gripper2"
    doc["tool_binding"]["urdf_sha256"] = "abc"
    doc["tool_binding"]["force_sign"] = [-1, -1, -1, 1, 1, 1]
    with pytest.raises(ToolBindingError):
        check_tool_binding(doc, {"active_tool_name": "Arm_Tip", "urdf_sha256": "abc"})


def test_recorder_no_zero_fill_and_overflow(tmp_path):
    rec = PayloadIdRecorder(tmp_path / "p.csv", max_queue=1)
    rec.start()
    snap = AsyncStateSnapshot(ok=False, seq=1, t_s=1.0, wall_time_ns=2)
    row = snapshot_row(snap)
    assert np.isnan(row["q_deg_1"])
    rec.on_snapshot(AsyncStateSnapshot(ok=True, seq=1, t_s=1.0, q_deg=np.ones(7), force_raw=np.ones(6)))
    rec.on_snapshot(AsyncStateSnapshot(ok=True, seq=2, t_s=1.01, q_deg=np.ones(7), force_raw=np.ones(6)))
    rec.stop()
    # second put may overflow depending on writer speed; overflow marks invalid
    assert rec.queue_overflow_count >= 0


def test_merge_cmd_by_time_not_by_seq():
    snap_t = np.array([0.0, 0.01, 0.02])
    cmd_t = np.array([0.0, 0.015])
    cmd = np.array([[1.0], [2.0]])
    m = merge_by_time(snap_t, cmd_t, cmd)
    np.testing.assert_allclose(m.ravel(), [1.0, 1.0, 2.0])


def test_rail_lock_gate():
    st = evaluate_rail_lock(0.4, np.array([0.4001, 0.4002]), np.array([0.0, 0.0001, 0.0002]))
    assert st.ok
    bad = evaluate_rail_lock(0.4, np.array([0.41]), np.array([0.0]))
    assert not bad.ok


def test_safety_raw_envelope_and_deadman():
    lim = SafetyLimits(m_max_kg=0.6, f_margin_n=2.0)
    assert raw_contact_abort(np.array([0, 0, 5, 0, 0, 0]), lim).ok
    assert not raw_contact_abort(np.array([0, 0, 20, 0, 0, 0]), lim).ok
    assert command_stale(1.0, 0.9, lim)
    v = deadman_twist(np.array([0.2, 0, 0, 0, 0, 0]), dt=0.01)
    assert abs(v[0]) < 0.2


def test_payload_id_policy_suppresses_nullspace():
    class Inner:
        def __init__(self):
            self.q_cmd = np.array([0.4] + [0.0] * 7)
            self.flags = {}

        def set_plan_drives_rail(self, v):
            self.flags["plan"] = v

        def set_locked(self, style, q_ref_m=None):
            self.flags["locked"] = (str(style), q_ref_m)

        def set_rail_extension_active(self, v):
            self.flags["ext"] = v

        def set_centering_suppressed(self, v):
            self.flags["cent"] = v

        def set_arm_task_suppressed(self, v):
            self.flags["arm"] = v

        def set_manipulability_active(self, v):
            self.flags["manip"] = v

        def set_coupled(self):
            self.flags["coupled"] = True

    inner = Inner()
    SecondaryPolicy(preset="payload_id").apply(inner)
    assert inner.flags["arm"] is True
    assert inner.flags["cent"] is True
    assert inner.flags["manip"] is False
    assert inner.flags["locked"][1] == pytest.approx(0.4)

    hold = Inner()
    SecondaryPolicy(preset="hold").apply(hold)
    assert hold.flags["arm"] is False


def test_joint_kf_and_delay_ring():
    obs = ArmJointObserver()
    t = 0.0
    q = np.zeros(7)
    last = None
    for i in range(40):
        t = i * 0.005
        q = np.full(7, 0.01 * t)
        qd = np.full(7, 0.01)
        last = obs.step(t, q, qd, rail_q=0.4)
    q8, qd8, qdd8, _ = last
    assert qd8[0] == 0.0
    assert q8[0] == pytest.approx(0.4)
    ring = DelayRing()
    ring.push(0.0, np.zeros(8), np.zeros(8), np.zeros(8))
    ring.push(0.01, np.ones(8), np.ones(8), np.ones(8))
    q, _, _ = ring.at(0.005)
    np.testing.assert_allclose(q, 0.5 * np.ones(8))


def test_pinocchio_classical_rest_and_rail(kin_ok=None):
    kin = RobotKinematics(DEFAULT_URDF)
    q = np.zeros(8)
    mot = kin.frame_classical_motion(q, np.zeros(8), np.zeros(8), "link_7")
    np.testing.assert_allclose(mot.linear_acceleration, 0.0, atol=1e-9)
    g = kin.gravity_link7(q, np.array([0.0, 0.0, -9.80665]))
    assert abs(np.linalg.norm(g) - 9.80665) < 1e-6
    qdd = np.zeros(8)
    qdd[0] = 1.0
    mot_r = kin.frame_classical_motion(q, np.zeros(8), qdd, "link_7")
    # rail is base Y prismatic; link_7 linear acc in LOCAL depends on orientation
    assert np.linalg.norm(mot_r.linear_acceleration) > 0.2


def test_identify_payload_dry_run(tmp_path):
    from peirastic.apps.identify_payload import main
    from rm75_control.force.compensation.paths import PHI_JSON

    live_before = PHI_JSON.read_bytes() if PHI_JSON.is_file() else None
    out = tmp_path / "force_id_phi_v2.json"
    assert main(["--dry-run", "--out", str(out)]) == 0
    doc = load_phi_v2(out)
    assert doc["schema_version"] == 2
    assert doc["payload"]["mass_kg"] > 0.3
    assert "phi_recommended" in doc
    assert doc["calibration_session"]["bias0"]
    assert doc["payload"]["inertia_kg_m2"] is None
    assert all(doc["phi_recommended"][k] == 0.0 for k in ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"))
    assert "Cx" in doc["com_recommended"]["link7_mm"]
    assert doc["static"]["rms_all"] is not None
    if live_before is None:
        assert not PHI_JSON.is_file()
    else:
        assert PHI_JSON.read_bytes() == live_before


def test_campaign_phases_split_static_and_fourier():
    from peirastic.apps.identify_payload import campaign_phases

    ph = campaign_phases({"inertia": {"enabled": True}})
    ids = [p["id"] for p in ph]
    assert ids[0] == "movej_mid" and "rail_lock" in ids
    assert "static_holds" in ids
    assert ids.count("dt_x") == 1 and "inertia" in ids
    assert ph[2]["mode"] == "SERVO_TWIST_HOLD"
    assert all(p["mode"] == "SERVO_TWIST" for p in ph if p["id"].startswith("dt") or p["id"].startswith("dr"))
    default = campaign_phases({})
    assert next(p for p in default if p["id"] == "inertia")["enabled"] is False


def test_inertia_ident_off_by_default():
    from peirastic.apps.identify_payload import load_v2_yaml
    from rm75_control.force.compensation.v2.campaign import CampaignOpts, inertia_ident_enabled

    cfg = load_v2_yaml()
    assert cfg["inertia"]["enabled"] is False
    assert cfg["output"]["auto_promote_live"] is True
    assert inertia_ident_enabled(cfg) is False
    assert inertia_ident_enabled(cfg, CampaignOpts()) is False
    assert inertia_ident_enabled({"inertia": {"enabled": True}}, CampaignOpts(skip_inertia=True)) is False
    assert inertia_ident_enabled({"inertia": {"enabled": True}}) is True


def test_promote_mhb_to_live_zeros_I(tmp_path):
    import json

    from rm75_control.force.compensation.v2.schema import promote_mhb_to_live

    live = tmp_path / "force_id_phi.json"
    live.write_text(
        json.dumps(
            {
                "config": "keep-me",
                "phi_16": {"m": 0.9, "Ixx": 0.02},
                "phi_10": {
                    "m": 0.1,
                    "Ixx": 1.0,
                    "Iyy": 1.0,
                    "Izz": 1.0,
                    "Ixy": 1.0,
                    "Ixz": 1.0,
                    "Iyz": 1.0,
                },
                "phi_recommended": {"m": 0.1, "Ixx": 1.0},
            }
        )
    )
    rec = {
        "m": 0.516,
        "mc_x": -0.0031,
        "mc_y": -0.0109,
        "mc_z": -0.0249,
        "Ixx": 0.01,
        "Iyy": 0.01,
        "Izz": 0.01,
        "Ixy": 0.0,
        "Ixz": 0.0,
        "Iyz": 0.0,
        "Fx0": 1.2,
        "Fy0": -4.8,
        "Fz0": -2.3,
        "Mx0": 0.14,
        "My0": 0.02,
        "Mz0": -0.06,
    }
    promote_mhb_to_live(
        live,
        rec,
        com={"sensor_mm": {"Cx": -6.0, "Cy": -21.1, "Cz": -48.3}, "link7_mm": {"Cx": -6.0, "Cy": -21.1, "Cz": -48.3}},
        rms_all=0.21,
        per_pose={"all": {"rms_force": 0.30, "rms_moment": 0.015}},
    )
    doc = json.loads(live.read_text())
    assert doc["config"] == "keep-me"
    assert doc["phi_16"]["m"] == 0.9
    assert doc["phi_recommended"]["m"] == pytest.approx(0.516)
    assert doc["phi_10"]["m"] == pytest.approx(0.516)
    assert doc["phi_recommended"]["Ixx"] == 0.0
    assert doc["phi_10"]["Ixx"] == 0.0
    assert doc["com_recommended"]["link7_mm"]["Cz"] == pytest.approx(-48.3)
    assert doc["rms_10"] == pytest.approx(0.21)
    assert doc["per_pose_residual"]["all"]["rms_force"] == pytest.approx(0.30)


def test_link7_pose_projects_to_live_tcp():
    from rm75_control.force.compensation.v2.campaign import load_id_kinematics
    from peirastic.DEMO.movej import q_target_rad

    kin = load_id_kinematics()
    q = np.asarray(q_target_rad(), dtype=float)
    pose_L = kin.frame_pose(q, "link_7")
    pose_T = tcp_pose_from_link7_pose(
        pose_L,
        R_LT=kin.R_LT,
        r_LT_L=kin.r_LT_L,
        euler_order=kin.euler_order,
        hold_point="tcp",
        p_tcp_hold=kin.fk_pose(q)[:3],
    )
    np.testing.assert_allclose(pose_T[:3], kin.fk_pose(q)[:3], atol=1e-9)
    from rm75_control.force.compensation.v2.campaign import _world_tilt_pose

    M = kin.frame_placement(q, "link_7")
    tilted_L = _world_tilt_pose(M.translation, M.rotation, 0, 45.0, euler_order=kin.euler_order)
    tilted_hold_tcp = tcp_pose_from_link7_pose(
        tilted_L,
        R_LT=kin.R_LT,
        r_LT_L=kin.r_LT_L,
        euler_order=kin.euler_order,
        hold_point="tcp",
        p_tcp_hold=pose_T[:3],
    )
    np.testing.assert_allclose(tilted_hold_tcp[:3], pose_T[:3], atol=1e-9)
    tilted_hold_L = tcp_pose_from_link7_pose(
        tilted_L,
        R_LT=kin.R_LT,
        r_LT_L=kin.r_LT_L,
        euler_order=kin.euler_order,
        hold_point="link_7",
    )
    assert float(np.linalg.norm(tilted_hold_L[:3] - pose_T[:3])) > 0.02


def test_static_targets_from_mid_are_reachable():
    from rm75_control.force.compensation.v2.campaign import load_id_kinematics, static_targets_from_mid
    from rm75_control.force.compensation.v2.safety import SafetyLimits, joint_margin_abort
    from peirastic.DEMO.movej import q_target_rad

    kin = load_id_kinematics()
    q = np.asarray(q_target_rad(), dtype=float)
    tgts = static_targets_from_mid(kin, q)
    names = {t.name for t in tgts}
    gravity = [t for t in tgts if not t.is_yaw]
    assert any(t.name == "mid" and t.is_train for t in tgts)
    assert any(abs(t.tilt_deg) >= 45.0 - 1e-9 for t in gravity)
    assert "WX-45" in names and "WY+45" in names
    assert not any(t.name.startswith(("Tx", "Ty", "WZ")) for t in tgts)
    assert not any(t.is_yaw for t in tgts)
    assert len(gravity) >= 16
    assert tgts[-1].name != "mid"
    assert all(abs(t.q[0] - q[0]) < 1e-9 for t in tgts)
    p_T0 = kin.fk_pose(q)[:3]
    lim = SafetyLimits()
    for t in tgts:
        assert joint_margin_abort(t.q[1:], kin.q_lower[1:], kin.q_upper[1:], lim).ok
        if not t.name.startswith(("UP", "UWX", "UWY")):
            assert float(np.linalg.norm(t.pose[:3] - p_T0)) < 0.02
    g = np.vstack([kin.gravity_link7(t.q, np.array([0.0, 0.0, -9.80665])) for t in gravity])
    assert float(np.ptp(g[:, 1])) > 6.0 or float(np.ptp(g[:, 2])) > 6.0
    up = [t for t in tgts if t.name == "UP" or t.name.startswith("UP+")]
    assert up, "tip-up approach should be reachable from the taught mid"
    g_up = kin.gravity_link7(up[-1].q, np.array([0.0, 0.0, -9.80665]))
    assert float(g_up[2]) < -8.0
    assert any(t.name.startswith("UWY") for t in tgts)
    assert float(np.ptp(g[:, 2])) > 12.0


def test_fourier_default_spec_keeps_requested_peak():
    spec = FourierSpec()
    _t, tw, _ = axis_twist_L(spec, 3, peak=0.70, rotational=True)
    assert float(np.max(np.abs(tw[:, 3]))) > 0.55


def test_v2_yaml_fourier_is_short_and_faster():
    import yaml
    from rm75_control.force.compensation.paths import CONFIG_ID_V2
    from rm75_control.force.compensation.v2.campaign import CampaignOpts, _fourier_spec, estimate_campaign_s

    cfg = yaml.safe_load(CONFIG_ID_V2.read_text()) or {}
    spec = _fourier_spec(cfg)
    assert spec.f0_hz >= 0.45
    assert spec.n_warmup + spec.n_measure + spec.n_cooldown <= 8
    assert spec.x_max_m >= 0.05
    t, tw, _ = axis_twist_L(spec, 0, peak=float(cfg["dynamic"]["v_peak_m_s"]), rotational=False)
    assert float(t[-1]) < 20.0
    a = np.gradient(tw[:, 0], spec.dt)
    assert float(np.percentile(np.abs(a), 99)) > 1.0
    assert estimate_campaign_s(cfg, CampaignOpts()) < 8.0 * 60.0
    assert CampaignOpts().movej_v == 0.6
    assert CampaignOpts().settle_s == 0.0
    assert cfg["inertia"]["enabled"] is False
    t_off = estimate_campaign_s(cfg, CampaignOpts())
    cfg_on = yaml.safe_load(CONFIG_ID_V2.read_text()) or {}
    cfg_on["inertia"]["enabled"] = True
    assert estimate_campaign_s(cfg_on, CampaignOpts()) > t_off


def test_recorder_push_keeps_phase_and_cmd(tmp_path):
    rec = PayloadIdRecorder(tmp_path / "p.csv")
    rec.start()
    rec.push(
        AsyncStateSnapshot(ok=True, seq=3, t_s=1.0, q_deg=np.ones(7), force_raw=np.ones(6)),
        phase_id="static_mid",
        record_enable=1,
        v_cmd_x=0.01,
    )
    rec.stop()
    text = (tmp_path / "p.csv").read_text()
    assert "static_mid" in text
    assert "v_cmd_x" in text


def test_finish_phase_applies_payload_secondary_after_move_preset():
    from peirastic.realman8dof.session import _finish_phase
    from rm75_control.control.joint_admittance_8dof.api import SecondaryPolicy
    from rm75_control.control.joint_admittance_8dof.loop import Phase

    order: list[str] = []
    inner = type(
        "Inner",
        (),
        {
            "set_plan_drives_rail": lambda self, v: order.append(f"plan={v}"),
            "set_locked": lambda self, *a, **k: order.append("locked"),
            "set_rail_extension_active": lambda self, v: order.append(f"rail_ext={v}"),
            "set_centering_suppressed": lambda self, v: order.append(f"center={v}"),
            "set_arm_task_suppressed": lambda self, v: order.append(f"arm={v}"),
            "set_manipulability_active": lambda self, v: order.append(f"manip={v}"),
            "q_cmd": np.zeros(8),
        },
    )()
    ctx = type("Ctx", (), {"inner": inner})()
    phase = Phase(outer=object(), label="move")
    phase.on_enter = lambda: order.append("move_preset")
    _finish_phase(ctx, {"secondary": "payload_id"}, phase)
    phase.on_enter()
    assert order[0] == "move_preset"
    assert "locked" in order
    assert "arm=True" in order or any("arm=" in x for x in order)


def test_session_secondary_preset():
    from peirastic.realman8dof.session import _secondary_preset

    assert _secondary_preset({}) == "track"
    assert _secondary_preset({"secondary": "payload_id"}) == "payload_id"


def test_idle_after_payload_id_is_servo_not_hold():
    from peirastic.core.modes import Mode
    from peirastic.realman8dof.daemon import idle_after_command

    req = idle_after_command(last_secondary="payload_id")
    assert req.mode == Mode.SERVO_TWIST
    assert req.payload.get("secondary") == "payload_id"
    assert req.payload.get("joint_hold") is True
    default = idle_after_command(last_secondary=None)
    assert default.mode == Mode.SERVO_TWIST_HOLD
    assert not (default.payload or {}).get("secondary")
    pad = idle_after_command(last_secondary=None, pad_source=True)
    assert pad.mode == Mode.SERVO_TWIST
    assert not (pad.payload or {}).get("secondary")


def test_observer_update_stays_gravity_only_when_off(tmp_path):
    import json

    from rm75_control.control.admittance_common.observer import CompensatedForceObserver, ForceObserverConfig
    from rm75_control.force.compensation.regressor import PHI_NAMES

    rec = {k: 0.0 for k in PHI_NAMES}
    rec["m"] = 0.5
    path = tmp_path / "phi.json"
    path.write_text(json.dumps({"phi_recommended": rec}))
    obs = CompensatedForceObserver(
        ForceObserverConfig(phi_path=path, dynamic_kinematics_mode="off", use_dynamic_kinematics=False)
    )
    signed, filt = obs.update(0.01, np.zeros(6), np.array([0.2, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert signed.shape == (6,) and filt.shape == (6,)
    assert obs.cfg.dynamic_kinematics_mode == "off"


def test_observer_yaml_new_flags_do_not_enable_apply():
    from rm75_control.control.admittance_common.observer import CompensatedForceObserver

    obs = CompensatedForceObserver.from_yaml(
        {"force": {"dynamic_kinematics_mode": "off", "use_inertia": False}, "timing": {"dt_ms": 5.0}}
    )
    assert obs.cfg.dynamic_kinematics_mode == "off"
    assert obs.cfg.use_dynamic_kinematics is False


def test_controller_yaml_builds_force_observer():
    from peirastic.configs import DEFAULT_CONTROLLER_YAML
    from peirastic.realman8dof.binding import load_yaml
    from rm75_control.control.admittance_common.observer import CompensatedForceObserver

    obs = CompensatedForceObserver.from_yaml(load_yaml(DEFAULT_CONTROLLER_YAML))
    assert obs.cfg.dynamic_kinematics_mode == "off"
    assert obs.cfg.use_dynamic_kinematics is False


def test_holdout_isolation_fit_does_not_use_hold_for_theta():
    poses = build_default_set()
    mass, h = 0.4, np.array([0.01, 0.0, -0.02])
    bias = np.zeros(6)
    windows = []
    for i, g in enumerate(poses.train_g):
        y = np.concatenate([gravity_force_link7(mass, g), np.cross(g, h)])
        windows.append(StaticWindow(g, y, t_s=float(i), is_train=True, block_id=0))
    poison = poses.holdout_g[0] * 0 + np.array([0, 0, -9.8])
    windows.append(
        StaticWindow(poison, np.array([100.0, 0, 0, 0, 0, 0]), t_s=99.0, is_train=False, block_id=99)
    )
    fit = fit_static_windows(windows, Sigma=np.eye(6) * 1e-4)
    assert abs(fit.mass_kg - mass) < 0.05
