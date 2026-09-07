"""Force-ID/online/monitor regressions at rotated, displaced sensor and TCP frames."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from rm75_control.control.admittance_common.observer import CompensatedForceObserver, ForceObserverConfig
from rm75_control.force.compensation.regressor import FrameConfig, com_from_phi, regressor_row
from rm75_control.force.compensation.v2.frames import (
    FrameContract, link7_pose_from_tcp_pose, tcp_pose_from_link7_pose,
    wrench_link7_to_sensor, wrench_link7_to_tcp, wrench_tcp_to_link7,
)
from rm75_control.force.compensation.v2.schema import phi16, phi_dict16, runtime_phi_link7, promote_mhb_to_live

_MONITOR_PATH = Path(__file__).resolve().parents[1] / "apps/force_compensation/force_monitor.py"
_spec = importlib.util.spec_from_file_location("force_monitor_frames_test", _MONITOR_PATH)
_monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_monitor)
MonitorEstimator = _monitor.MonitorEstimator


def _setup(tmp_path, *, sensor_rot=(0.28, -0.37, 0.61), sensor_xyz=(0.015, -0.02, 0.025)):
    config = tmp_path / "sensor.yaml"
    config.write_text(yaml.safe_dump({
        "force_sign": [-1] * 6, "euler_order": "xyz",
        "sensor_offset_euler_xyz_rad": list(sensor_rot),
        "sensor_origin_in_link7_m": list(sensor_xyz), "gravity_base": [0, 0, -9.80665],
    }))
    phi = phi16(0.5109, np.array([0.0032, 0.0109, 0.0245]), np.array([0.27, -4.07, -1.99, -0.087, 0.027, 0.087]))
    path = tmp_path / "phi.json"
    promote_mhb_to_live(path, phi_dict16(phi))
    obs = CompensatedForceObserver(ForceObserverConfig(phi_path=path, force_sensor=config, min_samples=1))
    offset = np.array([0, -0.0152, 0.1214, *np.radians([1, 49.9, -88.7])])
    return obs, phi, offset


def _sample(obs, phi, offset, pose_L, external_tcp=None):
    R_LT = Rotation.from_euler("xyz", offset[3:]).as_matrix()
    g_L = Rotation.from_euler("xyz", pose_L[3:]).as_matrix().T @ obs.contract.gravity_base()
    payload = regressor_row(np.zeros(3), g_L, np.zeros(3), np.zeros(3), use_inertia=False) @ phi
    if external_tcp is not None:
        payload += wrench_tcp_to_link7(external_tcp, R_LT=R_LT, r_LT_L=offset[:3])
    raw = -wrench_link7_to_sensor(payload, obs.contract)
    pose_T = tcp_pose_from_link7_pose(pose_L, R_LT=R_LT, r_LT_L=offset[:3], hold_point="link_7")
    return pose_T, raw


def test_free_air_zero_under_arbitrary_sensor_mount_and_live_tcp(tmp_path):
    obs, phi, offset = _setup(tmp_path)
    monitor = MonitorEstimator(obs, offset)
    poses = np.array([[0.3, 0.1, 0.5, 0, 0, 0], [0.2, -0.4, 0.7, 0.4, -1.0, 0.7], [0.5, 0.1, 0.4, -1.0, 0.6, -1.8]])
    for i, pose_L in enumerate(poses):
        pose_T, raw = _sample(obs, phi, offset, pose_L)
        signed, ext = monitor.sample(0.01 * i, pose_T, raw)
        np.testing.assert_allclose(ext, 0, atol=3e-14)
        recovered = link7_pose_from_tcp_pose(pose_T, R_LT=monitor.R_LT, r_LT_L=offset[:3])
        np.testing.assert_allclose(recovered, pose_L, atol=2e-14)


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("direction", [-1, 1])
def test_each_tcp_channel_is_preserved_without_cross_axis_residual(tmp_path, axis, direction):
    obs, phi, offset = _setup(tmp_path)
    monitor = MonitorEstimator(obs, offset)
    expected = np.zeros(6)
    expected[axis] = direction * (3 if axis < 3 else 0.12)
    pose_T, raw = _sample(obs, phi, offset, np.array([0.3, -0.1, 0.6, 0.7, -0.8, 0.9]), expected)
    _, ext = monitor.sample(0.01, pose_T, raw)
    np.testing.assert_allclose(ext, expected, atol=3e-14)


def test_monitor_gravity_only_does_not_differentiate_tcp_translation(tmp_path):
    obs, phi, offset = _setup(tmp_path)
    monitor = MonitorEstimator(obs, offset)
    for i in range(40):
        pose_L = np.array([0.4 + 0.01 * (-1)**i, -0.1, 0.6, 0.4, -0.6, 0.8])
        pose_T, raw = _sample(obs, phi, offset, pose_L)
        _, ext = monitor.sample(i * 0.005, pose_T, raw)
        np.testing.assert_allclose(ext, 0, atol=3e-14)


def test_legacy_sensor_parameters_move_to_link7_with_power_consistent_wrench(tmp_path):
    obs, _, _ = _setup(tmp_path)
    c = obs.contract
    p_s = phi16(0.7, [0.012, -0.02, 0.035], [0.1, 0.2, -0.4, 0.01, -0.02, 0.03], [0.003, 0.004, 0.005, 0.0001, -0.0002, 0.0003])
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"phi_recommended": phi_dict16(p_s)}))
    p_l, _ = runtime_phi_link7(path, "phi_recommended", c)
    R, r = c.R_LS_mat(), c.r_LS_L_vec()
    a, g, w, al = np.array([0.5, -0.7, 0.2]), np.array([2, -4, -8]), np.array([0.4, -0.6, 0.9]), np.array([-0.3, 0.7, 1.2])
    a_s = R.T @ (a + np.cross(al, r) + np.cross(w, np.cross(w, r)))
    ws = regressor_row(a_s, R.T @ g, R.T @ w, R.T @ al, use_inertia=True) @ p_s
    wl = regressor_row(a, g, w, al, use_inertia=True) @ p_l
    np.testing.assert_allclose(wrench_link7_to_sensor(wl, c), ws, atol=2e-14)
    rs, rl = com_from_phi(p_l, FrameConfig.from_yaml(obs.cfg.force_sensor), parameter_frame="link_7")
    np.testing.assert_allclose(rs, p_s[1:4] / p_s[0], atol=1e-14)
    np.testing.assert_allclose(rl, R @ rs + r, atol=1e-14)


def test_explicit_reload_never_zeros_from_contact_and_keeps_last_good_model(tmp_path):
    obs, phi, offset = _setup(tmp_path)
    before = obs.phi.copy()
    new_phi = phi.copy()
    new_phi[0] += 0.02
    promote_mhb_to_live(obs.cfg.phi_path, phi_dict16(new_phi))
    # A file change alone cannot mutate an active force observer.
    pose_T, raw = _sample(obs, phi, offset, np.zeros(6), np.array([0, 0, -4, 0, 0.1, 0]))
    MonitorEstimator(obs, offset).sample(0.01, pose_T, raw)
    np.testing.assert_array_equal(obs.phi, before)
    assert obs.reload_if_changed()
    np.testing.assert_allclose(obs.phi, new_phi)
    obs.cfg.phi_path.write_text('{"phi_recommended":')
    assert not obs.reload_if_changed()
    assert obs.reload_error
    np.testing.assert_allclose(obs.phi, new_phi)


def test_unknown_parameter_frame_is_rejected(tmp_path):
    obs, phi, _ = _setup(tmp_path)
    obs.cfg.phi_path.write_text(json.dumps({"phi_recommended": phi_dict16(phi), "parameter_frames": {"phi_recommended": "tcp"}}))
    with pytest.raises(ValueError, match="Unknown payload parameter frame"):
        runtime_phi_link7(obs.cfg.phi_path, "phi_recommended", obs.contract)


def test_unidentifiable_delay_does_not_prevent_static_payload_save(tmp_path):
    from rm75_control.force.compensation.v2.schema import write_phi_v2
    obs, phi, _ = _setup(tmp_path)
    doc = {"phi_recommended": phi_dict16(phi), "delay": {"delay_ci95_s": float("inf"), "delay_per_axis_s": [float("nan"), 0.03]}}
    write_phi_v2(obs.cfg.phi_path, doc)
    saved = json.loads(obs.cfg.phi_path.read_text())
    assert saved["delay"]["delay_ci95_s"] is None
    assert saved["delay"]["delay_per_axis_s"] == [None, 0.03]
    assert saved["phi_recommended"]["m"] == phi[0]
    before = obs.cfg.phi_path.read_bytes()
    doc["phi_recommended"]["m"] = float("nan")
    with pytest.raises(ValueError):
        write_phi_v2(obs.cfg.phi_path, doc)
    assert obs.cfg.phi_path.read_bytes() == before


def test_noise_covariance_excludes_between_pose_gravity_excitation():
    from rm75_control.force.compensation.v2.fit_staged import pooled_shrinkage_cov
    rng = np.random.default_rng(75)
    windows = [rng.normal(size=(80, 6)) * [0.05, 0.04, 0.06, 0.002, 0.003, 0.001] for _ in range(4)]
    shifts = np.array([[5, -4, 2, .1, -.2, .03], [-3, 1, -5, -.3, .1, -.05], [1, 5, 3, .2, .2, -.08], [-4, -2, -1, -.1, -.3, .1]])
    expected = pooled_shrinkage_cov(windows)
    actual = pooled_shrinkage_cov([w + shift for w, shift in zip(windows, shifts)])
    np.testing.assert_allclose(actual, expected, atol=2e-17)


def test_campaign_holdout_does_not_tune_covariance_or_payload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rm75_control.force.compensation.v2 import campaign
    from rm75_control.force.compensation.v2.fit_staged import StaticWindow
    from rm75_control.force.compensation.v2.static_select import build_default_set
    from rm75_control.force.compensation.paths import CONFIG_FORCE
    rng = np.random.default_rng(0)
    poses = build_default_set()
    phi = phi16(.55, [.005, .008, .025], [.2, -.1, .05, .01, -.02, .005])
    train, hold = [], []
    for is_train, gravities in [(True, poses.train_g), (False, poses.holdout_g)]:
        for i, g in enumerate(gravities):
            pred = regressor_row(np.zeros(3), g, np.zeros(3), np.zeros(3), use_inertia=False) @ phi
            samples = pred + rng.normal(size=(40, 6)) * [.02, .03, .01, .002, .001, .002]
            window = StaticWindow(g_L=g, wrench_L=samples.mean(axis=0), t_s=float(i), n_eff=40, samples=samples,
                                  is_train=is_train, name=f'{"train" if is_train else "hold"}_{i}')
            (train if is_train else hold).append(window)
    monkeypatch.setattr(campaign, "windows_from_csv", lambda *_: train + hold)
    monkeypatch.setattr(campaign, "_motion_from_csv", lambda *_args, **_kwargs: None)
    kin = SimpleNamespace(R_LT=np.eye(3), r_LT_L=np.zeros(3))
    cfg = {"output": {"auto_promote_live": False}}
    first = campaign.fit_hardware_log(tmp_path / "unused.csv", cfg, out_json=tmp_path / "first.json", kin=kin, contract=FrameContract.from_yaml(CONFIG_FORCE))
    for window in hold:
        window.samples += rng.normal(size=window.samples.shape) * [100, 100, 100, 10, 10, 10]
        window.wrench_L += 50
    second = campaign.fit_hardware_log(tmp_path / "unused.csv", cfg, out_json=tmp_path / "second.json", kin=kin, contract=FrameContract.from_yaml(CONFIG_FORCE))
    assert first["phi_recommended"] == second["phi_recommended"]
    assert second["static"]["per_pose_residual"]["holdout"]["rms_force"] > 50
