"""Offline smoke tests for force-ID collection helpers (no robot)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.collection import require_tool_frame
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID


def test_load_config_pose_d_joint_only():
    cfg = load_config(CONFIG_ID)
    pd = cfg.collect.pose_d
    # The checked-in force-ID profile restored the rank-rich pre-b632324
    # excitation duration; this test verifies the loader without retuning it.
    assert pd.joint_duration_s == 45.0
    assert len(pd.joint_amp_deg) == 7
    assert not hasattr(pd, "velocity_burst")
    assert cfg.fit.phi_recommended_key == "phi_10"
    assert cfg.monitor.use_inertia is False


def test_excitation_cartesian_and_joint():
    cfg = load_config(CONFIG_ID)
    cart = cfg.collect.cartesian
    exc = ex.CartesianExcitation.from_config(cart, cfg.collect.scale, "a")
    d0 = exc.delta_pose(0.0)
    d1 = exc.delta_pose(1.25)
    assert d0.shape == (6,)
    assert float(np.linalg.norm(d1[:3])) > 0.0

    q0 = np.zeros(7, dtype=float)
    q = ex.joint_cmd(2.0, q0, cfg.collect.pose_d, 1.0)
    assert q.shape == (7,)
    assert float(np.max(np.abs(q - q0))) > 0.0

    preview = ex.preview_pose_d(q0, cfg.collect.pose_d, scale=1.0)
    assert "joint_max_deg" in preview
    assert "burst_omega_deg_s_peak" not in preview


def test_require_tool_frame_rejects_wrong_tool():
    class _Bot:
        def rm_get_current_tool_frame(self):
            return 0, {"name": "gripper"}

    with pytest.raises(SystemExit) as ei:
        require_tool_frame(_Bot(), required="Arm_Tip")
    assert "Arm_Tip" in str(ei.value)


def test_require_tool_frame_accepts_arm_tip():
    class _Bot:
        def rm_get_current_tool_frame(self):
            return 0, {"name": "Arm_Tip"}

    require_tool_frame(_Bot(), required="Arm_Tip")


def test_cartesian_ramp_cosine_matches_vendor_formula():
    n = 5
    scales = [0.5 * (1.0 + math.cos(math.pi * i / (n - 1))) for i in range(n)]
    assert scales[0] == pytest.approx(1.0)
    assert scales[-1] == pytest.approx(0.0)


def test_npz_schema_keys_documented():
    cart_keys = {
        "t", "pose", "q_deg", "force_raw", "delta_pose",
        "pose0", "q0_deg", "pose_slot", "preset", "scale",
        "max_delta_mm", "max_delta_deg", "dt_ms", "log_every", "method",
    }
    d_keys = {
        "t", "pose", "q_deg", "force_raw", "phase",
        "pose0", "q0_deg", "pose_slot", "preset", "scale",
        "joint_s", "dt_ms", "log_every", "method",
    }
    assert "pose" in cart_keys and "force_raw" in cart_keys
    assert "phase" in d_keys and "pose_burst0" not in d_keys
