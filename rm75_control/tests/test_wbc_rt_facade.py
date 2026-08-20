"""Facade contract: python backend unchanged; native talks SHM when the binary exists."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary


_CFG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_SEED_Q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def test_protocol_sizes_match_packed_cxx() -> None:
    assert P.WBC_IN_SIZE == 608
    assert P.WBC_OUT_SIZE == 584
    binary = find_wbc_rt_binary()
    if binary is None:
        pytest.skip("wbc_rt binary not built")
    import subprocess

    out = subprocess.check_output([str(binary), "--sizes"], text=True).strip()
    inn, outn = out.split()
    assert int(inn) == P.WBC_IN_SIZE
    assert int(outn) == P.WBC_OUT_SIZE


def test_yaml_default_backend_is_native() -> None:
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    assert cfg.backend == "native"
    assert cfg.native_shm_prefix == "rm75_wbc"


def test_python_backend_facade_still_constructs() -> None:
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    assert inner._native is None
    inner.reset(_SEED_Q)
    inner.enable()
    inner.set_coupled()
    inner.set_plan_drives_rail(False)
    inner.set_direct_joint_ptp(False)
    inner.set_arm_task_suppressed(False)
    inner.set_centering_suppressed(False)
    inner.set_manipulability_active(False)
    inner.set_rail_extension_active(True)
    inner.set_mode("reach")
    inner.capture_rail_extension_ref()
    status = inner.step(np.zeros(6), q_meas=inner.q_cmd)
    assert status.v_cmd_received.shape == (6,)
    inner.stop()


def test_native_backend_skipped_without_binary(monkeypatch) -> None:
    monkeypatch.delenv("WBC_RT_BIN", raising=False)
    cfg = JointIkConfig(backend="native", native_bin="/no/such/wbc_rt")
    with pytest.raises(FileNotFoundError, match="wbc_rt"):
        JointIkController(RobotKinematics(), cfg)


@pytest.mark.skipif(find_wbc_rt_binary() is None, reason="wbc_rt binary not built")
def test_native_smoke_step_and_setters() -> None:
    raw = yaml.safe_load(_CFG.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "native"
    cfg.native_shm_prefix = f"rm75_wbc_smoke_{os.getpid()}"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    try:
        assert inner._native is not None
        inner.reset(_SEED_Q)
        np.testing.assert_allclose(inner.q_cmd, _SEED_Q, atol=1e-9)
        inner.enable()
        inner.set_rail_mode(RailMode.COUPLED)
        inner.set_locked(LockedStyle.HOLD, q_ref_m=float(_SEED_Q[0]))
        inner.set_coupled()
        inner.set_plan_drives_rail(False)
        inner.set_direct_joint_ptp(False)
        inner.set_arm_task_suppressed(False)
        inner.set_centering_suppressed(False)
        inner.set_rail_extension_active(True)
        inner.begin_hybrid_episode(_SEED_Q, np.zeros(8))
        q0 = inner.q_cmd.copy()
        for _ in range(8):
            inner.step(np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]), q_meas=inner.q_cmd)
        assert np.all(np.isfinite(inner.q_cmd))
        assert inner.q_cmd.shape == (8,)
        assert float(np.linalg.norm(inner.q_cmd - q0)) > 1e-6
        inner.stop()
    finally:
        if inner._native is not None:
            inner._native.shutdown()
