"""Tests for UDP realtime state observer helpers (no robot required)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from rm75_control.control.hybrid_motion.async_state import (
    AsyncStateSnapshot,
    arm_qdot_rad_s_from_snap,
    parse_realtime_push_config,
    pose_from_waypoint,
)


class _FakePos(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class _FakeEuler(ctypes.Structure):
    _fields_ = [("rx", ctypes.c_float), ("ry", ctypes.c_float), ("rz", ctypes.c_float)]


class _FakeWaypoint(ctypes.Structure):
    _fields_ = [("position", _FakePos), ("euler", _FakeEuler)]


def test_pose_from_waypoint():
    wp = _FakeWaypoint()
    wp.position = _FakePos(0.1, -0.2, 0.3)
    wp.euler = _FakeEuler(0.01, -0.02, 0.03)
    pose = pose_from_waypoint(wp)
    np.testing.assert_allclose(pose, [0.1, -0.2, 0.3, 0.01, -0.02, 0.03])


def test_parse_realtime_push_config_defaults():
    cfg = parse_realtime_push_config({"timing": {"dt_ms": 5.0}})
    assert cfg.cycle == 1
    assert cfg.port == 8098
    assert cfg.force_coordinate == 0


def test_parse_realtime_push_config_explicit():
    cfg = parse_realtime_push_config(
        {
            "timing": {"dt_ms": 10.0},
            "realtime_push": {"cycle": 2, "port": 9000, "ip": "10.0.0.5", "force_coordinate": 2},
        }
    )
    assert cfg.cycle == 2
    assert cfg.port == 9000
    assert cfg.ip == "10.0.0.5"
    assert cfg.force_coordinate == 2


def test_arm_qdot_rad_s_from_sdk_deg():
    snap = AsyncStateSnapshot(qdot_deg_s=np.array([10.0, 0.0, -5.0, 0.0, 0.0, 0.0, 20.0]))
    qdot = arm_qdot_rad_s_from_snap(snap)
    assert qdot is not None
    np.testing.assert_allclose(qdot[0], np.deg2rad(10.0))
    np.testing.assert_allclose(qdot[2], np.deg2rad(-5.0))
    np.testing.assert_allclose(qdot[6], np.deg2rad(20.0))


def test_arm_qdot_rad_s_rejects_bad_field():
    assert arm_qdot_rad_s_from_snap(AsyncStateSnapshot()) is None
    assert arm_qdot_rad_s_from_snap(
        AsyncStateSnapshot(qdot_deg_s=np.array([np.nan, 0, 0, 0, 0, 0, 0]))
    ) is None
    assert arm_qdot_rad_s_from_snap(
        AsyncStateSnapshot(qdot_deg_s=np.array([1.0, 2.0]))
    ) is None


def test_wrap_joint_delta_shortest_path():
    from rm75_control.control.joint_admittance.model import max_joint_err_deg, wrap_joint_delta

    q_from = np.deg2rad([-165.8, 0, 0, 0, 0, 0, 0])
    q_to = np.deg2rad([4.99, 0, 0, 0, 0, 0, 0])
    dq = wrap_joint_delta(q_from, q_to)
    assert np.rad2deg(dq[0]) == pytest.approx(170.79, abs=0.5)
    assert max_joint_err_deg(q_from, q_to) == pytest.approx(170.79, abs=0.5)
