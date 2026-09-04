"""Runtime RealMan tool -> Pinocchio tcp sync (offline apply, no robot)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    pose_distance,
)


def test_apply_link7_to_tcp_offset_updates_fk():
    kin = RobotKinematics()
    q = full_q_from_arm(deg2rad([0.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), rail_m=0.0)
    pose0 = kin.fk_pose(q)
    offset = np.array([0.0, 0.0, 0.220, 0.0, 1.5707963, 0.0])
    kin.apply_link7_to_tcp_offset(offset)
    pose1 = kin.fk_pose(q)
    _, d_rot = pose_distance(pose0, pose1, kin.euler_order)
    assert d_rot > 89.0
    np.testing.assert_allclose(kin.tcp_offset_pose, offset, atol=1e-9)


def test_sync_warns_when_tool_differs_from_urdf(capsys):
    from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config

    kin = RobotKinematics()
    urdf = kin.urdf_tcp_offset_pose.copy()
    offset = urdf.copy()
    offset[5] += np.deg2rad(1.5)
    maybe_sync_kin_tcp_from_config(
        kin,
        {"inner": {"sync_tcp_from_robot": False, "euler_order": "xyz"}},
        tcp_offset_pose=offset,
    )
    err = capsys.readouterr().out
    assert "synced tool frame differs from URDF" in err
    assert "1.5" in err or "1.50" in err
    assert "tcp sync:" not in err


def test_matching_tcp_apply_is_silent(capsys):
    from rm75_control.force.compensation.tool_pose import _apply_tcp_offset

    kin = RobotKinematics()
    _apply_tcp_offset(
        kin,
        kin.urdf_tcp_offset_pose.copy(),
        "tcp sync: cached tool='gripper2' should stay quiet",
    )
    assert capsys.readouterr().out == ""


def test_live_robot_overrides_stale_gripper2_cache(tmp_path, monkeypatch):
    import json

    from rm75_control.force.compensation.tool_pose import (
        maybe_sync_kin_tcp_from_config,
        write_tool_offset_cache,
    )

    cache = tmp_path / "rm75_tool_offset.json"
    monkeypatch.setattr(
        "rm75_control.force.compensation.tool_pose.tool_offset_cache_path",
        lambda: cache,
    )
    write_tool_offset_cache("gripper2", np.array([0.0, -0.015, 0.121, 0.0, 0.87, -1.55]))

    class _Bot:
        def rm_get_current_tool_frame(self):
            return 0, {"name": "probe45", "pose": [0.0, 0.0, 0.220, 0.0, 1.5707963, 0.0]}

    kin = RobotKinematics()
    name = maybe_sync_kin_tcp_from_config(
        kin,
        {"inner": {"sync_tcp_from_robot": True, "euler_order": "xyz"}},
        robot=_Bot(),
    )
    assert name == "probe45"
    np.testing.assert_allclose(kin.tcp_offset_pose[2], 0.220, atol=1e-9)
    stored = json.loads(cache.read_text())
    assert stored["name"] == "probe45"


def test_attach_mode_keeps_cache_when_no_robot(tmp_path, monkeypatch):
    from rm75_control.force.compensation.tool_pose import (
        maybe_sync_kin_tcp_from_config,
        write_tool_offset_cache,
    )

    cache = tmp_path / "rm75_tool_offset.json"
    monkeypatch.setattr(
        "rm75_control.force.compensation.tool_pose.tool_offset_cache_path",
        lambda: cache,
    )
    stale = np.array([0.0, -0.015, 0.121, 0.0, 0.87, -1.55])
    write_tool_offset_cache("gripper2", stale)
    kin = RobotKinematics()
    name = maybe_sync_kin_tcp_from_config(
        kin,
        {"inner": {"sync_tcp_from_robot": True, "euler_order": "xyz"}},
        attach_mode=True,
    )
    assert name == "gripper2"
    np.testing.assert_allclose(kin.tcp_offset_pose, stale, atol=1e-6)
