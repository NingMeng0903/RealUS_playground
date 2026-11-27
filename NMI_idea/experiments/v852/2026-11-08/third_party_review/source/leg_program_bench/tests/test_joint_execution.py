"""Contract tests for the offline joint-state execution adapter."""
import numpy as np
import pytest

from leg_program_bench.joint_execution import JointServoBackend, JointServoConfig


Q = np.r_[.4, np.deg2rad([-89.5, -94.5, 65.2, 96., 89.3, 61., 94.6])]


@pytest.fixture
def backend():
    pytest.importorskip("pinocchio")
    return JointServoBackend(JointServoConfig(backend="pinocchio_lag"))


def test_invalid_backend_and_timestep():
    with pytest.raises(ValueError):
        JointServoBackend(JointServoConfig(backend="teleport"))
    with pytest.raises(ValueError):
        JointServoBackend(JointServoConfig(dt=0))


def test_step_requires_initial_reset(backend):
    with pytest.raises(RuntimeError):
        backend.step(Q)


def test_actual_state_lags_command_and_clock_advances(backend):
    backend.reset(Q)
    target = Q.copy(); target[0] += .01
    row = backend.step(target)
    assert Q[0] < row["q_meas"][0] < target[0]
    assert row["sim_time_s"] == pytest.approx(.01)
    assert row["reset_count"] == 1
    assert row["safe"]
    assert np.max(np.abs(np.asarray(row["qdot"]) - 0)) <= .02 + 1e-9


def test_reject_out_of_limits_without_tick(backend):
    backend.reset(Q)
    target = Q.copy(); target[0] = .9
    with pytest.raises(ValueError):
        backend.step(target)
    assert backend.state()["sim_time_s"] == 0


def test_entry_transition_is_executed_and_recorded(backend):
    target = Q.copy(); target[0] += .006; target[1] += .01
    result = backend.execute_path(np.vstack((target, target + np.r_[.002, np.zeros(7)])), Q)
    assert result["success"]
    assert result["transition_duration_s"] > 0
    assert result["duration_s"] > result["transition_duration_s"]
    assert result["recorded_ticks"] > 10
    assert result["trajectory"][0]["q_meas"] != result["trajectory"][0]["q_cmd"]
    assert result["trajectory"][0]["phase"] == "entry"
    assert result["exit_q"][0] == pytest.approx(target[0] + .002, abs=.0002)
    assert result["reset_count"] == 1


def test_second_program_continues_actual_exit_without_reset(backend):
    target = Q.copy(); target[0] += .003
    first = backend.execute_path(np.array([target]), Q)
    target[0] += .002
    second = backend.execute_path(np.array([target]))
    assert second["actual_entry_q"] == first["exit_q"]
    assert second["reset_count"] == 1


def test_dense_collision_rejection_precedes_execution(backend, monkeypatch):
    backend.reset(Q)
    monkeypatch.setattr(backend, "_segment_clearance", lambda a, b: (-.001, 20))
    result = backend.execute_path(np.array([Q]))
    assert result["status"] == "entry_collision_rejection"
    assert result["duration_s"] == 0
    assert result["recorded_ticks"] == 0


def test_pose_step_uses_fk_state_and_does_not_reset(backend):
    backend.reset(Q)
    target = np.array(backend.state()["T_image_world"])
    target[0, 3] += .002
    row = backend.step_pose(target)
    assert row["reset_count"] == 1
    assert row["step_count"] == 1
    assert row["command_pose_error_m"] == pytest.approx(.002)
    assert np.asarray(row["T_image_world"]).shape == (4, 4)


def test_nontrivial_world_image_tcp_bridge(backend):
    from scipy.spatial.transform import Rotation
    backend.base_world[:3, :3] = Rotation.from_euler("z", .6).as_matrix()
    backend.base_world[:3, 3] = [.2, -.4, .1]
    backend.T_image_tcp[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    backend.T_image_tcp[:3, 3] = [.01, -.02, .03]
    backend.reset(Q)
    state = backend.state()
    np.testing.assert_allclose(np.asarray(state["T_image_world"]) @ backend.T_image_tcp,
                               backend.base_world @ backend.fk(), atol=1e-12)


def test_out_of_limit_path_is_finite_json_rejection(backend):
    import json
    backend.reset(Q)
    invalid = Q.copy(); invalid[0] = .9
    result = backend.execute_path(np.array([invalid]))
    assert result["status"] == "path_joint_limit_rejection"
    assert result["preflight_min_clearance_m"] is None
    json.dumps(result, allow_nan=False)

