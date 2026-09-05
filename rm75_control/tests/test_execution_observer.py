"""Exercise the real send-path model without hardware or controller feedback."""
import json
import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.tasks.execution_observer import (
    ActuatorModel, ExecutionObserver,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge, RailServoConfig, RailServoSample,
)
from rm75_control.hw.lw100.drive import LW100Drive


def test_write_counter_counts_io_not_acceptance_or_unchanged_commands():
    drive = LW100Drive.__new__(LW100Drive)
    drive._last_rpm_cmd = 0
    drive._max_speed_rpm = 1200
    writes = []
    drive.write_param = lambda *args: writes.append(args)
    drive.set_velocity_rpm(100)
    first_time = drive.velocity_write_mono_ns
    drive.set_velocity_rpm(100)
    assert drive.velocity_write_seq == len(writes) == 1
    assert drive.velocity_write_mono_ns == first_time
    drive.set_velocity_rpm(100, force=True)
    assert drive.velocity_write_seq == len(writes) == 2
    def fail(*_):
        raise IOError("write failed")
    drive.write_param = fail
    with pytest.raises(IOError):
        drive.set_velocity_rpm(200)
    assert drive.velocity_write_seq == 2


def test_received_command_does_not_ack_worker_write():
    bridge = RailServoBridge(RailServoConfig(enabled=False))
    bridge._calibrated = bridge._armed = True
    bridge._measured_m = 0.4
    for _ in range(3):
        assert bridge.set_target_m(0.4, 0.003)
    feedback = bridge.execution_feedback
    assert feedback.command_rx_seq == 3
    assert feedback.command_processed_seq == feedback.command_written_seq == 0
    bridge._servo_sample = RailServoSample(command_processed_seq=2,
                                           command_written_seq=1, drive_write_seq=4)
    feedback = bridge.execution_feedback
    assert (feedback.command_rx_seq, feedback.command_processed_seq,
            feedback.command_written_seq, feedback.drive_write_seq) == (3, 2, 1, 4)


def test_arm_position_delay_is_causal_and_rail_uses_write_history():
    observer = ExecutionObserver([ActuatorModel(.05, 0)] * 7, ActuatorModel(0, 0))
    observer.reset(0.0, np.zeros(8))
    J = np.zeros((6, 8))
    J[1, 0] = J[1, 1] = 1.0
    observer.record_arm_send(.01, np.r_[0.0, -.001, np.zeros(6)])
    observer.record_rail_write(.01, 1, .02)
    assert observer.sample(.04, J)[1] == pytest.approx(.02)
    assert observer.sample(.06, J)[1] == pytest.approx(-.03)
    assert observer.mode == "observe"


def test_repeated_input_command_can_have_multiple_distinct_worker_writes():
    observer = ExecutionObserver([ActuatorModel(0, .02)] * 7, ActuatorModel(0, 0))
    observer.reset(0.0, np.zeros(8))
    J = np.zeros((6, 8)); J[1, 0] = 1.0
    for seq in range(1, 5):
        observer.record_rail_write(seq * .02, seq, seq * .001)
        assert observer.sample(seq * .02, J)[1] == pytest.approx(seq * .001)
    observer.record_rail_write(.08, 4, 99.0)  # same write seen again at 200 Hz
    assert observer.sample(.085, J)[1] == pytest.approx(.004)


def test_model_requires_provenance_and_never_enables_control(tmp_path):
    path = tmp_path / "model.json"
    raw = {"schema_version": 1, "arm_position": [{"delay_s": .02, "tau_s": .01}] * 7,
           "rail_velocity": {"delay_s": .03, "tau_s": .05}}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fitting data"):
        ExecutionObserver.from_file(path)
    raw.update(provenance={"run_id": "independent-run"}, validated=True)
    path.write_text(json.dumps(raw))
    model = ExecutionObserver.from_file(path)
    assert model.validated and model.mode == "observe" and model.model_hash
    raw["mode"] = "control"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="mode=observe"):
        ExecutionObserver.from_file(path)
