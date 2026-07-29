"""Tests for phase program IPC (window A hub / window C client)."""

from __future__ import annotations

import time

import pytest

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCmd,
    PhaseCommandClient,
    PhaseCommandHub,
    PhaseStatus,
    SinToolYTaskParams,
    phase_ipc_hub_ready,
)


def test_phase_ipc_start_done_roundtrip():
    hub = PhaseCommandHub(ctl_name="test_phase_ctl", payload_name="test_phase_payload")
    client = PhaseCommandClient(ctl_name="test_phase_ctl", payload_name="test_phase_payload")
    try:
        client.wait_for_hub(timeout_s=1.0)
        params = SinToolYTaskParams(
            config_path="configs/joint_admittance_8dof.yaml",
            slot="d",
            q0_rad=[0.0] * 8,
            q_target_rad=[0.1] * 8,
            pose_d=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            plan_duration_s=2.0,
        )
        cmd_seq = client.start(params)
        polled = hub.poll()
        assert polled is not None
        cmd, seq, decoded = polled
        assert cmd == PhaseCmd.START
        assert seq == cmd_seq
        assert decoded is not None
        assert decoded.slot == "d"
        hub.set_done(cmd_seq)
        hub.ack(cmd_seq)
        final = client.wait_for_cmd(cmd_seq, timeout_s=1.0, poll_s=0.01)
        assert final == PhaseStatus.DONE
        hub.set_idle()
    finally:
        client.close()
        hub.close()


def test_phase_ipc_stop_while_idle():
    hub = PhaseCommandHub(ctl_name="test_phase_ctl2", payload_name="test_phase_payload2")
    client = PhaseCommandClient(ctl_name="test_phase_ctl2", payload_name="test_phase_payload2")
    try:
        client.wait_for_hub(timeout_s=1.0)
        client.stop()
        assert hub.should_stop()
        hub.ack(0)
        assert not hub.should_stop()
    finally:
        client.close()
        hub.close()


def test_phase_ipc_owner_pid_and_force_kill_self_guard():
    hub = PhaseCommandHub(ctl_name="test_phase_ctl_pid", payload_name="test_phase_payload_pid")
    client = PhaseCommandClient(
        ctl_name="test_phase_ctl_pid", payload_name="test_phase_payload_pid"
    )
    try:
        client.wait_for_hub(timeout_s=1.0)
        assert client.hub_pid() == int(hub._ctl["owner_pid"])
        assert client.hub_pid() > 1
        # Hub runs in this process: force_kill_hub must refuse self-kill.
        assert client.force_kill_hub() is False
    finally:
        client.close()
        hub.close()


def test_subscriber_exit_keeps_hub_alive():
    hub = PhaseCommandHub(ctl_name="test_phase_ctl3", payload_name="test_phase_payload3")
    client = PhaseCommandClient(ctl_name="test_phase_ctl3", payload_name="test_phase_payload3")
    try:
        client.wait_for_hub(timeout_s=1.0)
        client.close()
        assert phase_ipc_hub_ready("test_phase_ctl3")
        client2 = PhaseCommandClient(ctl_name="test_phase_ctl3", payload_name="test_phase_payload3")
        client2.wait_for_hub(timeout_s=1.0)
        client2.close()
    finally:
        hub.close()
