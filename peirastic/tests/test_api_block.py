"""block polls done_seq; ESTOP and timeout map to RM_API2 codes."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_PLAYGROUND = Path(__file__).resolve().parents[2]
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

import os
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.api import ERR_CONTROLLER, ERR_STOPPED, ERR_TIMEOUT, OK, PeirasticArm
from peirastic.core.ipc import CommandClient, CommandHub, IpcMigrationError, Status
from peirastic.core.modes import DofRequest, Mode, ModeRequest
from peirastic.core.session import request_dof, stop_before_dof
from rm75_control.control.admittance_common.shm_util import close_named_shm, create_named_shm


def test_block_waits_for_done_seq() -> None:
    prefix = f"peir_block_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:

        def _ack_and_done() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.01)
                    continue
                _cmd, seq, _req = polled
                hub.ack(seq)
                time.sleep(0.05)
                hub.publish(
                    status=Status.DONE,
                    mode=Mode.MOVEJ,
                    done_seq=seq,
                    err_code=0,
                    install_seq=seq,
                )
                return

        worker = threading.Thread(target=_ack_and_done, daemon=True)
        worker.start()
        ret = arm.movej(np.zeros(8), v=0.2, block=2)
        worker.join(timeout=2.0)
        assert ret == OK
    finally:
        client.close()
        hub.close()


def test_block_zero_returns_immediately() -> None:
    prefix = f"peir_block0_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:
        seen: list[tuple] = []

        def _install() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.005)
                    continue
                seen.append(polled)
                _cmd, seq, _req = polled
                hub.ack(seq)
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.SERVO_TWIST,
                    install_seq=seq,
                )
                return

        worker = threading.Thread(target=_install, daemon=True)
        worker.start()
        t0 = time.monotonic()
        ret = arm.track_twist([0.0] * 6)
        worker.join(timeout=2.0)
        assert ret == OK
        assert time.monotonic() - t0 < 0.5
        assert seen
        assert seen[0][2] is not None
        assert seen[0][2].mode == Mode.SERVO_TWIST
    finally:
        client.close()
        hub.close()


def test_install_wait_rejects_a_superseded_mode_command() -> None:
    prefix = f"peir_install_superseded_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:
        def _supersede() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.005)
                    continue
                _cmd, first_seq, _req = polled
                hub.ack(first_seq)
                second = client.set_mode(
                    ModeRequest(Mode.TRACK_CARTESIAN, {"reference": "hold"})
                )
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.TRACK_CARTESIAN,
                    install_seq=second,
                )
                return

        worker = threading.Thread(target=_supersede, daemon=True)
        worker.start()
        assert arm.track_twist([0.0] * 6) == ERR_CONTROLLER
        worker.join(timeout=2.0)
    finally:
        client.close()
        hub.close()


def test_block_timeout_returns_minus_5() -> None:
    prefix = f"peir_blockt_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:

        def _ack_only() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.01)
                    continue
                hub.ack(polled[1])
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.MOVEJ,
                    install_seq=polled[1],
                )
                return

        worker = threading.Thread(target=_ack_only, daemon=True)
        worker.start()
        ret = arm.movej(np.zeros(8), v=0.2, block=0.25)
        worker.join(timeout=2.0)
        assert ret == ERR_TIMEOUT
    finally:
        client.close()
        hub.close()


def test_block_estop_returns_minus_6() -> None:
    prefix = f"peir_blocke_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:

        def _estop() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.01)
                    continue
                seq = polled[1]
                hub.ack(seq)
                hub.publish(
                    status=Status.ESTOP,
                    mode=Mode.MOVEJ,
                    estop=True,
                    done_seq=seq,
                    err_code=-6,
                )
                return

        worker = threading.Thread(target=_estop, daemon=True)
        worker.start()
        ret = arm.movej(np.zeros(8), v=0.2, block=2)
        worker.join(timeout=2.0)
        assert ret == ERR_STOPPED
    finally:
        client.close()
        hub.close()


def test_cartesian_track_block_zero_is_async() -> None:
    prefix = f"peir_track0_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:
        seen: list[tuple] = []

        def _install() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.005)
                    continue
                seen.append(polled)
                _cmd, seq, _req = polled
                hub.ack(seq)
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.TRACK_CARTESIAN,
                    install_seq=seq,
                )
                return

        worker = threading.Thread(target=_install, daemon=True)
        worker.start()
        t0 = time.monotonic()
        ret = arm.cartesian_track(
            amplitude_x_m=0.05,
            amplitude_y_m=0.15,
            max_vel_m_s=0.04,
            duration_s=8.0,
            block=0,
        )
        worker.join(timeout=2.0)
        assert ret == OK
        assert time.monotonic() - t0 < 0.5
        assert seen
        assert seen[0][2] is not None
        assert seen[0][2].mode == Mode.TRACK_CARTESIAN
    finally:
        client.close()
        hub.close()


def test_cartesian_track_collects_errors() -> None:
    prefix = f"peir_trackerr_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:

        def _ack_run_done() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.01)
                    continue
                seq = polled[1]
                hub.ack(seq)
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.TRACK_CARTESIAN,
                    install_seq=seq,
                    track_err_mm=1.25,
                )
                time.sleep(0.05)
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.TRACK_CARTESIAN,
                    track_err_mm=2.5,
                )
                time.sleep(0.05)
                hub.publish(
                    status=Status.DONE,
                    mode=Mode.TRACK_CARTESIAN,
                    done_seq=seq,
                    err_code=0,
                    track_err_mm=2.5,
                )
                return

        worker = threading.Thread(target=_ack_run_done, daemon=True)
        worker.start()
        errors: list[float] = []
        ret = arm.cartesian_track(
            amplitude_x_m=0.05,
            amplitude_y_m=0.15,
            max_vel_m_s=0.04,
            duration_s=1.0,
            block=2,
            errors=errors,
        )
        worker.join(timeout=2.0)
        assert ret == OK
        assert errors
        assert min(errors) >= 1.0
    finally:
        client.close()
        hub.close()


def test_snapshot_exposes_done_seq() -> None:
    prefix = f"peir_snap_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        seq = client.set_mode(ModeRequest(Mode.SERVO_TWIST, {}))
        hub.poll()
        hub.ack(seq)
        hub.publish(status=Status.DONE, mode=Mode.SERVO_TWIST, done_seq=seq, err_code=0)
        snap = client.snapshot()
        assert snap["done_seq"] == seq
        assert snap["err_code"] == 0
        assert snap["ack_seq"] == seq
    finally:
        client.close()
        hub.close()


def test_control_ipc_version_is_published_and_legacy_is_rejected() -> None:
    prefix = f"peir_abi_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        snap = client.snapshot()
        assert snap["abi_magic"] == b"PEIRAST2"
        assert snap["abi_version"] == 2
    finally:
        client.close()
        hub.close()

    legacy_name = prefix + "peirastic_ctl"
    legacy = create_named_shm(legacy_name, 64)
    try:
        with pytest.raises(IpcMigrationError, match="legacy PEIRASTIC IPC"):
            CommandClient(prefix=prefix)
    finally:
        close_named_shm(legacy)

    short_name = prefix + "peirastic_ctl_v2"
    short = create_named_shm(short_name, 1)
    try:
        with pytest.raises(IpcMigrationError, match="too small"):
            CommandClient(prefix=prefix)
    finally:
        close_named_shm(short)

    hub = CommandHub(prefix=prefix)
    try:
        hub._ctl[0]["abi_version"] = 99
        with pytest.raises(IpcMigrationError, match="unsupported PEIRASTIC IPC ABI"):
            CommandClient(prefix=prefix)
    finally:
        hub.close()


def test_remote_get_dof_requires_fresh_healthy_snapshot() -> None:
    prefix = f"peir_remote_dof_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:
        assert arm.get_dof() == (OK, 8)
        hub.publish(
            status=Status.RUNNING,
            mode=Mode.SERVO_TWIST,
            dof=7,
            dof_pending=-1,
            dof_effective=7,
            dof_status=Status.DONE,
        )
        assert arm.get_dof() == (OK, 7)

        hub._ctl[0]["t_mono"] = time.monotonic() - 1.0
        assert arm.get_dof() == (ERR_CONTROLLER, 0)

        hub.publish(status=Status.ERROR, mode=Mode.SERVO_TWIST, dof=7)
        assert arm.get_dof() == (ERR_CONTROLLER, 0)
        hub.publish(
            status=Status.ESTOP,
            mode=Mode.SERVO_TWIST,
            dof=7,
            estop=True,
        )
        assert arm.get_dof() == (ERR_STOPPED, 0)
    finally:
        client.close()
        hub.close()


def test_dof_ipc_rejects_fractional_and_string_structure_values() -> None:
    with pytest.raises(ValueError, match="exactly 7 or 8"):
        DofRequest(7.2)
    with pytest.raises(ValueError, match="exactly 7 or 8"):
        DofRequest.from_json({"dof": 7.2, "after_current": True})
    with pytest.raises(ValueError, match="task boundary"):
        DofRequest.from_json({"dof": 7, "after_current": "false"})

    prefix = f"peir_bad_dof_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        with pytest.raises(ValueError, match="exactly 7 or 8"):
            client.set_dof(7.2)
        blob = b'{"dof":7.2,"after_current":true}'
        client._pay[: len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        client._ctl[0]["payload_len"] = len(blob)
        client._ctl[0]["cmd"] = 5
        client._ctl[0]["cmd_seq"] = 1
        with pytest.raises(ValueError, match="exactly 7 or 8"):
            hub.poll()
    finally:
        client.close()
        hub.close()


def test_set_dof_round_trip_has_pending_state() -> None:
    prefix = f"peir_dof_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        assert client.snapshot()["dof"] == 8
        assert client.snapshot()["dof_pending"] == -1
        assert client.snapshot()["dof_effective"] == 8
        assert client.snapshot()["dof_requested"] == 8
        seq = client.set_dof(7)
        polled = hub.poll()
        assert polled is not None
        cmd, got_seq, req = polled
        assert int(cmd) == 5
        assert got_seq == seq
        assert isinstance(req, DofRequest)
        assert req.dof == 7
        hub.ack(seq)
        hub.publish(
            status=Status.RUNNING,
            mode=Mode.SERVO_TWIST,
            dof=8,
            dof_pending=7,
        )
        assert client.snapshot()["dof_pending"] == 7
        pending = client.snapshot()
        assert pending["dof_requested"] == 7
        assert pending["dof_effective"] == 8
        assert pending["dof_request_seq"] == seq
        assert pending["dof_status"] == int(Status.RUNNING)
        hub.publish(
            status=Status.DONE,
            mode=Mode.SERVO_TWIST,
            done_seq=seq,
            err_code=0,
            dof=7,
            dof_pending=-1,
        )
        snap = client.snapshot()
        assert snap["dof"] == 7
        assert snap["dof_pending"] == -1
        assert snap["done_seq"] == seq
        assert snap["dof_effective"] == 7
        assert snap["dof_requested"] == 7
        assert snap["dof_done_seq"] == seq
        assert snap["dof_status"] == int(Status.DONE)
    finally:
        client.close()
        hub.close()


def test_set_dof_preserves_an_explicit_stop_boundary() -> None:
    prefix = f"peir_dof_stop_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        client.stop()
        assert client.set_dof(7) > 0
        assert bool(hub._ctl[0]["stop_req"])
        polled = hub.poll()
        assert polled is not None and isinstance(polled[2], DofRequest)
    finally:
        client.close()
        hub.close()


def test_stop_before_dof_preserves_unacknowledged_request_but_stops_acknowledged_pending() -> None:
    prefix = f"peir_dof_boundary_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        seq = client.set_dof(7)
        # Window A has not consumed the one-slot request yet; STOP would
        # overwrite it and must therefore be deferred.
        stop_before_dof(client)
        assert not bool(hub._ctl[0]["stop_req"])
        assert int(hub._ctl[0]["cmd_seq"]) == seq

        polled = hub.poll()
        assert polled is not None and polled[1] == seq
        hub.ack(seq)
        hub.publish(
            status=Status.RUNNING,
            mode=Mode.SERVO_TWIST,
            dof=8,
            dof_pending=7,
            dof_requested=7,
            dof_request_seq=seq,
            dof_status=Status.RUNNING,
        )
        # The daemon now owns the pending request, so STOP opens its task
        # boundary without erasing the already accepted DOF transition.
        stop_before_dof(client)
        assert bool(hub._ctl[0]["stop_req"])
    finally:
        client.close()
        hub.close()


def test_stop_before_dof_does_not_overwrite_when_snapshot_is_unreadable() -> None:
    class _BrokenClient:
        def __init__(self) -> None:
            self.stops = 0

        def snapshot(self):
            raise OSError("controller unavailable")

        def stop(self):
            self.stops += 1

    client = _BrokenClient()
    stop_before_dof(client)
    assert client.stops == 0


def test_request_dof_rejects_unversioned_snapshot_instead_of_defaulting_to_8() -> None:
    class _LegacyClient:
        def snapshot(self):
            return {"dof": 8, "dof_pending": -1}

        def set_dof(self, _dof):
            raise AssertionError("legacy snapshot must not send a replacement request")

    with pytest.raises(RuntimeError, match="telemetry"):
        request_dof(_LegacyClient(), 7, timeout_s=0.05)


def test_request_dof_waits_for_commit_ack() -> None:
    prefix = f"peir_dof_wait_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    try:
        def _commit() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.005)
                    continue
                _cmd, seq, _req = polled
                hub.ack(seq)
                hub.publish(
                    status=Status.DONE,
                    mode=Mode.SERVO_TWIST,
                    done_seq=seq,
                    err_code=0,
                    dof=7,
                    dof_pending=-1,
                )
                return

        worker = threading.Thread(target=_commit, daemon=True)
        worker.start()
        assert request_dof(client, 7, timeout_s=2.0) == 8
        worker.join(timeout=2.0)
        assert client.snapshot()["dof"] == 7
    finally:
        client.close()
        hub.close()


def test_set_dof_block_ignores_unrelated_global_done_seq() -> None:
    prefix = f"peir_dof_seq_{os.getpid()}_"
    hub = CommandHub(prefix=prefix)
    client = CommandClient(prefix=prefix)
    arm = PeirasticArm(client=client, attach=False)
    try:
        def _publish_unrelated_done() -> None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                polled = hub.poll()
                if polled is None:
                    time.sleep(0.005)
                    continue
                _cmd, seq, _req = polled
                hub.ack(seq)
                hub.publish(
                    status=Status.RUNNING,
                    mode=Mode.SERVO_TWIST,
                    dof=8,
                    dof_pending=7,
                )
                # A later mode completion must not satisfy this DOF wait.
                hub.publish(
                    status=Status.DONE,
                    mode=Mode.SERVO_TWIST,
                    done_seq=seq + 1,
                    err_code=0,
                )
                return

        worker = threading.Thread(target=_publish_unrelated_done, daemon=True)
        worker.start()
        assert arm.set_dof(7, block=0.15) == ERR_TIMEOUT
        worker.join(timeout=2.0)
    finally:
        client.close()
        hub.close()


def test_dof_commit_never_treats_stale_feedback_as_stationary() -> None:
    from peirastic.realman8dof.daemon import ControllerService

    svc = object.__new__(ControllerService)
    svc.inner = SimpleNamespace(core=SimpleNamespace(qdot_prev=np.zeros(8)))
    rail = SimpleNamespace(enabled=False)
    fresh = SimpleNamespace(
        ok=True,
        t_s=time.monotonic(),
        qdot_deg_s=np.zeros(7),
    )
    stale = SimpleNamespace(
        ok=True,
        t_s=time.monotonic() - 1.0,
        qdot_deg_s=np.zeros(7),
    )
    invalid = SimpleNamespace(
        ok=False,
        t_s=time.monotonic(),
        qdot_deg_s=np.zeros(7),
    )
    assert ControllerService._dof_stationary(svc, SimpleNamespace(read=lambda: fresh), rail)
    assert not ControllerService._dof_stationary(svc, SimpleNamespace(read=lambda: stale), rail)
    assert not ControllerService._dof_stationary(svc, SimpleNamespace(read=lambda: invalid), rail)


def test_dof_commit_seeds_once_from_fresh_arm_and_rail_feedback() -> None:
    from peirastic.realman8dof.daemon import ControllerService

    class _Inner:
        kin = SimpleNamespace(nv=8)
        q_cmd = np.zeros(8)
        is_locked_hold = False
        _peirastic_dof = 8

        def __init__(self) -> None:
            self.reset_seeds: list[np.ndarray] = []

        def reset(self, q) -> None:
            q = np.asarray(q, dtype=float).copy()
            self.reset_seeds.append(q)
            self.q_cmd = q

        def set_plan_drives_rail(self, value) -> None:
            del value

        def set_locked(self, *args, **kwargs) -> None:
            del args, kwargs
            self.is_locked_hold = True

        def set_coupled(self) -> None:
            self.is_locked_hold = False

        def set_rail_extension_active(self, value) -> None:
            del value

    now = time.monotonic()
    q_deg = np.array([10.0, 20.0, -30.0, 40.0, 50.0, -60.0, 70.0])
    bus = SimpleNamespace(
        read=lambda: SimpleNamespace(
            ok=True,
            t_s=time.monotonic(),
            q_deg=q_deg,
            qdot_deg_s=np.zeros(7),
        )
    )
    feedback = SimpleNamespace(
        valid=True,
        position_m=0.512,
        sample_mono_s=now,
        sample_age_s=0.0,
        v_meas_m_s=0.0,
    )
    rail = SimpleNamespace(
        enabled=True,
        command=SimpleNamespace(v_ff_m_s=float("nan")),
        measured_speed_m_s=0.0,
        measured_m=0.512,
        execution_feedback=feedback,
    )
    published: list[dict] = []
    svc = object.__new__(ControllerService)
    svc.inner = _Inner()
    svc._dof = 8
    svc._pending_dof = (7, 19)
    svc.ctx = SimpleNamespace(dof=8)
    svc.mode = Mode.SERVO_TWIST
    svc.hub = SimpleNamespace(publish=lambda **kw: published.append(kw))
    svc.panel = SimpleNamespace(event=lambda *args, **kwargs: None)

    assert ControllerService._commit_pending_dof(svc, bus, rail)
    assert svc._pending_dof is None
    assert svc._dof == 7
    assert svc.ctx.dof == 7
    assert len(svc.inner.reset_seeds) == 1
    seed = svc.inner.reset_seeds[0]
    assert seed[0] == pytest.approx(0.512)
    assert seed[1] == pytest.approx(np.deg2rad(10.0))
    assert published[-1]["dof_status"] == Status.DONE


def test_dof_request_does_not_stop_a_live_servo_phase() -> None:
    from peirastic.realman8dof.daemon import ControllerService

    stop_calls: list[bool] = []
    publishes: list[dict] = []

    class _Hub:
        def request_stop(self) -> None:
            stop_calls.append(True)

        def publish(self, **kwargs) -> None:
            publishes.append(kwargs)

    svc = object.__new__(ControllerService)
    svc._dof = 8
    svc._pending_dof = None
    svc._live = object()
    svc.mode = Mode.SERVO_TWIST
    svc.hub = _Hub()
    svc.inner = SimpleNamespace(core=SimpleNamespace(qdot_prev=np.zeros(8)))
    svc.panel = SimpleNamespace(event=lambda *args, **kwargs: None)
    ControllerService._queue_dof(svc, DofRequest(7), 4, None, SimpleNamespace(enabled=False))
    assert not stop_calls
    assert svc._pending_dof == (7, 4)
    assert publishes[-1]["dof_pending"] == 7
