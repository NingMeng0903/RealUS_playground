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

import numpy as np

from peirastic.api import ERR_STOPPED, ERR_TIMEOUT, OK, PeirasticArm
from peirastic.core.ipc import CommandClient, CommandHub, Status
from peirastic.core.modes import Mode, ModeRequest


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
        t0 = time.monotonic()
        ret = arm.track_twist([0.0] * 6)
        assert ret == OK
        assert time.monotonic() - t0 < 0.5
        polled = hub.poll()
        assert polled is not None
        assert polled[2] is not None
        assert polled[2].mode == Mode.SERVO_TWIST
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
                hub.publish(status=Status.RUNNING, mode=Mode.MOVEJ)
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
        t0 = time.monotonic()
        ret = arm.cartesian_track(
            amplitude_x_m=0.05,
            amplitude_y_m=0.15,
            max_vel_m_s=0.04,
            duration_s=8.0,
            block=0,
        )
        assert ret == OK
        assert time.monotonic() - t0 < 0.5
        polled = hub.poll()
        assert polled is not None
        assert polled[2] is not None
        assert polled[2].mode == Mode.TRACK_CARTESIAN
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
