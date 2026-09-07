"""Exercise the real SHM client and native wakeup channel without robot I/O."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.admittance_common.shm_util import create_named_shm
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import NativeWbcClient


@pytest.fixture(scope="module")
def peer_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("transport peer requires a C++ compiler")
    root = Path(__file__).resolve().parents[1] / "native" / "wbc_rt"
    binary = tmp_path_factory.mktemp("notification-peer") / "peer"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root / "include"),
                    str(root / "tests" / "notification_peer.cpp"), "-o", str(binary)], check=True)
    return binary


@pytest.fixture
def client(peer_binary):
    class Controller:
        cfg = SimpleNamespace(native_shm_prefix=f"notification_test_{os.getpid()}",
                              dt=0.005, rail_refresh_dt_s=0.02)
        q_cmd = np.arange(8, dtype=float) / 10.0
        rail_observer = SimpleNamespace(_initialized=False)
        core = None
        rail_ext_task = None
        posture_retarget = None
        kin = SimpleNamespace(jacobian=lambda q: np.zeros((6, 8)))

    ctrl = Controller()  # Retain the owner: client only keeps a weak reference.
    native = NativeWbcClient(ctrl)
    native._shm_in = create_named_shm(native.in_name, P.WBC_IN_SIZE)
    native._shm_out = create_named_shm(native.out_name, P.WBC_OUT_SIZE)
    native._in = P.view_in(native._shm_in.buf)
    native._out = P.view_out(native._shm_out.buf)
    native._notify, child = socket.socketpair()
    native._notify.setblocking(False)
    native._proc = subprocess.Popen([str(peer_binary), str(child.fileno()), native.in_name,
                                     native.out_name], pass_fds=(child.fileno(),))
    child.close()
    try:
        assert select.select([native._notify], [], [], 2.0)[0]
        assert native._notify.recv(1)
        assert int(native._out["status"][0]) == P.STATUS_READY
        native._started = True
        yield native
    finally:
        if native._proc is not None and native._proc.poll() is None:
            os.kill(native._proc.pid, signal.SIGCONT)
        native.shutdown()


def test_notifications_wake_despite_native_timer_slack(client):
    # Several real process round trips, each within the unchanged 20-ms bound.
    waits = []
    for _ in range(20):
        assert client._command(P.CMD_RESET, q_meas=client.ctrl.q_cmd, timeout_s=0.020)
        waits.append(client._last_wait_s)
    assert max(waits) < 0.020
    assert int(client._out["cmd_ack"][0]) == client._seq


def test_timeout_never_returns_unacknowledged_data_and_requires_reset(client):
    os.kill(client._proc.pid, signal.SIGSTOP)
    # Wait for the process stop, rather than racing its next transaction.
    os.waitpid(client._proc.pid, os.WUNTRACED)
    client._out["q_cmd"] = 999.0
    client._out["qdot"] = 888.0
    client._out["solve_ms"] = 777.0
    cpu_start = time.thread_time()
    step = client.update(np.ones(6), q_meas=client.ctrl.q_cmd, auto_commit=False)
    assert time.thread_time() - cpu_start < 0.005  # The waiter yields its CPU.
    assert step.solver_fault_latched and step.fallback_reason == "native_timeout"
    assert np.isnan(step.qp_solver_solve_ms)
    np.testing.assert_array_equal(step.q_send, client.ctrl.q_cmd)
    np.testing.assert_array_equal(step.qdot, np.zeros(8))
    assert client._published_q_cmd is None
    failed_seq = client._seq
    os.kill(client._proc.pid, signal.SIGCONT)
    assert client._wait_seq(failed_seq, timeout_s=0.5)  # Late reply arrives.
    assert client.update(np.zeros(6), q_meas=client.ctrl.q_cmd).solver_fault_latched
    assert client._seq == failed_seq  # No automatic retry/commit of stale work.
    client.reset(client.ctrl.q_cmd)
    assert not client._fault_latched
    recovered = client.update(np.zeros(6), q_meas=client.ctrl.q_cmd, auto_commit=False)
    assert not recovered.solver_fault_latched


def test_peer_death_wakes_waiter(client):
    client._proc.kill()
    started = time.monotonic()
    assert not client._wait_seq(999, timeout_s=0.5)
    assert time.monotonic() - started < 0.1
    assert client._last_wait_reason == "process_exit"


def test_concurrent_setters_cannot_overwrite_inflight_request(client):
    def commands(value):
        return [client._command(P.CMD_RESET, q_meas=np.full(8, value), timeout_s=0.020)
                for _ in range(10)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(commands, 1.0)
        second = pool.submit(commands, 2.0)
        assert all(first.result()) and all(second.result())
    assert client._seq == 20


def test_peer_exits_when_owner_closes_notification_socket(client):
    client._notify.close()
    client._notify = None
    client._proc.wait(timeout=0.5)
    assert client._proc.returncode == 0


def test_controller_close_releases_only_owned_cached_solvers():
    calls = []
    first, second, other = object(), object(), object()
    controller = JointIkController.__new__(JointIkController)
    controller._native = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    controller.core = SimpleNamespace(backend=first, _backend_qp2=second)
    controller.kin = SimpleNamespace(_qp_backend_cache={1: first, 2: second, 3: other})
    controller.close()
    controller.close()
    assert calls == ["shutdown"]
    assert controller.core is None
    assert controller.kin._qp_backend_cache == {3: other}
