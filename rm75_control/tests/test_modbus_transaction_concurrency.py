"""Exercise worker/stop overlap without a network connection or hardware."""

from concurrent.futures import ThreadPoolExecutor
import socket
import struct
import threading

import pytest

from rm75_control.hw.lw100.modbus_rtu_tcp import (
    ModbusRtuError,
    ModbusRtuTcpClient,
    ModbusRtuTcpConfig,
    append_crc,
)


class ControlledSocket:
    def __init__(self, *, hold_receive=False):
        self.blocking = True
        self.closed = False
        self.sent = []
        self.drain_count = 0
        self.receive_started = threading.Event()
        self.second_send = threading.Event()
        self.release_receive = threading.Event()
        if not hold_receive:
            self.release_receive.set()

    def setsockopt(self, *args):
        pass

    def settimeout(self, timeout):
        self.blocking = timeout != 0

    def setblocking(self, blocking):
        self.blocking = blocking

    def sendall(self, request):
        if self.closed:
            raise OSError("closed")
        self.sent.append(request)
        if len(self.sent) > 1:
            self.second_send.set()

    def recv(self, size):
        if not self.blocking:
            self.drain_count += 1
            raise BlockingIOError()
        self.receive_started.set()
        if not self.release_receive.wait(2):
            raise socket.timeout()
        if self.closed:
            raise OSError("closed")
        request = self.sent[-1]
        if request[1] == 3:
            return append_crc(bytes([request[0], 3, 2]) + struct.pack(">H", 123))
        return request  # FC06 echoes the write.

    def shutdown(self, how):
        self.closed = True

    def close(self):
        self.closed = True


def client_with(sock, *, retries=1):
    client = ModbusRtuTcpClient(ModbusRtuTcpConfig(
        host="unused", timeout_s=1, retries=retries, inter_frame_delay_s=0,
    ))
    client._sock = sock
    return client


class ObservedTransactionLock:
    """Signal when the second caller reaches the transaction queue."""

    def __init__(self):
        self.lock = threading.RLock()
        self.attempt_lock = threading.Lock()
        self.attempts = 0
        self.queued = threading.Event()

    def __enter__(self):
        with self.attempt_lock:
            self.attempts += 1
            if self.attempts == 2:
                self.queued.set()
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()


def test_stop_write_waits_for_worker_read_response():
    sock = ControlledSocket(hold_receive=True)
    client = client_with(sock)
    stop_started = threading.Event()

    def stop_write():
        stop_started.set()
        client.write_register(24, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(client.read_holding_registers, 0x1000, 1)
        try:
            assert sock.receive_started.wait(1)
            stop = pool.submit(stop_write)
            assert stop_started.wait(1)
            assert not sock.second_send.wait(.05)
        finally:
            sock.release_receive.set()
        assert read.result(timeout=1) == [123]
        assert stop.result(timeout=1) is None
    assert [request[1] for request in sock.sent] == [3, 6]


def test_emergency_close_and_reconnect_do_not_replay_interrupted_command(monkeypatch):
    old = ControlledSocket(hold_receive=True)
    client = client_with(old, retries=3)
    new = ControlledSocket()
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: new)

    with ThreadPoolExecutor(max_workers=3) as pool:
        # Simulate a pre-stop nonzero command waiting for its ACK.
        stale = pool.submit(client.write_register, 24, 900)
        try:
            assert old.receive_started.wait(1)
            close = pool.submit(client.close)
            assert close.result(timeout=.2) is None
            assert old.closed
            assert not stale.done()  # close did not wait for receive to unwind.
            client.connect()
            stopped = pool.submit(client.write_register, 24, 0)
        finally:
            old.release_receive.set()
        with pytest.raises(ModbusRtuError, match="closed"):
            stale.result(timeout=1)
        assert stopped.result(timeout=1) is None

    assert len(old.sent) == 1
    assert new.sent == [append_crc(struct.pack(">BBHH", 1, 6, 24, 0))]


@pytest.mark.parametrize("queued_operation", ["write", "recover"])
def test_reconnect_invalidates_operations_queued_on_old_connection(monkeypatch, queued_operation):
    old = ControlledSocket(hold_receive=True)
    client = client_with(old)
    queue_lock = ObservedTransactionLock()
    client._transaction_lock = queue_lock
    new = ControlledSocket()
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: new)

    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(client.read_holding_registers, 0x1000, 1)
        try:
            assert old.receive_started.wait(1)
            if queued_operation == "write":
                queued = pool.submit(client.write_register, 24, 900)
            else:
                queued = pool.submit(client.recover)
            assert queue_lock.queued.wait(1)
            client.close()
            client.connect()
            connect_drains = new.drain_count
        finally:
            old.release_receive.set()
        with pytest.raises(ModbusRtuError, match="closed"):
            read.result(timeout=1)
        if queued_operation == "write":
            with pytest.raises(ModbusRtuError, match="connection changed"):
                queued.result(timeout=1)
        else:
            assert queued.result(timeout=1) is None

    assert new.sent == []
    assert new.drain_count == connect_drains
    client.write_register(24, 0)  # A call originating after reconnect is valid.
    assert new.sent == [append_crc(struct.pack(">BBHH", 1, 6, 24, 0))]
