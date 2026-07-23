"""Modbus RTU client over TCP transparent serial (USR-TCP232-304)."""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS (poly 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(frame: bytes) -> bytes:
    crc = crc16_modbus(frame)
    return frame + struct.pack("<H", crc)


def verify_crc(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    payload, crc_bytes = frame[:-2], frame[-2:]
    expected = struct.unpack("<H", crc_bytes)[0]
    return crc16_modbus(payload) == expected


class ModbusRtuError(RuntimeError):
    """Modbus exception or transport failure."""


@dataclass
class ModbusRtuTcpConfig:
    host: str
    port: int = 8234
    slave_id: int = 1
    timeout_s: float = 1.0
    retries: int = 2
    # At 115200, 3.5 RTU chars ≈ 0.3 ms. USR-TCP232 needs a few ms of turnaround.
    # 50 ms was historically used for flaky links but capped the rail loop at ~10 Hz
    # (read+write), which looked like one update per motor revolution in the twin.
    inter_frame_delay_s: float = 0.002


def _expected_response_len(request: bytes) -> int | None:
    """Exact RTU response length for a well-formed request (incl. CRC).

    Returning early on *any* CRC-valid prefix was the desync bug: a stale
    7-byte write-ACK (FC06) or 1-register read would satisfy ``verify_crc``
    while we were still waiting for a 2-register read → 'unexpected function
    code 6' / 'unexpected read length'.
    """
    if len(request) < 2:
        return None
    fc = request[1]
    if fc == 0x03:  # read holding
        if len(request) < 6:
            return None
        count = struct.unpack(">H", request[4:6])[0]
        return 5 + 2 * int(count)  # id+fc+bc+data+crc
    if fc == 0x06:  # write single
        return 8
    if fc == 0x10:  # write multiple
        return 8
    return None


class ModbusRtuTcpClient:
    """Send Modbus RTU ADUs through a TCP serial server (no MBAP header)."""

    FC_READ_HOLDING = 0x03
    FC_WRITE_SINGLE = 0x06
    FC_WRITE_MULTIPLE = 0x10

    def __init__(self, config: ModbusRtuTcpConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None
        # One TCP/RTU link — all callers (rail worker, safety, start) must serialize.
        self._io_lock = threading.RLock()

    def __enter__(self) -> ModbusRtuTcpClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _drain_rx(self) -> None:
        """Discard any stale bytes in the TCP receive buffer. Caller holds ``_io_lock``."""
        if self._sock is None:
            return
        self._sock.setblocking(False)
        try:
            while True:
                chunk = self._sock.recv(256)
                if not chunk:
                    break
        except (BlockingIOError, InterruptedError):
            pass
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self.config.timeout_s)

    def connect(self) -> None:
        with self._io_lock:
            if self._sock is not None:
                return
            sock = socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.timeout_s,
            )
            sock.settimeout(self.config.timeout_s)
            self._sock = sock
            # Drop any bytes left from a previous crashed client on the USR.
            time.sleep(0.02)
            self._drain_rx()

    def close(self) -> None:
        """Drop the TCP link. Unblocks any thread blocked in ``recv`` via shutdown."""
        with self._io_lock:
            sock = self._sock
            self._sock = None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    def reconnect(self) -> None:
        """Close + open, draining stale USR RX (call after hard desync / Ctrl+C)."""
        self.close()
        time.sleep(0.05)
        self.connect()

    def recover(self) -> None:
        """Drain + short quiet period after a framing error."""
        with self._io_lock:
            if self._sock is None:
                return
            try:
                self._drain_rx()
            except Exception:
                pass
        time.sleep(max(0.005, float(self.config.inter_frame_delay_s) * 3.0))
        with self._io_lock:
            if self._sock is None:
                return
            try:
                self._drain_rx()
            except Exception:
                pass

    def send_raw(self, request: bytes) -> bytes:
        """Send RTU ADU (without CRC) and return raw bytes from TCP (diagnostics)."""
        return self._send_receive(request)

    def _send_receive(self, request: bytes) -> bytes:
        req = append_crc(request)
        expected = _expected_response_len(request)
        last_err: Exception | None = None
        for attempt in range(max(1, self.config.retries)):
            sock: socket.socket | None = None
            try:
                with self._io_lock:
                    if self._sock is None:
                        raise ModbusRtuError("not connected")
                    self._drain_rx()
                    self._sock.sendall(req)
                    sock = self._sock
                # Recv OUTSIDE the lock so close()/Ctrl+C can shutdown the socket.
                time.sleep(self.config.inter_frame_delay_s)
                response = self._read_response_on(
                    sock, expected_len=expected
                )
                if response[0] != request[0]:
                    raise ModbusRtuError(
                        f"slave mismatch: sent id={request[0]}, got id={response[0]}"
                    )
                fc = response[1]
                if fc & 0x80:
                    exc_code = response[2] if len(response) > 2 else -1
                    raise ModbusRtuError(
                        f"Modbus exception fc=0x{request[1]:02x} code={exc_code}"
                    )
                if (fc & 0x7F) != request[1]:
                    raise ModbusRtuError(
                        f"unexpected function code {fc} (want {request[1]})"
                    )
                return response
            except (TimeoutError, socket.timeout, OSError, ModbusRtuError, AttributeError) as err:
                last_err = err if isinstance(err, ModbusRtuError) else ModbusRtuError(str(err))
                # Brief recover without holding lock across long sleeps.
                try:
                    self.recover()
                except Exception:
                    pass
                if attempt + 1 < self.config.retries:
                    continue
                raise ModbusRtuError(str(last_err)) from last_err
        raise ModbusRtuError("unreachable")

    def _read_response_on(
        self, sock: socket.socket | None, *, expected_len: int | None = None
    ) -> bytes:
        if sock is None:
            raise ModbusRtuError("not connected")
        buf = bytearray()
        deadline = time.monotonic() + self.config.timeout_s
        sid = int(self.config.slave_id) & 0xFF
        try:
            sock.settimeout(self.config.timeout_s)
        except Exception:
            raise ModbusRtuError("not connected")
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(256)
            except socket.timeout:
                break
            except OSError as exc:
                raise ModbusRtuError(f"socket closed: {exc}") from exc
            if not chunk:
                break
            if not buf:
                chunk = chunk.lstrip(b"\x00")
                if not chunk:
                    continue
            buf.extend(chunk)
            while buf and buf[0] != sid:
                del buf[0]
            if not buf:
                continue
            if expected_len is not None:
                if len(buf) >= expected_len:
                    frame = bytes(buf[:expected_len])
                    if verify_crc(frame):
                        return frame
                    del buf[0]
                continue
            if len(buf) >= 5 and verify_crc(bytes(buf)):
                return bytes(buf)
            if len(buf) > 64:
                buf.clear()
        if not buf:
            raise ModbusRtuError("response timeout")
        if expected_len is not None and len(buf) != expected_len:
            raise ModbusRtuError(
                f"short response ({len(buf)}/{expected_len}): {buf.hex()}"
            )
        if not verify_crc(buf):
            raise ModbusRtuError(f"CRC mismatch on response: {buf.hex()}")
        return bytes(buf)

    def _read_response_raw(self, *, expected_len: int | None = None) -> bytes:
        with self._io_lock:
            sock = self._sock
        try:
            return self._read_response_on(sock, expected_len=expected_len)
        except ModbusRtuError:
            return b""

    def _read_response(self, *, expected_len: int | None = None) -> bytes:
        with self._io_lock:
            sock = self._sock
        return self._read_response_on(sock, expected_len=expected_len)

    def read_holding_registers(self, address: int, count: int = 1) -> list[int]:
        addr = int(address) & 0xFFFF
        cnt = int(count) & 0xFFFF
        req = struct.pack(
            ">BBHH",
            self.config.slave_id,
            self.FC_READ_HOLDING,
            addr,
            cnt,
        )
        resp = self._send_receive(req)
        byte_count = resp[2]
        data = resp[3 : 3 + byte_count]
        if len(data) != byte_count or byte_count != 2 * count:
            raise ModbusRtuError(f"unexpected read length: {resp.hex()}")
        return list(struct.unpack(f">{count}H", data))

    def write_register(self, address: int, value: int) -> None:
        addr = int(address) & 0xFFFF
        val = int(value) & 0xFFFF
        req = struct.pack(
            ">BBHH",
            self.config.slave_id,
            self.FC_WRITE_SINGLE,
            addr,
            val,
        )
        self._send_receive(req)

    def write_registers(self, address: int, values: list[int]) -> None:
        if not values:
            return
        addr = int(address) & 0xFFFF
        payload = b"".join(struct.pack(">H", int(v) & 0xFFFF) for v in values)
        req = struct.pack(
            ">BBHHB",
            self.config.slave_id,
            self.FC_WRITE_MULTIPLE,
            addr,
            len(values),
            len(payload),
        ) + payload
        self._send_receive(req)
