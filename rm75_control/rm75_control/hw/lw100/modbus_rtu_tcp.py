"""Modbus RTU client over TCP transparent serial (USR-TCP232-304)."""

from __future__ import annotations

import socket
import struct
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
    inter_frame_delay_s: float = 0.05


class ModbusRtuTcpClient:
    """Send Modbus RTU ADUs through a TCP serial server (no MBAP header)."""

    FC_READ_HOLDING = 0x03
    FC_WRITE_SINGLE = 0x06
    FC_WRITE_MULTIPLE = 0x10

    def __init__(self, config: ModbusRtuTcpConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.timeout_s,
        )
        sock.settimeout(self.config.timeout_s)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> ModbusRtuTcpClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _drain_rx(self) -> None:
        """Discard any stale bytes in the TCP receive buffer."""
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

    def send_raw(self, request: bytes) -> bytes:
        """Send RTU ADU (without CRC) and return raw bytes from TCP (diagnostics)."""
        if self._sock is None:
            raise ModbusRtuError("not connected")
        self._drain_rx()
        req = append_crc(request)
        self._sock.sendall(req)
        time.sleep(self.config.inter_frame_delay_s)
        return self._read_response_raw()

    def _send_receive(self, request: bytes) -> bytes:
        if self._sock is None:
            raise ModbusRtuError("not connected")
        req = append_crc(request)
        last_err: Exception | None = None
        for attempt in range(max(1, self.config.retries)):
            try:
                self._drain_rx()
                self._sock.sendall(req)
                time.sleep(self.config.inter_frame_delay_s)
                response = self._read_response()
                if not verify_crc(response):
                    raise ModbusRtuError(f"CRC mismatch on response: {response.hex()}")
                if response[0] != request[0]:
                    raise ModbusRtuError(
                        f"slave mismatch: sent id={request[0]}, got id={response[0]}"
                    )
                fc = response[1]
                if fc & 0x80:
                    exc_code = response[2] if len(response) > 2 else -1
                    raise ModbusRtuError(f"Modbus exception fc=0x{request[1]:02x} code={exc_code}")
                return response
            except (TimeoutError, socket.timeout, OSError, ModbusRtuError) as err:
                last_err = err
                if attempt + 1 < self.config.retries:
                    time.sleep(self.config.inter_frame_delay_s)
                    continue
                raise ModbusRtuError(str(last_err)) from last_err
        raise ModbusRtuError("unreachable")

    def _read_response_raw(self) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(256)
            except socket.timeout:
                break
            if not chunk:
                break
            # Drop leading idle/noise nulls before a real ADU starts.
            if not buf:
                chunk = chunk.lstrip(b"\x00")
                if not chunk:
                    continue
            buf.extend(chunk)
            if len(buf) >= 5 and verify_crc(bytes(buf)):
                return bytes(buf)
            # Truncate leading noise if slave id never appears.
            if len(buf) > 64:
                buf.clear()
        return bytes(buf)

    def _read_response(self) -> bytes:
        buf = self._read_response_raw()
        if not buf:
            raise ModbusRtuError("response timeout")
        if not verify_crc(buf):
            raise ModbusRtuError(f"CRC mismatch on response: {buf.hex()}")
        return buf

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
        if resp[1] != self.FC_READ_HOLDING:
            raise ModbusRtuError(f"unexpected function code {resp[1]}")
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
        resp = self._send_receive(req)
        if resp[1] != self.FC_WRITE_SINGLE:
            raise ModbusRtuError(f"unexpected function code {resp[1]}")

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
        resp = self._send_receive(req)
        if resp[1] != self.FC_WRITE_MULTIPLE:
            raise ModbusRtuError(f"unexpected function code {resp[1]}")
