"""Unit tests for LW100 geometry / Modbus RTU helpers (no hardware)."""

from __future__ import annotations

import struct

import pytest

from rm75_control.hw.lw100.geometry import mm_to_position_command, position_command_to_mm
from rm75_control.hw.lw100.modbus_rtu_tcp import append_crc, crc16_modbus, verify_crc
from rm75_control.hw.lw100.registers import (
    P_FA5_SPEED_KP_HZ,
    P_FA6_SPEED_TI_MS,
    P_FA7_TORQUE_FILTER,
    P_FA71_SLAVE,
    P_FA8_SPEED_FILTER,
    P_FC15_DI_FORCE1,
    P_FD0_ABS_INC,
    P_FD2_P1_REVS,
    RegisterMap,
)


def test_crc16_known_vector():
    # Classic Modbus test vector: "123456789" -> 0x4B37
    data = b"123456789"
    assert crc16_modbus(data) == 0x4B37


def test_append_and_verify_crc():
    frame = b"\x01\x03\x00\x00\x00\x01"
    full = append_crc(frame)
    assert verify_crc(full)
    assert len(full) == len(frame) + 2


def test_register_map_defaults():
    m = RegisterMap()
    assert m.addr(P_FA5_SPEED_KP_HZ) == 5
    assert m.addr(P_FA6_SPEED_TI_MS) == 6
    assert m.addr(P_FA7_TORQUE_FILTER) == 7
    assert m.addr(P_FA8_SPEED_FILTER) == 8
    assert m.addr(P_FA71_SLAVE) == 71
    assert m.addr(P_FD0_ABS_INC) == 512
    assert m.addr(P_FD2_P1_REVS) == 514
    assert m.addr(P_FC15_DI_FORCE1) == 256 + 15


def test_register_map_custom_bases():
    m = RegisterMap(bases={"FA": 0, "FD": 512, "FC": 300})
    assert m.addr(P_FC15_DI_FORCE1) == 315


def test_mm_to_position_command_1610():
    cmd = mm_to_position_command(5.0, lead_mm=10.0, gear_ratio=1.0, pulses_per_rev=10_000)
    assert cmd.revolutions == 0
    assert cmd.pulses == 5000
    assert position_command_to_mm(cmd) == pytest.approx(5.0)


def test_mm_to_position_command_whole_rev():
    cmd = mm_to_position_command(10.0, lead_mm=10.0, pulses_per_rev=10_000)
    assert cmd.revolutions == 1
    assert cmd.pulses == 0


def test_mm_negative():
    cmd = mm_to_position_command(-2.5, lead_mm=10.0, pulses_per_rev=10_000)
    assert cmd.revolutions == 0
    assert cmd.pulses == -2500


def test_write_single_request_crc():
    req = struct.pack(">BBHH", 1, 0x06, 71, 1)
    full = append_crc(req)
    assert verify_crc(full)


def test_read_motion_fast_is_three_registers():
    from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
    from rm75_control.hw.lw100.registers import MONITOR_SPEED_RPM

    class _Client:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def read_holding_registers(self, addr, count):
            self.calls.append((int(addr), int(count)))
            return [0] * int(count)

    drive = LW100Drive.__new__(LW100Drive)
    drive._client = _Client()
    drive._counts_bias = 0
    drive._counts0 = 0
    drive.config = LW100DriveConfig()
    rpm, rail_m = drive.read_motion_fast()
    assert drive._client.calls == [(MONITOR_SPEED_RPM, 3)]
    assert rpm == 0
    assert rail_m == pytest.approx(0.0)
    drive._client.calls.clear()
    drive.read_motion_and_di_fast()
    assert drive._client.calls == [(MONITOR_SPEED_RPM, 16)]


def test_modbus_connect_sets_tcp_nodelay(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuTcpClient, ModbusRtuTcpConfig

    opts: list[tuple[int, int, int]] = []

    class _Sock:
        def setsockopt(self, level, opt, value):
            opts.append((int(level), int(opt), int(value)))

        def settimeout(self, _timeout):
            return None

        def setblocking(self, _flag):
            return None

        def recv(self, _n):
            raise BlockingIOError

    def _connect(_addr, timeout=None):
        return _Sock()

    monkeypatch.setattr(socket, "create_connection", _connect)
    client = ModbusRtuTcpClient(ModbusRtuTcpConfig(host="127.0.0.1", port=8234))
    client.connect()
    assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in opts
