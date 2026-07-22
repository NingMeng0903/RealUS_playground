"""Unit tests for LW100 geometry / Modbus RTU helpers (no hardware)."""

from __future__ import annotations

import struct

import pytest

from rm75_control.hw.lw100.geometry import mm_to_position_command, position_command_to_mm
from rm75_control.hw.lw100.modbus_rtu_tcp import append_crc, crc16_modbus, verify_crc
from rm75_control.hw.lw100.registers import (
    P_FA71_SLAVE,
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
