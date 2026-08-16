"""USB vs Bluetooth Xbox layout classification and remap."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
    map_pad_to_world_lin_tool_ang,
)
from rm75_control.control.joint_admittance_8dof.teleop.pad_layout import (
    LAYOUT_BT_REST0,
    LAYOUT_BT_XPADNEO,
    LAYOUT_WIRED_XPAD,
    apply_layout,
    classify_layout,
    layout_bt_rest0,
    layout_bt_xpadneo,
    layout_wired_xpad,
    pick_device_index,
)
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import PadState


def _cmd(axes, buttons=None, layout=None):
    raw_btn = np.zeros(8) if buttons is None else np.asarray(buttons, dtype=float)
    if layout is None:
        layout = classify_layout(axes)
    mapped_ax, mapped_btn = apply_layout(axes, raw_btn, layout)
    return map_pad_to_world_lin_tool_ang(
        PadState(axes=mapped_ax, buttons=mapped_btn),
        GamepadTwistConfig(trans_m_s=0.12, rot_rad_s=0.60, deadzone=0.18),
    )


def test_pick_device_prefers_usb_over_bluetooth() -> None:
    names = ["Xbox Wireless Controller", "Microsoft X-Box 360 pad"]
    assert pick_device_index(names) == 1
    assert pick_device_index(["Xbox Wireless Controller"]) == 0


def test_classify_matches_logged_rest_poses() -> None:
    wired = np.array([0.02, 0.02, -1.0, 0.01, 0.02, -1.0])
    bt_neo = np.array([0.02, -0.01, 0.0, 0.0, -1.0, -1.0])
    bt_zero = np.zeros(6)
    assert classify_layout(wired, name="Microsoft X-Box 360 pad").name == LAYOUT_WIRED_XPAD
    assert classify_layout(bt_neo, name="Xbox Wireless Controller").name == LAYOUT_BT_XPADNEO
    assert classify_layout(bt_zero, name="Xbox Wireless Controller").name == LAYOUT_BT_REST0


def test_rest_does_not_invent_motion() -> None:
    cases = (
        (np.array([0.02, 0.02, -1.0, 0.01, 0.02, -1.0]), layout_wired_xpad()),
        (np.array([0.02, -0.01, 0.0, 0.0, -1.0, -1.0]), layout_bt_xpadneo()),
        (np.zeros(6), layout_bt_rest0()),
    )
    for axes, layout in cases:
        v, w = _cmd(axes, layout=layout)
        assert np.allclose(v, 0.0, atol=1e-12), (layout.name, v)
        assert np.allclose(w, 0.0, atol=1e-12), (layout.name, w)


def test_bt_xpadneo_left_stick_left_is_plus_y() -> None:
    # Physical: LX LY RX RY RT LT. Left stick left = axis0 = -1.
    axes = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, -1.0])
    v, w = _cmd(axes, layout=layout_bt_xpadneo())
    assert v[1] > 0.05
    assert abs(v[0]) < 1e-12
    assert abs(v[2]) < 1e-12
    assert np.allclose(w, 0.0)


def test_bt_xpadneo_physical_axis5_is_lt_minus_z() -> None:
    # This Series X: axis5 = LT (−Z), axis4 = RT (−wz).
    rest = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
    lt = rest.copy()
    lt[5] = 1.0
    v, w = _cmd(lt, layout=layout_bt_xpadneo())
    assert v[2] < -0.05
    assert np.allclose(w, 0.0)
    rt = rest.copy()
    rt[4] = 1.0
    v2, w2 = _cmd(rt, layout=layout_bt_xpadneo())
    assert abs(v2[2]) < 1e-12
    assert w2[2] < -0.05


def test_bt_rest0_lt_press_is_minus_z() -> None:
    axes = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # wired-order, 0..1 trigger
    v, _w = _cmd(axes, layout=layout_bt_rest0())
    assert v[2] < -0.05


def test_bt_xpadneo_lb_is_physical_button_6() -> None:
    axes = np.array([0.02, 0.0, 0.0, 0.0, -1.0, -1.0])
    buttons = np.zeros(16)
    buttons[6] = 1.0
    v, _w = _cmd(axes, buttons=buttons, layout=layout_bt_xpadneo())
    assert v[2] > 0.05
    buttons[6] = 0.0
    buttons[4] = 1.0  # wired LB index must not lift on BT
    v_wrong, _ = _cmd(axes, buttons=buttons, layout=layout_bt_xpadneo())
    assert abs(v_wrong[2]) < 1e-12
