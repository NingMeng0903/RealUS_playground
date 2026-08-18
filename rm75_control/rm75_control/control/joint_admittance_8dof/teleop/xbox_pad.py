"""Minimal Xbox / pygame joystick reader for 8-DOF QPIK teleop.

Physical SDL order depends on USB vs Bluetooth.  This module picks the
device (wired wins) and remaps into the logical order consumed by
``gamepad_twist``: left stick 0/1, LT 2, right stick 3/4, RT 5; LB=4, RB=5.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.teleop.pad_layout import (
    PadLayout,
    apply_layout,
    classify_layout,
    load_pinned_layout,
    pick_device_index,
    transport_from_name,
)

XBOX_BUTTON_A = 0
XBOX_BUTTON_B = 1
XBOX_BUTTON_X = 2
XBOX_BUTTON_Y = 3
XBOX_BUTTON_LB = 4
XBOX_BUTTON_RB = 5
XBOX_BUTTON_BACK = 6
XBOX_BUTTON_START = 7

_N_AXES = 6
_N_BUTTONS = 8


def _require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise ImportError("pygame is required for gamepad teleop") from exc
    return pygame


def _init_joystick_pygame():
    """Full pygame init. SDL steals SIGINT — give it back afterwards."""
    pygame = _require_pygame()
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        pygame.init()
        pygame.joystick.init()
    finally:
        try:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
        except Exception:
            pass
    return pygame


@dataclass
class PadState:
    axes: np.ndarray
    buttons: np.ndarray

    def button(self, index: int) -> bool:
        idx = int(index)
        if idx < 0 or idx >= int(self.buttons.size):
            return False
        return bool(self.buttons[idx])


def list_joystick_names() -> list[str]:
    pygame = _init_joystick_pygame()
    names = []
    for i in range(int(pygame.joystick.get_count())):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        names.append(joy.get_name())
        joy.quit()
    return names


class XboxPad:
    """Read Xbox axes/buttons, remapped to the wired logical layout."""

    def __init__(
        self,
        *,
        device_index: int | None = 0,
        allow_missing: bool = True,
        auto_select: bool = True,
        layout: PadLayout | None = None,
        pin_layout: bool = True,
    ) -> None:
        pygame = _init_joystick_pygame()
        self._pygame = pygame
        self._joy = None
        self._closed = False
        self._layout = layout
        self.name = ""
        self.transport = "unknown"
        self.device_index = -1
        count = int(pygame.joystick.get_count())
        names = [pygame.joystick.Joystick(i).get_name() for i in range(count)]
        if auto_select and names:
            idx = pick_device_index(names)
        else:
            idx = 0 if device_index is None else int(device_index)
        if count <= idx or idx < 0:
            if not allow_missing:
                raise RuntimeError(
                    f"no joystick at index {idx} (found {count})"
                )
            return
        self._joy = pygame.joystick.Joystick(int(idx))
        self._joy.init()
        self.device_index = int(idx)
        self.name = self._joy.get_name()
        self.transport = transport_from_name(self.name)
        guid = ""
        try:
            guid = str(self._joy.get_guid())
        except Exception:
            pass
        if self._layout is None and pin_layout:
            self._layout = load_pinned_layout(name=self.name, guid=guid or None)

    @property
    def connected(self) -> bool:
        return self._joy is not None and not self._closed

    @property
    def layout(self) -> PadLayout | None:
        return self._layout

    def describe(self) -> str:
        layout_name = self._layout.name if self._layout is not None else "pending"
        return (
            f"pad[{self.device_index}] {self.name!r} "
            f"transport={self.transport} layout={layout_name}"
        )

    def read(self) -> PadState:
        axes = np.zeros(_N_AXES, dtype=float)
        buttons = np.zeros(_N_BUTTONS, dtype=float)
        if self._joy is None or self._closed:
            return PadState(axes=axes, buttons=buttons)
        self._pygame.event.pump()
        n_ax = int(self._joy.get_numaxes())
        n_btn = int(self._joy.get_numbuttons())
        raw_ax = np.array(
            [float(self._joy.get_axis(i)) for i in range(n_ax)], dtype=float
        )
        raw_btn = np.array(
            [1.0 if self._joy.get_button(i) else 0.0 for i in range(n_btn)],
            dtype=float,
        )
        if self._layout is None:
            self._layout = classify_layout(raw_ax, name=self.name)
        mapped_ax, mapped_btn = apply_layout(raw_ax, raw_btn, self._layout)
        axes[: min(_N_AXES, mapped_ax.size)] = mapped_ax[:_N_AXES]
        buttons[: min(_N_BUTTONS, mapped_btn.size)] = mapped_btn[:_N_BUTTONS]
        return PadState(axes=axes, buttons=buttons)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._joy is not None:
                self._joy.quit()
        except Exception:
            pass
        self._joy = None


class FakePad:
    """Deterministic pad for tests / dry-run. Axes are already logical."""

    def __init__(
        self,
        axes: np.ndarray | None = None,
        buttons: np.ndarray | None = None,
    ) -> None:
        self.axes = (
            np.zeros(_N_AXES, dtype=float)
            if axes is None
            else np.asarray(axes, dtype=float).reshape(-1)
        )
        self.buttons = (
            np.zeros(_N_BUTTONS, dtype=float)
            if buttons is None
            else np.asarray(buttons, dtype=float).reshape(-1)
        )
        self.closed = False
        self.name = "fake"
        self.transport = "fake"
        self.device_index = 0

    @property
    def connected(self) -> bool:
        return not self.closed

    def read(self) -> PadState:
        ax = np.zeros(_N_AXES, dtype=float)
        btn = np.zeros(_N_BUTTONS, dtype=float)
        ax[: min(_N_AXES, self.axes.size)] = self.axes[:_N_AXES]
        btn[: min(_N_BUTTONS, self.buttons.size)] = self.buttons[:_N_BUTTONS]
        return PadState(axes=ax, buttons=btn)

    def describe(self) -> str:
        return "pad[fake] transport=fake layout=logical"

    def close(self) -> None:
        self.closed = True
