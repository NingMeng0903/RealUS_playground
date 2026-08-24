"""Minimal Xbox / pygame joystick reader for 8-DOF QPIK teleop.

Physical SDL order depends on USB vs Bluetooth.  Default pick still
prefers USB when both are present.  Peirastic teleop passes
``require_transport="bluetooth"`` and uses the kernel Bus/GUID, not the
pygame display name: this bench is kernel ``Xbox Wireless Controller``
and SDL ``Xbox Series X Controller``.
Logical order consumed by ``gamepad_twist``: left stick 0/1, LT 2, right
stick 3/4, RT 5; LB=4, RB=5.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.teleop.pad_layout import (
    PadLayout,
    apply_layout,
    bluetooth_link_live,
    classify_layout,
    classify_link_transport,
    load_pinned_layout,
    pick_device_index,
)

XBOX_BUTTON_A = 0
XBOX_BUTTON_B = 1
XBOX_BUTTON_X = 2
XBOX_BUTTON_Y = 3
XBOX_BUTTON_LB = 4
XBOX_BUTTON_RB = 5
XBOX_BUTTON_L3 = 6
XBOX_BUTTON_R3 = 7

_N_AXES = 6
_N_BUTTONS = 16
_PYGAME_READY = False


def _require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise ImportError("pygame is required for gamepad teleop") from exc
    return pygame


def _init_joystick_pygame():
    """Init pygame once. SDL steals SIGINT — give it back afterwards."""
    global _PYGAME_READY
    pygame = _require_pygame()
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        if not _PYGAME_READY:
            pygame.init()
            _PYGAME_READY = True
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
        require_transport: str | None = None,
    ) -> None:
        pygame = _init_joystick_pygame()
        self._pygame = pygame
        self._joy = None
        self._closed = False
        self._layout = layout
        self._pin_layout = bool(pin_layout)
        self._allow_missing = bool(allow_missing)
        self._auto_select = bool(auto_select)
        self._requested_index = device_index
        self.require_transport = (
            None if require_transport is None else str(require_transport)
        )
        self.name = ""
        self.guid = ""
        self.transport = "unknown"
        self.link_transport = "none"
        self.device_index = -1
        self._instance_id = None
        self._next_open_s = 0.0
        self._open_or_refresh(force=True)

    @property
    def connected(self) -> bool:
        if self._joy is None or self._closed:
            return False
        if self.require_transport:
            return self.link_transport == self.require_transport
        return True

    @property
    def layout(self) -> PadLayout | None:
        return self._layout

    def describe(self) -> str:
        layout_name = self._layout.name if self._layout is not None else "pending"
        req = self.require_transport or "any"
        return (
            f"pad[{self.device_index}] {self.name!r} "
            f"transport={self.link_transport} layout={layout_name} "
            f"require={req} live={int(self.connected)}"
        )

    def read(self) -> PadState:
        axes = np.zeros(_N_AXES, dtype=float)
        buttons = np.zeros(_N_BUTTONS, dtype=float)
        if self._closed:
            return PadState(axes=axes, buttons=buttons)
        pump = getattr(getattr(self._pygame, "event", None), "pump", None)
        if pump is not None:
            pump()
        self._open_or_refresh()
        if self._joy is None or not self.connected:
            return PadState(axes=axes, buttons=buttons)
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
        self._drop_joy()

    def _enumerate(self) -> list[dict]:
        out: list[dict] = []
        count = int(self._pygame.joystick.get_count())
        for i in range(count):
            joy = self._pygame.joystick.Joystick(i)
            name = str(joy.get_name())
            guid = ""
            try:
                guid = str(joy.get_guid())
            except Exception:
                pass
            kind = classify_link_transport(name=name, guid=guid)
            out.append(
                {"index": i, "name": name, "guid": guid, "transport": kind}
            )
        return out

    def _open_or_refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        removed = self._removed_instance()
        if self._joy is not None:
            live = self._probe_live()
            if removed or not live:
                self._drop_joy()
            else:
                return
        if not force and now < self._next_open_s:
            return
        self._next_open_s = now + 0.25
        found = self._enumerate()
        if not found:
            return
        names = [row["name"] for row in found]
        kinds = [row["transport"] for row in found]
        try:
            if self._auto_select:
                idx = pick_device_index(
                    names,
                    transports=kinds,
                    require_transport=self.require_transport,
                )
            else:
                idx = 0 if self._requested_index is None else int(self._requested_index)
                if idx < 0 or idx >= len(found):
                    raise ValueError("joystick index out of range")
                if (
                    self.require_transport
                    and kinds[idx] != self.require_transport
                ):
                    raise ValueError("joystick is not the required transport")
        except ValueError:
            if not self._allow_missing:
                raise
            return
        row = found[idx]
        if self.require_transport and row["transport"] != self.require_transport:
            return
        if self.require_transport == "bluetooth" and not bluetooth_link_live(
            guid=row["guid"]
        ):
            return
        joy = self._pygame.joystick.Joystick(int(row["index"]))
        joy.init()
        self._joy = joy
        self.device_index = int(row["index"])
        self.name = str(row["name"])
        self.guid = str(row["guid"])
        self.link_transport = str(row["transport"])
        self.transport = self.link_transport
        try:
            self._instance_id = int(joy.get_instance_id())
        except Exception:
            self._instance_id = None
        if self._layout is None and self._pin_layout:
            self._layout = load_pinned_layout(name=self.name, guid=self.guid or None)

    def _probe_live(self) -> bool:
        if self._joy is None:
            self.link_transport = "none"
            return False
        guid = self.guid
        try:
            guid = str(self._joy.get_guid())
            self.guid = guid
        except Exception:
            self.link_transport = "none"
            return False
        kind = classify_link_transport(name=self.name, guid=guid)
        self.link_transport = kind
        self.transport = kind
        if self.require_transport == "bluetooth":
            return kind == "bluetooth" and bluetooth_link_live(guid=guid)
        if self.require_transport:
            return kind == self.require_transport
        return True

    def _removed_instance(self) -> bool:
        if self._joy is None or self._instance_id is None:
            return False
        removed = getattr(self._pygame, "JOYDEVICEREMOVED", None)
        ev = getattr(self._pygame, "event", None)
        if removed is None or ev is None:
            return False
        getter = getattr(ev, "get", None)
        if getter is None:
            return False
        for event in getter():
            if getattr(event, "type", None) != removed:
                continue
            ev_id = getattr(event, "instance_id", None)
            if ev_id is None or int(ev_id) == int(self._instance_id):
                return True
        return False

    def _drop_joy(self) -> None:
        try:
            if self._joy is not None:
                self._joy.quit()
        except Exception:
            pass
        self._joy = None
        self._instance_id = None
        self.device_index = -1
        self.name = ""
        self.guid = ""
        self.transport = "none"
        self.link_transport = "none"


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
        self.link_transport = "fake"
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
