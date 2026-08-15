"""Minimal Xbox / pygame joystick reader for 8-DOF QPIK teleop.

Axis layout matches the Linux SDL Xbox mapping used by the Genesis teleop
stack (left stick 0/1, LT 2, right stick 3/4, RT 5; LB=4, RB=5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

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


@dataclass
class PadState:
    axes: np.ndarray
    buttons: np.ndarray

    def button(self, index: int) -> bool:
        idx = int(index)
        if idx < 0 or idx >= int(self.buttons.size):
            return False
        return bool(self.buttons[idx])


class XboxPad:
    """Read raw Xbox axes/buttons. Missing device emits zeros."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        allow_missing: bool = True,
    ) -> None:
        pygame = _require_pygame()
        if os.environ.get("DISPLAY") is None and os.environ.get("SDL_VIDEODRIVER") is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()
        self._pygame = pygame
        self._joy = None
        self._closed = False
        count = int(pygame.joystick.get_count())
        if count <= int(device_index):
            if not allow_missing:
                pygame.quit()
                raise RuntimeError(
                    f"no joystick at index {device_index} (found {count})"
                )
            return
        self._joy = pygame.joystick.Joystick(int(device_index))
        self._joy.init()

    @property
    def connected(self) -> bool:
        return self._joy is not None and not self._closed

    def read(self) -> PadState:
        axes = np.zeros(_N_AXES, dtype=float)
        buttons = np.zeros(_N_BUTTONS, dtype=float)
        if self._joy is None or self._closed:
            return PadState(axes=axes, buttons=buttons)
        self._pygame.event.pump()
        n_ax = int(self._joy.get_numaxes())
        n_btn = int(self._joy.get_numbuttons())
        for i in range(min(_N_AXES, n_ax)):
            axes[i] = float(self._joy.get_axis(i))
        for i in range(min(_N_BUTTONS, n_btn)):
            buttons[i] = 1.0 if self._joy.get_button(i) else 0.0
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
        try:
            self._pygame.quit()
        except Exception:
            pass


class FakePad:
    """Deterministic pad for tests / dry-run."""

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

    @property
    def connected(self) -> bool:
        return not self.closed

    def read(self) -> PadState:
        ax = np.zeros(_N_AXES, dtype=float)
        btn = np.zeros(_N_BUTTONS, dtype=float)
        ax[: min(_N_AXES, self.axes.size)] = self.axes[:_N_AXES]
        btn[: min(_N_BUTTONS, self.buttons.size)] = self.buttons[:_N_BUTTONS]
        return PadState(axes=ax, buttons=btn)

    def close(self) -> None:
        self.closed = True
