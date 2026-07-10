from __future__ import annotations

import os
from dataclasses import dataclass

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def _require_pygame_joystick():
    try:
        import pygame  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("Install pygame for Xbox/gamepad support: pip install pygame") from exc
    return pygame


@dataclass(frozen=True)
class XboxAxisMap:
    """Indices for pygame ``joystick.get_axis(i)``; negative values disable a channel.

    On Linux Xbox pads, trigger axes often rest at ``-1`` instead of ``0``. Do not map them as
    zero-centered velocity axes unless they are explicitly normalized first.
    """

    trans_x: int = 0
    trans_y: int = 1
    trans_z: int = 2
    rot_x: int = 3
    rot_y: int = 4
    rot_z: int = 5


# Left stick XY; RB/RT (R1/R2) -> trans_z (see ``trigger_trans_z_scale``); right stick XY -> rot_x/rot_y.
AXIS_PROFILE_LINUX_XBOX = XboxAxisMap(trans_x=0, trans_y=1, trans_z=-1, rot_x=3, rot_y=4, rot_z=-1)

# Same as linux_xbox plus D-pad X -> rot_z (``hat_rot_scale_z`` in ``build_xbox_gamepad``).
AXIS_PROFILE_LINUX_XBOX_HYBRID = AXIS_PROFILE_LINUX_XBOX


AXIS_PROFILE_SDL_GENERIC = XboxAxisMap()

# SDL2 / pygame joystick button indices for typical Xbox layouts on Linux.
XBOX_BUTTON_A = 0
XBOX_BUTTON_B = 1
XBOX_BUTTON_X = 2
XBOX_BUTTON_Y = 3
XBOX_BUTTON_LB = 4
XBOX_BUTTON_RB = 5
XBOX_BUTTON_BACK = 6
XBOX_BUTTON_START = 7


class XboxGamepad:
    def __init__(
        self,
        *,
        device_index: int = 0,
        deadzone: float = 0.12,
        axis_map: XboxAxisMap | None = None,
        lt_axis_idx: int = 2,
        rt_axis_idx: int = 5,
        hat_rot_scale_y: float = 0.0,
        hat_rot_scale_z: float = 0.0,
        trigger_rot_y_scale: float = 0.0,
        trigger_trans_z_scale: float = 0.0,
        trigger_normalize_rest_neg_one: bool = True,
        allow_missing: bool = True,
    ) -> None:
        pygame = _require_pygame_joystick()
        if os.environ.get("DISPLAY") is None and os.environ.get("SDL_VIDEODRIVER") is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= device_index:
            if allow_missing:
                _logger.warning(
                    "No joystick at index %s (found %s); teleop will emit zero action until a device appears.",
                    device_index,
                    pygame.joystick.get_count(),
                )
                self._pygame = pygame
                self._joy = None
                self._deadzone = float(deadzone)
                self._map = axis_map or XboxAxisMap()
                self._lt_ax = int(lt_axis_idx)
                self._rt_ax = int(rt_axis_idx)
                self._hat_scale_y = float(hat_rot_scale_y)
                self._hat_scale_z = float(hat_rot_scale_z)
                self._trigger_rot_scale = float(trigger_rot_y_scale)
                self._trigger_trans_z_scale = float(trigger_trans_z_scale)
                self._trigger_norm = bool(trigger_normalize_rest_neg_one)
                self._prev_buttons: tuple[int, ...] = ()
                return
            pygame.quit()
            raise RuntimeError(
                f"No joystick at index {device_index} (found {pygame.joystick.get_count()}). "
                "Connect an Xbox controller and retry."
            )
        self._pygame = pygame
        self._joy = pygame.joystick.Joystick(device_index)
        self._joy.init()
        self._deadzone = float(deadzone)
        self._map = axis_map or XboxAxisMap()
        self._lt_ax = int(lt_axis_idx)
        self._rt_ax = int(rt_axis_idx)
        self._hat_scale_y = float(hat_rot_scale_y)
        self._hat_scale_z = float(hat_rot_scale_z)
        self._trigger_rot_scale = float(trigger_rot_y_scale)
        self._trigger_trans_z_scale = float(trigger_trans_z_scale)
        self._trigger_norm = bool(trigger_normalize_rest_neg_one)
        self._prev_buttons: tuple[int, ...] = ()

    @staticmethod
    def _normalize_trigger_rest_neg_one(raw: float) -> float:
        """Axis in [-1,1] resting at -1 mapped to [0,1]; 0 at rest."""
        return float(max(0.0, min(1.0, (float(raw) + 1.0) * 0.5)))

    def _dz(self, v: float) -> float:
        if abs(v) < self._deadzone:
            return 0.0
        return float(v)

    def _refresh_buttons(self) -> tuple[int, ...]:
        if self._joy is None:
            self._prev_buttons = ()
            return ()
        self._pygame.event.pump()
        count = int(self._joy.get_numbuttons())
        pressed = tuple(int(self._joy.get_button(i)) for i in range(count))
        self._prev_buttons = pressed
        return pressed

    def read_button(self, button_index: int) -> bool:
        if self._joy is None or int(button_index) < 0:
            return False
        self._pygame.event.pump()
        count = int(self._joy.get_numbuttons())
        idx = int(button_index)
        if idx >= count:
            return False
        return bool(self._joy.get_button(idx))

    def button_rising_edge(self, button_index: int) -> bool:
        if self._joy is None or int(button_index) < 0:
            self._prev_buttons = ()
            return False
        self._pygame.event.pump()
        count = int(self._joy.get_numbuttons())
        idx = int(button_index)
        if idx >= count:
            prev = self._prev_buttons[idx] if idx < len(self._prev_buttons) else 0
            self._prev_buttons = tuple(
                int(self._joy.get_button(i)) if i < count else 0 for i in range(count)
            )
            return False
        current = tuple(int(self._joy.get_button(i)) for i in range(count))
        rising = int(current[idx]) == 1 and (
            idx >= len(self._prev_buttons) or int(self._prev_buttons[idx]) == 0
        )
        self._prev_buttons = current
        return rising

    def poll_button_rising_edges(self) -> tuple[int, ...]:
        """Return button indices with rising edges this frame (single pump + prev update)."""
        if not self._ensure_joy():
            self._prev_buttons = ()
            return ()
        self._pygame.event.pump()
        count = int(self._joy.get_numbuttons())
        current = tuple(int(self._joy.get_button(i)) if i < count else 0 for i in range(count))
        rising: list[int] = []
        for idx in range(count):
            if current[idx] == 1 and (idx >= len(self._prev_buttons) or int(self._prev_buttons[idx]) == 0):
                rising.append(idx)
        self._prev_buttons = current
        return tuple(rising)

    def read_raw_axes(self, max_axes: int = 8) -> list[float]:
        if not self._ensure_joy():
            return []
        self._pygame.event.pump()
        count = min(int(self._joy.get_numaxes()), int(max_axes))
        return [float(self._joy.get_axis(i)) for i in range(count)]

    @staticmethod
    def list_joysticks() -> list[str]:
        """Probe connected joystick names without closing active ``XboxGamepad`` instances."""
        pygame = _require_pygame_joystick()
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        names: list[str] = []
        for i in range(int(pygame.joystick.get_count())):
            joy = pygame.joystick.Joystick(i)
            if not joy.get_init():
                joy.init()
            names.append(str(joy.get_name()))
        return names

    def _ensure_joy(self) -> bool:
        if self._joy is None:
            return False
        get_init = getattr(self._joy, "get_init", None)
        if callable(get_init) and not get_init():
            self._joy.init()
        return True

    def read_action_vector(self) -> np.ndarray:
        if not self._ensure_joy():
            return np.zeros(6, dtype=np.float32)
        self._pygame.event.pump()
        m = self._map
        ax = self._joy.get_numaxes()

        def g(idx: int) -> float:
            if idx < 0 or idx >= ax:
                return 0.0
            return self._dz(float(self._joy.get_axis(idx)))

        tx = g(m.trans_x)
        ty = g(m.trans_y)
        tz = g(m.trans_z)
        rx = g(m.rot_x)
        ry = g(m.rot_y)
        rz = g(m.rot_z)

        lt_raw = float(self._joy.get_axis(self._lt_ax)) if 0 <= self._lt_ax < ax else -1.0
        rt_raw = float(self._joy.get_axis(self._rt_ax)) if 0 <= self._rt_ax < ax else -1.0
        lt_n = self._normalize_trigger_rest_neg_one(lt_raw) if self._trigger_norm else max(0.0, lt_raw)
        rt_n = self._normalize_trigger_rest_neg_one(rt_raw) if self._trigger_norm else max(0.0, rt_raw)

        if self._trigger_trans_z_scale > 1e-6 and m.trans_z < 0:
            # RB (R1) up, RT (R2) down before teleop sign flip on tz.
            rb_on = 1.0 if self.read_button(XBOX_BUTTON_RB) else 0.0
            tz += self._trigger_trans_z_scale * float(rb_on - rt_n)

        if self._trigger_rot_scale > 1e-6 and m.rot_y < 0:
            ry += self._trigger_rot_scale * float(lt_n - rt_n)

        if (self._hat_scale_y > 1e-6 or self._hat_scale_z > 1e-6) and self._joy.get_numhats() > 0:
            hx, hy = self._joy.get_hat(0)
            if m.rot_z < 0:
                rz += float(self._hat_scale_z) * float(hx)
            if m.rot_y < 0:
                ry += float(self._hat_scale_y) * float(hy)

        return np.array(
            [
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
            ],
            dtype=np.float32,
        )

    def close(self) -> None:
        try:
            if self._joy is not None:
                self._joy.quit()
        except Exception:
            pass
        try:
            self._pygame.quit()
        except Exception:
            pass


def build_xbox_gamepad(
    *,
    device_index: int = 0,
    deadzone: float = 0.12,
    axis_profile: str = "linux_xbox",
    allow_missing: bool = True,
    hat_rot_scale_z: float = 1.0,
    hat_rot_scale_y: float = 0.0,
) -> XboxGamepad:
    """Construct ``XboxGamepad`` with Linux Xbox defaults (RB/RT vertical, right stick rotation)."""

    profile = str(axis_profile).strip().lower()
    if profile == "sdl_generic":
        return XboxGamepad(
            device_index=int(device_index),
            deadzone=float(deadzone),
            axis_map=AXIS_PROFILE_SDL_GENERIC,
            allow_missing=bool(allow_missing),
        )
    axis_map = AXIS_PROFILE_LINUX_XBOX_HYBRID if profile == "linux_xbox_hybrid" else AXIS_PROFILE_LINUX_XBOX
    kwargs: dict[str, object] = {
        "device_index": int(device_index),
        "deadzone": float(deadzone),
        "axis_map": axis_map,
        "allow_missing": bool(allow_missing),
        "trigger_trans_z_scale": 1.0,
    }
    if profile == "linux_xbox_hybrid":
        kwargs["hat_rot_scale_z"] = float(hat_rot_scale_z)
        kwargs["hat_rot_scale_y"] = float(hat_rot_scale_y)
    elif profile == "linux_xbox":
        kwargs["hat_rot_scale_z"] = float(hat_rot_scale_z)
    return XboxGamepad(**kwargs)
