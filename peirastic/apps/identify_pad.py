#!/usr/bin/env python3
"""Press left stick (L3) then right stick (R3). Prints physical button indices."""

from __future__ import annotations

import os
import time

import numpy as np


def _pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required") from exc
    return pygame


def _open():
    pygame = _pygame()
    if os.environ.get("DISPLAY") is None and os.environ.get("SDL_VIDEODRIVER") is None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.joystick.init()
    n = int(pygame.joystick.get_count())
    if n <= 0:
        pygame.quit()
        raise SystemExit("no joystick")
    joy = pygame.joystick.Joystick(0)
    joy.init()
    return pygame, joy


def _buttons(joy) -> np.ndarray:
    n = int(joy.get_numbuttons())
    return np.array([1.0 if joy.get_button(i) else 0.0 for i in range(n)], dtype=float)


def _wait_press(pygame, joy, label: str, rest: np.ndarray) -> int:
    print(f"[STATE] release all, then press {label} (stick click, not tilt)", flush=True)
    while True:
        pygame.event.pump()
        now = _buttons(joy)
        delta = now - rest
        hits = np.nonzero(delta > 0.5)[0]
        if hits.size == 1:
            idx = int(hits[0])
            print(f"[OK] {label} physical button {idx}", flush=True)
            while True:
                pygame.event.pump()
                if float(_buttons(joy)[idx]) < 0.5:
                    break
                time.sleep(0.02)
            return idx
        time.sleep(0.02)


def main() -> int:
    pygame, joy = _open()
    print(f"[STATE] pad {joy.get_name()!r} buttons={joy.get_numbuttons()}", flush=True)
    time.sleep(0.4)
    pygame.event.pump()
    rest = _buttons(joy)
    l3 = _wait_press(pygame, joy, "LEFT stick press (L3)", rest)
    time.sleep(0.3)
    pygame.event.pump()
    rest = _buttons(joy)
    r3 = _wait_press(pygame, joy, "RIGHT stick press (R3)", rest)
    print(f"[OK] map button_index l3={l3} r3={r3}", flush=True)
    print("write these into var/gamepad_layout.json layout.button_index", flush=True)
    joy.quit()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
