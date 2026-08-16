#!/usr/bin/env python3
"""Walk one Xbox pad through sticks / triggers / face buttons.

Each step: release → HOLD the named control → we record which physical
axis or button moved. Writes ``var/gamepad_layout.json`` for Window A.

  source env.sh
  python apps/joint_admittance_8dof/identify_gamepad.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.teleop.pad_layout import (
    DEFAULT_LAYOUT_PATH,
    LOGICAL_AXES,
    PadLayout,
    classify_layout,
    pick_device_index,
    save_identify_result,
    transport_from_name,
)


STEPS: tuple[tuple[str, str, str], ...] = (
    ("axis", "lx_left", "左摇杆 向左推满并按住"),
    ("axis", "lx_right", "左摇杆 向右推满并按住"),
    ("axis", "ly_up", "左摇杆 向上推满并按住"),
    ("axis", "ly_down", "左摇杆 向下推满并按住"),
    ("axis", "rx_left", "右摇杆 向左推满并按住"),
    ("axis", "rx_right", "右摇杆 向右推满并按住"),
    ("axis", "ry_up", "右摇杆 向上推满并按住"),
    ("axis", "ry_down", "右摇杆 向下推满并按住"),
    ("button", "lb", "按下 LB（左肩键）并按住"),
    ("axis", "lt", "按下 LT（左扳机）并按住"),
    ("button", "rb", "按下 RB（右肩键）并按住"),
    ("axis", "rt", "按下 RT（右扳机）并按住"),
    ("button", "y", "按下 Y 并按住"),
    ("button", "b", "按下 B 并按住"),
    ("button", "a", "按下 A 并按住"),
    ("button", "x", "按下 X 并按住"),
)


def _require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required") from exc
    return pygame


def _open_joy(device_index: int | None):
    pygame = _require_pygame()
    if os.environ.get("DISPLAY") is None and os.environ.get("SDL_VIDEODRIVER") is None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.joystick.init()
    count = int(pygame.joystick.get_count())
    names = []
    for i in range(count):
        j = pygame.joystick.Joystick(i)
        j.init()
        names.append(j.get_name())
        j.quit()
    if count <= 0:
        pygame.quit()
        raise SystemExit("no joystick found")
    idx = pick_device_index(names) if device_index is None else int(device_index)
    if idx < 0 or idx >= count:
        pygame.quit()
        raise SystemExit(f"device-index {idx} out of range (found {count})")
    joy = pygame.joystick.Joystick(idx)
    joy.init()
    return pygame, joy, names, idx


def _sample(pygame, joy) -> tuple[np.ndarray, np.ndarray]:
    pygame.event.pump()
    n_ax = int(joy.get_numaxes())
    n_btn = int(joy.get_numbuttons())
    axes = np.array([float(joy.get_axis(i)) for i in range(n_ax)], dtype=float)
    buttons = np.array(
        [1.0 if joy.get_button(i) else 0.0 for i in range(n_btn)], dtype=float
    )
    return axes, buttons


def _mean_sample(pygame, joy, *, seconds: float, hz: float = 40.0):
    n = max(4, int(float(seconds) * float(hz)))
    dt = 1.0 / float(hz)
    axes_rows = []
    btn_rows = []
    for _ in range(n):
        ax, btn = _sample(pygame, joy)
        axes_rows.append(ax)
        btn_rows.append(btn)
        time.sleep(dt)
    # Pad ragged lengths (should not happen mid-run).
    ax_w = max(a.size for a in axes_rows)
    btn_w = max(b.size for b in btn_rows)
    axes = np.zeros((len(axes_rows), ax_w), dtype=float)
    buttons = np.zeros((len(btn_rows), btn_w), dtype=float)
    for i, (a, b) in enumerate(zip(axes_rows, btn_rows)):
        axes[i, : a.size] = a
        buttons[i, : b.size] = b
    return axes.mean(axis=0), buttons.mean(axis=0), axes, buttons


def _peak_axis(rest: np.ndarray, held: np.ndarray) -> tuple[int, float]:
    delta = np.asarray(held, dtype=float) - np.asarray(rest, dtype=float)
    if delta.size == 0:
        return 0, 0.0
    idx = int(np.argmax(np.abs(delta)))
    return idx, float(delta[idx])


def _peak_button(rest: np.ndarray, held: np.ndarray) -> tuple[int, float]:
    held_m = np.asarray(held, dtype=float)
    rest_m = np.asarray(rest, dtype=float)
    if held_m.size == 0:
        return 0, 0.0
    delta = held_m - rest_m
    # Rising edge first. If the user already held through the rest sample
    # (common on LB/RB), accept the currently-on button.
    if float(np.max(delta)) >= 0.4:
        idx = int(np.argmax(delta))
        return idx, float(delta[idx])
    if float(np.max(held_m)) >= 0.5:
        idx = int(np.argmax(held_m))
        return idx, float(held_m[idx])
    idx = int(np.argmax(delta))
    return idx, float(delta[idx])


def _build_layout(hits: dict[str, tuple[int, float]], rest_axes: np.ndarray) -> PadLayout:
    axis_index = {"lx": 0, "ly": 1, "lt": 2, "rx": 3, "ry": 4, "rt": 5}
    axis_sign = {k: 1 for k in LOGICAL_AXES}
    button_index = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 4, "rb": 5}

    def _pair(neg_key: str, pos_key: str, logical: str) -> None:
        i_neg, _d_neg = hits[neg_key]
        i_pos, _d_pos = hits[pos_key]
        if i_neg != i_pos:
            raise SystemExit(
                f"{logical}: opposite directions hit axes {i_neg} and {i_pos}; redo"
            )
        axis_index[logical] = int(i_pos)
        axis_sign[logical] = 1

    _pair("lx_left", "lx_right", "lx")
    _pair("ly_up", "ly_down", "ly")
    _pair("rx_left", "rx_right", "rx")
    _pair("ry_up", "ry_down", "ry")
    for key in ("lt", "rt"):
        idx, _delta = hits[key]
        axis_index[key] = int(idx)
        axis_sign[key] = 1
    for key in ("lb", "rb", "y", "b", "a", "x"):
        idx, delta = hits[key]
        if delta < 0.4:
            raise SystemExit(f"{key}: no button rose (delta={delta:.2f}); redo")
        button_index[key] = int(idx)

    rest = np.zeros(6, dtype=float)
    rest[: min(6, rest_axes.size)] = rest_axes[:6]
    lt_i = axis_index["lt"]
    rt_i = axis_index["rt"]
    lt_rest = float(rest[lt_i]) if lt_i < rest.size else 0.0
    rt_rest = float(rest[rt_i]) if rt_i < rest.size else 0.0
    trigger_rest = -1.0 if (lt_rest < -0.5 and rt_rest < -0.5) else 0.0
    return PadLayout(
        name="identified",
        axis_index=axis_index,
        axis_sign=axis_sign,
        trigger_rest=trigger_rest,
        button_index=button_index,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--hold-s", type=float, default=2.2)
    ap.add_argument("--rest-s", type=float, default=1.4)
    ap.add_argument("--out", type=Path, default=DEFAULT_LAYOUT_PATH)
    args = ap.parse_args()

    pygame, joy, names, idx = _open_joy(args.device_index)
    try:
        name = joy.get_name()
        guid = ""
        try:
            guid = str(joy.get_guid())
        except Exception:
            pass
        print("=== gamepad identify ===", flush=True)
        for i, n in enumerate(names):
            mark = "  <-- selected" if i == idx else ""
            print(
                f"  [{i}] {n!r}  transport={transport_from_name(n)}{mark}",
                flush=True,
            )
        print("松开全部摇杆和按键…", flush=True)
        time.sleep(0.6)
        rest_ax, rest_btn, _, _ = _mean_sample(pygame, joy, seconds=args.rest_s)
        guessed = classify_layout(rest_ax, name=name)
        print(
            f"rest axes={np.array2string(rest_ax[:6], precision=3)}  "
            f"guess={guessed.name}",
            flush=True,
        )
        hits: dict[str, tuple[int, float]] = {}
        for kind, key, prompt in STEPS:
            print(f"\n>>> 3 秒内：{prompt}", flush=True)
            time.sleep(0.8)
            held_ax, held_btn, _, _ = _mean_sample(pygame, joy, seconds=args.hold_s)
            if kind == "axis":
                i_ax, delta = _peak_axis(rest_ax, held_ax)
                hits[key] = (i_ax, delta)
                print(f"    axis[{i_ax}] Δ={delta:+.3f}", flush=True)
            else:
                i_btn, delta = _peak_button(rest_btn, held_btn)
                hits[key] = (i_btn, delta)
                print(f"    button[{i_btn}] Δ={delta:+.3f}", flush=True)
            print("    松开，下一拍…", flush=True)
            time.sleep(0.7)
            rest_ax, rest_btn, _, _ = _mean_sample(pygame, joy, seconds=0.6)

        layout = _build_layout(hits, rest_ax)
        payload = {
            "name": name,
            "guid": guid,
            "device_index": idx,
            "transport": transport_from_name(name),
            "rest_axes": [float(x) for x in rest_ax.tolist()],
            "hits": {k: {"index": int(i), "delta": float(d)} for k, (i, d) in hits.items()},
            "layout": layout.to_json(),
        }
        path = save_identify_result(args.out, payload)
        print(f"\nOK  layout={layout.name}  trigger_rest={layout.trigger_rest}", flush=True)
        print(f"    axis_index={layout.axis_index}", flush=True)
        print(f"    button_index={layout.button_index}", flush=True)
        print(f"    wrote {path}", flush=True)
    finally:
        try:
            joy.quit()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    main()
