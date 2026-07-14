#!/usr/bin/env python3
"""Recover controller after force-mode conflict or latched system errors (e.g. 4119)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rm75f_default.yaml"

ERR_4119 = "4119"
ERR_4119_MSG = (
    "4119 = six-axis force external-load data validation failed. On the pendant: "
    "Config -> Arm Config -> Force Sensor Config -> redo six-axis force center-of-mass "
    "calibration (do not touch external objects at the tool tip during calibration); "
    "clear the error in System Info, then retry."
)


def read_system_err(robot) -> list[str]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        return [f"get_state_failed:{ret}"]
    err = st.get("err", {})
    return list(err.get("err", []))[: int(err.get("err_len", 0))]


def main() -> int:
    from rm75_control import RobotSession

    print("Recovering controller...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        before = read_system_err(bot.robot)
        print(f"  system err before: {before or 'none'}", flush=True)
        if ERR_4119 in before:
            print(f"  !! {ERR_4119_MSG}", flush=True)

        clr = bot.robot.rm_clear_system_err()
        print(f"  rm_clear_system_err: {clr}", flush=True)
        time.sleep(0.5)
        after_clear = read_system_err(bot.robot)
        print(f"  system err after clear: {after_clear or 'none'}", flush=True)

        if ERR_4119 in after_clear:
            print(
                f"\n4119 still latched — movev/force control will not move. Calibrate the "
                f"force sensor center of mass on the pendant and clear the error, then rerun "
                f"this script.\n{ERR_4119_MSG}",
                file=sys.stderr,
                flush=True,
            )
            return 1

        for attempt in range(6):
            diag = bot.prepare_for_force_stream(settle_s=1.0)
            ret = bot.robot.rm_start_force_position_move()
            print(f"  attempt {attempt}: {diag} force_stream_start={ret}", flush=True)
            if ret == 0:
                bot.robot.rm_stop_force_position_move()
                time.sleep(0.5)
                print("Recovered.", flush=True)
                return 0

        print(
            "Force-stream probe still failing (may be OK if you only need movej/movep). "
            "Check pendant for running programs; power-cycle if needed.",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
