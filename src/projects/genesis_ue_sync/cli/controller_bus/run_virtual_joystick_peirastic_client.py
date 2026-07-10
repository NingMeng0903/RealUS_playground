from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if parent.name == "src" and (parent / "common" / "project.py").is_file():
            sp = str(parent)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map joystick axes to PEIRASTIC OSC_POSE deltas via FrankaInterface.")
    parser.add_argument(
        "--interface-yaml",
        type=Path,
        default=None,
        help="PEIRASTIC interface YAML (defaults to ref_code_library/PEIRASTIC_control/config/local-host.yml).",
    )
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument(
        "--axis-profile",
        type=str,
        default="linux_xbox" if sys.platform.startswith("linux") else "sdl_generic",
        choices=("linux_xbox", "linux_xbox_hybrid", "sdl_generic"),
    )
    parser.add_argument("--trans-scale", type=float, default=0.20, help="Metres/sec equivalent stick scale.")
    parser.add_argument("--rot-scale", type=float, default=0.9, help="Radians/sec equivalent rotation stick scale.")
    parser.add_argument("--duration-s", type=float, default=3600.0)
    return parser.parse_args()


def default_iface_yaml() -> Path:
    from common.project import project_paths

    return project_paths(__file__).root / "ref_code_library" / "PEIRASTIC_control" / "config" / "local-host.yml"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _ensure_src_on_path()
    args = parse_args()
    iface = Path(args.interface_yaml or default_iface_yaml()).expanduser().resolve()
    if not iface.is_file():
        logging.error("Missing interface yaml: %s", iface)
        return 2

    try:
        from peirastic.franka_interface import FrankaInterface
        from peirastic.utils.config_utils import get_default_controller_config
    except ImportError as exc:
        logging.error("Install PEIRASTIC_control (pip install -e ref_code_library/PEIRASTIC_control): %s", exc)
        return 4

    from projects.genesis_ue_sync.sim_platform.control.teleop import (
        AXIS_PROFILE_LINUX_XBOX,
        AXIS_PROFILE_LINUX_XBOX_HYBRID,
        AXIS_PROFILE_SDL_GENERIC,
    )
    from projects.genesis_ue_sync.sim_platform.control.teleop.xbox_gamepad import XboxGamepad

    if args.axis_profile == "sdl_generic":
        axis_map = AXIS_PROFILE_SDL_GENERIC
    elif args.axis_profile == "linux_xbox_hybrid":
        axis_map = AXIS_PROFILE_LINUX_XBOX_HYBRID
    else:
        axis_map = AXIS_PROFILE_LINUX_XBOX

    if args.axis_profile == "linux_xbox_hybrid":
        pad = XboxGamepad(
            device_index=int(args.device_index),
            deadzone=float(args.deadzone),
            axis_map=axis_map,
            hat_rot_scale_y=1.08,
            hat_rot_scale_z=1.08,
            trigger_rot_y_scale=3.25,
        )
    else:
        pad = XboxGamepad(device_index=int(args.device_index), deadzone=float(args.deadzone), axis_map=axis_map)

    robot = FrankaInterface(str(iface), has_gripper=False, use_visualizer=False)
    if not robot.wait_for_state(timeout=15.0):
        logging.error("No robot state received from Genesis sim server.")
        pad.close()
        robot.close()
        return 5

    cfg_osc = get_default_controller_config("OSC_POSE")
    dt = 1.0 / max(float(args.rate_hz), 1e-6)
    deadline = time.perf_counter() + float(args.duration_s)

    logging.info(
        "Virtual joystick client running (iface=%s profile=%s). Left stick XY translation; right stick rot tweak.",
        iface,
        args.axis_profile,
    )
    try:
        while time.perf_counter() < deadline:
            vec = pad.read_action_vector()
            lx = float(vec[0])
            ly = float(vec[1])
            lz = float(vec[2])
            rx = float(vec[3])
            ry = float(vec[4])
            rz = float(vec[5])
            dx = -ly * float(args.trans_scale) * dt
            dy = -lx * float(args.trans_scale) * dt
            dz = -lz * float(args.trans_scale) * dt
            rax = rx * float(args.rot_scale) * dt
            ray = ry * float(args.rot_scale) * dt
            raz = rz * float(args.rot_scale) * dt
            try:
                robot.control(
                    "OSC_POSE",
                    [dx, dy, dz, rax, ray, raz],
                    controller_cfg=cfg_osc,
                )
            except Exception as exc:
                logging.warning("control failed: %s", exc)
            time.sleep(dt)
    finally:
        pad.close()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
