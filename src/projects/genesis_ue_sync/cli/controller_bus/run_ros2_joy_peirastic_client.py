from __future__ import annotations

import argparse
import logging
import sys
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
    parser = argparse.ArgumentParser(description="Subscribe ROS2 Joy and forward OSC_POSE deltas via FrankaInterface.")
    parser.add_argument(
        "--interface-yaml",
        type=Path,
        default=None,
        help="PEIRASTIC interface YAML (defaults to ref_code_library/PEIRASTIC_control/config/local-host.yml).",
    )
    parser.add_argument("--joy-topic", type=str, default="/joy")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--trans-scale", type=float, default=0.20)
    parser.add_argument("--rot-scale", type=float, default=0.9)
    parser.add_argument(
        "--axis-profile",
        type=str,
        default="linux_xbox",
        choices=("linux_xbox", "linux_xbox_hybrid", "sdl_generic"),
        help="Same axis indices as pygame XboxGamepad profiles; hybrid adds LT/RT -> rot_y and optional hat axes.",
    )
    parser.add_argument("--joy-hat-axis-x", type=int, default=None, help="If set with --joy-hat-axis-y, analog hat X -> rot_z.")
    parser.add_argument("--joy-hat-axis-y", type=int, default=None, help="If set with --joy-hat-axis-x, analog hat Y -> rot_y.")
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
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Joy
    except ImportError as exc:
        logging.error("ROS2 Python deps missing (rclpy, sensor_msgs): %s", exc)
        return 3

    try:
        from peirastic.franka_interface import FrankaInterface
        from peirastic.utils.config_utils import get_default_controller_config
    except ImportError as exc:
        logging.error("Install PEIRASTIC_control (pip install -e ref_code_library/PEIRASTIC_control): %s", exc)
        return 4

    from projects.genesis_ue_sync.integrations.controller_bus.joy_axis_mapping import (
        augment_linux_xbox_hybrid_inplace,
        axes_with_deadzone,
    )
    from projects.genesis_ue_sync.sim_platform.control.teleop import (
        AXIS_PROFILE_LINUX_XBOX,
        AXIS_PROFILE_LINUX_XBOX_HYBRID,
        AXIS_PROFILE_SDL_GENERIC,
    )

    if args.axis_profile == "sdl_generic":
        axis_map = AXIS_PROFILE_SDL_GENERIC
    else:
        axis_map = AXIS_PROFILE_LINUX_XBOX_HYBRID if args.axis_profile == "linux_xbox_hybrid" else AXIS_PROFILE_LINUX_XBOX
    cfg_osc = get_default_controller_config("OSC_POSE")

    robot = FrankaInterface(str(iface), has_gripper=False, use_visualizer=False)
    if not robot.wait_for_state(timeout=15.0):
        logging.error("No robot state received from Genesis sim server.")
        robot.close()
        return 5

    class _JoyBridge(Node):
        def __init__(self) -> None:
            super().__init__("amongus_ros2_joy_peirastic_client")
            self.create_subscription(Joy, str(args.joy_topic), self._cb, 10)

        def _cb(self, msg: Joy) -> None:
            vec = axes_with_deadzone(msg.axes, axis_map, deadzone=float(args.deadzone))
            if args.axis_profile == "linux_xbox_hybrid":
                hat_pair = None
                if args.joy_hat_axis_x is not None and args.joy_hat_axis_y is not None:
                    hat_pair = (int(args.joy_hat_axis_x), int(args.joy_hat_axis_y))
                augment_linux_xbox_hybrid_inplace(vec, msg.axes, axis_map, hat_axis_pair=hat_pair)
            dt = 1.0 / max(float(args.rate_hz), 1e-6)
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
                robot.control("OSC_POSE", [dx, dy, dz, rax, ray, raz], controller_cfg=cfg_osc)
            except Exception as exc:
                logging.warning("control failed: %s", exc)

    rclpy.init()
    node = _JoyBridge()
    logging.info("ROS2 joy bridge on topic=%s iface=%s profile=%s", args.joy_topic, iface, args.axis_profile)
    try:
        rclpy.spin(node)
    finally:
        robot.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
