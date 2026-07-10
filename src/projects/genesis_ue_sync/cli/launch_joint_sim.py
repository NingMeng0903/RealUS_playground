"""Print orchestration checklist for Genesis + optional UE consumers."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genesis–UE joint simulation launch hints.")
    parser.add_argument("--markdown", action="store_true", help="Print MD path reference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.markdown:
        print("See MD/genesis_ue_joint_sim.md for authoritative instructions.")
        return 0
    print(
        """
Genesis–UE joint simulation (quick checklist):

1) Export AMONGUS_SESSION_ID + AMONGUS_GENESIS_CANONICAL_ZMQ_BIND as needed.
2) PYTHONPATH=src python src/projects/genesis_ue_sync/cli/peirastic/run_genesis_franka_sim_server.py ...
3a) PYTHONPATH=src python .../cli/controller_bus/run_virtual_joystick_peirastic_client.py
    OR 3b) ros2 run joy joy_node + PYTHONPATH=src python .../cli/controller_bus/run_ros2_joy_peirastic_client.py
4) Launch UE editor watcher + apply_scene_to_level once (existing tooling).
5) PYTHONPATH=src python .../cli/render/unreal/run_canonical_zmq_ue_bridge.py ...
6) PYTHONPATH=src python .../cli/render/unreal/amongus_ue_tcp_camera_mux.py ...
7) Enable AmongUsTcpCaptureComponent inside UE (assign SceneCapture actors).

Full narrative + payload contracts live in MD/genesis_ue_joint_sim.md.
""".strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
