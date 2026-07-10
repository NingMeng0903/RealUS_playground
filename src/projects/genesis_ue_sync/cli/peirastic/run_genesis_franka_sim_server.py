from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from projects.genesis_ue_sync.integrations.controller_bus.peirastic_robot_sim_bridge import (
    GenesisRobotSimPeirasticBridge,
    GenesisRobotSimPeirasticConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEIRASTIC-compatible Genesis Franka simulation ZMQ server (NUC role).")
    parser.add_argument(
        "--interface-yaml",
        type=Path,
        default=None,
        help="Path to PEIRASTIC interface YAML (e.g. config/local-host.yml). Defaults to ref_code_library/PEIRASTIC_control/config/local-host.yml.",
    )
    parser.add_argument(
        "--scene-spec",
        type=Path,
        default=None,
        help="Optional SyncSceneSpec path for Genesis Panda placement.",
    )
    parser.add_argument("--backend", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--state-rate-hz", type=float, default=None, help="Override CONTROL.STATE_PUBLISHER_RATE in YAML.")
    parser.add_argument("--gravity-compensation", type=float, default=1.0, help="Genesis material gravity compensation scale [0,1].")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument(
        "--peirastic-repo",
        type=Path,
        default=None,
        help="Root of PEIRASTIC_control checkout if peirastic is not installed.",
    )
    parser.add_argument(
        "--control-backend",
        type=str,
        default="cartesian_follow",
        choices=("cartesian_follow", "osc_impedance"),
        help="OSC_PEIRASTIC Cartesian follow (default) or torque OSC impedance (protobuf stiffness).",
    )
    parser.add_argument(
        "--osc-impedance-yaml",
        type=Path,
        default=None,
        help="Optional YAML overriding configs/controllers/franka_panda_osc_impedance_default.yaml.",
    )
    parser.add_argument(
        "--stream-human-canonical-motion",
        action="store_true",
        help="Advance SyncSceneSpec motion sequence_npz into amongus_canonical_human each Genesis step.",
    )
    parser.add_argument(
        "--ue-anim-sequence-path",
        type=str,
        default="",
        help="Optional /Game/... AnimSequence for UE skeletal playback during human streaming.",
    )
    return parser.parse_args()


def default_interface_yaml() -> Path:
    from common.project import project_paths

    root = project_paths(__file__).root
    return root / "ref_code_library" / "PEIRASTIC_control" / "config" / "local-host.yml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    iface = args.interface_yaml or default_interface_yaml()
    cfg = GenesisRobotSimPeirasticConfig(
        interface_yaml=iface,
        scene_spec=args.scene_spec,
        backend=args.backend,
        state_rate_hz=float(args.state_rate_hz) if args.state_rate_hz is not None else None,
        gravity_compensation=float(args.gravity_compensation),
        show_viewer=bool(args.show_viewer),
        peirastic_repo=args.peirastic_repo,
        control_backend=str(args.control_backend),
        osc_impedance_yaml=args.osc_impedance_yaml,
        stream_human_canonical_motion=bool(args.stream_human_canonical_motion),
        ue_anim_sequence_path=str(args.ue_anim_sequence_path or ""),
    )
    server = GenesisRobotSimPeirasticBridge(cfg)
    try:
        server.run_forever()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
    sys.exit(0)
