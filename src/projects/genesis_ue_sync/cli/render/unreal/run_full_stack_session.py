#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common.project import project_paths


PROJECT_PATHS = project_paths(__file__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch the Genesis-as-source UE LiveSync stack.")
    p.add_argument("--scene-spec", type=Path, default=PROJECT_PATHS.default_scene_spec_path)
    p.add_argument(
        "--augmentation-spec",
        type=Path,
        default=None,
        help="Optional augmentation yaml/json (Bedlam avatar overrides, render/motion tweaks).",
    )
    p.add_argument(
        "--amass-npz",
        type=Path,
        default=Path("dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz"),
        help="AMASS clip for Genesis GT driver (§7).",
    )
    p.add_argument("--session-dir", type=Path, default=PROJECT_PATHS.outputs_root / "ue_sessions" / "full_stack")
    p.add_argument("--backend", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--scene-init-bind", type=str, default="tcp://127.0.0.1:5588")
    p.add_argument("--canonical-bind", type=str, default="tcp://127.0.0.1:5599")
    p.add_argument("--camera-port", type=int, default=17355)
    p.add_argument("--camera-pub", type=str, default="tcp://127.0.0.1:17356")
    p.add_argument("--scene-apply-mode", type=str, default="prepare", choices=("apply", "prepare"))
    p.add_argument("--teleop-control-mode", type=str, default="cartesian")
    p.add_argument("--teleop-rate-hz", type=float, default=100.0)
    p.add_argument(
        "--robot-model",
        type=str,
        default="",
        help="Override robot model_id for Genesis + UE scene init (e.g. rm75_6f).",
    )
    p.add_argument(
        "--xbox-teleop",
        action="store_true",
        help="Enable Xbox Cartesian teleop in Genesis driver (requires --show-viewer).",
    )
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-genesis", action="store_true")
    p.add_argument(
        "--no-track-worker",
        action="store_true",
        help="Do not launch the dedicated multiview track worker process.",
    )
    p.add_argument("--print-commands", action="store_true")
    return p.parse_args()


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    repo = str(PROJECT_PATHS.root)
    src = str(PROJECT_PATHS.src_root)
    env.update(
        {
            "REPO": repo,
            "SRC": src,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": src,
            "AMONGUS_SESSION_DIR": str(args.session_dir.expanduser().resolve()),
            "AMONGUS_GENESIS_CANONICAL_ZMQ_BIND": str(args.canonical_bind),
            "AMONGUS_GENESIS_CANONICAL_STATE_JSONL": str(args.session_dir.expanduser().resolve() / "genesis_canonical.jsonl"),
            "AMONGUS_UE_DRIVE_HUMAN_BONES": env.get("AMONGUS_UE_DRIVE_HUMAN_BONES", "1"),
            "AMONGUS_UE_SPAWN_HUMAN_ANCHOR_MARKER": env.get("AMONGUS_UE_SPAWN_HUMAN_ANCHOR_MARKER", "0"),
            "AMONGUS_UE_DISABLE_BACKGROUND_THROTTLE": env.get("AMONGUS_UE_DISABLE_BACKGROUND_THROTTLE", "1"),
            "AMONGUS_UE_MAX_FPS": env.get("AMONGUS_UE_MAX_FPS", "120"),
            "AMONGUS_GENESIS_TRACK_SUBSCRIBE": env.get(
                "AMONGUS_GENESIS_TRACK_SUBSCRIBE",
                "tcp://127.0.0.1:5598",
            ),
        }
    )
    robot_model = str(args.robot_model or "").strip()
    if robot_model:
        env["AMONGUS_ROBOT_MODEL"] = robot_model
    return env


def _cmds(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    session = str(args.session_dir.expanduser().resolve())
    scene = str(args.scene_spec)
    amass = str(args.amass_npz)
    robot_model = str(args.robot_model or "").strip()
    scene_init_publisher_cmd = [
        py,
        "-m",
        "projects.genesis_ue_sync.cli.render.unreal.run_scene_init_publisher",
        "--scene-spec",
        scene,
        "--bind",
        str(args.scene_init_bind),
        "--repeat-s",
        "2.0",
    ]
    if robot_model:
        scene_init_publisher_cmd.extend(["--robot-model", robot_model])
    if args.augmentation_spec is not None:
        scene_init_publisher_cmd.extend(
            ["--augmentation-spec", str(args.augmentation_spec.expanduser().resolve())]
        )
    cmds: list[tuple[str, list[str]]] = [
        (
            "ue_editor",
            [
                py,
                "-m",
                "projects.genesis_ue_sync.cli.render.unreal.run_ue_scene_session",
                "--session-dir",
                session,
                "--watcher-only",
                "--clear-pending-commands",
            ],
        ),
        (
            "camera_mux",
            [
                py,
                "-m",
                "projects.genesis_ue_sync.cli.render.unreal.amongus_ue_tcp_camera_mux",
                "--listen-host",
                "127.0.0.1",
                "--listen-port",
                str(int(args.camera_port)),
                "--pub-bind",
                str(args.camera_pub),
            ],
        ),
        ("scene_init_publisher", scene_init_publisher_cmd),
        (
            "scene_init_bridge",
            [
                py,
                "-m",
                "projects.genesis_ue_sync.cli.render.unreal.run_scene_init_zmq_ue_bridge",
                "--connect",
                str(args.scene_init_bind),
                "--session-dir",
                session,
                "--scene-apply-mode",
                str(args.scene_apply_mode),
                "--exit-after-first",
            ],
        ),
        (
            "canonical_bridge",
            [
                py,
                "-m",
                "projects.genesis_ue_sync.cli.render.unreal.run_canonical_zmq_ue_bridge",
                "--canonical-connect",
                str(args.canonical_bind),
                "--session-dir",
                session,
                "--diagnose-every",
                "200",
            ],
        ),
    ]
    if not args.no_preview:
        cmds.append(
            (
                "camera_preview",
                [
                    py,
                    "-m",
                    "projects.genesis_ue_sync.cli.render.unreal.watch_ue_camera_frames",
                    "--connect",
                    str(args.camera_pub),
                    "--camera-names",
                    "cam_left",
                    "cam_right",
                    "cam_top",
                    "cam_front",
                    "cam_front_left",
                    "cam_front_right",
                ],
            )
        )
    if not args.no_genesis:
        genesis_cmd = [
            py,
            "-m",
            "projects.genesis_ue_sync.sim_platform.apps.genesis_viz.amass_bed_capsule_demo",
            "--backend",
            str(args.backend),
            "--loop",
            "--frame-step",
            "4",
            "--scene-spec",
            scene,
            "--amass-npz",
            amass,
        ]
        if not args.no_viewer:
            genesis_cmd.append("--show-viewer")
        if robot_model:
            genesis_cmd.extend(["--robot-model", robot_model])
        if args.xbox_teleop:
            genesis_cmd.append("--xbox-teleop")
            genesis_cmd.extend(["--teleop-control-mode", str(args.teleop_control_mode)])
        cmds.append(("genesis_driver", genesis_cmd))
    if not args.no_track_worker and not args.no_genesis:
        track_pub = os.environ.get("AMONGUS_GENESIS_TRACK_SUBSCRIBE", "tcp://127.0.0.1:5598")
        cmds.append(
            (
                "multiview_track_worker",
                [
                    py,
                    "-m",
                    "projects.genesis_ue_sync.multiview_realtime.cli.run_multiview_track_worker",
                    "--config",
                    "configs/tracking/multiview_realtime_dwpose_triangulation.yaml",
                    "--pub-bind",
                    str(track_pub),
                ],
            )
        )
    return cmds


def main() -> int:
    args = parse_args()
    args.session_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    env = _env(args)
    cmds = _cmds(args)
    if args.print_commands:
        for name, cmd in cmds:
            print(f"# {name}\n" + " ".join(cmd))
        return 0
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for name, cmd in cmds:
            print(f"[full_stack] start {name}: {' '.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_PATHS.root), env=env, start_new_session=True)
            if name == "ue_editor":
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"{name} exited with code {code}")
                print("[full_stack] ue_editor session ready", flush=True)
                continue
            if name == "scene_init_bridge":
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"{name} exited with code {code}")
                print("[full_stack] scene_init_bridge completed initial scene apply", flush=True)
                continue
            procs.append((name, proc))
            time.sleep(0.5)
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"{name} exited with code {code}")
                    print(f"[full_stack] {name} exited cleanly; stopping remaining children", flush=True)
                    return 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[full_stack] interrupted; stopping children", flush=True)
    finally:
        for _name, proc in reversed(procs):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    proc.terminate()
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
