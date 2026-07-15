#!/usr/bin/env python3
"""Genesis mirror (window B): read-only SHM subscriber, no robot TCP.

  source env_viewer.sh   # or RealUS env.sh + genesis
  python apps/joint_admittance_8dof/run_with_twin.py

Optional human overlay (requires RealUS src on PYTHONPATH):
  --track-subscribe tcp://127.0.0.1:5598
  --canonical-human-source fitted
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from rm75_control.control.admittance_common.state_relay import RelayStateBus, relay_shm_has_publisher
from rm75_control.control.joint_admittance_8dof.viewer import DigitalTwinMirror, RailGenesisConfig, RailGenesisScene


def _run_subscribe_loop(
    *,
    bus: RelayStateBus,
    twin: DigitalTwinMirror,
    shm_name: str,
) -> None:
    stop = False
    last_session_id = 0

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    while not stop:
        if twin.viewer_closed:
            stop = True
            break

        if not relay_shm_has_publisher(shm_name):
            time.sleep(0.2)
            continue

        live = bus.is_live()
        sid = int(bus.session_id) if live else last_session_id

        if sid != last_session_id and sid != 0:
            if last_session_id != 0:
                print("rm75 twin: reconnected to controller", flush=True)
            else:
                print("rm75 twin: running", flush=True)
            last_session_id = sid
            twin.sync_once()

        time.sleep(0.1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--subscribe",
        metavar="NAME",
        default="rm75_state",
        help="SHM segment from window A (default rm75_state)",
    )
    ap.add_argument("--headless", action="store_true", help="No Genesis window (kinematic sync only)")
    ap.add_argument("--twin-hz", type=float, default=60.0)
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--track-subscribe", type=str, default="", help="ZMQ track overlay (e.g. tcp://127.0.0.1:5598)")
    ap.add_argument("--anatomy-subscribe", type=str, default="tcp://127.0.0.1:5601")
    ap.add_argument("--canonical-bind", type=str, default="tcp://127.0.0.1:5599")
    ap.add_argument(
        "--canonical-human-source",
        type=str,
        default="none",
        choices=["none", "robot", "fitted"],
        help="none=off; robot=5599 robot only (no human); fitted=5599 robot+EasyMocap human",
    )
    ap.add_argument("--smplx-npz", type=Path, default=None, help="Optional static smplx_result.npz for anatomy/canonical")
    ap.add_argument("--no-anatomy", action="store_true")
    ap.add_argument(
        "--track-mesh-alpha",
        type=int,
        default=120,
        metavar="0-255",
        help="Orange SMPL-X skin opacity (default 120; anatomy draws underneath in solid pass)",
    )
    args = ap.parse_args()

    if args.dry_run:
        return 0

    if args.backend == "cuda":
        from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import (
            ensure_cuda_driver_for_taichi,
        )

        ensure_cuda_driver_for_taichi(require_gpu=True)

    twin: DigitalTwinMirror | None = None
    overlay = None

    scene = RailGenesisScene(
        RailGenesisConfig(
            backend=args.backend,
            show_viewer=not args.headless,
        )
    )
    try:
        scene.build()
    except ImportError as exc:
        print(f"Genesis unavailable: {exc}", file=sys.stderr, flush=True)
        return 1

    shm_name = str(args.subscribe)
    print(f"rm75 twin: waiting for {shm_name!r} …", flush=True)
    bus = RelayStateBus(shm_name)
    try:
        twin = DigitalTwinMirror(bus, scene, hz=args.twin_hz)
        twin.start_background()

        if args.track_subscribe or args.canonical_human_source in ("fitted", "robot") or args.smplx_npz:
            try:
                from rm75_control.control.joint_admittance_8dof.viewer.human_overlay import (
                    TwinHumanOverlay,
                    TwinHumanOverlayConfig,
                )

                alpha = max(0, min(255, int(args.track_mesh_alpha)))
                overlay = TwinHumanOverlay(
                    scene,
                    TwinHumanOverlayConfig(
                        track_subscribe=str(args.track_subscribe or "tcp://127.0.0.1:5598"),
                        anatomy_subscribe=str(args.anatomy_subscribe),
                        canonical_bind=str(args.canonical_bind),
                        canonical_human_source=str(args.canonical_human_source),
                        smplx_npz=args.smplx_npz,
                        track_mesh_rgba=(250, 122, 31, alpha),
                        enable_track=bool(args.track_subscribe) or args.canonical_human_source == "fitted",
                        enable_anatomy=not args.no_anatomy,
                        enable_canonical=args.canonical_human_source in ("fitted", "robot"),
                    ),
                )
                overlay.set_robot_q_provider(lambda: bus.q_meas_8dof(0.0) if bus.is_live() else None)
                overlay.start()
                print(
                    f"rm75 twin: human overlay track={args.track_subscribe or '-'} "
                    f"canonical={args.canonical_human_source}",
                    flush=True,
                )
            except Exception as exc:
                print(f"rm75 twin: human overlay disabled ({exc})", flush=True)

        _run_subscribe_loop(bus=bus, twin=twin, shm_name=shm_name)
    finally:
        if overlay is not None:
            overlay.stop()
        bus.stop()
        if twin is not None:
            twin.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
