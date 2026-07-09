#!/usr/bin/env python3
"""Genesis mirror (window B): read-only SHM subscriber, no robot TCP.

  source env.sh
  python apps/joint_admittance_8dof/run_with_twin.py
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

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
    args = ap.parse_args()

    if args.dry_run:
        return 0

    if args.backend == "cuda":
        from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import (
            ensure_cuda_driver_for_taichi,
        )

        ensure_cuda_driver_for_taichi(require_gpu=True)

    twin: DigitalTwinMirror | None = None

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
        _run_subscribe_loop(bus=bus, twin=twin, shm_name=shm_name)
    finally:
        bus.stop()
        if twin is not None:
            twin.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
