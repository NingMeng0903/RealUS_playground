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

import numpy as np

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
            # Controller down (limit recovery / restart): freeze rail filter so
            # the next live sample hard-teleports to the real encoder pose.
            twin.freeze_rail()
            if last_session_id != 0:
                print("rm75 twin: controller offline — freezing rail pose", flush=True)
                last_session_id = 0
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
            twin.reset_rail_filter()
            try:
                twin.sync_once()
                rail = getattr(bus, "last_rail_m", float("nan"))
                if rail == rail:  # finite
                    print(
                        f"rm75 twin: rail locked to encoder @ {float(rail) * 1000:.1f} mm",
                        flush=True,
                    )
            except AssertionError as exc:
                # Genesis/quadrants fastcache can trip after controller restart
                # while the viewer process is reused / partially stale.
                print(
                    f"rm75 twin: Genesis cache glitch ({exc}); "
                    "continuing (background sync will retry). "
                    "If it keeps failing: close viewer, rm -rf ~/.cache/quadrants, restart B.",
                    flush=True,
                )

        time.sleep(0.1)


def main() -> int:
    # Quadrants fastcache can assert on viewer reopen after controller restart.
    import os

    os.environ.setdefault("GS_ENABLE_FASTCACHE", "0")

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
    ap.add_argument(
        "--no-track-subscribe",
        action="store_true",
        help="Do not subscribe to orange SMPL-X mesh (robot-only twin)",
    )
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

    track_subscribe = ""
    if not args.no_track_subscribe:
        track_subscribe = str(args.track_subscribe or os.environ.get("AMONGUS_GENESIS_TRACK_SUBSCRIBE", "")).strip()

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
        # Seed Genesis from the first real encoder rail — never show URDF rail_y=0
        # then animate to the true pose (looks like a snap back to 0 / post_home).
        print("rm75 twin: waiting for first encoder rail sample …", flush=True)
        seeded = False
        t_seed = time.monotonic()
        while time.monotonic() - t_seed < 120.0:
            if bus.is_live():
                q8 = bus.q_meas_8dof()
                if q8 is not None:
                    q = np.asarray(q8, dtype=float).reshape(-1)
                    if q.size >= 1 and np.isfinite(q[0]) and -0.05 <= float(q[0]) <= 0.85:
                        scene.set_joint_positions(q)
                        scene.step()
                        print(
                            f"rm75 twin: seeded @ rail={float(q[0]) * 1000:.1f} mm "
                            "(live encoder, not 0 / post_home)",
                            flush=True,
                        )
                        seeded = True
                        break
            time.sleep(0.05)
        if not seeded:
            print(
                "rm75 twin: WARN no encoder rail yet — start window A first; "
                "viewer will stay at URDF default until live",
                flush=True,
            )

        twin = DigitalTwinMirror(
            bus,
            scene,
            hz=args.twin_hz,
            rail_extrapolate_s=0.12,
        )
        if seeded:
            twin.reset_rail_filter()
            # Match filter to already-displayed pose so the first sync does not
            # invent velocity from a stale 0.
            try:
                q_now = bus.q_meas_8dof()
                if q_now is not None:
                    twin._rail_x = float(q_now[0])
                    twin._rail_sample = float(q_now[0])
                    twin._rail_v = 0.0
                    twin._rail_t = time.monotonic()
                    twin._rail_have = True
            except Exception:
                pass
        twin.start_background()
        print(
            f"rm75 twin: rail display extrapolate≤120 ms @ {args.twin_hz:.0f} Hz",
            flush=True,
        )

        if track_subscribe or args.canonical_human_source in ("fitted", "robot") or args.smplx_npz:
            try:
                from rm75_control.control.joint_admittance_8dof.viewer.human_overlay import (
                    TwinHumanOverlay,
                    TwinHumanOverlayConfig,
                )

                alpha = max(0, min(255, int(args.track_mesh_alpha)))
                overlay = TwinHumanOverlay(
                    scene,
                    TwinHumanOverlayConfig(
                        track_subscribe=str(track_subscribe or "tcp://127.0.0.1:5598"),
                        anatomy_subscribe=str(args.anatomy_subscribe),
                        canonical_bind=str(args.canonical_bind),
                        canonical_human_source=str(args.canonical_human_source),
                        smplx_npz=args.smplx_npz,
                        track_mesh_rgba=(250, 122, 31, alpha),
                        enable_track=bool(track_subscribe) or args.canonical_human_source == "fitted",
                        enable_anatomy=not args.no_anatomy,
                        enable_canonical=args.canonical_human_source in ("fitted", "robot"),
                    ),
                )
                overlay.set_robot_q_provider(lambda: bus.q_meas_8dof(0.0) if bus.is_live() else None)
                overlay.start()
                print(
                    f"rm75 twin: human overlay track={track_subscribe or '-'} "
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
