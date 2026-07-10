#!/usr/bin/env python3
"""Publish anatomy asset display/lifecycle control messages to Genesis."""

from __future__ import annotations

import argparse
import json
import time

from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_ANATOMY_ASSET_PUB_BIND,
    TOPIC_ANATOMY_ASSET_V1,
    anatomy_asset_control_to_dict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", type=str, default=DEFAULT_ANATOMY_ASSET_PUB_BIND)
    p.add_argument("--model-id", type=str, default="patient_anatomy")
    p.add_argument(
        "--action",
        type=str,
        required=True,
        choices=("delete", "clear_all", "set_visible", "set_opacity", "restore_opacity", "set_render_mode"),
    )
    p.add_argument("--visible", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--opacity", type=float, default=None)
    p.add_argument("--mode", type=str, default=None, choices=("hidden", "transparent", "opaque"))
    p.add_argument("--duration-s", type=float, default=1.0)
    p.add_argument("--rate-hz", type=float, default=10.0)
    return p.parse_args()


def main() -> int:
    import zmq

    args = parse_args()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(str(args.bind))
    payload = anatomy_asset_control_to_dict(
        action=str(args.action),
        model_id=str(args.model_id),
        visible=bool(args.visible) if str(args.action) == "set_visible" else None,
        opacity=args.opacity,
        mode=args.mode,
        timestamp_ns=time.time_ns(),
    )
    topic = TOPIC_ANATOMY_ASSET_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    time.sleep(0.2)
    end = time.time() + max(0.1, float(args.duration_s))
    interval = 1.0 / max(1.0, float(args.rate_hz))
    sent = 0
    while time.time() < end:
        sock.send_multipart([topic, body])
        sent += 1
        time.sleep(interval)
    sock.close(0)
    print(f"published anatomy control action={args.action} model_id={args.model_id} sent={sent} bind={args.bind}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
