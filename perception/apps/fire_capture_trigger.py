#!/usr/bin/env python3
"""Send one capture trigger to run_capture_trigger_service."""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--connect", type=str, default="tcp://127.0.0.1:17357")
    ap.add_argument("--topic", type=str, default="realus_capture_trigger_v1")
    ap.add_argument("--label", type=str, default="manual")
    args = ap.parse_args()

    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.connect(str(args.connect))
    time.sleep(0.15)
    payload = {"schema_version": 1, "label": str(args.label), "wall_time_ns": time.time_ns()}
    sock.send_multipart([str(args.topic).encode("utf-8"), json.dumps(payload).encode("utf-8")])
    sock.close(0)
    print(f"trigger sent connect={args.connect} label={args.label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
