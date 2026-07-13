#!/usr/bin/env python3
"""ZMQ trigger service: fire Window 8 SMPL-X capture on demand (Phase 1 optional).

Publish a trigger message to bind endpoint; service runs run_smplx_capture once per trigger.

Example trigger (another terminal)::

  python -m perception.apps.fire_capture_trigger --connect tcp://127.0.0.1:17357
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

TOPIC = "realus_capture_trigger_v1"


def _repo_root() -> Path:
    return Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def _smplx_output_root(repo: Path) -> Path:
    return Path(os.environ.get("REALUS_SMPLX_OUTPUT_ROOT", repo / "smplx_outputs"))


def _run_capture(repo: Path, *, config: Path, connect: str, extra_argv: list[str]) -> int:
    py = os.environ.get("PY", sys.executable)
    cmd = [
        py,
        str(repo / "perception/apps/run_smplx_capture.py"),
        "--config",
        str(config),
        "--connect",
        str(connect),
        "--output-root",
        str(_smplx_output_root(repo)),
        "--publish-kind",
        "smplx_mesh",
        "--publish-genesis",
        *extra_argv,
    ]
    env = dict(os.environ)
    src = str((repo / "src").resolve())
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
    logging.info("capture exec: %s", " ".join(cmd))
    return int(subprocess.call(cmd, cwd=str(repo), env=env))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind", type=str, default="tcp://127.0.0.1:17357")
    ap.add_argument("--topic", type=str, default=TOPIC)
    ap.add_argument("--config", type=Path, default=repo / "configs/tracking/realus_dwpose_easymocap.yaml")
    ap.add_argument("--camera-connect", type=str, default="tcp://127.0.0.1:17356")
    ap.add_argument("--cooldown-s", type=float, default=3.0)
    args, extra = ap.parse_known_args()

    try:
        import zmq
    except ImportError as exc:
        logging.error("pyzmq required: %s", exc)
        return 2

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.bind(str(args.bind))
    topic_b = str(args.topic).encode("utf-8")
    sock.setsockopt(zmq.SUBSCRIBE, topic_b)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    logging.info("capture trigger listening bind=%s topic=%s", args.bind, args.topic)

    last_fire = 0.0
    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            continue
        except KeyboardInterrupt:
            break
        if len(parts) < 2:
            continue
        now = time.monotonic()
        if now - last_fire < float(args.cooldown_s):
            logging.info("trigger ignored (cooldown)")
            continue
        last_fire = now
        label = "manual"
        try:
            payload = json.loads(parts[-1].decode("utf-8"))
            label = str(payload.get("label", label))
        except Exception:
            pass
        logging.info("capture trigger received label=%s", label)
        rc = _run_capture(repo, config=args.config, connect=str(args.camera_connect), extra_argv=list(extra))
        logging.info("capture finished exit=%s", rc)

    sock.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
