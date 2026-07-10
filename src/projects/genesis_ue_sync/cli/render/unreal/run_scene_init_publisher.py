"""Publish a SceneInitMessageV1 over ZMQ so the UE bridge can stand up the scene.

Genesis is the source of truth for what UE should build. This CLI reads a scene yaml
(and optional augmentation), normalises path fields to repo-relative form, attaches a
SHA256 hash and session id, and publishes the dict on the configured ZMQ PUB endpoint.

Late joiners (UE bridge launched after this publisher) get the latest payload via the
``--repeat-s`` re-publish loop. Pass ``--once`` for one-shot mode.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in _THIS.parents if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import TOPIC_SCENE_INIT_V1
from projects.genesis_ue_sync.sim_platform.state.scene_init import (
    build_scene_init_message,
    scene_init_message_to_dict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-spec", type=Path, required=True, help="Path to scene yaml/json (Genesis source of truth).")
    p.add_argument("--augmentation-spec", type=Path, default=None, help="Optional augmentation yaml/json.")
    p.add_argument("--bind", type=str, default="tcp://127.0.0.1:5588", help="ZMQ PUB bind endpoint.")
    p.add_argument("--topic", type=str, default=TOPIC_SCENE_INIT_V1)
    p.add_argument("--repeat-s", type=float, default=2.0, help="Re-publish interval (seconds). Use 0 with --once for one-shot.")
    p.add_argument("--once", action="store_true", help="Publish once and exit.")
    p.add_argument("--session-id", type=str, default=None, help="Override session id (defaults to AMONGUS_SESSION_ID env).")
    p.add_argument(
        "--robot-model",
        type=str,
        default="",
        help="Override robot model_id (panda_urdf, rm75_6f). Also AMONGUS_ROBOT_MODEL env.",
    )
    p.add_argument("--print-payload", action="store_true", help="Print the JSON payload to stdout (one-shot).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import zmq
    except ImportError as exc:
        logging.error("pyzmq required: %s", exc)
        return 2

    message = build_scene_init_message(
        scene_spec_path=args.scene_spec,
        augmentation_spec_path=args.augmentation_spec,
        session_id=args.session_id,
        robot_model=str(args.robot_model or ""),
    )
    payload = scene_init_message_to_dict(message)
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    topic_bytes = str(args.topic).encode("utf-8")

    if args.print_payload:
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:4096])

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(str(args.bind))
    logging.info(
        "scene_init publisher bind=%s topic=%s payload_hash=%s session_id=%s",
        args.bind,
        args.topic,
        message.payload_hash_sha256[:12],
        message.session_id or "(none)",
    )
    time.sleep(0.2)

    publish_count = 0
    try:
        while True:
            sock.send_multipart([topic_bytes, body])
            publish_count += 1
            if args.once:
                logging.info("scene_init published once (count=%s); exiting.", publish_count)
                break
            if publish_count % 5 == 1:
                logging.info("scene_init heartbeat publish_count=%s", publish_count)
            time.sleep(max(float(args.repeat_s), 0.05))
    except KeyboardInterrupt:
        logging.info("scene_init publisher interrupted (count=%s).", publish_count)
    finally:
        sock.close(linger=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
