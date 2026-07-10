from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import sys
import threading
import time
from typing import Any

from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import TOPIC_CAMERA_FRAME_V1

_AGENT_DEBUG_LOG = "/home/camp/.cursor/debug-logs/debug-05706c.log"
_AGENT_LOGGED_CAMERAS: set[str] = set()


def _agent_debug_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "05706c",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _read_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _handle_connection(conn: socket.socket, publisher: Any, topic: bytes, publish_lock: threading.Lock) -> None:
    try:
        while True:
            meta_len_bytes = _read_exact(conn, 4)
            meta_len = struct.unpack(">I", meta_len_bytes)[0]
            meta_bytes = _read_exact(conn, meta_len)
            meta = json.loads(meta_bytes.decode("utf-8"))
            camera_name = str(meta.get("camera_name") or meta.get("camera_frame_id") or "")
            if camera_name and camera_name not in _AGENT_LOGGED_CAMERAS:
                _AGENT_LOGGED_CAMERAS.add(camera_name)
                extrinsics = dict(meta.get("extrinsics") or {})
                # #region agent log
                _agent_debug_log(
                    hypothesis_id="C" if camera_name == "cam_top" else "E",
                    location="amongus_ue_tcp_camera_mux.py:_handle_connection",
                    message="first UE TCP camera frame metadata",
                    data={
                        "camera_name": camera_name,
                        "scene_capture_flip_u": bool(meta.get("scene_capture_flip_u", False)),
                        "scene_capture_flip_v": bool(meta.get("scene_capture_flip_v", False)),
                        "ue_location_cm": extrinsics.get("ue_location_cm"),
                        "ue_rotation_deg": extrinsics.get("ue_rotation_deg"),
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                    },
                )
                # #endregion
            img_len_bytes = _read_exact(conn, 4)
            img_len = struct.unpack(">I", img_len_bytes)[0]
            img_bytes = _read_exact(conn, img_len)
            # ZMQ sockets are not thread-safe; serialize multipart sends so concurrent
            # per-camera connection threads cannot interleave/corrupt each other's frames.
            with publish_lock:
                publisher.send_multipart([topic, json.dumps(meta, ensure_ascii=True).encode("utf-8"), img_bytes])
    except Exception as exc:
        logging.debug("connection ended: %s", exc)
    finally:
        try:
            conn.close()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCP ingress from UE capture plugin -> ZMQ multipart publisher.")
    parser.add_argument("--listen-host", type=str, default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=17355)
    parser.add_argument("--pub-bind", type=str, default="tcp://127.0.0.1:17356")
    parser.add_argument("--topic", type=str, default=TOPIC_CAMERA_FRAME_V1)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import zmq
    except ImportError as exc:
        logging.error("pyzmq required: %s", exc)
        return 2

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(str(args.pub_bind))
    topic_bytes = str(args.topic).encode("utf-8")
    publish_lock = threading.Lock()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((str(args.listen_host), int(args.listen_port)))
    server.listen(8)
    logging.info("Listening TCP %s:%s publishing ZMQ %s topic=%s", args.listen_host, args.listen_port, args.pub_bind, args.topic)

    try:
        while True:
            conn, addr = server.accept()
            logging.info("accepted %s", addr)
            threading.Thread(
                target=_handle_connection, args=(conn, pub, topic_bytes, publish_lock), daemon=True
            ).start()
    finally:
        server.close()
        pub.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
