from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import TOPIC_CAMERA_FRAME_V1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview realtime UE camera frames from amongus_ue_tcp_camera_mux.py.")
    parser.add_argument("--connect", type=str, default="tcp://127.0.0.1:17356")
    parser.add_argument("--topic", type=str, default=TOPIC_CAMERA_FRAME_V1)
    parser.add_argument("--camera-names", type=str, nargs="*", default=[])
    parser.add_argument("--tile-width", type=int, default=426)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--window-name", type=str, default="AmongUS UE Cameras")
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument(
        "--separate-windows",
        action="store_true",
        help="Open one OS window per camera (default: a single tiled window).",
    )
    return parser.parse_args()


def _decode_jpeg(image_bytes: bytes) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install opencv-python to preview UE camera frames.") from exc
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame.")
    return frame


def _tile_frames(frames: dict[str, tuple[dict, np.ndarray]], names: list[str], *, size: tuple[int, int]) -> np.ndarray:
    import cv2

    w, h = int(size[0]), int(size[1])
    if not names:
        names = sorted(frames)
    if not names:
        return np.zeros((h, w, 3), dtype=np.uint8)
    tiles: list[np.ndarray] = []
    for name in names:
        item = frames.get(name)
        if item is None:
            tile = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(tile, f"{name}: waiting", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            tiles.append(tile)
            continue
        meta, frame = item
        tile = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        label = f"{name} f={meta.get('frame_index', '?')} t={meta.get('sim_time_ns', 0)}"
        cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import cv2
        import zmq
    except ImportError as exc:
        logging.error("Required dependency missing: %s", exc)
        return 2

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 250)
    sock.connect(str(args.connect))
    sock.setsockopt(zmq.SUBSCRIBE, str(args.topic).encode("utf-8"))

    wanted = [str(x) for x in args.camera_names]
    frames: dict[str, tuple[dict, np.ndarray]] = {}
    count = 0
    last_draw = 0.0
    separate_windows = bool(args.separate_windows)
    logging.info(
        "Watching UE camera frames endpoint=%s topic=%s mode=%s",
        args.connect,
        args.topic,
        "separate_windows" if separate_windows else "tiled",
    )

    def _draw_separate(now: float) -> bool:
        names = wanted or sorted(frames)
        for name in names:
            item = frames.get(name)
            if item is None:
                placeholder = np.zeros((int(args.tile_height), int(args.tile_width), 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{name}: waiting", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                cv2.imshow(name, placeholder)
                continue
            meta, frame = item
            label = f"{name} f={meta.get('frame_index', '?')} t={meta.get('sim_time_ns', 0)}"
            annotated = frame.copy()
            cv2.putText(annotated, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(name, annotated)
        return cv2.waitKey(1) & 0xFF in (27, ord("q"))

    def _draw_tiled() -> bool:
        canvas = _tile_frames(frames, wanted, size=(int(args.tile_width), int(args.tile_height)))
        cv2.imshow(str(args.window_name), canvas)
        return cv2.waitKey(1) & 0xFF in (27, ord("q"))

    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            now = time.perf_counter()
            if now - last_draw > 0.25:
                quit_signal = _draw_separate(now) if separate_windows else _draw_tiled()
                if quit_signal:
                    break
                last_draw = now
            continue
        if len(parts) < 3:
            continue
        try:
            meta = json.loads(parts[1].decode("utf-8"))
            name = str(meta.get("camera_name") or meta.get("camera_frame_id") or "camera")
            frames[name] = (meta, _decode_jpeg(parts[2]))
        except Exception as exc:
            logging.warning("Skipping bad camera frame: %s", exc)
            continue
        count += 1
        if args.log_every > 0 and count % int(args.log_every) == 0:
            logging.info("received=%s cameras=%s", count, sorted(frames))
        quit_signal = _draw_separate(time.perf_counter()) if separate_windows else _draw_tiled()
        if quit_signal:
            break

    sock.close(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
