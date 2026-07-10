from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
for _parent in _THIS.parents:
    if _parent.name == "src" and (_parent / "common" / "project.py").is_file():
        _src = str(_parent)
        if _src not in sys.path:
            sys.path.insert(0, _src)
        break

from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import TOPIC_CANONICAL_SCENE_V1
from projects.genesis_ue_sync.integrations.ue.session import (
    CANONICAL_UDP_DEFAULT_HOST,
    CANONICAL_UDP_DEFAULT_PORT,
    CanonicalSceneTickUdpSender,
    EditorCommandResult,
    EditorSessionPaths,
    enqueue_apply_canonical_scene_tick,
    parse_canonical_udp_endpoint,
)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if parent.name == "src" and (parent / "common" / "project.py").is_file():
            sp = str(parent)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe canonical ZMQ and enqueue UE LiveSync ticks (optional).")
    parser.add_argument(
        "--canonical-connect",
        type=str,
        default="tcp://127.0.0.1:5599",
        help="Connect address for Genesis canonical ZMQ PUB (SUB connects here).",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="UE editor session directory (default: AMONGUS_SESSION_DIR or SESSION_DIR env).",
    )
    parser.add_argument("--topic", type=str, default=TOPIC_CANONICAL_SCENE_V1)
    parser.add_argument("--recv-timeout-ms", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--max-send-hz",
        type=float,
        default=60.0,
        help="Max canonical ticks per second forwarded to UE (0 disables bridge-side rate limiting).",
    )
    parser.add_argument(
        "--zmq-rcvhwm",
        type=int,
        default=4,
        help="ZMQ receive high-water mark; low values keep only recent visual sync states.",
    )
    parser.add_argument(
        "--diagnose-every",
        type=int,
        default=500,
        help="Print UE-side last-result robot_updates once per N enqueued ticks (0 disables).",
    )
    parser.add_argument(
        "--ue-canonical-udp",
        type=str,
        default=f"{CANONICAL_UDP_DEFAULT_HOST}:{CANONICAL_UDP_DEFAULT_PORT}",
        help=(
            "UDP endpoint (host:port) where the UE editor watcher listens for canonical ticks. "
            "Defaults to 127.0.0.1:5601; pass empty to force the legacy file-based path."
        ),
    )
    parser.add_argument(
        "--no-ue-canonical-udp",
        action="store_true",
        help="Disable the UDP fast path and fall back to the legacy file-based command queue.",
    )
    return parser.parse_args()


def _resolve_session_dir(raw: Path | None) -> Path:
    if raw is not None and str(raw).strip():
        return Path(raw).expanduser().resolve()
    for key in ("AMONGUS_SESSION_DIR", "SESSION_DIR"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return Path(value).expanduser().resolve()
    raise SystemExit(
        "missing session dir: pass --session-dir or export AMONGUS_SESSION_DIR / SESSION_DIR"
    )


def _log_recent_command_result(paths: EditorSessionPaths, last_command_id: str | None) -> None:
    if not last_command_id:
        return
    result = EditorCommandResult.load(paths, last_command_id)
    if result is None:
        return
    payload = result.payload or {}
    robot_updates = payload.get("robot_updates") or []
    errors = [item for item in robot_updates if isinstance(item, dict) and item.get("error")]
    if errors:
        logging.warning(
            "UE canonical tick had robot_update errors: %s",
            json.dumps(errors)[:512],
        )
    human = payload.get("human") or {}
    if isinstance(human, dict) and human.get("reason") not in (None, "", "ok"):
        logging.debug("UE canonical tick human note: %s", json.dumps(human)[:256])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _ensure_src_on_path()
    args = parse_args()

    try:
        import zmq
    except ImportError as exc:
        logging.error("pyzmq required: %s", exc)
        return 2

    session_dir = _resolve_session_dir(args.session_dir)
    paths = EditorSessionPaths(session_dir)
    topic_bytes = str(args.topic).encode("utf-8")

    try:
        from projects.genesis_ue_sync.sim_platform.sync import human_align_diag as _human_align_diag
    except Exception:
        _human_align_diag = None

    udp_sender: CanonicalSceneTickUdpSender | None = None
    udp_endpoint = "" if args.no_ue_canonical_udp else str(args.ue_canonical_udp or "").strip()
    if udp_endpoint:
        host, port = parse_canonical_udp_endpoint(udp_endpoint)
        udp_sender = CanonicalSceneTickUdpSender(host=host, port=port)
        logging.info("canonical UDP sender ready -> %s:%s (file path disabled).", host, port)
    else:
        logging.info("canonical UDP path disabled by flag; using legacy JSON-file queue.")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, int(max(args.recv_timeout_ms, 1)))
    try:
        sock.setsockopt(zmq.RCVHWM, int(max(args.zmq_rcvhwm, 1)))
    except Exception:
        pass
    sock.connect(str(args.canonical_connect))
    sock.setsockopt(zmq.SUBSCRIBE, topic_bytes)

    processed = 0
    skipped_for_rate = 0
    last_dispatch_mono = 0.0
    min_dispatch_interval_s = 0.0
    if float(args.max_send_hz) > 0.0:
        min_dispatch_interval_s = 1.0 / max(float(args.max_send_hz), 1.0e-6)
    last_request_id: str | None = None
    started_mono = time.monotonic()
    last_idle_warn_mono = 0.0
    logging.info(
        "UE canonical bridge listening topic=%s endpoint=%s max_send_hz=%.3f rcvhwm=%s",
        args.topic,
        args.canonical_connect,
        float(args.max_send_hz),
        int(max(args.zmq_rcvhwm, 1)),
    )

    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                now_idle = time.monotonic()
                if (
                    processed == 0
                    and now_idle - started_mono >= 5.0
                    and now_idle - last_idle_warn_mono >= 5.0
                ):
                    logging.warning(
                        "no canonical ZMQ messages on %s yet — terminal 7 (amass_bed_capsule_demo) "
                        "must PUB here (default --canonical-zmq-bind tcp://127.0.0.1:5599 or export "
                        "AMONGUS_GENESIS_CANONICAL_ZMQ_BIND in terminal 7, not terminal 6)",
                        args.canonical_connect,
                    )
                    last_idle_warn_mono = now_idle
                continue
            except zmq.ZMQError as exc:
                logging.warning("zmq error (retrying): %s", exc)
                time.sleep(0.5)
                continue

            if len(parts) < 2:
                logging.debug("skip short multipart len=%s", len(parts))
                continue
            if parts[0] != topic_bytes:
                continue
            body = parts[-1]
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                logging.warning("skip bad json: %s", exc)
                continue

            now = time.monotonic()
            if min_dispatch_interval_s > 0.0 and last_dispatch_mono > 0.0:
                if now - last_dispatch_mono < min_dispatch_interval_s:
                    skipped_for_rate += 1
                    continue

            try:
                if udp_sender is not None:
                    last_request_id = udp_sender.send_canonical_state(payload)
                else:
                    cmd = enqueue_apply_canonical_scene_tick(paths, payload)
                    last_request_id = cmd.request_id
            except OSError as exc:
                logging.warning("dispatch failed (session dir / udp): %s", exc)
                continue
            except Exception as exc:
                logging.warning("dispatch failed: %s", exc)
                continue

            processed += 1
            last_dispatch_mono = now
            _br_every = 20
            try:
                _br_every = max(int(os.environ.get("AMONGUS_DEBUG_BRIDGE_EVERY", "20") or "20"), 1)
            except ValueError:
                _br_every = 20
            if processed <= 4 or processed % _br_every == 0:
                try:
                    from projects.genesis_ue_sync.sim_platform.sync.human_align_diag import (
                        agent_debug_ndjson,
                        amongus_debug_ndjson_enabled,
                    )

                    if amongus_debug_ndjson_enabled():
                        hu = payload.get("human") if isinstance(payload.get("human"), dict) else {}
                        rt = hu.get("root_translation_world_m")
                        agent_debug_ndjson(
                            hypothesis_id="BRIDGE_EGRESS",
                            location="run_canonical_zmq_ue_bridge.py:main_loop",
                            message="canonical payload human root at bridge dispatch",
                            data={
                                "processed": int(processed),
                                "sim_step_index": payload.get("sim_step_index"),
                                "frame_index": payload.get("frame_index"),
                                "motion_frame_index": hu.get("motion_frame_index"),
                                "root_translation_world_m": (
                                    [float(rt[i]) for i in range(3)]
                                    if isinstance(rt, (list, tuple)) and len(rt) >= 3
                                    else None
                                ),
                                "has_body_pose": bool(hu.get("smpl_body_pose_axis_angle")),
                            },
                        )
                except Exception:
                    pass
            if args.log_every > 0 and processed % int(args.log_every) == 0:
                if udp_sender is not None:
                    logging.info(
                        "udp canonical ticks=%s sent=%s drops=%s rate_skipped=%s last_step=%s",
                        processed,
                        udp_sender.send_count,
                        udp_sender.dropped_too_large,
                        skipped_for_rate,
                        payload.get("sim_step_index"),
                    )
                else:
                    logging.info(
                        "enqueued canonical ticks=%s rate_skipped=%s last_step=%s",
                        processed,
                        skipped_for_rate,
                        payload.get("sim_step_index"),
                    )
            if (
                _human_align_diag is not None
                and _human_align_diag.human_align_diag_enabled()
            ):
                every = _human_align_diag.human_align_diag_every(200)
                if every > 0 and processed % int(every) == 0:
                    hu = payload.get("human") if isinstance(payload.get("human"), dict) else {}
                    rt = hu.get("root_translation_world_m")
                    if isinstance(rt, (list, tuple)) and len(rt) >= 3:
                        logging.info(
                            "%s (bridge tick sim_step=%s)",
                            _human_align_diag.human_root_genesis_vs_ue_line(
                                rt, step=int(payload.get("sim_step_index", 0) or 0)
                            ),
                            payload.get("sim_step_index"),
                        )
            if args.diagnose_every > 0 and processed % int(args.diagnose_every) == 0:
                _log_recent_command_result(paths, last_request_id)
    finally:
        if udp_sender is not None:
            udp_sender.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
