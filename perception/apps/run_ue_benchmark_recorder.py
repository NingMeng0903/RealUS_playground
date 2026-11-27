#!/usr/bin/env python3
"""Record applied UE canonical frames and strict four-camera JPEG groups.

The UE watcher must be running with ``prepare_tpose_avatar`` (or an already
loaded scene) and the AmongUs TCP mux must publish the camera stream.  Each
canonical JSONL row is sent to the watcher; the recorder waits for UE's
apply/capture ACK, then waits for all camera packets carrying the same
``capture_group_id`` before writing the frame manifest.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from projects.genesis_ue_sync.multiview_realtime.benchmark.ue_benchmark_recorder import (  # noqa: E402
    DEFAULT_ACK_CONNECT,
    DEFAULT_RUNTIME_ACK,
    UEBenchmarkRecorder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", type=Path)
    parser.add_argument(
        "--recording-dir",
        type=Path,
        help="Directory containing genesis_canonical.jsonl (convenience for pose/runtime recordings).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--loop-id", default="amass")
    parser.add_argument(
        "--camera-connect",
        default=os.environ.get("REALUS_UE_CAMERA_CONNECT", "tcp://127.0.0.1:17366"),
    )
    parser.add_argument("--ack-connect", default=DEFAULT_ACK_CONNECT)
    parser.add_argument("--camera-names", nargs="+", default=["cam1", "cam2", "cam3", "cam4"])
    parser.add_argument("--topic", default="amongus_camera_frame_v1")
    parser.add_argument("--runtime-ack-connect", default=DEFAULT_RUNTIME_ACK)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--capture-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--keep-duplicate-motion-frames",
        action="store_true",
        help="Record every canonical row instead of one row per (loop_id,motion_frame_index).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical_jsonl = args.canonical_jsonl
    if canonical_jsonl is None and args.recording_dir is not None:
        root = args.recording_dir.expanduser().resolve()
        candidates = (
            root / "genesis_canonical.jsonl",
            root / "canonical_state.jsonl",
            root / "canonical.jsonl",
        )
        canonical_jsonl = next((candidate for candidate in candidates if candidate.is_file()), None)
        if canonical_jsonl is None:
            matches = sorted(root.glob("**/*canonical*.jsonl"))
            canonical_jsonl = matches[0] if matches else None
    if canonical_jsonl is None:
        raise SystemExit("one of --canonical-jsonl or --recording-dir with a canonical JSONL log is required")
    recorder = UEBenchmarkRecorder(
        canonical_jsonl=canonical_jsonl,
        output_root=args.output_root,
        session_id=args.session_id,
        loop_id=args.loop_id,
        camera_connect=args.camera_connect,
        ack_connect=args.ack_connect,
        camera_names=args.camera_names,
        topic=args.topic,
        runtime_ack_connect=(args.runtime_ack_connect or None),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_frames=args.max_frames,
        timeout_s=args.timeout_s,
        capture_timeout_s=args.capture_timeout_s,
        deduplicate_motion_frames=not args.keep_duplicate_motion_frames,
    )
    summary = recorder.run()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
