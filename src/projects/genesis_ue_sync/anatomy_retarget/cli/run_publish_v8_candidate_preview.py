#!/usr/bin/env python3
"""Export and publish a rejected V8 subject as a Genesis preview overlay.

This command is intentionally separate from trusted publication.  It requires
an explicit acknowledgement flag and labels the exported rig as a
non-publishable candidate so visual inspection cannot update trusted latest.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_shape_hash
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import save_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_subject_runtime,
)
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_ANATOMY_ASSET_PUB_BIND,
    TOPIC_ANATOMY_ASSET_V1,
    anatomy_asset_control_to_dict,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--bind", default=DEFAULT_ANATOMY_ASSET_PUB_BIND)
    parser.add_argument("--model-id", default="patient_anatomy_v8_preview")
    parser.add_argument("--color-rgba", default="0.65,0.15,0.95,0.78")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--allow-rejected-candidate-preview",
        action="store_true",
        help="Required acknowledgement: this does not publish trusted latest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.allow_rejected_candidate_preview:
        raise SystemExit("--allow-rejected-candidate-preview is required")
    subject_path = args.subject.expanduser().resolve()
    subject = load_subject_runtime(subject_path)
    runtime_digest = subject.runtime_digest(validate=False)
    shape_hash = smplx_shape_hash(subject.betas, gender=subject.gender)
    metadata = dict(subject.rigged_asset.metadata or {})
    metadata.update(
        {
            "shape_hash": shape_hash,
            "v8_candidate_preview": True,
            "v8_publishable": False,
            "v8_operator_runtime_digest": subject.operator_runtime_digest,
            "v8_subject_runtime_digest": runtime_digest,
        }
    )
    preview = replace(subject.rigged_asset, metadata=metadata)
    output = (
        args.output_npz.expanduser().resolve()
        if args.output_npz is not None
        else subject_path.parent / f"{subject_path.name}_genesis_preview.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_rigged_asset(output, preview)
    print(
        f"exported rejected V8 candidate preview subject={runtime_digest} "
        f"shape_hash={shape_hash} -> {output}",
        flush=True,
    )
    if args.export_only:
        return 0

    rgba = tuple(float(value.strip()) for value in args.color_rgba.split(","))
    if len(rgba) != 4:
        raise SystemExit("--color-rgba must contain r,g,b,a")
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(str(args.bind))
    payload = anatomy_asset_control_to_dict(
        action="upsert",
        model_id=str(args.model_id),
        asset_npz=str(output),
        color_rgba=rgba,
        timestamp_ns=time.time_ns(),
    )
    topic = TOPIC_ANATOMY_ASSET_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    time.sleep(0.25)
    end = time.time() + max(0.25, float(args.duration_s))
    interval = 1.0 / max(1.0, float(args.rate_hz))
    sent = 0
    while time.time() < end:
        socket.send_multipart([topic, body])
        sent += 1
        time.sleep(interval)
    socket.close(0)
    print(
        f"published preview action=upsert model_id={args.model_id} "
        f"sent={sent} bind={args.bind}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
