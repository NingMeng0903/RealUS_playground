#!/usr/bin/env python3
"""Cache full official HybrIK-X RLE predictions for saved multiview RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from perception.benchmark.hybrik_rgb import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, OfficialHybrIKXRGB, select_rgb_bbox,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--observation-sequence-dir", type=Path, required=True,
                        help="Saved RGB detector/DWPose moment.json metadata; 3D/GT not read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--flip-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-views", action="store_true",
                        help="One full official forward for all available views; timings are batch total and per-view amortization")
    parser.add_argument("--frame-indices", nargs="+", type=int)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-passes", type=int, default=1)
    args = parser.parse_args()
    if args.max_frames < 0 or args.warmup_passes < 0:
        parser.error("max-frames and warmup-passes must be nonnegative")
    recording, source, output_dir = args.recording_dir.resolve(), args.observation_sequence_dir.resolve(), args.output_dir.resolve()
    source_doc = json.loads((source / "index.json").read_text())
    source_rows = sorted(source_doc.get("index", []) + source_doc.get("rejected_index", []), key=lambda row: int(row["frame_index"]))
    if args.frame_indices is not None:
        selected = set(args.frame_indices)
        source_rows = [row for row in source_rows if int(row["frame_index"]) in selected]
    if args.max_frames:
        source_rows = source_rows[:args.max_frames]
    if not source_rows:
        raise ValueError("No requested RGB frames")
    records = {int(row["frame_index"]): row for line in (recording / "manifest.jsonl").read_text().splitlines()
               if (row := json.loads(line)).get("status") == "ok"}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "frames").mkdir()
    model = OfficialHybrIKXRGB(checkpoint=args.checkpoint, config=args.config,
                              device=args.device, flip_test=args.flip_test)
    manifest = {"schema_version": 1, "recording_dir": str(recording),
                "observation_sequence_dir": str(source), "output_dir": str(output_dir),
                "rgb_model": model.metadata, "requested_frames": len(source_rows),
                "batch_views": args.batch_views,
                "timing_policy": ("One synchronized full official RGB forward per frame; rgb_forward_batch_elapsed_s is measured batch total; rgb_forward_elapsed_s is batch-total / successful-view-count amortization, NOT individually measured latency; decode/crop measured separately; initialization and RGB-only batch warmup excluded; not live end-to-end latency" if args.batch_views else "Per-view synchronized full RGB forward measured separately; RGB decode and official crop measured separately; initialization and RGB-only warmup excluded; not live end-to-end latency"),
                "warmup_passes": args.warmup_passes, "warmup_elapsed_s": 0.0,
                "gt_usage": "No canonical/GT files opened; only manifest image paths and RGB detector moment.json metadata read",
                "index": []}
    warmed = False
    for source_row in source_rows:
        frame = int(source_row["frame_index"])
        record = records.get(frame)
        row = {"frame_index": frame, "timestamp_ns": int(source_row.get("timestamp_ns", 0)),
               "status": "rejected", "npz_path": None, "camera_ids": [], "failed_views": []}
        if record is None or "moment_dir" not in source_row:
            row["reason"] = "recording_or_rgb_metadata_unavailable"
        else:
            moment = Path(source_row["moment_dir"])
            moment = moment if moment.is_absolute() else source / moment
            moment_file = moment / "moment.json"
            if moment_file.is_file():
                moment_metadata = json.loads(moment_file.read_text())
            else:
                moment_metadata = source_row
            detector = moment_metadata.get("detection_meta", {})
            views, ids, bbox_metadata, timings = [], [], [], []
            prepared_views = []
            batch_elapsed_s = None
            expected_cameras = sorted(record["image_paths"])
            for camera in expected_cameras:
                try:
                    image_path = (recording / record["image_paths"][camera]).resolve()
                    started = time.perf_counter()
                    bgr = cv2.imread(str(image_path))
                    if bgr is None:
                        raise ValueError("RGB image could not be decoded")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    decode_s = time.perf_counter() - started
                    bbox, bbox_meta = select_rgb_bbox(detector.get(camera, {}), image_size=(rgb.shape[1], rgb.shape[0]))
                    crop, expanded_bbox, center, crop_s = model.prepare(rgb, bbox)
                    if args.batch_views:
                        prepared_views.append((camera, crop, bbox, expanded_bbox, center, decode_s, crop_s,
                                               {"camera_id": camera, "image_path": str(image_path), **bbox_meta}))
                        continue
                    if not warmed:
                        started = time.perf_counter()
                        for _ in range(args.warmup_passes):
                            model.forward_prepared(crop, expanded_bbox, center)
                        manifest["warmup_elapsed_s"] = time.perf_counter() - started
                        warmed = True
                    prediction, forward_s = model.forward_prepared(crop, expanded_bbox, center)
                    prediction.update(bbox_input=bbox[None], bbox_used=expanded_bbox[None], image_center=center[None],
                                      rgb_decode_elapsed_s=np.asarray([decode_s]), crop_elapsed_s=np.asarray([crop_s]),
                                      rgb_forward_elapsed_s=np.asarray([forward_s]))
                    views.append(prediction)
                    ids.append(camera)
                    bbox_metadata.append({"camera_id": camera, "image_path": str(image_path), **bbox_meta})
                    timings.append(forward_s)
                except (ValueError, RuntimeError, AssertionError) as exc:
                    row["failed_views"].append({"camera_id": camera, "reason": str(exc)})
            if args.batch_views and prepared_views:
                try:
                    crops = [view[1] for view in prepared_views]
                    expanded_boxes = np.stack([view[3] for view in prepared_views])
                    centers = np.stack([view[4] for view in prepared_views])
                    if not warmed:
                        started = time.perf_counter()
                        for _ in range(args.warmup_passes):
                            model.forward_prepared_batch(crops, expanded_boxes, centers)
                        manifest["warmup_elapsed_s"] = time.perf_counter() - started
                        warmed = True
                    prediction, batch_elapsed_s = model.forward_prepared_batch(crops, expanded_boxes, centers)
                    amortized_s = batch_elapsed_s / len(prepared_views)
                    for i, view in enumerate(prepared_views):
                        camera, _, bbox, expanded_bbox, center, decode_s, crop_s, bbox_meta = view
                        per_view = {key: value[i:i + 1] for key, value in prediction.items()}
                        per_view.update(bbox_input=bbox[None], bbox_used=expanded_bbox[None], image_center=center[None],
                                        rgb_decode_elapsed_s=np.asarray([decode_s]), crop_elapsed_s=np.asarray([crop_s]),
                                        rgb_forward_elapsed_s=np.asarray([amortized_s]))
                        views.append(per_view)
                        ids.append(camera)
                        bbox_metadata.append(bbox_meta)
                        timings.append(amortized_s)
                except (ValueError, RuntimeError, AssertionError) as exc:
                    row["failed_views"].extend({"camera_id": view[0], "reason": f"batch_forward_failed: {exc}"}
                                               for view in prepared_views)
            if views:
                keys = set(views[0])
                if any(set(view) != keys for view in views):
                    raise ValueError("Official per-view cache fields disagree")
                fields = {key: np.concatenate([view[key] for view in views], axis=0) for key in keys}
                fields.update(camera_ids=np.asarray(ids), frame_index=np.int64(frame),
                              timestamp_ns=np.int64(row["timestamp_ns"]))
                if args.batch_views:
                    fields.update(rgb_forward_batch_elapsed_s=np.float64(batch_elapsed_s),
                                  rgb_forward_batch_size=np.int64(len(ids)),
                                  rgb_forward_timing_kind=np.asarray("batch_total_divided_by_view_count"))
                relative_path = f"frames/frame_{frame:08d}.npz"
                np.savez_compressed(output_dir / relative_path, **fields)
                row.update(status="ok" if len(ids) == len(expected_cameras) else "partial",
                           npz_path=relative_path, camera_ids=ids, expected_camera_ids=expected_cameras,
                           bbox_metadata=bbox_metadata, rgb_forward_elapsed_s=timings)
                if args.batch_views:
                    row.update(rgb_forward_batch_elapsed_s=batch_elapsed_s, rgb_forward_batch_size=len(ids),
                               rgb_forward_timing_kind="batch_total_divided_by_view_count")
        manifest["index"].append(row)
        (output_dir / "index.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"frame_index": frame, "status": row["status"], "views": len(row["camera_ids"]),
                          "failed_views": row["failed_views"]}), flush=True)
    manifest["complete_frames"] = sum(row["status"] == "ok" for row in manifest["index"])
    manifest["cached_views"] = sum(len(row["camera_ids"]) for row in manifest["index"])
    manifest["complete"] = manifest["complete_frames"] == len(source_rows)
    if args.batch_views:
        manifest["rgb_forward_batch_measured_total_s"] = sum(row.get("rgb_forward_batch_elapsed_s", 0.0)
                                                            for row in manifest["index"])
    (output_dir / "index.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ("complete_frames", "cached_views", "complete", "warmup_elapsed_s")}))


if __name__ == "__main__":
    main()
