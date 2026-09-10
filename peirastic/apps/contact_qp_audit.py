"""Read-only raw/aligned human-data audit using the production confidence extractor.

Use the genesis environment (h5py, OpenCV, SciPy). Results are written only to
the supplied output directory; source H5 and prior analyses are never edited.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor


def nearest(t, query):
    hi = np.clip(np.searchsorted(t, query), 0, len(t)-1)
    lo = max(0, hi-1)
    return int(lo if abs(t[lo]-query) <= abs(t[hi]-query) else hi)


def audit(root, output, *, stride=1, max_files=0):
    import cv2
    import h5py

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.h5"))
    if max_files:
        files = files[:max_files]
    config = FeatureConfig()
    extractor = FeatureExtractor(config)
    summary, elapsed, total = [], [], 0
    examples = []
    for file_number, path in enumerate(files):
        with h5py.File(path, "r") as f:
            if "ultrasound/jpeg" not in f or not bool(f.attrs.get("complete", False)):
                continue
            schema = str(f.attrs.get("schema", ""))
            aligned = schema == "realus-ultrasound-aligned-v1"
            if schema not in ("realus_icra_v1", "realus-ultrasound-aligned-v1"):
                raise ValueError(f"unsupported H5 schema: {schema}")
            ut = np.asarray(f["ultrasound/timestamp_ns"], dtype=np.float64)*1e-9
            ft = np.asarray(f["force/timestamp_ns"], dtype=np.float64)*1e-9
            tt = np.asarray(f["tcp/timestamp_ns"], dtype=np.float64)*1e-9
            wrench = np.asarray(f["force/wrench_tcp"])
            for name, ts in (("ultrasound", ut), ("force", ft), ("tcp", tt)):
                if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                    raise ValueError(f"{path}: invalid {name} timestamps")
            relative = path.relative_to(root)
            destination = output / relative.with_suffix("")
            destination.mkdir(parents=True, exist_ok=True)
            rows = []
            for i in range(0, len(ut), stride):
                before = time.perf_counter()
                im = cv2.imdecode(np.asarray(f["ultrasound/jpeg"][i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    raise ValueError(f"JPEG decode failed: {path}:{i}")
                meta = json.loads(f["ultrasound/metadata_json"][i]) if "ultrasound/metadata_json" in f else {}
                capture = 1.+float(ut[i]-ut[0])
                observation, confidence = extractor.extract(
                    im, frame_seq=i, source_id=str(relative), capture_time_s=capture,
                    received_time_s=capture, crop_box=meta.get("crop_box"),
                    hflip=meta.get("hflip", False), already_aligned=aligned,
                    clock_domain="offline_aligned" if aligned else "host_monotonic")
                # Aligned files are already on their image grid. Raw force's
                # effective phase differs from TCP by 0.677747 ms.
                tq = ut[i] if aligned else ut[i]-config.effective_delay_s
                fq = ut[i] if aligned else ut[i]-.1512859033580012
                fi, ti = nearest(ft, fq), nearest(tt, tq)
                covered = bool(ft[0] <= fq <= ft[-1] and tt[0] <= tq <= tt[-1]
                               and abs(ft[fi]-fq) <= .01 and abs(tt[ti]-tq) <= .01)
                elapsed.append(time.perf_counter()-before)
                q = observation.quality
                row = dict(frame=i, image_time_s=capture-1., effective_time_s=observation.effective_time_s-1.,
                           left=q[0], center=q[1], right=q[2], valid=bool(observation.valid.all()),
                           force_n=float(wrench[fi, 2]) if covered else "",
                           torque_y_nm=float(wrench[fi, 4]) if covered else "", aligned_covered=covered,
                           window_version=observation.window_version,
                           registration_version=observation.registration_version)
                rows.append(row)
                if i == len(ut)//2 or (not examples and file_number == 0):
                    preview = cv2.resize(im, (config.width, config.height))
                    panel = np.concatenate((preview, np.round(confidence*255).astype(np.uint8)), axis=1)
                    cv2.imwrite(str(destination/f"frame_{i:04d}_image_confidence.png"), panel)
                    examples.append(str(destination/f"frame_{i:04d}_image_confidence.png"))
            with (destination/"features.csv").open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            total += len(rows)
            covered_rows = [r for r in rows if r["aligned_covered"]]
            summary.append(dict(file=str(relative), schema=schema, frames=len(ut), processed=len(rows),
                                aligned_frames=len(covered_rows),
                                mean_quality=np.mean([[r[n] for n in ("left","center","right")] for r in rows], axis=0).tolist(),
                                low_torque_fraction=float(np.mean([abs(r["torque_y_nm"]) <= .025 for r in covered_rows]))
                                if covered_rows else None))
            print(f"{file_number+1}/{len(files)} {relative}: {len(rows)} frames", flush=True)
    result = dict(schema_version=1, source=str(root.resolve()), files=len(summary), processed_frames=total,
                  config=asdict(config), window_version=config.window_version,
                  config_sha256=hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
                  processing_ms={p: float(np.percentile(elapsed, q)*1000) for p,q in (("median",50),("p95",95),("max",100))}
                  if elapsed else {}, scans=summary,
                  evidence_limits=["confidence is not mechanical contact truth", "no new controller commands in H5",
                                   "effective delay is not certified hardware latency", "no physical aperture/image-axis calibration"])
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"README.md").write_text(
        "# Human-data confidence audit\n\n"
        f"Read {len(summary)} scans; processed {total} images with fixed random-walk parameters. "
        "Per-frame CSV keeps quality, alignment coverage and configuration versions. "
        "Paired preview panels show image and confidence.\n\n"
        "These data identify acoustic/torque disagreement; they do not establish mechanical separation "
        "truth or the causal efficacy of the new controller. Raw and already-aligned files are handled "
        "separately. No controller was connected, and no source data were changed.\n")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=1); p.add_argument("--max-files", type=int, default=0)
    args = p.parse_args(argv)
    if args.stride < 1 or args.max_files < 0:
        p.error("stride must be positive and max-files nonnegative")
    audit(args.data, args.output, stride=args.stride, max_files=args.max_files)


if __name__ == "__main__":
    main()
