"""Verify which designed acoustic windows were traversed by actual student scans."""

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from storage import atomic_json, read_lines, shards


def angular_distance(a, b):
    return abs((a - b + 12.) % 24. - 12.)


def audit(directory):
    samples = []
    totals, by_group = Counter(), {}
    for shard in shards(directory):
        rows = list(read_lines(shard / "episodes.jsonl.gz"))
        selected = [(i, row) for i, row in enumerate(rows) if row.get("recoverable_window")]
        totals["complete_scans"] += len(rows)
        if not selected:
            continue
        with np.load(shard / "public_frames.npz", allow_pickle=False) as stream:
            offsets, public = stream["offsets"], stream["pose_quality"]
        for i, row in selected:
            window = row["recoverable_window"]
            near = public[offsets[i]:offsets[i + 1], :2]
            near = near[abs(near[:, 0] - window["station"]) < 1.]
            at_ref = int(np.count_nonzero(angular_distance(near[:, 1], window["sector"]) < 1.))
            at_alternate = int(np.count_nonzero(angular_distance(near[:, 1], window["alternate_sector"]) < 1.))
            if at_ref != row.get("acquired_in_ambiguous_window", 0):
                raise RuntimeError(f"Recomputed exposure differs for {row['identifier']}")
            key = f"source_{row['source']}|{row['profile']}|{row['task']}"
            group = by_group.setdefault(key, Counter())
            for dest in (totals, group):
                dest["verified_recoverable_window_cases"] += 1
                dest["cases_reaching_window_station"] += bool(len(near))
                dest["cases_acquiring_reference_window"] += bool(at_ref)
                dest["cases_acquiring_alternate_window"] += bool(at_alternate)
                dest["cases_acquiring_both_windows"] += bool(at_ref and at_alternate)
                dest["reference_frames"] += at_ref
                dest["alternate_frames"] += at_alternate
                dest["proof_frame_gain"] += window["alternate_resolved"] - window["reference_resolved"]
            samples.append({"episode": row["identifier"], "at_reference": at_ref, "at_alternate": at_alternate,
                            "minimum_sector_gap_at_station": float(angular_distance(near[:, 1], window["sector"]).min()) if len(near) else None})
    return {"totals": totals, "by_source_profile_task": by_group, "cases": samples,
            "policy_window_match": "within one station and one circumferential sector of specified window",
            "design_proof": "independent measurement tape on same fixed anatomical layout"}


if __name__ == "__main__":
    path = ROOT / "training" / "seed_276382173" / "teacher_ppo" / "batches"
    results = audit(path)
    atomic_json(ROOT / "diagnostics" / "window_exposure.json", results)
    print(results["totals"])
