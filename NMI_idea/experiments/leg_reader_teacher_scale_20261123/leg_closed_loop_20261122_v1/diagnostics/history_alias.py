#!/usr/bin/env python3
"""Audit public-history aliases in committed privileged teacher shards.

The teacher branch outcomes are used only to define the two audit classes.
Public actor features and the optional raw-frame summary are read without
opening evaluator label files.  No training, threshold, or holdout input is
modified by this diagnostic.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = (
    "commit.json", "episodes.jsonl.gz", "teacher.jsonl.gz", "public_frames.npz",
)
FEATURE_COUNT = 69
CONTROL_DISTANCE_MAX = .55
WORLD_FEATURE_DISTANCE_MM = 5.


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_batches(root):
    batches = []
    for directory in sorted(root.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else -1):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        if not all((directory / name).is_file() for name in REQUIRED):
            continue
        commit = json.loads((directory / "commit.json").read_text())
        if commit.get("episodes") != 256 or commit.get("complete") not in (None, 256):
            continue
        batches.append((directory, commit))
    return batches


def robust_outcome(branch, continue_branch):
    if not branch.get("feasible", False) or len(branch.get("replays", ())) != 2:
        return None
    cont_replays = continue_branch.get("replays", ())
    active_replays = branch["replays"]
    if len(cont_replays) != 2:
        return None
    cont_outcomes = [int(item.get("outcome", -1)) for item in cont_replays]
    active_outcomes = [int(item.get("outcome", -1)) for item in active_replays]
    cont_utilities = [float(item.get("utility", 0.)) for item in cont_replays]
    active_utilities = [float(item.get("utility", 0.)) for item in active_replays]
    gains = [active - base for active, base in zip(active_utilities, cont_utilities)]
    rescue = all(outcome == 1 for outcome in active_outcomes) and all(outcome != 1 for outcome in cont_outcomes)
    harm = all(outcome == 1 for outcome in cont_outcomes) and all(active < base for active, base in zip(active_utilities, cont_utilities))
    if rescue:
        return "rescue", gains, cont_outcomes, active_outcomes
    if harm:
        return "harm", gains, cont_outcomes, active_outcomes
    return None


def raw_public_summary(pack, prefix_frames):
    """Make a label-free compact summary from a prefix in public_frames.npz."""
    offsets = pack["offsets"]
    index = int(pack["_episode_index"])
    start = int(offsets[index])
    stop = min(start + max(0, int(prefix_frames)), int(offsets[index + 1]))
    if stop <= start:
        return None
    quality = np.asarray(pack["pose_quality"][start:stop, 2], dtype=float)
    pose = np.asarray(pack["pose_quality"][start:stop, :2], dtype=float)
    echoes = np.asarray(pack["echo_mm"][start:stop], dtype=float)
    posterior = np.asarray(pack["target_unknown_mixed"][start:stop], dtype=float)
    valid_echo = np.isfinite(echoes).all(axis=-1)
    valid_posterior = np.isfinite(posterior).all(axis=-1)
    echo_count = valid_echo.sum(axis=1)
    unknown = np.where(valid_posterior, posterior[:, :, 1], np.nan)
    mixed = np.where(valid_posterior, posterior[:, :, 2], np.nan)
    target = np.where(valid_posterior, posterior[:, :, 0], np.nan)
    unique_views = {(round(float(row[0])), round(float(row[1]))) for row in pose}
    return np.asarray((
        float(np.nanmean(quality)), float(np.nanmax(quality)),
        float(np.nanmean(echo_count)), float(np.nanstd(echo_count)),
        float(np.nanmean(target)), float(np.nanmean(unknown)),
        float(np.nanmean(mixed)), float(len(unique_views) / max(1, stop - start)),
        float(stop - start), float(quality[-1]),
    ), dtype=float)


def robust_scale(rows):
    values = np.asarray(rows, dtype=float)
    center = np.nanmedian(values, axis=0)
    low, high = np.nanpercentile(values, (25., 75.), axis=0)
    scale = high - low
    scale[~np.isfinite(scale) | (scale < 1.e-6)] = 1.
    return center, scale


def feature_distance(first, second, center, scale):
    left = np.clip((np.asarray(first, dtype=float) - center) / scale, -5., 5.)
    right = np.clip((np.asarray(second, dtype=float) - center) / scale, -5., 5.)
    return float(np.linalg.norm(left - right) / np.sqrt(len(left)))


def make_record(row, branch, meta, batch, raw):
    return {
        "episode": row["episode"], "batch": int(batch["batch"]),
        "layout": row.get("fixed_layout_id") or meta.get("layout_id"),
        "task": meta.get("task", "unknown"), "shape": meta.get("shape"),
        "source": meta.get("source"), "profile": meta.get("profile"),
        "region": int(row.get("prefix_region", -1)),
        "stage": int(row.get("stage", -1)), "skill": int(branch["skill"]),
        "controls": [float(value) for value in branch.get("controls", (0., 0., 0.))],
        "features": np.asarray(row["features"], dtype=float),
        "raw_features": raw,
        "prefix_frames": int(row.get("prefix_frames", 0)),
        "prefix_station": float(row.get("prefix_station", 0.)),
        "prefix_sector": float(row.get("prefix_sector", 0.)),
        "feasible": bool(branch.get("feasible", False)),
    }


def collect(root):
    batches = complete_batches(root)
    records = []
    skipped = []
    per_batch = []
    for directory, commit in batches:
        episodes = read_jsonl(directory / "episodes.jsonl.gz")
        episode_map = {row.get("identifier"): row for row in episodes}
        public = np.load(directory / "public_frames.npz", allow_pickle=False)
        public_map = {key: public[key] for key in public.files}
        public_map["_episode_index"] = 0
        teacher_rows = read_jsonl(directory / "teacher.jsonl.gz")
        count = 0
        for row in teacher_rows:
            if not row.get("eligible_prefix") or len(row.get("features", ())) != FEATURE_COUNT:
                skipped.append({"batch": commit["batch"], "episode": row.get("episode"), "reason": "missing_or_ineligible_features"})
                continue
            meta = episode_map.get(row.get("episode"))
            if meta is None:
                skipped.append({"batch": commit["batch"], "episode": row.get("episode"), "reason": "episode_not_found"})
                continue
            try:
                index = episodes.index(meta)
                public_map["_episode_index"] = index
                raw = raw_public_summary(public_map, row.get("prefix_frames", 0))
            except (IndexError, KeyError, ValueError):
                raw = None
            continue_branch = next((branch for branch in row.get("branches", ()) if branch.get("origin") == "continue"), None)
            if continue_branch is None:
                skipped.append({"batch": commit["batch"], "episode": row.get("episode"), "reason": "no_continue_branch"})
                continue
            for branch in row.get("branches", ()):
                if branch.get("origin") == "continue" or int(branch.get("skill", 0)) == 0:
                    continue
                outcome = robust_outcome(branch, continue_branch)
                if outcome is None:
                    continue
                label, gains, cont_outcomes, active_outcomes = outcome
                record = make_record(row, branch, meta, commit, raw)
                record.update({"label": label, "paired_gains": gains,
                               "continue_outcomes": cont_outcomes,
                               "active_outcomes": active_outcomes,
                               "branch_origin": branch.get("origin")})
                records.append(record)
                count += 1
        public.close()
        per_batch.append({"batch": int(commit["batch"]), "episodes": len(episodes),
                          "teacher_rows": len(teacher_rows), "robust_records": count,
                          "commit_sha256": sha256(directory / "commit.json")})
    return batches, records, skipped, per_batch


def nearest_conflicts(records):
    rescues = [record for record in records if record["label"] == "rescue"]
    harms = [record for record in records if record["label"] == "harm"]
    all_features = [record["features"] for record in records]
    center, scale = robust_scale(all_features) if all_features else (np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT))
    raw_rows = [record["raw_features"] for record in records if record["raw_features"] is not None]
    raw_center, raw_scale = robust_scale(raw_rows) if raw_rows else (np.zeros(10), np.ones(10))
    pairs = []
    for rescue in rescues:
        candidates = []
        for harm in harms:
            if rescue["task"] != harm["task"] or rescue["region"] != harm["region"]:
                continue
            if rescue["skill"] != harm["skill"] or rescue["feasible"] != harm["feasible"]:
                continue
            control_distance = float(np.linalg.norm(np.asarray(rescue["controls"]) - np.asarray(harm["controls"])))
            if control_distance > CONTROL_DISTANCE_MAX:
                continue
            distance = feature_distance(rescue["features"], harm["features"], center, scale)
            raw_distance = None
            if rescue["raw_features"] is not None and harm["raw_features"] is not None:
                raw_distance = feature_distance(rescue["raw_features"], harm["raw_features"], raw_center, raw_scale)
            candidates.append((0 if rescue["layout"] != harm["layout"] else 1, distance, raw_distance, control_distance, harm))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            layout_priority, distance, raw_distance, control_distance, harm = candidates[0]
            pairs.append({
                "feature_distance": distance, "raw_feature_distance": raw_distance,
                "control_distance": control_distance,
                "different_layout": bool(layout_priority == 0),
                "task": rescue["task"], "region": rescue["region"], "skill": rescue["skill"],
                "rescue": {key: rescue[key] for key in ("episode", "batch", "layout", "shape", "source", "prefix_station", "prefix_sector", "paired_gains", "branch_origin", "skill", "controls")},
                "harm": {key: harm[key] for key in ("episode", "batch", "layout", "shape", "source", "prefix_station", "prefix_sector", "paired_gains", "branch_origin", "skill", "controls")},
            })
    pairs.sort(key=lambda item: item["feature_distance"])
    unique = []
    seen = set()
    for pair in pairs:
        key = (pair["rescue"]["episode"], pair["rescue"]["branch_origin"], tuple(pair["rescue"]["controls"]),
               pair["harm"]["episode"], pair["harm"]["branch_origin"], tuple(pair["harm"]["controls"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique, {"center": center.tolist(), "scale": scale.tolist(),
                   "raw_center": raw_center.tolist(), "raw_scale": raw_scale.tolist()}


def held_shape_summary(records, pairs):
    shapes = sorted({record["shape"] for record in records if record["shape"] is not None})
    result = {}
    for shape in shapes:
        held = [pair for pair in pairs
                if pair["rescue"]["shape"] == shape or pair["harm"]["shape"] == shape]
        cross_layout = [pair for pair in held if pair["different_layout"]]
        result[str(shape)] = {
            "records_with_shape": sum(record["shape"] == shape for record in records),
            "nearest_conflicts_touching_shape": len(held),
            "different_layout_conflicts": len(cross_layout),
            "minimum_feature_distance": min((pair["feature_distance"] for pair in held), default=None),
            "minimum_raw_feature_distance": min((pair["raw_feature_distance"] for pair in held if pair["raw_feature_distance"] is not None), default=None),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, default=Path("leg_closed_loop_20261122_v1/training/seed_276382173/teacher_ppo/batches"))
    parser.add_argument("--out", type=Path, default=Path("leg_closed_loop_20261122_v1/diagnostics/history_alias.json"))
    args = parser.parse_args()
    batches, records, skipped, per_batch = collect(args.batches)
    pairs, scaling = nearest_conflicts(records)
    rescue_count = sum(record["label"] == "rescue" for record in records)
    harm_count = sum(record["label"] == "harm" for record in records)
    output = {
        "status": "complete",
        "diagnostic": "public_history_alias",
        "command": f"python {Path(__file__).as_posix()} --batches {args.batches.as_posix()} --out {args.out.as_posix()}",
        "inputs": {"batch_root": args.batches.as_posix(), "required_files": list(REQUIRED),
                   "latest_complete_batch": max((commit["batch"] for _directory, commit in batches), default=None),
                   "batches_read": [int(commit["batch"]) for _directory, commit in batches],
                   "evaluator_labels_opened": False, "blender_opened": False},
        "definitions": {
            "rescue": "continue outcome is not correct on both tapes and an executable active branch is correct on both",
            "harm": "continue is correct on both tapes and the comparable active branch has lower utility on both",
            "matching": {"task": "exact", "region": "exact", "skill": "exact", "feasible": "exact",
                         "control_l2_max": CONTROL_DISTANCE_MAX},
            "different_layout_priority": "nearest pair selection prefers different fixed_layout_id when available",
            "raw_public_summary": "10-dimensional aggregate from prefix public_frames.npz, with no evaluator label file",
        },
        "counts": {"complete_batches": len(batches), "teacher_records": len(records),
                   "rescue_records": rescue_count, "harm_records": harm_count,
                   "matching_conflict_pairs": len(pairs),
                   "different_layout_pairs": sum(pair["different_layout"] for pair in pairs),
                   "records_skipped": len(skipped)},
        "uncertainty": {
            "robust_record_scarcity": len(records) < 100,
            "conflict_scarcity": len(pairs) < 20,
            "note": "Teacher rows are a selected privileged subset, not a prevalence estimate. Feature proximity is descriptive and does not establish causal aliasing.",
        },
        "per_batch": per_batch,
        "closest_conflicts": pairs[:25],
        "held_shape": held_shape_summary(records, pairs),
        "scaling": scaling,
        "skipped_examples": skipped[:25],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": args.out.as_posix(), "batches": len(batches), "records": len(records),
                      "rescue": rescue_count, "harm": harm_count, "pairs": len(pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
