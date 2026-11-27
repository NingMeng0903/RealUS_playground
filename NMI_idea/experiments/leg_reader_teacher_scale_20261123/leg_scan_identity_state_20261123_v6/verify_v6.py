"""Recompute the provenance and paired phase totals from committed shards."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from v6_data import DEFAULT_PACK, ROOT, STAGES, V5_RUN, atomic_json, cases, digest, load_shard, read_json, readonly_input
from live_programs import PROGRAMS, SCORE_STREAMS, TEACH_ACTIONS
import numpy as np


def physical_shards(out, stage):
    paths = sorted((out / stage).glob("*/commit.json"))
    rows = []
    for path in paths:
        data, ledger = load_shard(path.parent); index = read_json(path)
        if (digest(data) != index["data_sha256"] or digest(path.parent / "cases.jsonl.gz") != index["metadata_sha256"]
                or len(ledger) != index["cases"] or not all(r["complete"] for r in ledger)):
            raise RuntimeError(f"Committed shard did not validate: {path}")
        if stage != "reader_fresh":
            streams = ("student_exploration",) if stage == "closed_dev" else SCORE_STREAMS
            scores = [[[int(m["outcomes"][band][arm]["strict"]) for arm in
                        ((m["student"],) if stage == "closed_dev" else PROGRAMS)] for band in streams] for m in ledger]
            with np.load(data, allow_pickle=False) as z:
                np.testing.assert_array_equal(scores, z["strict"])
        rows.extend(ledger)
    if [r["case"] for r in rows] != [asdict(case) for case in cases(stage)]:
        raise RuntimeError(f"Cohort roster differs from manifest: {stage}")
    return rows, len(paths)


def count_branches(rows):
    checked = Counter()
    for row in rows:
        arms = row["outcomes"]
        for group in arms.values():
            if not all(a["completed_legal"] and a["layout_id"] == row["layout_id"] for a in group.values()):
                raise RuntimeError("A complete evaluation did not preserve legal motion or anatomy")
            if row["trigger"] and len(group) == len(PROGRAMS):
                names = ("equal_time_hold", "v5_public", "prefix_frozen", "unary_dynamic", "objects_dynamic")
                if len({(group[name]["at_exit"][0], group[name]["at_exit"][2]) for name in names}) != 1:
                    raise RuntimeError("Fixed-anchor control arms did not share clock and station")
            used = {a["program"] for a in group.values()}
            checked["scoring_executions"] += len(used)
            for a in group.values():
                if a["joint"] != (a["old_pool_correct"] and a["suffix_valid"]):
                    raise RuntimeError("Joint outcome did not satisfy the old and suffix requirements")
        if row["paired_samples"] is not None:
            prefix_pass = len(TEACH_ACTIONS) if row["trigger"] else 2
            checked["teacher_search_executions"] += 2 * prefix_pass
            if len(row["adaptive_actions"]) > 1:
                checked["teacher_search_executions"] += 2 * len(row["adaptive_actions"])
            checked["all_action_paired_observations"] += sum(len(s) for s in row["paired_samples"].values())
    return dict(checked)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run/confirmed")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    args = p.parse_args(); out = args.out.resolve()
    fr, fc, fs = (out / n for n in ("reader_freeze.json", "closed_freeze.json", "student_manifest.json"))
    reader, closed, student = map(read_json, (fr, fc, fs))
    atlas = readonly_input(args.atlas_pack)
    if atlas["sha256"] != reader["atlas_sha256"] or atlas["sha256"] != closed["atlas_sha256"]:
        raise RuntimeError("Current atlas differs from the frozen cohort")
    if readonly_input(V5_RUN / "reader_freeze.json")["sha256"] != reader["v5_reader_sha256"]:
        raise RuntimeError("The pinned v5 reader differs from the archived source")
    results, readout = (read_json(out / n) for n in ("closed_results.json", "reader_results.json"))
    if (readout["reader_freeze_sha256"] != digest(fr) or closed["reader_freeze_sha256"] != digest(fr)
            or results["protocol_sha256"] != digest(fc) or student["closed_protocol"] != digest(fc)
            or results["student_freeze_sha256"] != digest(fs) or student["bundle_sha256"] != digest(out / "student_bundle.json")
            or reader["usability_sha256"] != digest(out / "usability_bundle.json")):
        raise RuntimeError("Frozen sources, parameters or result identity changed")
    for key, manifest in (("reader_source_sha256", reader), ("source_sha256", closed)):
        if any(digest(ROOT / name) != expected for name, expected in manifest[key].items()):
            raise RuntimeError("Published source differs from the evaluated module")
    by_stage, anatomy, world = {}, set(), set()
    for stage in STAGES:
        rows, shard_count = physical_shards(out, stage)
        ids = {r["layout_id"] for r in rows}; seeds = {r["case"]["world_seed"] for r in rows}
        if len(ids) != len(rows) or anatomy & ids or world & seeds:
            raise RuntimeError("Same layout or seed used in separate stages")
        anatomy.update(ids); world.update(seeds)
        by_stage[stage] = {"layouts": len(rows), "verified_shards": shard_count,
                           "verified_layout_counts_by_route": dict(Counter(r["case"]["task"] for r in rows))}
        if stage != "reader_fresh":
            by_stage[stage] |= count_branches(rows)
    actual = sum(r.get("scoring_executions", 0) for r in by_stage.values())
    if actual != results["complete_scoring_trajectories"] or len(anatomy) != results["new_layouts"]:
        raise RuntimeError("Reported full episode count differs from the actual logs")
    result = {"stages": by_stage, "unique_fresh_layouts": len(anatomy), "complete_scoring_trajectories": actual,
              "reader_freeze_sha256": digest(fr), "closed_freeze_sha256": digest(fc), "student_manifest_sha256": digest(fs),
              "verified_student_bundle_sha256": student["bundle_sha256"], "verified_usability_bundle_sha256": reader["usability_sha256"]}
    atomic_json(out / "verification.json", result)
    print(f"Verified {len(anatomy)} distinct fresh layouts and {actual} complete scoring trajectories", flush=True)


if __name__ == "__main__":
    main()
