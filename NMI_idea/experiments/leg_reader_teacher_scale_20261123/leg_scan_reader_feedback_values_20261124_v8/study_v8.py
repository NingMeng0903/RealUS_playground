"""Disjoint cohorts, frozen input checks and resumable v8 simulation shards."""

from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
import sys

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "leg_scan_public_evidence_20261124_v7"
BASE = "2f2c704eb0085f78e638d41cd2467c436713b695"
sys.path.insert(1, str(V7))

from study_v7 import (DEFAULT_PACK, ROUTES, V4_RUN, V5_RUN, V6_RUN, atomic_json, cases_for, commit_shard,
                      digest, freeze, load_shard, pinned, read_json)


COHORTS = {
    "reader_test": (24, 72),
    "evidence_dev": (16, 44),
    "evidence_test": (16, 48),
    "phase2_fit": (12, 28),
    "phase1_fit": (12, 28),
    "student_test": (16, 32),
}
SEEDS = {name: 2_612_408_000_000_000 + 12_000_000 * i for i, name in enumerate(COHORTS)}


def roster(name):
    quota = dict(zip(ROUTES, COHORTS[name], strict=True))
    family = "reader_test" if name.endswith("_test") else "fit"
    for i, source in enumerate(cases_for(family, 2 * max(quota.values()))):
        if quota[source.task] == 0:
            continue
        quota[source.task] -= 1
        seed = SEEDS[name] + 19 * i
        yield replace(source, identifier=f"v8_{name}_{i:05d}", split=name, world_seed=seed,
                      tape_seed=seed+1, action_seed=seed+2, challenge=False)


def verify_shard(folder):
    path, rows = load_shard(folder)
    lock = read_json(folder / "commit.json")
    if digest(path) != lock["data_sha256"] or digest(folder / "cases.jsonl.gz") != lock["metadata_sha256"]:
        raise RuntimeError(f"Shard failed digest verification at {folder}")
    return path, rows


def committed(out, name):
    rows = [r for lock in sorted((out/name).glob("*/commit.json")) for r in verify_shard(lock.parent)[1]]
    if [r["case"] for r in rows] != [asdict(c) for c in roster(name)] or not all(r["complete"] for r in rows):
        raise RuntimeError(f"Incomplete or unexpected cohort {name}")
    if len({r["layout_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"Duplicate synthetic layout in {name}")
    return rows


def shards(pool, out, name, worker, pack, *, size=16):
    cases, began = list(roster(name)), monotonic()
    for offset in range(0, len(cases), size):
        batch = cases[offset:offset+size]
        folder = out / name / f"{offset//size:04d}"
        if not (folder/"commit.json").exists():
            acquired = list(pool.map(worker, batch, chunksize=1))
            commit_shard(folder, pack(acquired), [r[1] for r in acquired])
        _, rows = verify_shard(folder)
        if [r["case"] for r in rows] != [asdict(c) for c in batch] or not all(r["complete"] for r in rows):
            raise RuntimeError(f"Unexpected completed roster at {folder}")
        print(f"{name}: {offset+len(batch)}/{len(cases)} layouts in {monotonic()-began:.1f}s", flush=True)
    return committed(out, name)


def hashes(names):
    return {name: digest(ROOT/name) for name in names}


def check_hashes(expected):
    if hashes(expected) != expected:
        raise RuntimeError("Frozen v8 source changed")
