"""Disjoint atlas-only v9 rosters and hash-checked resumable shards."""

from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = "0e498eb2f4702f2efed997389cc8e09917f3dc47"
V8 = ROOT.parent / "leg_scan_reader_feedback_values_20261124_v8"
sys.path.insert(1, str(V8))

from study_v8 import (DEFAULT_PACK, ROUTES, V4_RUN, V5_RUN, V6_RUN, atomic_json, cases_for,
                      commit_shard, digest, freeze, pinned, read_json, verify_shard)


SEEDS = {"smoke": 6_419_240_000_000_000, "bc": 6_419_250_000_000_000,
         "train": 6_419_260_000_000_000, "dev": 6_419_270_000_000_000,
         "test": 6_419_280_000_000_000, "reader_confirm": 6_419_290_000_000_000}
PROOF_TAPES = (None, 9_100_000_000_000, 9_200_000_000_000)
TEACHER_TAPES = (10_100_000_000_000, 10_200_000_000_000)


def roster(split, n, batch=0, *, full=None):
    if n < 2 or split not in SEEDS or batch < 0:
        raise ValueError("Invalid v9 roster")
    full = n - n//3 if full is None else full
    quota = dict(zip(ROUTES, (n-full, full), strict=True))
    family = "reader_test" if split in ("test", "reader_confirm") else "fit"
    source = list(cases_for(family, 2 * n * (batch + 1)))
    start = 2*n*batch
    for ordinal, template in enumerate(source[start:], start):
        if quota[template.task] <= 0:
            continue
        quota[template.task] -= 1
        seed = SEEDS[split] + ordinal*29
        yield replace(template, identifier=f"v9_{split}_{ordinal:06d}", split=f"v9_{split}",
                      world_seed=seed, tape_seed=seed+1, action_seed=seed+2, challenge=False)


def hashes(names):
    return {name: digest(ROOT/name) for name in names}


def assert_hashes(expected):
    if hashes(expected) != expected:
        raise RuntimeError("Frozen v9 source changed")


def take_shard(folder, cases, batch_results, pack):
    if not (folder/"commit.json").exists():
        commit_shard(folder,pack(batch_results),[metadata for _,metadata in batch_results])
    path, rows=verify_shard(folder)
    if [r["case"] for r in rows] != [asdict(c) for c in cases] or not all(r["complete"] for r in rows):
        raise RuntimeError(f"Invalid v9 shard roster at {folder}")
    if len(rows) != len({r["layout_id"] for r in rows}):
        raise RuntimeError("A layout was repeated in one batch")
    return path,rows
