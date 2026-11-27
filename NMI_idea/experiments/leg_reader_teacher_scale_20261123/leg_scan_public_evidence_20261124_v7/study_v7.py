"""Disjoint v7 cohorts and immutable v6 dependencies."""

from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
import sys

ROOT = Path(__file__).resolve().parent
V6 = ROOT.parent / "leg_scan_identity_state_20261123_v6"
V6_RUN = V6 / "run/confirmed"
BASE = "60a77233b60686cfeed377757503c1fa95f8ff6f"
sys.path.insert(1, str(V6))

from v6_data import (DEFAULT_PACK, ROUTES, V4_RUN, V5_RUN, atomic_json, cases_for, commit_shard, digest,
                     load_shard, read_json)
from reader_study import frozen_reader, warm_bundle
from live_programs import init_live


COHORTS = {
    "confirmation": (32, 96), "evidence_dev": (16, 40), "mechanism_test": (12, 36),
    "student_wave0": (12, 28), "student_wave1": (12, 28), "student_test": (12, 28),
}
SEEDS = {name: 2_107_240_000_000_000 + i * 10_000_000 for i, name in enumerate(COHORTS)}


def roster(cohort):
    quotas = dict(zip(ROUTES, COHORTS[cohort], strict=True))
    family = "reader_test" if cohort.endswith("test") or cohort == "confirmation" else "fit"
    for ordinal, case in enumerate(cases_for(family, 2 * max(quotas.values()))):
        if not quotas[case.task]:
            continue
        quotas[case.task] -= 1
        seed = SEEDS[cohort] + 17 * ordinal
        yield replace(case, identifier=f"v7_{cohort}_{ordinal:05d}", split=cohort, world_seed=seed,
                      tape_seed=seed+1, action_seed=seed+2, challenge=False)


def shards(pool, out, name, worker, pack, *, size=16):
    planned = list(roster(name)); out = out / name
    start = monotonic(); rows = []
    for index in range(0, len(planned), size):
        whole = planned[index:index+size]; folder = out / f"{index//size:04d}"
        if not (folder / "commit.json").is_file():
            batch = list(pool.map(worker, whole, chunksize=1))
            commit_shard(folder, pack(batch), [row[1] for row in batch])
        data, entries = load_shard(folder)
        index_row = read_json(folder / "commit.json")
        if (digest(data) != index_row["data_sha256"] or digest(folder / "cases.jsonl.gz") != index_row["metadata_sha256"]
                or [r["case"] for r in entries] != [asdict(c) for c in whole] or not all(r["complete"] for r in entries)):
            raise RuntimeError(f"Study shard invalid {folder}")
        rows.extend(entries)
        print(f"{name}: {len(rows)}/{len(planned)} layouts, {monotonic()-start:.1f}s", flush=True)
    if len({r["layout_id"] for r in rows}) != len(planned):
        raise RuntimeError("Layouts were repeated")
    return rows


def pinned(atlas):
    reader_path = V6_RUN / "reader_freeze.json"
    old = read_json(reader_path)
    if (old["atlas_sha256"] != digest(atlas) or old["usability_sha256"] != digest(V6_RUN / "usability_bundle.json")
            or old["v5_reader_sha256"] != digest(V5_RUN / "reader_freeze.json")):
        raise RuntimeError("Locked v6 model/data hashes differ")
    for name, sha in old["reader_source_sha256"].items():
        if digest(V6 / name) != sha:
            raise RuntimeError(f"Locked v6 reader file differs: {name}")
    for name, sha in read_json(V6_RUN / "closed_freeze.json")["source_sha256"].items():
        if digest(V6 / name) != sha:
            raise RuntimeError(f"Locked v6 action file differs: {name}")
    return frozen_reader(warm_bundle(V6_RUN, atlas), old)


def init_pinned(atlas, readers):
    init_live(atlas, readers)


def freeze(out, name, rule):
    path = out / name
    if path.exists():
        from json import dumps, loads
        if read_json(path) != loads(dumps(rule)):
            raise RuntimeError(f"Frozen protocol already exists and differs: {name}")
    else:
        atomic_json(path, rule)
    return digest(path)
