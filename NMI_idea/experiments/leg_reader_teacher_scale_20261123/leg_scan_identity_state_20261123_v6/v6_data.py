"""Frozen inputs and resumable, disjoint v6 simulation cohorts."""

from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
import sys

ROOT = Path(__file__).resolve().parent
V5 = ROOT.parent / "leg_scan_anchor_state_20261123_v5"
V5_RUN = V5 / "run"
BASE = "ba2871673325b714d877b8004acef0f58139b959"
sys.path.insert(1, str(V5))

from study_data import (DEFAULT_PACK, ROUTES, V4_RUN, atomic_json, cases_for, commit_shard, current_scan, digest,
                        init_worlds, load_shard, pack_acquired, read_json)


# Each group is independent by both world seed and measurement seed. The
# main route has more held-out layouts; the comparator route is kept separate.
STAGES = {
    "reader_fresh": {"femoral": 64, "groin_to_distal_calf": 96},
    "closed_dev": {"femoral": 16, "groin_to_distal_calf": 32},
    "closed_test": {"femoral": 32, "groin_to_distal_calf": 96},
}
SEED_BASES = {key: 2_084_500_000_000_000 + j * 10_000_000 for j, key in enumerate(STAGES)}
# A second reader roster avoids reusing the exploratory panel after pinning
# the final usability export. Closed-loop rosters retain their own bands.
SEED_BASES["reader_fresh"] += 1_000_000


def cases(stage, counts=None):
    required = STAGES[stage] if counts is None else counts
    family = "fit" if stage == "closed_dev" else "reader_test"
    tally = {key: 0 for key in required}
    for index, old in enumerate(cases_for(family, 2 * max(required.values()))):
        route = old.task
        if tally[route] >= required[route]:
            continue
        seed = SEED_BASES[stage] + 11 * index
        yield replace(old, identifier=f"v6_{stage}_{index:05d}", split=stage, world_seed=seed,
                      tape_seed=seed + 1, action_seed=seed + 2, challenge=False)
        tally[route] += 1


def collect(pool, out, stage, runner, packer, *, size=16, counts=None):
    planned = list(cases(stage, counts))
    paths, meta = [], []
    begin = monotonic()
    for start in range(0, len(planned), size):
        batch = planned[start:start + size]
        shard = out / stage / f"{start // size:05d}"
        if not (shard / "commit.json").exists():
            rows = list(pool.map(runner, batch, chunksize=1))
            commit_shard(shard, packer(rows), [row[1] for row in rows])
        path, rows = load_shard(shard)
        commit = read_json(shard / "commit.json")
        if (commit["data_sha256"] != digest(path) or commit["metadata_sha256"] != digest(shard / "cases.jsonl.gz")
                or [r["case"] for r in rows] != [asdict(c) for c in batch] or not all(r["complete"] for r in rows)):
            raise RuntimeError(f"Invalid committed simulation shard {shard}")
        paths.append(path)
        meta.extend(rows)
        print(f"{stage}: {len(meta)}/{len(planned)} completed layouts, {monotonic() - begin:.1f}s", flush=True)
    if len({r["layout_id"] for r in meta}) != len(planned):
        raise RuntimeError(f"Layout repeated in {stage}")
    return paths, meta


def readonly_input(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return {"path": str(p), "sha256": digest(p)}
