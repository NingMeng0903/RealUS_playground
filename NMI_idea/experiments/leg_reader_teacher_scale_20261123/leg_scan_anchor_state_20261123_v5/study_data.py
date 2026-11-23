"""Atlas-only acquisition for causal association and downstream-anchor tests."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import multiprocessing as mp
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / "leg_scan_reader_feedback_20261123_v4"
sys.path.insert(0, str(V4))

from study_worlds import (ARM, DEFAULT_PACK, FEATURES, ROUTES, V1, V2, V3, cases_for, complete, digest,
                          held_frames, init_worlds, load_shard, motion, pack_acquired, read_json, read_plan,
                          scene_links, start, stash_public)
from scale_worlds import atomic_json, commit_shard

import numpy as np


EXPERIMENT = "leg_scan_anchor_state_20261123_v5"
V4_RUN = V4 / "run"
BASE = "c4e4b321d68c011d2c6e5d80b03390e39c8ee5ea"
STAGE_COUNTS = {"fit_revisit": 1536, "reader_test": 512, "anchor_dev": 96, "anchor_test": 192}
V5_SEEDS = {name: 2_046_300_000_000_000 + i * 10_000_000 for i, name in enumerate(STAGE_COUNTS)}
SHARD = 64


def roster(stage, count=None):
    count = STAGE_COUNTS[stage] if count is None else count
    seed_base = V5_SEEDS[stage]
    family = "reader_test" if stage.endswith("_test") else "fit"
    for i, case in enumerate(cases_for(family, count)):
        seed = seed_base + 7 * i
        yield replace(case, identifier=f"v5_{stage}_{i:06d}", split=stage, world_seed=seed, tape_seed=seed + 1,
                      action_seed=seed + 2, challenge=False)


def current_scan(case):
    env, spec, node, trial, limits = start(case)
    rng = np.random.default_rng(case.action_seed)
    # The full-route acquisitions target region 3 as well as the approach
    # to the named popliteal/posterior tibial transition.
    desired = int(np.argmin(abs(env.scanner.stations - (-750. if case.task == "femoral" else -910.))))
    where = (desired - 1, spec[1] + int((spec[2] - spec[1]) * .45), desired + 2)[case.band % 3]
    where = int(np.clip(where, spec[1] + 3, spec[2] - 2))
    complete(env, spec, node, trial, limits, where)
    cut = len(node.reader.frames)
    legacy = case.split != "fit_revisit"
    saved = [stash_public(env, spec, node, trial, "before", cut, include_legacy=legacy)]
    plan, caps = read_plan(env, spec, node, limits, rng)
    realized_tick_gap = 0
    if plan is not None:
        branch = node.fork()
        # Acquire two separated visits to the same local anatomy. A temporal
        # gap may recur at an otherwise nearby physical station.
        if not legacy:
            reserve = limits.reserve_ticks[int(node.station) - spec[1], int(node.sector)]
            spare = max(0, limits.ticks - node.tick - int(reserve) - plan.ticks)
            held_frames(env, spec, branch, trial, min(24, spare))
            caps = limits.probe(branch, spec)
            realized_tick_gap = branch.tick - node.tick
        result = env.material.run_probe(branch, spec, trial, plan, *caps, use_dird=False, copy_node=False)
        if not result.legal:
            raise RuntimeError(f"The observed revisit failed to rejoin: {result.reason}")
        probe = result.node
        if legacy:
            h = held_frames(env, spec, node.fork(), trial, plan.ticks)
            saved.append(stash_public(env, spec, h, trial, "after_hold", cut, include_legacy=legacy))
        saved.append(stash_public(env, spec, probe, trial, "after_probe", cut, include_legacy=legacy))
        if not legacy:
            node = probe
    complete(env, spec, node, trial, limits)
    saved.append(stash_public(env, spec, node, trial, "terminal", cut, include_legacy=legacy))
    return saved, {"case": asdict(case), "layout_id": trial.layout.identifier, "complete": True,
                   "scope": "actual_revisit" if not legacy else "ordinary_mpc",
                   "histories": [h for _, h in saved], "probe_in_region": motion.region_of(spec, where),
                   "probe_skill": plan.skill if plan is not None else 0, "same_pose_wait_ticks": realized_tick_gap}


def collect(pool, out, stage, worker, pack, *, size=SHARD, count=None):
    cases = list(roster(stage, count))
    paths, rows = [], []
    began = monotonic()
    for j in range(0, len(cases), size):
        part = cases[j:j + size]
        shard = out / stage / f"{j // size:05d}"
        if not (shard / "commit.json").exists():
            output = list(pool.map(worker, part, chunksize=1))
            commit_shard(shard, pack(output), [m for _, m in output])
        path, ledger = load_shard(shard)
        index = read_json(shard / "commit.json")
        if index["data_sha256"] != digest(path) or index["metadata_sha256"] != digest(shard / "cases.jsonl.gz"):
            raise RuntimeError("Study shard hash changed")
        if [m["case"] for m in ledger] != [asdict(c) for c in part]:
            raise RuntimeError("Study shard case schedule changed")
        paths.append(path)
        rows.extend(ledger)
        print(f"{stage}: {len(rows)}/{len(cases)} cases, {monotonic() - began:.1f}s", flush=True)
    if len({m["layout_id"] for m in rows}) != len(cases):
        raise RuntimeError("Synthetic layouts are not disjoint")
    return paths, rows


def fit_inventory(out):
    paths = []
    ledger = []
    for expected in range(0, STAGE_COUNTS["fit_revisit"], SHARD):
        folder = out / "fit_revisit" / f"{expected // SHARD:05d}"
        path, rows = load_shard(folder)
        if digest(path) != read_json(folder / "commit.json")["data_sha256"]:
            raise RuntimeError("Independent fit shard failed its hash")
        paths.append(path)
        ledger.extend(rows)
    if [m["case"] for m in ledger] != [asdict(c) for c in roster("fit_revisit")]:
        raise RuntimeError("Fit cases differ from the frozen roster")
    return paths, ledger


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ROOT / "run", type=Path)
    parser.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    count = args.smoke or STAGE_COUNTS["fit_revisit"]
    source = {str(p.relative_to(ROOT.parent)): digest(p) for p in (Path(__file__), V4 / "study_worlds.py",
                                                                  V4 / "public_features.py", V3 / "scale_worlds.py")}
    protocol = {"experiment": EXPERIMENT, "base_commit": BASE, "atlas_sha256": digest(args.atlas_pack),
                "fit_count": count, "stages": STAGE_COUNTS, "case_seed_bases": V5_SEEDS, "source_sha256": source,
                "dird": False}
    if (args.out / "acquisition.json").exists() and read_json(args.out / "acquisition.json") != protocol:
        raise RuntimeError("Collection protocol changed. Use a new output directory.")
    atomic_json(args.out / "acquisition.json", protocol)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worlds,
                             initargs=(args.atlas_pack.resolve(),)) as pool:
        collect(pool, args.out, "fit_revisit", current_scan, pack_acquired, count=count)


if __name__ == "__main__":
    main()
