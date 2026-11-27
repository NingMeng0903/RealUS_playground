"""Gather the region-V reader deck on fixed, actually acquired trajectories."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from math import ceil
from multiprocessing import get_context
from pathlib import Path
import hashlib

import numpy as np

from replay_v11 import ROOT, actor_path
from closed_scan import Engine, execute_program, programs, snapshot
from recovery_policy import read_bundle
from recovery_reader import load_bundle
from recovery_task import DEFAULT_PACK, Live, initialize, roster
from run_recovery import _save_snapshots, _snapshots_verified
from run_v11 import READER, outcome
from study_v9 import atomic_json, commit_shard, digest, freeze, read_json, verify_shard
import study_worlds


SEEDS = {stage: 8_202_730_000_000_000 + i*1_000_000
         for i, stage in enumerate(("fit", "cal", "reader_dev", "teach_fit", "teach_cal", "policy_dev", "test", "femoral"))}
FIT_COUNTS = {"fit": 168, "cal": 42, "reader_dev": 48}
_ENGINE = None
_AUGMENT = False


def cases(stage, n):
    source = "test" if stage in ("test", "femoral") else "dev" if stage in ("reader_dev", "policy_dev") else "reader_train"
    target = "femoral" if stage == "femoral" else "groin_to_distal_calf"
    templates = (c for c in roster(source, max(2, 2*ceil(n*1.75))) if c.task == target)
    for j, c in zip(range(n), templates):
        seed = SEEDS[stage] + 31*j
        yield replace(c, identifier=f"v12_{stage}_{j:05d}", split=f"v12_{stage}", world_seed=seed,
                      tape_seed=seed+1, action_seed=seed+2, challenge=False)


class RelativeLayouts:
    def __init__(self, base):
        self.base = base

    def build(self, source, seed, profile, split):
        layout = self.base.build(source, seed, profile, split)
        if split != "v12_fit" or source != 2 or seed % 3 == 0:
            return layout
        surface = self.base.scanner.surface
        centers = layout.centers
        rng = np.random.default_rng(seed + 819)
        x = layout.stations
        t = np.maximum(0., 1. - ((x - rng.uniform(-1085., -1040.))/85.)**2)**3
        touched = np.flatnonzero((t > 0.) & np.isfinite(centers[:, :2]).all(axis=(1, 2)))
        if not len(touched):
            return layout
        axis = rng.normal(size=2)
        axis /= np.linalg.norm(axis)
        move = t[touched, None]*axis[None, :]*rng.uniform(1.5, 4.5)
        origin = surface.cross_origin[touched]
        old = centers[touched, :2]
        old_gap = np.linalg.norm(old[:, 0] - old[:, 1], axis=1)
        # Keep the anatomical support, the signed vessel order near contact,
        # and the public bone/skin clearance. All transformations are synthetic.
        for scale in (1., .6, .35):
            proposal = old + scale*move[:, None, :]*np.array([1., -.85])[None, :, None]
            local = proposal - origin[:, None, :]
            angle = np.arctan2(local[..., 1], local[..., 0]) / (np.pi/12.)
            a = np.rint(angle).astype(int) % 24
            radial = np.linalg.norm(local, axis=-1)
            skin = np.linalg.norm(surface.contact[touched[:, None], a] - origin[:, None, :], axis=-1)
            # Variants move complete linked centerlines and retain all masks.
            if np.any(radial >= skin-2.) or np.any(np.linalg.norm(proposal[:, 0]-proposal[:, 1], axis=1) <
                                                   np.minimum(2.8, old_gap)):
                continue
            field = np.array([surface.occupancy(int(s), pts) for s, pts in zip(touched, proposal, strict=True)])
            base = np.array([surface.occupancy(int(s), pts) for s, pts in zip(touched, old, strict=True)])
            if np.any(field > np.maximum(.35, base + .04)):
                continue
            changed = centers.copy()
            changed[touched, :2] = proposal
            ident = hashlib.sha256(layout.identifier.encode() + changed.tobytes()).hexdigest()[:20]
            return replace(layout, centers=changed, identifier=f"v12_relative_{ident}",
                           metadata=layout.metadata | {"synthetic_relative": True, "displacement_mm": float(np.max(
                               np.linalg.norm(proposal - old, axis=-1)))})
        return layout


def init(atlas, actor, edge=None, decoder=None, blend=0., augment=False):
    global _ENGINE, _AUGMENT
    initialize(atlas, READER)
    _ENGINE = Engine(read_bundle(actor), load_bundle(READER), edge, decoder, blend)
    _AUGMENT = augment


def acquire(case):
    engine = _ENGINE
    env = study_worlds._WORLDS.environment(case.shape)
    if _AUGMENT and not isinstance(env.layout_factory, RelativeLayouts):
        env.layout_factory = RelativeLayouts(env.layout_factory)
    live = Live(case)
    is_fit = case.split == "v12_fit"
    engine.to_frontier(live, 4)
    initial = engine.read(live, original=True)
    layout = live.scan.tape.layout
    recorded = []
    if is_fit:
        recorded.append(snapshot(live, initial, "before"))
    end_branch = live.fork()
    end, base = engine.finish(end_branch, update=False)
    outcomes = {"plain": outcome(base)}
    recorded.append(snapshot(end_branch, end, "plain"))
    choices = programs(live, initial)
    agenda = [(4, "window"), (4, "trace_a"), (4, "trace_b")]
    if is_fit:
        side = 3 if case.action_seed % 3 else 4
        agenda += [(side, "repeat" if side == 4 else "window")]
    for pick in (x for x in choices if x.key in agenda):
        branch, effects = execute_program(live, initial, pick)
        end, grade = engine.finish(branch, update=False)
        label = f"{pick.region+1}_{pick.name}"
        outcomes[label] = outcome(grade) | {"execution": effects}
        recorded.append(snapshot(branch, end, label))
    for rec in recorded:
        rec.update(shape=int(case.shape), split=case.split, relative=bool(layout.metadata.get("synthetic_relative", False)))
    current = snapshot(live, initial, "truth_audit")
    labels = current["labels"]
    q = initial.hypotheses.get(4)
    i, j = q.indices if q is not None else (-1, -1)
    pair = [int(labels[k]) if k >= 0 else -99 for k in (i, j)]
    alias = i >= 0 and j >= 0 and current["instance"][i] >= 0 and current["instance"][i] == current["instance"][j]
    return {"case": asdict(case), "layout_id": layout.identifier, "relative": recorded[0]["relative"],
            "r5_pair": pair, "r5_same_object": bool(alias), "arms": outcomes}, recorded


def _check(records):
    if not all(r["arms"]["plain"]["completed_legal"] for r in records):
        raise RuntimeError("Incomplete fixed reader trajectory")
    return {
        "layouts": len({r["layout_id"] for r in records}),
        "relative_layouts": sum(r["relative"] for r in records),
        "r5_a_wrong_b_right": sum(a != 0 and b == 0 for a, b in (r["r5_pair"] for r in records)),
        "r5_aliases": sum(r["r5_same_object"] for r in records),
        "programs": Counter(arm for r in records for arm in r["arms"]),
        "r5_plain_wrong": sum(r["arms"]["plain"]["selected_ids"][4] != 0 for r in records),
    }


def collect(args):
    root = args.out
    atlas = args.atlas
    actor = actor_path()
    source_names = ("replay_v11.py", "track_state.py", "closed_scan.py", "data_v12.py")
    protocol = {"atlas_sha256": digest(atlas), "actor_sha256": digest(actor), "reader_sha256": digest(READER),
                "sources": {name: digest(ROOT/name) for name in source_names},
                "layout_seeds": SEEDS, "cohorts": FIT_COUNTS,
                "acquisition": "common v10 history to the end of R5, legal sound-window and bridge traces, same v10 continuation"}
    freeze(root, "reader_data_freeze.json", protocol)
    result = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=init,
                             initargs=(atlas, actor, None, None, 0., True)) as pool:
        for stage, n in FIT_COUNTS.items():
            group = list(cases(stage, n))
            if len(group) != n:
                raise RuntimeError("Reader case generator produced too few cases")
            rows = []
            for i in range(0, n, 12):
                path = root / "reader_data" / stage / f"{i//12:04d}"
                batch = group[i:i+12]
                if not (path/"commit.json").is_file():
                    meta, saved = zip(*pool.map(acquire, batch), strict=True)
                    _save_snapshots(path/"observer", [a for panel in saved for a in panel])
                    commit_shard(path, {"rows": np.arange(len(meta))}, meta)
                _, committed = verify_shard(path)
                if [x["case"] for x in committed] != [asdict(c) for c in batch] or not _snapshots_verified(path/"observer"):
                    raise RuntimeError("A reader shard differs from its scheduled acquisition")
                rows.extend(committed)
                print(stage, len(rows), "/", n, flush=True)
            result[stage] = _check(rows)
    if sum(s["layouts"] for s in result.values()) != sum(FIT_COUNTS.values()):
        raise RuntimeError("Reader layout duplications")
    atomic_json(root/"reader_data_summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"run")
    p.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=10)
    collect(p.parse_args())
