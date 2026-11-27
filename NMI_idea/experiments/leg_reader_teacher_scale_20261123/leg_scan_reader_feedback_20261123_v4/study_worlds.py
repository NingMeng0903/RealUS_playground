"""Atlas-only physical trajectories and causal public snapshots for v4."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V3 = ROOT.parent / "leg_reader_teacher_scale_20261123_v3"
sys.path.insert(0, str(V3))

from scale_worlds import (ARM, DEFAULT_PACK, FEATURES, ROUTES, V1, V2, Worlds, atomic_json, complete,
                          commit_shard, digest, load_shard, motion, read_json)
from environment import Case, spec_for, station_bins
from route_rollouts import CALIBRATION_SHAPES, EVALUATION_SHAPES, FEMORAL_SOURCES, PROFILES, WHOLE_SOURCES
from public_features import CorrectedPublicHistoryReader, public_arrays, replay_reader

import numpy as np


EXPERIMENT = "leg_scan_reader_feedback_20261123_v4"
BASE_COMMIT = "22d1f9c8c1e048f64ba6546c166c0b748e9d3893"
PRIOR = V3 / "run" / "reader_weights" / "scaled_identity.pt"
PRIOR_MANIFEST = V3 / "run" / "reader_freeze.json"
SCOPES = ("before", "after_hold", "after_probe", "terminal")
SPLITS = ("smoke", "fit", "calibrate", "select_reader", "reader_test", "search_dev", "search_test", "feedback_test")
SEED_BASES = {stage: 1_988_630_000_000_000 + i * 10_000_000 for i, stage in enumerate(SPLITS)}
READER_SCANS = {"fit": 8192, "calibrate": 512, "select_reader": 512, "reader_test": 1024}
SHARD = 96


class StudyWorlds(Worlds):
    def environment(self, shape):
        env = super().environment(shape)
        env.scanner.reader_type = CorrectedPublicHistoryReader
        return env


def cases_for(stage, count):
    if count < 2 or count % 2 or count >= 1_000_000:
        raise ValueError("Each stage needs an even number of paired-route cases")
    shapes = EVALUATION_SHAPES if stage.endswith("_test") else CALIBRATION_SHAPES
    for i in range(count):
        task = ROUTES[i % 2]
        slot = i // 2
        sources = FEMORAL_SOURCES if task == "femoral" else WHOLE_SOURCES
        profile = PROFILES[slot % 2]
        source = sources[(slot // 2) % len(sources)]
        shape = shapes[(slot // (2 * len(sources))) % len(shapes)]
        band = slot // (2 * len(sources) * len(shapes))
        seed = SEED_BASES[stage] + i * 7
        yield Case(f"v4_{stage}_{i:06d}", stage, source, shape, task, profile, seed, seed + 1, seed + 2, band)


_WORLDS = None


def init_worlds(pack):
    global _WORLDS
    _WORLDS = StudyWorlds(pack)


def start(case):
    env = _WORLDS.environment(case.shape)
    spec = spec_for(env.scanner, case)
    node, trial, limits = env.start(case, spec)
    return env, spec, node, trial, limits


def material_widths(env, spec):
    widths = np.zeros(len(env.scanner.stations), dtype=np.float32)
    widths[spec[1]:spec[2] + 1] = station_bins(spec, env.scanner.surface.y)
    return widths


def scene_links(env):
    return np.asarray(env.scanner.surface.reference_links)


def read_prefix(env, spec, node, *, legacy=False):
    reader = replay_reader(node.reader, node.reader.frames, node.reader.mixed_posterior, legacy=True) if legacy else node.reader
    return public_arrays(reader, spec, env.material, legacy=legacy)


def truth_at(node, public, trial):
    labels = np.asarray([node.reader.frames[i][3][j] for i, j in zip(public["frames"], public["slots"], strict=True)], dtype=np.int8)
    # Roles 3 and 4 can include several named vessels along an atlas. A new
    # instance starts at every source-unsupported edge. Positive temporal
    # links are supervised only within the locally connected component.
    links = trial.layout.links
    cuts = np.r_[np.zeros((1, links.shape[1]), dtype=bool), ~links].cumsum(axis=0)
    bins = np.clip(np.rint(public["stations"]).astype(int), 0, len(cuts) - 1)
    instances = np.full(len(labels), -1, dtype=np.int16)
    valid = labels >= 0
    roles = labels[valid].astype(np.int32)
    instances[valid] = roles * len(cuts) + cuts[bins[valid], roles]
    return {"labels": labels, "instance": instances}


def stash_public(env, spec, node, trial, tag, old_frames, *, include_legacy):
    p = read_prefix(env, spec, node)
    truth = truth_at(node, p, trial)
    names = ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality")
    snap = {name: p[name] for name in names} | truth
    snap["features"] = snap["features"].astype(np.float16)
    if include_legacy:
        old = read_prefix(env, spec, node, legacy=True)
        for key in ("frames", "slots", "regions"):
            np.testing.assert_array_equal(p[key], old[key])
        snap["legacy_features"] = old["features"].astype(np.float16)
    return snap, {"scope": tag, "first_frame_count": old_frames, "frames": len(node.reader.frames),
                  "complete": node.station == spec[2]}


def read_plan(env, spec, node, limits, rng):
    caps = limits.probe(node, spec)
    skill = 1 if rng.random() < .5 else 2
    for turn in rng.permutation([-.82, .82, -.50, .50]):
        cmd = np.array([turn, .73 if skill == 2 else 0., -.18 * turn])
        plan = env.plan(node, spec, skill, cmd, caps, ARM)
        if plan is not None and (abs(plan.goal[1] - plan.start[1]) >= .50 or abs(plan.goal[0] - plan.start[0]) >= .50):
            return plan, caps
    return None, caps


def held_frames(env, spec, node, trial, ticks):
    for _ in range(ticks):
        node.tick += 1
        env.scanner._observe(node, spec, trial, node.station, node.sector, node.tick)
    return node


def acquire(case):
    env, spec, node, trial, limits = start(case)
    rng = np.random.default_rng(case.action_seed)
    if case.task != "femoral" and case.band % 3 != 0:
        stations = np.flatnonzero(spec[4] == 2) + spec[1]
        at = int(rng.choice(stations))
    else:
        fraction = (.15, .30, .47, .67, .84)[case.band % 5]
        at = spec[1] + int(fraction * (spec[2] - spec[1]))
    complete(env, spec, node, trial, limits, at)
    old_frames = len(node.reader.frames)
    legacy = case.split != "fit"
    snaps = [stash_public(env, spec, node, trial, "before", old_frames, include_legacy=legacy)]
    plan, caps = read_plan(env, spec, node, limits, rng)
    probe = None
    executed_skill = 0
    if plan is not None:
        motion_result = env.material.run_probe(node, spec, trial, plan, *caps, use_dird=False)
        if not motion_result.legal:
            raise RuntimeError(f"An admitted readout acquisition failed: {motion_result.reason}")
        probe = motion_result.node
        executed_skill = plan.skill
        # Measure old-image revision on actual probe and equal-time hold paths.
        if case.split != "fit":
            hold = held_frames(env, spec, node.fork(), trial, plan.ticks)
            snaps.append(stash_public(env, spec, hold, trial, "after_hold", old_frames, include_legacy=legacy))
        snaps.append(stash_public(env, spec, probe, trial, "after_probe", old_frames, include_legacy=legacy))
    # Evaluation terminal trajectories are regular MPC, paired across readers.
    # Training also acquires entire feedback probe tracks across the route.
    final = probe if case.split == "fit" and probe is not None and case.band % 2 else node
    complete(env, spec, final, trial, limits)
    snaps.append(stash_public(env, spec, final, trial, "terminal", old_frames, include_legacy=legacy))
    meta = {"case": asdict(case), "layout_id": trial.layout.identifier, "complete": True,
            "probe_skill": executed_skill, "probe_in_region": motion.region_of(spec, at),
            "probe_ticks": 0 if plan is None else plan.ticks,
            "probe_goal": None if plan is None else plan.goal.tolist(),
            "histories": [m for _, m in snaps], "frame_count": len(final.reader.frames)}
    return snaps, meta


def pack_acquired(items):
    snapshots = [p for item, _ in items for p, _ in item]
    payload = {name: np.concatenate([row[name] for row in snapshots], axis=0) for name in snapshots[0]}
    payload["offsets"] = np.cumsum([0, *(len(row["frames"]) for row in snapshots)], dtype=np.int64)
    payload["feature_order"] = np.asarray(FEATURES)
    payload["snapshot_case"] = np.repeat(np.arange(len(items)), [len(item) for item, _ in items])
    return payload


def collect(pool, base, stage, count, worker, pack, shard_size):
    from time import monotonic
    roster = list(cases_for(stage, count))
    paths, meta = [], []
    began = monotonic()
    for start in range(0, count, shard_size):
        folder = base / stage / f"{start // shard_size:05d}"
        batch = roster[start:start + shard_size]
        if not (folder / "commit.json").is_file():
            run = list(pool.map(worker, batch, chunksize=1))
            commit_shard(folder, pack(run), [m for _, m in run])
        file, rows = load_shard(folder)
        receipt = read_json(folder / "commit.json")
        if digest(file) != receipt["data_sha256"] or digest(folder / "cases.jsonl.gz") != receipt["metadata_sha256"]:
            raise RuntimeError(f"Shard checksum mismatch in {folder}")
        if [row["case"] for row in rows] != [asdict(case) for case in batch]:
            raise RuntimeError(f"Existing {stage} shard differs from the frozen case schedule")
        paths.append(file)
        meta.extend(rows)
        print(f"{stage}: committed {len(meta)}/{count} cases, {monotonic() - began:.1f}s", flush=True)
    if len({m["layout_id"] for m in meta}) != len(meta):
        raise RuntimeError(f"Duplicate layout in {stage}")
    return paths, meta


def completed(base, stage, count, shard_size):
    paths, meta = [], []
    for begin in range(0, count, shard_size):
        path, rows = load_shard(base / stage / f"{begin // shard_size:05d}")
        receipt = read_json(path.parent / "commit.json")
        if digest(path) != receipt["data_sha256"] or digest(path.parent / "cases.jsonl.gz") != receipt["metadata_sha256"]:
            raise RuntimeError(f"Immutable input shard did not pass its hash: {path}")
        paths.append(path)
        meta.extend(rows)
    if ([m["case"] for m in meta] != [asdict(c) for c in cases_for(stage, count)]
            or len({m["layout_id"] for m in meta}) != count):
        raise RuntimeError(f"Ledger for {stage} changed")
    return paths, meta
