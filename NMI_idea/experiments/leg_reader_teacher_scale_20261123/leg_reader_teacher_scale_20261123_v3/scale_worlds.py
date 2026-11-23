"""Reproducible atlas-only worlds and packed candidate tracks for the scale study."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "leg_closed_loop_20261122_v1" / "src"
V2 = ROOT.parent / "leg_scan_target_routes_20261123_v2"
for directory in (V1, V2):
    sys.path.insert(0, str(directory))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["LEG_ENABLE_DIRD"] = "0"
os.environ["LEG_ENABLE_MATERIAL_ATLAS"] = "1"

import numpy as np

import motion
from environment import Arm, Case, ScanEnvironment, spec_for
from online_reader import PublicHistoryReader, _angular_distance
from public_embedding import BASELINE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES
from route_rollouts import (CALIBRATION_SHAPES, DEFAULT_PACK, EVALUATION_SHAPES, FEMORAL_SOURCES,
                            PROFILES, ROUTES, WHOLE_SOURCES, AtlasLayoutFactory, _install_input_guard,
                            _scanner_from_atlas, _validate_pack)
from storage import atomic_json, digest, json_lines, npz, read_json, read_lines


EXPERIMENT = "leg_reader_teacher_scale_20261123_v3"
PRIOR = V2 / "fixed_pilot" / "weights" / "full_public_embedding.pt"
BASE = len(BASELINE_FEATURE_NAMES)
TAIL = EXTENDED_FEATURE_NAMES[BASE:]
FEATURES = (BASELINE_FEATURE_NAMES + tuple(f"available_{name}" for name in BASELINE_FEATURE_NAMES)
            + TAIL + tuple(f"available_{name}" for name in TAIL))
ARM = Arm("shared_public_continuation", dird=False, stop=False)
SEED_RANGES = {name: 1_760_340_000_000_000 + i * 10_000_000
               for i, name in enumerate(("smoke", "reader_fit", "reader_calibration", "reader_test",
                                          "teacher_fit", "teacher_calibration", "teacher_test"))}
FUTURE_SEEDS = {"fit": (1_000_000_000_000, 2_000_000_000_000),
                "search": (3_000_000_000_000, 4_000_000_000_000),
                "proof": (5_000_000_000_000, 6_000_000_000_000)}
LEGACY_THRESHOLD = .82


class VectorHistoryReader(PublicHistoryReader):
    """Same v1 public support reductions with the per-candidate masks batched."""

    def _pair_support(self, new_ids, old_ids):
        if not new_ids or not old_ids:
            return
        old = [self._candidates[i] for i in old_ids]
        new = [self._candidates[i] for i in new_ids]

        def columns(name):
            return np.asarray([x[name] for x in old])[:, None], np.asarray([x[name] for x in new])[None, :]

        os_, ns = columns("station")
        oa, na = columns("sector")
        ds = abs(os_ - ns)
        da = _angular_distance(oa, na, motion.SECTORS)
        correlation = np.exp(-.5 * (ds / .55) ** 2 - .5 * (da / .75) ** 2)
        for ids, scores in ((old_ids, correlation.max(axis=1)), (new_ids, correlation.max(axis=0))):
            for cid, value in zip(ids, scores):
                self._repetition[cid] = max(self._repetition[cid], float(value))
        valid = ((ds >= .65) & (ds <= self._support_radius)) | ((ds <= 2.05) & (da >= self._support_angle))
        if not valid.any():
            return
        ow, nw = columns("world")
        oq, nq = columns("quality")
        op, np_ = columns("posterior")
        region_o, region_n = columns("region")
        distance = np.linalg.norm(ow - nw, axis=-1)
        support = np.exp(-.5 * (distance / (4. + ds)) ** 2) * (.65 + .35 * np.minimum(oq, nq))
        support *= (.55 + .45 * np.sqrt(np.minimum(op, np_)))
        support *= 1. - .72 * correlation
        support = np.where(valid, support, 0.)
        across = np.where(region_o != region_n, support, 0.)
        for ids, axis in ((old_ids, 1), (new_ids, 0)):
            for cid, score, cross in zip(ids, support.max(axis=axis), across.max(axis=axis)):
                self._supports[cid] = max(self._supports[cid], float(score))
                self._cross_supports[cid] = max(self._cross_supports[cid], float(cross))


class Worlds:
    def __init__(self, pack):
        _install_input_guard()
        self.data = _validate_pack(Path(pack))
        self.sha256 = digest(pack)
        self.envs = {}

    def environment(self, shape):
        if shape not in self.envs:
            scanner = _scanner_from_atlas(self.data, shape)
            factory = AtlasLayoutFactory(self.data, scanner, shape, self.sha256)
            env = ScanEnvironment(scanner, factory)
            scanner.reader_type = VectorHistoryReader
            self.envs[shape] = env
        return self.envs[shape]


def cases_for(stage, count):
    """Each split has its own layout, measurement and action seed range."""
    if not 2 <= count < 1_000_000 or count % 2:
        raise ValueError("Case counts must be positive even numbers below one million")
    shapes = EVALUATION_SHAPES if stage.endswith("_test") else CALIBRATION_SHAPES
    for i in range(count):
        task = ROUTES[i % 2]
        sources = FEMORAL_SOURCES if task == "femoral" else WHOLE_SOURCES
        slot = i // 2
        profile = PROFILES[slot % len(PROFILES)]
        source = sources[slot // len(PROFILES) % len(sources)]
        shape = shapes[slot // (len(PROFILES) * len(sources)) % len(shapes)]
        band = slot // (len(PROFILES) * len(sources) * len(shapes))
        base = SEED_RANGES[stage] + i * 7
        challenge = band % 3 == 1
        yield Case(f"v3_{stage}_{i:06d}", stage, source, shape, task, profile,
                   base, base + 1, base + 2, band, challenge)


def complete(env, spec, node, trial, limits, until=None):
    goal = spec[2] if until is None else until
    while node.station < goal:
        env.route.advance(node, spec, trial, limits)
    if node.station != goal or node.tick > limits.ticks or node.distance > limits.mm + 1.e-6:
        raise RuntimeError("Common MPC continuation exceeded its terminal envelope")
    return node


def vectors(env, spec, node):
    from fast_embedding import extract

    observed = extract(node.reader, spec, env.material)
    full, present = observed["extended"], observed["extended_mask"]
    return {
        "features": np.c_[full[:, :BASE], present[:, :BASE], full[:, BASE:], present[:, BASE:]].astype(np.float32),
        "frames": observed["frame_index"], "slots": observed["slot"], "regions": observed["region"],
        "v1_scores": observed["baseline"][:, 0],
    }


def private_outcomes(node, observed):
    """Labels reach this evaluator after all public features have been constructed."""
    f = observed["frames"]
    return {"labels": np.asarray([node.reader.frames[i][3][j] for i, j in zip(f, observed["slots"])], dtype=np.int8),
            "stations": np.asarray([node.reader.frames[i][0] for i in f], dtype=np.float32),
            "sectors": np.asarray([node.reader.frames[i][1] for i in f], dtype=np.float32)}


_WORLDS = None


def init_worlds(pack):
    global _WORLDS
    _WORLDS = Worlds(pack)


def start(case):
    env = _WORLDS.environment(case.shape)
    spec = spec_for(env.scanner, case)
    node, trial, budget = env.start(case, spec)
    return env, spec, node, trial, budget


def windows_visited(node, trial):
    window = trial.window
    if window is None:
        return False, False
    visits = [frame[1] for frame in node.reader.frames if abs(frame[0] - window.station) <= .60]
    return tuple(any(abs(motion._sector_delta(angle, sector)) <= .60 for angle in visits)
                 for sector in (window.sector, window.alternate_sector))


def collect_reader_case(case):
    """Physically acquire every candidate on a complete, contact-checked scan."""
    env, spec, node, trial, limits = start(case)
    skill = 0
    name = "mpc"
    # Test and calibration use the reference path. Training sees additional
    # legal feedback poses distributed across the route.
    if case.split in ("reader_fit", "smoke"):
        rng = np.random.default_rng(case.action_seed)
        skill = int(rng.choice(3, p=[.50, .25, .25]))
        if skill:
            onset = int(rng.integers(spec[1] + 2, spec[2]))
            complete(env, spec, node, trial, limits, onset)
            caps = limits.probe(node, spec)
            for turn in rng.permutation([-.79, -.46, .46, .79]):
                controls = np.array([turn, .72 if skill == 2 else 0., -.27 * turn])
                plan = env.plan(node, spec, skill, controls, caps, ARM)
                if plan is not None:
                    result = env.material.run_probe(node, spec, trial, plan, *caps, use_dird=False)
                    if not result.legal:
                        raise RuntimeError(f"Reader acquisition failed during probe: {result.reason}")
                    node = result.node
                    name = "window" if skill == 1 else "revisit"
                    break
    complete(env, spec, node, trial, limits)
    observed = vectors(env, spec, node)
    data = observed | private_outcomes(node, observed)
    ref_window, alt_window = windows_visited(node, trial)
    meta = {"case": asdict(case), "layout_id": trial.layout.identifier, "acquisition": name,
            "ticks": node.tick, "frames": len(node.reader.frames), "distance_mm": node.distance,
            "window_generated": trial.window is not None, "reference_window_visited": ref_window,
            "alternative_window_visited": alt_window, "complete": True,
            "harmonic_candidates": int(data["features"][:, FEATURES.index("available_harmonic_supported")].sum())}
    return data, meta


def pack_tracks(rows):
    fields = rows[0][0].keys()
    data = {name: np.concatenate([row[name] for row, _ in rows], axis=0) for name in fields}
    data["features"] = data["features"].astype(np.float16)
    data["offsets"] = np.cumsum([0, *(len(row["labels"]) for row, _ in rows)], dtype=np.int64)
    data["feature_order"] = np.asarray(FEATURES)
    return data


def commit_shard(directory, arrays, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = directory / "data.npz", directory / "cases.jsonl.gz"
    pending = directory / "data.pending"
    npz(pending, arrays)
    pending.replace(data_path)
    pending = directory / "cases.pending"
    json_lines(pending, metadata)
    pending.replace(meta_path)
    atomic_json(directory / "commit.json", {"cases": len(metadata), "data_sha256": digest(data_path),
                                         "metadata_sha256": digest(meta_path)})


def load_shard(directory):
    commit = read_json(directory / "commit.json")
    metadata = list(read_lines(directory / "cases.jsonl.gz"))
    if len(metadata) != commit["cases"]:
        raise RuntimeError(f"Sharded case ledger disagrees with its commit: {directory}")
    return directory / "data.npz", metadata


def collect(pool, stage, count, out, worker, pack_rows, shard_size):
    candidates = list(cases_for(stage, count))
    paths, all_meta = [], []
    began = monotonic()
    for index in range(0, count, shard_size):
        folder = out / stage / f"{index // shard_size:05d}"
        batch = candidates[index:index + shard_size]
        if not (folder / "commit.json").exists():
            rows = list(pool.map(worker, batch, chunksize=1))
            data, meta = pack_rows(rows), [x[1] for x in rows]
            commit_shard(folder, data, meta)
        else:
            commit = read_json(folder / "commit.json")
            for filename, key in (("data.npz", "data_sha256"), ("cases.jsonl.gz", "metadata_sha256")):
                if digest(folder / filename) != commit[key]:
                    raise RuntimeError(f"Committed shard checksum failed: {folder / filename}")
        path, meta = load_shard(folder)
        if [row["case"] for row in meta] != [asdict(case) for case in batch]:
            raise RuntimeError(f"Existing {stage} shard differs from the frozen case schedule")
        paths.append(path)
        all_meta.extend(meta)
        print(f"{stage}: committed {len(all_meta)}/{count} cases, {monotonic() - began:.1f}s", flush=True)
    if len(set(row["layout_id"] for row in all_meta)) != count:
        raise RuntimeError(f"Duplicate synthetic layout in {stage}")
    return paths, all_meta
