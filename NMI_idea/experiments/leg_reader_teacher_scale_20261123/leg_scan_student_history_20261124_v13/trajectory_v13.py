"""Collect current-student trajectories without changing the frozen v12 policy."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V12 = ROOT.parent / "leg_scan_region_evidence_20261124_v12"
sys.path.insert(1, str(V12))

import data_v12
from closed_scan import execute_program, programs
from evidence import fork_on_tape
from fit_v12 import hydrate
from recovery_task import Live
from region_policy import select
from replay_v11 import actor_path
from run_v11 import READER, outcome
from study_v9 import digest, read_json
from study_worlds import truth_at
import study_worlds


RUN12 = V12 / "run"
STAGES = {"fit": 1024, "cal": 64, "reader_dev": 72, "teach": 120, "teach_cal": 36,
          "policy_dev": 48, "test": 108, "femoral": 36}
SEEDS = {s: 8_420_130_000_000_000 + i*1_000_000 for i, s in enumerate(STAGES)}
EDIT_TAPES = (222_668_201, 222_668_271)
EPS = .002


def cases(stage, n):
    # Reuse the established source, shape and profile templates, with new
    # layout seeds and independent observation/action streams.
    source = stage if stage in ("fit","cal","reader_dev","policy_dev","test","femoral") else "fit"
    for i, c in enumerate(data_v12.cases(source, n)):
        seed = SEEDS[stage] + 41*i
        yield replace(c, identifier=f"v13_{stage}_{i:05d}", split=f"v13_{stage}",
                      world_seed=seed, tape_seed=seed+1, action_seed=seed+2)


def _ordinal(case):
    return int(case.identifier.rsplit("_", 1)[-1])


def inputs():
    cfg = read_json(RUN12/"test_freeze.json")
    weights = RUN12/"models"
    files = {"edge": weights/"association.json", "reader": weights/"reader.json",
             "student": weights/f"student_{cfg['model'][:2]}.json", "v10_reader": READER,
             "v10_actor": actor_path()}
    data = read_json(RUN12/"protocol.json")
    expected = (cfg["association_sha256"], cfg["reader_sha256"], cfg["actor_sha256"],
                data["v10_reader_sha256"], data["actor_sha256"])
    if [digest(p) for p in files.values()] != list(expected):
        raise RuntimeError("The frozen v12 or v10 model files changed")
    return cfg, files


class LocalVariants:
    """Move connected centerlines by a smooth local relative deformation."""

    def __init__(self, base, stage_idx):
        self.base = base
        self.index = stage_idx

    def build(self, source, seed, profile, split):
        layout = self.base.build(source, seed, profile, split)
        if source != 2 or self.index % 8 not in (0, 1):
            return layout
        surface = self.base.scanner.surface
        rng = np.random.default_rng(seed + 1021)
        axis = rng.normal(size=2)
        axis /= np.linalg.norm(axis)
        center = rng.uniform(-1217., -1132.) if self.index % 4 == 0 else rng.uniform(-1104., -1016.)
        weight = np.maximum(0., 1. - ((layout.stations - center)/69.)**2)**3
        finite = np.isfinite(layout.centers[:, :2]).all(axis=(1, 2))
        stations = np.flatnonzero((weight > 0.) & finite)
        if not len(stations):
            return layout
        original = layout.centers[stations, :2]
        origin = surface.cross_origin[stations, None, :]
        delta = weight[stations, None] * rng.uniform(1.5, 4.5)*axis
        for factor in (1., .55, .30):
            changed = original + factor*delta[:, None, :]*np.array([1., -.80])[None, :, None]
            local = changed - origin
            angle = np.rint(np.arctan2(local[...,1], local[...,0])*12./np.pi).astype(int)%24
            radial = np.linalg.norm(local, axis=-1)
            skin = np.linalg.norm(surface.contact[stations[:,None],angle]-origin, axis=-1)
            gap = np.linalg.norm(changed[:,0]-changed[:,1], axis=-1)
            old_gap = np.linalg.norm(original[:,0]-original[:,1], axis=-1)
            if np.any(radial >= skin-2.) or np.any(gap < np.minimum(2.8, old_gap)):
                continue
            bone = np.stack([surface.occupancy(int(s), p) for s,p in zip(stations,changed,strict=True)])
            baseline = np.stack([surface.occupancy(int(s), p) for s,p in zip(stations,original,strict=True)])
            if np.any(bone > np.maximum(.35, baseline + .04)):
                continue
            new = layout.centers.copy()
            new[stations,:2] = changed
            displace = float(np.max(np.linalg.norm(changed-original,axis=-1)))
            ident = hashlib.sha256(layout.identifier.encode()+new.tobytes()).hexdigest()[:20]
            return replace(layout, centers=new, identifier=f"v13_relative_{ident}", metadata=layout.metadata | {
                "v13_relative": True, "public_bone_skin_checked": True, "displacement_mm": displace})
        return layout


def init(atlas, switches=None):
    global ENGINE, FROZEN, CONFIG, EXTRA
    cfg, paths = inputs()
    data_v12.init(atlas, paths["v10_actor"], hydrate(read_json(paths["edge"])),
                  hydrate(read_json(paths["reader"])), cfg["blend"])
    ENGINE = data_v12._ENGINE
    FROZEN = hydrate(read_json(paths["student"]))
    CONFIG = cfg["rule"]
    EXTRA = switches


def live_case(case, offset=None, *, augment=False):
    env = study_worlds._WORLDS.environment(case.shape)
    factory = env.layout_factory
    try:
        if augment:
            env.layout_factory = LocalVariants(factory, _ordinal(case))
        live = Live(case, offset)
    finally:
        env.layout_factory = factory
    if augment and _ordinal(case)%8 in (3, 4):
        site = int(live.ends[5] - (7 if _ordinal(case)%8 == 3 else 13))
        legal = np.flatnonzero(live.scan.env.scanner.valid[site])
        sector = int(legal[(_ordinal(case)//8) % len(legal)])
        live.scan.tape.window = SimpleNamespace(station=site, sector=sector)
    return live


@dataclass
class Result:
    live: Live
    read: object
    actions: list
    seen: frozenset
    prefix: object = None


def menu(live, current, seen):
    return programs(live, current, seen)


def roll(live, chooser, *, seen=frozenset(), before_action=None, after_action=None, stop=None, first_read=None):
    # The read immediately following a review is also the next decision's
    # read. Caching it preserves the exact v12 input and saves a graph rebuild.
    done, current, actions = set(seen), first_read, []
    while not live.done:
        if live.frontier < live.ends[3]:
            ENGINE.step(live)
            current = None
            continue
        current = current if current is not None else ENGINE.read(live)
        choices = menu(live, current, done)
        if stop is not None and stop(live, current, choices, done):
            return Result(live, current, actions, frozenset(done), prefix=True)
        action = chooser(live, current, choices)
        if action is None:
            ENGINE.step(live)
            if not live.done:
                current = None
            continue
        previous = current
        captured = None if before_action is None else before_action(live, previous, action)
        done.add(action.key)
        live, detail = execute_program(live, current, action)
        current = ENGINE.read(live)
        detail["before_last"] = bool(live.frontier < live.scan.spec[2])
        if after_action is not None:
            after_action(live, previous, current, action, captured, detail)
        actions.append(detail)
    if current is None:
        current = ENGINE.read(live)
    return Result(live, current, actions, frozenset(done))


def v12_choice(live, read, choices):
    return select(live, read, choices, FROZEN, "joint", CONFIG)


def score(result, prediction=None):
    read = result.read
    grade = result.live.scan.grade(read.public, read.decision if prediction is None else prediction)[0]
    if not grade["completed_legal"]:
        raise RuntimeError("A teacher or student trajectory did not legally finish")
    return grade


def _indexes(public):
    return {k: j for j,k in enumerate(zip(public["frames"],public["slots"],strict=True))}


def old_logits(read):
    score = np.clip(1.-read.old["null"], EPS, 1.-EPS)
    return np.log(score / (1.-score)).astype(np.float32)


def observation(read):
    p = read.public
    return np.column_stack((read.x, read.posterior, read.reliability, p["quality"])).astype(np.float16)


def save_terminal(result, scope):
    read = result.read
    present = read.public["regions"] >= 3
    truth = truth_at(result.live.scan.node, read.public, result.live.scan.tape)
    return {"scope": scope, "x": observation(read)[present].astype(np.float16), "logit": old_logits(read)[present],
            "region": read.public["regions"][present].astype(np.int8), "label": truth["labels"][present],
            "old_winner": read.old["score"][present].astype(np.float32),
            "v12_winner": read.decision["score"][present].astype(np.float32),
            "track": read.decision["track_id"][present].astype(np.int32),
            "frame": read.public["frames"][present], "slot": read.public["slots"][present],
            "early_ok": all(score(result, read.old)["selected_ids"][i] == 0 for i in range(3)),
            "clock": result.live.scan.node.tick}


def _views(public, membership):
    return frozenset(zip(np.rint(public["stations"][membership]).astype(int),
                         np.rint(public["sectors"][membership]).astype(int)%24))


def _pair(live, before, after, action, detail):
    first = before.hypotheses[action.region]
    pre = _indexes(before.public)
    post = _indexes(after.public)
    truth = truth_at(live.scan.node, before.public, live.scan.tape)
    xb, xa = observation(before), observation(after)
    lb, la = old_logits(before), old_logits(after)
    matched = []
    for key in first.keys:
        if key is None:
            continue
        b,a = pre[key],post[key]
        old, new = before.decision["track_id"], after.decision["track_id"]
        before_views = _views(before.public, old == old[b])
        new_views = _views(after.public, (new == new[a]) & (after.public["frames"] >= detail["start_tick"]))
        matched.append((xb[b], xa[a], lb[b], la[a], int(truth["labels"][b]),
                        bool(new_views-before_views), action.region, key))
    return matched


def edit_suffix(result, failed_region, offset):
    """Mine a real legal correction and a retaped confirmation at its origin."""
    base, read = result.live, result.read
    available = [a for a in menu(base, read, frozenset()) if a.region == failed_region]
    if not available:
        return None
    candidate = max(available, key=lambda a:(a.name == "trace_b", a.name == "window", a.visibility))
    alt = fork_on_tape(base, offset)
    if alt.scan.tape.window is None:
        # The editing distribution varies the observed acoustic window. One
        # perturbation stays fixed through the complete branch.
        alt.scan.tape.window = SimpleNamespace(station=candidate.station, sector=(candidate.sector+3)%24)
    branch, detail = execute_program(alt, read, candidate)
    renewed = ENGINE.read(branch)
    branch.done = True
    scene = Result(branch, renewed, [detail], result.seen)
    return scene, score(scene), _pair(branch, read, renewed, candidate, detail)


def collect_case(case):
    fit = case.split == "v13_fit"
    live = live_case(case, augment=fit)
    paired = []

    def observed(active, before, after, action, _, log):
        if fit:
            paired.extend(_pair(active, before, after, action, log))

    trace = roll(live, v12_choice, after_action=observed if fit else None)
    base, current = score(trace, trace.read.old), score(trace)
    snapshots = [save_terminal(trace, "student")]
    raw = trace.live.scan.tape
    wrong = [r for r in (5, 4, 3) if current["selected_ids"][r] != 0]
    editing = {}
    if fit and wrong and _ordinal(case)%3 != 1:
        all_edits = [edit_suffix(trace, wrong[0], t) for t in EDIT_TAPES]
        if all(q is not None for q in all_edits):
            first, proof = all_edits
            snapshots.append(save_terminal(first[0], "failure_edit"))
            paired.extend(first[2])
            editing = {"region": wrong[0]+1, "proof_strict": proof[1]["strict"],
                       "train_strict": first[1]["strict"],
                       "proof_region_fixed": proof[1]["selected_ids"][wrong[0]] == 0,
                       "train_region_fixed": first[1]["selected_ids"][wrong[0]] == 0,
                       "proof_wrong": proof[1]["wrong_branch"]}
    q = trace.read.hypotheses.get(5)
    truth = truth_at(trace.live.scan.node, trace.read.public, raw)
    labels = [int(truth["labels"][j]) if j >= 0 else -99 for j in q.indices] if q else []
    events = [(x["region"], x["program"], x["start_tick"], x["end_tick"]) for x in trace.actions]
    meta = {"case": asdict(case), "layout_id": raw.layout.identifier,
            "relative": bool(raw.layout.metadata.get("v13_relative",False)),
            "acoustic": raw.window is not None, "v10": outcome(base), "v12": outcome(current),
            "actions": events, "r6_candidate_labels": labels, "edit": editing}
    return meta, snapshots, paired
