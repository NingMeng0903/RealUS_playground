"""Run the four paired v16 segment choices from the same actual leg prefix."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np

import physics_v16 as physics
import segment_v16 as segment

ROOT = Path(__file__).resolve().parent
V15 = ROOT.parent / "leg_scan_shared_belief_20261126_v15"
sys.path.insert(1, str(V15))
import data_v15 as data
import identity_v15
from control_v15 import Controller, _Windows, _merit
from environment import Case, spec_for
from fit_v12 import hydrate
from recovery_motion import execute as move, plan
from recovery_task import motion
from variants_v16 import LocalShear

PARENT = "e8837c832c2d6abda2151092c787377b81aed770"
EXPERIMENT = ROOT.name
ARMS = ("masked_mean", "masked_full", "read_mean", "read_full")
SEEDS = {"smoke": 8_616_260_000_000_000, "development": 8_616_260_100_000_000,
         "blind": 8_616_260_200_000_000, "shear": 8_616_260_300_000_000}
COUNTS = {"smoke": 6, "development": 18, "blind": 108, "shear": 54}
BANDS = (None, 338_160_257)
_CONTROLLER = _FOLLOW = None


def roster(stage):
    # One case assigned to each region/profile/shape cell per group of 18.
    for i in range(COUNTS[stage]):
        cell, region = divmod(i, 3)
        group, profile = divmod(cell, 2)
        shape = (3, 4, 6)[group % 3]
        seed = SEEDS[stage] + 53 * i
        yield Case(f"v16_{stage}_{i:05d}", f"v16_{stage}", 2, shape,
                   "groin_to_distal_calf", ("ordinary", "stress")[profile],
                   seed, seed + 1, seed + 2, group // 3, False), region + 3


def initialize(atlas):
    global _CONTROLLER, _FOLLOW
    data.init_base(atlas)
    frozen = data.read_json(V15 / "run/test_freeze.json")
    def load(name):
        path = V15 / "run/models" / name
        if data.digest(path) != frozen["models"][name]:
            raise RuntimeError(f"Frozen v15 asset differs: {name}")
        return path
    reader = identity_v15.load_reader(load("reader.json"))
    _CONTROLLER = Controller(reader, data.policy_v13.READER, data.policy_v13.MODE)
    _FOLLOW = _CONTROLLER.chooser(hydrate(data.read_json(load("student.json"))),
                                  frozen["development"]["rule"])


def _key(public, i):
    return int(public["frames"][i]), int(public["slots"][i])


def anchor_keys(read, region, station):
    public, decision = read.public, read.decision
    s, xy = public["stations"], public["points"]
    local = np.flatnonzero((public["regions"] == region) & (np.abs(s - station) <= 2.5))
    if not len(local):
        return ()
    priority = decision["candidate_probability"][local] - .002 * abs(s[local] - station)
    priority += .001 * public["quality"][local]
    order = local[np.argsort(-priority, kind="stable")]
    a = order[0]
    same = decision["object_id"][order] == decision["object_id"][a]
    dis = np.linalg.norm(xy[order] - xy[a], axis=1)
    rival = order[(~same) & (dis >= 4.5) & (dis <= 30.)]
    return (_key(public, a), _key(public, rival[0])) if len(rival) else (_key(public, a),)


def _path_fit(surface, public, rows, station):
    """Branch geometry from anonymous local observations and public atlas derivatives."""
    center_y = surface.y[station]
    s = public["stations"]
    rows = rows[abs(s[rows] - station) <= 6.]
    # Redundant pulses at the same address carry no extra slope evidence.
    address = np.column_stack((np.rint(2. * s[rows]), np.rint(public["sectors"][rows])))
    order = np.argsort(-public["quality"][rows], kind="stable")
    _, selected = np.unique(address[order], axis=0, return_index=True)
    rows = rows[order[selected]]
    y = np.interp(s[rows], np.arange(len(surface.y)), surface.y) - center_y
    xz = public["points"][rows].astype(np.float64)
    w = (.2 + public["quality"][rows]) * np.exp(-.5 * (y / 18.) ** 2)
    w /= w.sum()
    y_bar = float(w @ y)
    x_bar = w @ xz
    spread = w @ (y - y_bar) ** 2
    observed = (w * (y - y_bar)) @ (xz - x_bar)
    i, j = max(0, station - 2), min(len(surface.y) - 1, station + 2)
    slopes = (surface.priors[:, j, 0] - surface.priors[:, i, 0]) / (surface.y[j] - surface.y[i])
    if surface.reference_links is not None:
        support = surface.reference_links[:, i:j, 0].all(axis=1)
        slopes = np.where(support[:, None], slopes, np.nan)
    valid = np.isfinite(slopes).all(axis=1)
    default = slopes[valid].mean(axis=0) if valid.any() else np.zeros(2)
    slopes = np.where(valid[:, None], slopes, default)
    m = (observed + 100. * slopes) / (spread + 100.)
    xz = x_bar - y_bar * m
    centers = np.column_stack((xz[:, 0], np.full(len(xz), center_y), xz[:, 1]))
    tangents = segment.unit(np.column_stack((m[:, 0], np.ones(len(m)), m[:, 1])))
    return centers, tangents


def _lookup_rows(public, keys):
    addresses = {_key(public, i): i for i in range(len(public["frames"]))}
    if any(key not in addresses for key in keys):
        raise RuntimeError("A previously captured object address disappeared")
    return [addresses[key] for key in keys]


def anchor_geometry(read, surface, station, keys):
    public = read.public
    ids = read.decision["object_id"]
    hypotheses = []
    for row in _lookup_rows(public, keys):
        c, t = _path_fit(surface, public, np.flatnonzero(ids == ids[row]), station)
        hypotheses.append((c.mean(axis=0), segment.unit(t.mean(axis=0))))
    return hypotheses


def _associate(read, surface, station, keys, reference):
    """Map object probability once per region into fixed, measured A/B tracks."""
    p, d = read.public, read.decision
    ids = d["object_id"]
    anchors = _lookup_rows(p, keys)
    region = p["regions"][anchors[0]]
    rows = np.flatnonzero((p["regions"] == region) & (abs(p["stations"] - station) <= 6.))
    c, t = map(np.stack, zip(*reference))
    y = np.interp(p["stations"][rows], np.arange(len(surface.y)), surface.y)
    pred = c[None, :, :] + ((y[:, None, None] - c[None, :, 1:2])
                           * (t / t[:, 1:2])[None, :, :])
    distances = np.linalg.norm(p["points"][rows, None] - pred[..., (0, 2)], axis=-1)
    mass = np.zeros(len(keys))
    fits = [[] for _ in keys]
    for group in np.unique(ids[rows]):
        keep = ids[rows] == group
        group_rows = rows[keep]
        # If the reader merged both old objects into one group, leave that
        # ambiguous mass unresolved. Unanchored groups must match one curve.
        pinned = [i for i, anchor in enumerate(anchors) if ids[anchor] == group]
        if len(pinned) > 1:
            continue
        if pinned:
            selected = pinned[0]
        else:
            local = distances[keep]
            w = p["quality"][group_rows] * np.exp(-.5 * ((p["stations"][group_rows] - station) / 2.5) ** 2)
            average = (w @ local) / max(1.e-8, w.sum())
            order = np.argsort(average, kind="stable")
            selected = order[0]
            if average[selected] > 5.5 or (len(keys) > 1 and average[order[1]] - average[selected] < 1.75):
                continue
        nearby = group_rows[distances[keep, selected] <= 8.]
        if not len(nearby):
            continue
        mass[selected] += float(d["candidate_probability"][group_rows[0]])
        fits[selected].extend(nearby)
    return mass, fits


def belief_at(read, live, stationary, keys, reference):
    public, decision = read.public, read.decision
    surface = live.scan.env.scanner.surface
    weights = live.scan.node.reader.w.reshape(len(surface.priors), -1).sum(axis=1)
    weights = weights / weights.sum()
    centers, tangents, mass, objects, probabilities = [], [], [], [], []
    grouped, fits = _associate(read, surface, stationary, keys, reference)
    if grouped.sum() > 1.000001:
        raise RuntimeError("A public object probability entered the local belief twice")
    for tag, p, rows in zip(("A", "B"), grouped, fits):
        probabilities.append(p)
        if p <= 1.e-9:
            continue
        c, t = _path_fit(surface, public, np.asarray(rows, dtype=int), stationary)
        centers.extend(c)
        tangents.extend(t)
        mass.extend(p * weights)
        objects.append((tag, len(weights)))
    if len(keys) == 1:
        probabilities.append(0.)
    remainder = float(np.clip(1. - sum(probabilities), 0., 1.))
    return segment.Belief(np.asarray(centers).reshape(-1, 3), np.asarray(tangents).reshape(-1, 3),
                          np.asarray(mass), tuple(objects), (*probabilities, remainder))


def pose(scan):
    return [int(scan.node.station), int(scan.node.sector), int(scan.node.tick), float(scan.node.distance)]


def sha(data_object):
    content = json.dumps(data_object, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def record_read(read):
    p = read.public
    h = hashlib.sha256()
    for name in ("frames", "slots", "stations", "sectors", "points", "quality"):
        h.update(np.ascontiguousarray(p[name]).tobytes())
    return h.hexdigest()


def _approaches(scan, s):
    available = np.flatnonzero(scan.env.scanner.valid[s])
    return {int(a): path for a in available if (path := plan(scan, s, int(a), dwell=1)) is not None}


def _prepare(case, region):
    station = None
    env = data.tr.study_worlds._WORLDS.environment(case.shape)
    factory = env.layout_factory
    if case.split == "v16_shear":
        spec = spec_for(env.scanner, case)
        sites = np.flatnonzero(spec[4] == region) + spec[1]
        station = int(sites[len(sites) // 2])
        env.layout_factory = LocalShear(factory, station)
    try:
        live = data.tr.live_case(case)
    finally:
        env.layout_factory = factory
    result = _CONTROLLER.roll(live, _FOLLOW, stop=lambda x, r, m, seen: x.frontier >= x.ends[region])
    live, read = result.live, result.read
    q = read.hypotheses.get(region)
    if q is None or q.indices[0] < 0:
        return live, None, "no_observed_region_target"
    first, last = int(live.ends[region - 1] + 1), int(live.ends[region])
    s = min(last - 1, max(first + 1, q.station if station is None else station))
    surface = live.scan.env.scanner.surface
    valid = np.flatnonzero(surface.valid[s])
    costs = abs((valid - q.sector + 12) % 24 - 12)
    for a in valid[np.argsort(costs, kind="stable")]:
        path = plan(live.scan, s, int(a), dwell=1)
        if path is not None:
            move(live.scan, path)
            live.cache = None
            return live, _CONTROLLER.read(live), None
    return live, None, "no_contact_approach"


def _probe_choices(live, before, station, approaches):
    sector = int(live.scan.node.sector)
    q = before.hypotheses.get(motion.region_of(live.scan.spec, station))
    f = _merit(_Windows(live, before).features(station, q))
    directions = [a for a in approaches if abs((a - sector + 12) % 24 - 12) >= 2]
    directions.sort(key=lambda a: (-f[a], a))
    for a in directions:
        path = plan(live.scan, station, a, dwell=5)
        if path is not None:
            return path
    return None


def _probe(live, offset, station, probe):
    old = pose(live.scan)
    branch = live.fork() if offset is None else data.tr.fork_on_tape(live, offset)
    first = move(branch.scan, probe)
    rejoin = plan(branch.scan, station, old[1], dwell=1)
    if rejoin is None:
        raise RuntimeError("The common contact probe could not rejoin")
    last = move(branch.scan, rejoin)
    branch.cache = None
    end = pose(branch.scan)
    if old[:2] != end[:2] or end[2] <= old[2]:
        raise RuntimeError("The common probe must return with new echoes")
    return branch, {"start": old, "exit": end, "target_sector": probe.sector,
                    "outbound": first["trajectory"], "return": last["trajectory"]}


def _oblique_parity(scene, surface, station, sector):
    # A neutral 3-D slice must have the same intersections as the v15 axial chart.
    axial = surface.project(station, sector, scene.layout.at(station))
    actual = scene.intersect(surface.world(station, sector), segment.base_frame(sector))
    if not np.array_equal(np.isfinite(axial), np.isfinite(actual)):
        raise RuntimeError("Neutral oblique renderer changed the supported axial roles")
    delta = float(np.nan_to_num(np.abs(axial - actual)).max())
    if delta > 1.e-6:
        raise RuntimeError(f"Neutral oblique renderer disagrees with v15 axial projection by {delta}")
    return delta


def _execute_arms(live, station, menu, approaches, before, after):
    scan = live.scan
    surface = scan.env.scanner.surface
    actual = {}
    for arm in ARMS:
        belief = after if arm.startswith("read_") else before
        chosen, scores = segment.choose(surface, station, menu, approaches, belief, arm.endswith("_full"))
        actual[arm] = {"index": chosen, "scores": scores}
    executed = {}
    for i in sorted({result["index"] for result in actual.values()}):
        a = menu[i]
        fork = scan.fork(scan.tape)
        path = plan(fork, station, a.sector, dwell=1)
        frozen = approaches[a.sector]
        if path is None or path.travel_ticks != frozen.travel_ticks or len(path._segments) != len(frozen._segments):
            raise RuntimeError("The frozen contact menu changed after the evidence treatment")
        for new, old in zip(path._segments, frozen._segments, strict=True):
            if not np.array_equal(new.positions, old.positions):
                raise RuntimeError("The selected approach departed from the frozen physical path")
        result = move(fork, path)
        settle = physics.settle_rotations(a, motion.CLOCK_S)
        executed[i] = (fork, a, path, result, len(settle))
    first_pulse = max(fork.node.tick + settle for fork, _, _, _, settle in executed.values()) + 1
    # Hidden anatomy is accessed after every arm's action has been selected.
    scene = physics.ObliqueScene(surface, scan.tape.layout, station)
    parity = _oblique_parity(scene, surface, station, int(scan.node.sector))
    graded = {i: physics.acquire(scene, f, a, first_pulse, p, t, motion.CLOCK_S)
              for i, (f, a, p, t, _) in executed.items()}
    for arm in actual:
        actual[arm]["result"] = graded[actual[arm]["index"]]
    # Geometry-only opportunity audit, after the decisions and chosen pulses.
    true_passes = [i for i, a in enumerate(menu)
                   if scene.true_rule(surface.world(station, a.sector), a)["pass"]]
    return {"arms": actual, "axial_renderer_max_error_mm": parity,
            "shared_pulse_start": first_pulse, "unique_executed": len(graded),
            "true_geometry_passing_actions": true_passes}


def record_belief(b):
    return {"probabilities": b.probabilities, "weights": b.weights.tolist(), "centers": b.centers.tolist(),
            "tangents": b.tangents.tolist(), "objects": b.objects}


def job(case, region, band):
    live, before, reason = _prepare(case, region)
    scan = live.scan
    out = {"case": asdict(case), "region": region + 1, "band": band,
           "layout_id": scan.tape.layout.identifier, "prefix_pose": pose(scan),
           "shear": scan.tape.layout.metadata.get("v16_shear")}
    if reason:
        return out | {"skip": reason}
    station = int(scan.node.station)
    keys = anchor_keys(before, region, station)
    if not keys:
        return out | {"skip": "no_local_observed_object"}
    reference = anchor_geometry(before, scan.env.scanner.surface, station, keys)
    masked = belief_at(before, live, station, keys, reference)
    if masked.known <= 1.e-8:
        return out | {"skip": "no_public_object_probability"}
    approaches = _approaches(scan, station)
    menu = segment.candidates(scan.env.scanner.surface, station, approaches, masked)
    probe = _probe_choices(live, before, station, approaches)
    if len(menu) < 2 or probe is None:
        return out | {"skip": "no_common_probe_or_candidate_menu"}
    fixed = [{"sector": a.sector, "tag": a.tag, "orientation": a.middle.tolist(),
              "travel_ticks": approaches[a.sector].travel_ticks} for a in menu]
    menu_id = sha(fixed)
    public_prefix = record_read(before)
    branch, probe_log = _probe(live, BANDS[band], station, probe)
    read = _CONTROLLER.read(branch)
    received = belief_at(read, branch, station, keys, reference)
    # A denied evidence block cannot update either the object belief or atlas weights.
    if record_read(before) != public_prefix or branch.scan.tape.layout is not scan.tape.layout:
        raise RuntimeError("The shared prefix was mutated during its probe")
    answer = _execute_arms(branch, station, menu, approaches, masked, received)
    if sha(fixed) != menu_id or public_prefix == record_read(read):
        raise RuntimeError("Menu mutated or the real probe did not acquire observations")
    truth = [int(scan.node.reader.frames[f][3][j]) for f, j in keys]
    return out | {"object_keys": keys, "old_object_truth_audit": truth, "candidate_menu": fixed,
                  "menu_sha256": menu_id, "prefix_sha256": public_prefix, "post_sha256": record_read(read),
                  "masked": record_belief(masked), "read": record_belief(received), "probe": probe_log,
                  "new_frames": int(branch.scan.node.tick - scan.node.tick),
                  "new_echoes": int(np.sum(read.public["frames"] >= scan.node.tick))} | answer


def write_gzip(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as file, gzip.GzipFile(fileobj=file, mode="wb", filename="", mtime=0) as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
    temp.replace(path)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def freeze(args):
    frozen = V15 / "run/test_freeze.json"
    protocol = {"experiment": EXPERIMENT, "parent_commit": PARENT,
                "source_sha256": {p.name: data.digest(p) for p in sorted(ROOT.glob("*.py"))},
                "v15_source_sha256": {p.name: data.digest(p) for p in sorted(V15.glob("*.py"))},
                "v15_freeze_sha256": data.digest(frozen), "v15_models": data.read_json(frozen)["models"],
                "atlas_sha256": data.digest(args.atlas), "test_seeds": SEEDS, "counts": COUNTS,
                "bands": BANDS, "arms": ARMS, "segment_residual_scales": segment.SCALE.tolist(),
                "motion": "Inherited contact chart and clock, local orientation envelope, no added time or distance budget",
                "alpha": segment.ALPHA, "sweep_deg": segment.SWEEP_DEG, "sweep_pulses": segment.PULSES,
                "probe_attitude_deg": segment.MAX_ATTITUDE_DEG,
                "angular_rate_deg_s": segment.ANGULAR_RATE_DEG_S,
                "gates": "Full-segment endpoint Taylor enclosure over finite hypotheses; unresolved mass stays unresolved",
                "endpoint": "Public fresh target echo AND continuously valid true short-axis condition at assigned material station",
                "source": "Open3DModel right synthetic named route, independent layout and noise seeds; no Blender"}
    path = args.out / "protocol.json"
    if path.exists():
        if sha(data.read_json(path)) != sha(protocol):
            raise RuntimeError("Frozen v16 experimental inputs changed")
    else:
        data.atomic_json(path, protocol)
    return protocol


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=tuple(COUNTS))
    p.add_argument("--atlas", type=Path, default=data.DEFAULT_PACK)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze(args)
    path = args.out / "shards" / args.stage
    path.mkdir(parents=True, exist_ok=True)
    bands = (0, 1) if args.stage in ("blind", "shear") else (0,)
    inputs = [(case, region, band, path / f"{case.identifier}_{band}.json.gz")
              for case, region in roster(args.stage) for band in bands]
    pending = [item for item in inputs if not item[-1].exists()]
    print(f"{args.stage}: {len(inputs) - len(pending)}/{len(inputs)} complete, {args.workers} CPU workers", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                 initializer=initialize, initargs=(args.atlas,)) as pool:
            futures = {pool.submit(job, c, r, b): f for c, r, b, f in pending}
            for n, future in enumerate(as_completed(futures), len(inputs) - len(pending) + 1):
                try:
                    row = future.result()
                except Exception as error:
                    raise RuntimeError(f"v16 simulation failed on {futures[future].name}") from error
                write_gzip(futures[future], row)
                if n == 1 or n % 6 == 0 or n == len(inputs):
                    print(f"{args.stage}: {n}/{len(inputs)}", flush=True)
    rows = [read_gzip(f) for _, _, _, f in inputs]
    write_gzip(args.out / f"{args.stage}_paired.json.gz", rows)


if __name__ == "__main__":
    main()
