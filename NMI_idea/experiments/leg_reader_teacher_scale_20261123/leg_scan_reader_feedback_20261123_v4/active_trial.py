"""Run equal-time evidence, search, and graph-driven control on real simulator paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from study_worlds import (ARM, complete, held_frames, init_worlds, material_widths, motion, read_prefix, scene_links,
                          start, truth_at)
import numpy as np
from material_motion import MaterialMotion
from scale_teacher import ACTIONS as REVIEW_ACTIONS, probe_menu, visible_prefix
from scale_worlds import FUTURE_SEEDS
from scan_kernel import Trial
from windows import AcousticWindow
from public_features import public_arrays, replay_reader
from readout_state import confidence, earlier_image, evaluate, make_state, measured_length


ACTIONS = ("continue", "equal_time_hold", *REVIEW_ACTIONS[2:])
TARGETS = ("utility", "strict", "wrong_branch", "wrong_delivery", "correct_regions", "correct_mm",
           "old_correct", "old_wrong", "longest_contiguous_correct_mm")
TAPES = ("actual", "search_0", "search_1", "proof_0", "proof_1")
DEV_TAPES = ("search_0", "search_1", "proof_0", "proof_1")
_READERS = None


def init_action_study(pack, readers):
    global _READERS
    init_worlds(pack)
    _READERS = readers


def live_state(env, spec, node, reader):
    public = read_prefix(env, spec, node, legacy=reader["method"] == "v3_scaled")
    state = make_state(public, reader["model"], scene_links(env), reader["method"])
    return public, state


class LiveControl:
    """Use a short moving public history for frame-rate control inference."""

    horizon = 14

    def __init__(self, env, spec, reader):
        self.env, self.spec, self.reader = env, spec, reader
        self.weights_changes = 0
        self.feedback_changes = 0

    def target_posterior(self, node):
        original = node.reader
        legacy = self.reader["method"] == "v3_scaled"
        view = replay_reader(original, original.frames[-self.horizon:], original.mixed_posterior[-self.horizon:], legacy=legacy)
        pub = public_arrays(view, self.spec, self.env.material, legacy=legacy)
        state = make_state(pub, self.reader["model"], scene_links(self.env), self.reader["method"])
        newest = pub["frames"] == len(view.frames) - 1
        p = np.zeros(len(view.frames[-1][2]), dtype=np.float32)
        p[pub["slots"][newest]] = state["probabilities"][newest, 0]
        return p

    def weight(self, node):
        reader = node.reader
        frame = reader.frames[-1]
        p = self.target_posterior(node)
        if not len(p) or p.sum() <= 1.e-7:
            return reader.w
        means, _, _, visible, _, _ = reader.model.view(frame[0], frame[1])
        d = np.linalg.norm(means[:, 0, None] - frame[2][None], axis=-1)
        plausible = np.exp(-.5 * (d / 12.) ** 2) @ (p / p.sum())
        posterior = reader.w * (.07 + plausible * visible[:, 0])
        alpha = .50 * frame[4] * float(np.max(p))
        revised = (1. - alpha) * reader.w + alpha * posterior / posterior.sum()
        self.weights_changes += int(np.max(abs(revised - reader.w)) > 1.e-5)
        return revised

    def feedback(self, node, frame):
        p = self.target_posterior(node)
        self.feedback_changes += int(np.max(abs(p - frame[6]), initial=0.) > 1.e-5)
        return MaterialMotion._feedback_shape((*frame[:6], p, *frame[7:]))


class ReaderContactMotion(MaterialMotion):
    """Preserve the checked body-chart executor with an optional reader hook."""

    _feedback_hook = None

    def _feedback_shape(self, frame):
        if self._feedback_hook is None:
            return super()._feedback_shape(frame)
        return self._feedback_hook(frame)

    def run_with_reader(self, node, spec, trial, plan, caps, control):
        self._feedback_hook = lambda frame: control.feedback(node, frame)
        try:
            return self.run_probe(node, spec, trial, plan, *caps, use_dird=False, copy_node=False)
        finally:
            self._feedback_hook = None


def weighted_step(env, spec, node, trial, limits, weights):
    """Use the common MPC objective and path executor with public reader weights."""
    route = env.route
    s, a = int(node.station), int(node.sector)
    legal = route.bounded_candidates(node, spec, limits)
    if not legal.any():
        raise RuntimeError("No feasible next pose with the shared reserve")
    sectors = route.sectors[a]
    forecast = route.predictive(node.reader.model, s + 1)[sectors] @ weights
    score = forecast[:, 0] + .19 * forecast[:, 1] - .013 * route.distance[s, a]
    if s + 1 < spec[2]:
        next_detection = route.predictive(node.reader.model, s + 2)[:, 0] @ weights
        future = next_detection[route.sectors[sectors]] - .013 * route.distance[s + 1, sectors]
        score += .43 * np.max(np.where(route.legal[s + 1, sectors], future, -1.), axis=1)
    slot = int(np.argmax(np.where(legal, score, -np.inf)))
    key = s, a, slot
    path = route.cache.get(key)
    if path is None:
        ticks = route.clock[s, a, slot]
        fraction = np.arange(ticks + 1) / ticks
        positions = np.column_stack((s + fraction, a + route.deltas[slot] * fraction))
        lengths = env.scanner._path_metrics(positions, spec)
        if lengths is None:
            raise RuntimeError("Graph-guided candidate violated the shared chart")
        path = route.cache[key] = (positions[1:], lengths)
    for (station, sector), length in zip(*path, strict=True):
        node.tick += 1
        node.distance += float(length)
        env.scanner._observe(node, spec, trial, float(station), float(sector), node.tick)
    node.station, node.sector = s + 1, int(sectors[slot])


def advance_to_end(env, spec, node, trial, limits, control=None):
    if control is None:
        return complete(env, spec, node, trial, limits)
    while node.station < spec[2]:
        weighted_step(env, spec, node, trial, limits, control.weight(node))
    if node.station != spec[2] or node.tick > limits.ticks or node.distance > limits.mm + 1.e-6:
        raise RuntimeError("Reader-guided continuation exceeded its shared envelope")
    return node


def review(env, spec, prefix, tape, action, plans, slot, caps, control=None):
    cur = prefix.fork()
    trajectory = np.empty((0, 5), dtype=np.float64)
    if action in plans:
        plan = plans[action]
        if control is None:
            result = env.material.run_probe(cur, spec, tape, plan, *caps, use_dird=False, copy_node=False)
        else:
            if not isinstance(env.material, ReaderContactMotion):
                env.material = ReaderContactMotion(env.scanner)
            result = env.material.run_with_reader(cur, spec, tape, plan, caps, control)
        if not result.legal:
            raise RuntimeError(f"A verified ProDMP program failed to rejoin: {result.reason}")
        cur, trajectory = result.node, result.trajectory
    if action:
        held_frames(env, spec, cur, tape, prefix.tick + slot - cur.tick)
        if (cur.station, cur.sector, cur.tick) != (prefix.station, prefix.sector, prefix.tick + slot):
            raise RuntimeError("Matched reviews did not rejoin at the common boundary")
    return cur, trajectory


@dataclass
class Prefix:
    env: object
    spec: tuple
    node: object
    trial: Trial
    limits: object
    plans: dict
    slot: int
    caps: tuple
    seam: int
    reader: dict
    scene_band: str


def reached_prefix(case, band="natural"):
    env, spec, node, trial, caps = start(case)
    junction = int(np.argmin(abs(env.scanner.stations - (-750. if case.task == "femoral" else -910.))))
    region3 = np.flatnonzero(spec[4] == 2) + spec[1]
    other = spec[1] + (spec[2] - spec[1]) * (1 if case.band % 2 else 4) // 5
    at = (junction + 1, int(region3[len(region3) // 2]), other)[case.band % 3]
    at = int(np.clip(at, spec[1] + 2, spec[2] - 1))
    arm_name = "graph" if case.split == "feedback_test" else "selected"
    reader = _READERS[case.task][arm_name]
    # Start from a real visited node around the selected event. When the
    # public prefix looks decisive, also inspect the next valid anchor.
    complete(env, spec, node, trial, caps, max(spec[1] + 1, at - 1))
    prior = node
    public, state = live_state(env, spec, prior, reader)
    region = motion.region_of(spec, prior.station)
    idx = np.flatnonzero(public["regions"] == region)
    scores = state["probabilities"][:, 0]
    best = idx[np.argmax(scores[idx])] if len(idx) else -1
    others = idx[state["track_ids"][idx] != state["track_ids"][best]] if len(idx) else idx
    comparison = np.max(scores[others], initial=0.)
    ambiguous = best < 0 or scores[best] < .77 or scores[best] - comparison < .13
    if band == "stress_window" and ambiguous:
        # An acoustic blur condition begins here at the actual reached pose,
        # before any new view is acquired. This designed arm is development
        # only and holds identical scene and time across all action forks.
        trial.window = AcousticWindow(station=int(prior.station), sector=int(prior.sector),
                                     alternate_sector=int(prior.sector), competing_role=-1, validation_ticks=0,
                                     reference_resolved=0, alternate_resolved=0, reference_mixed=0, alternate_mixed=0)
        held_frames(env, spec, prior, trial, 3)
    elif not ambiguous and at < spec[2]:
        complete(env, spec, prior, trial, caps, at)
    ref = junction if prior.station == junction + 1 else int(prior.station - 1)
    plans, slot, probe_caps = probe_menu(env, spec, prior, caps, ref)
    tag = "stress_window" if trial.window is not None else "natural"
    return Prefix(env, spec, prior, trial, caps, plans, slot, probe_caps, junction, reader, tag)


def score_branch(info, node, cut, region, *, complete_scan):
    env, spec = info.env, info.spec
    public, state = live_state(env, spec, node, info.reader)
    private = truth_at(node, public, info.trial)
    labels = private["labels"]
    reader = info.reader
    score = confidence(public, state, reader["risk"])
    old = earlier_image(public, state, labels, reader["risk"], reader["threshold"], cut, region)
    if not complete_scan:
        return old
    final = evaluate(public, labels, score, reader["threshold"], complete=True)
    cover, longest = measured_length(public, state, labels, reader["position_threshold"], material_widths(env, spec))
    old_wrong = old not in (-99, 0)
    utility = (3. * final["strict"] + .38 * final["correct_regions"] + .002 * cover
               - .66 * final["wrong_region_count"] - .22 * final["wrong_branch_regions"]
               + .26 * (old == 0) - .32 * old_wrong)
    targets = (utility, final["strict"], final["wrong_branch"], final["wrong_delivery"], final["correct_regions"],
               cover, old == 0, old_wrong, longest)
    detail = np.stack((final["selected_ids"], final["capture"]), axis=-1).astype(np.int8)
    return targets, detail, old


def run_search(case):
    depth = "stress_window" if case.split == "search_dev" and case.band % 2 else "natural"
    q = reached_prefix(case, depth)
    env, spec, node, trial, limits = q.env, q.spec, q.node, q.trial, q.limits
    public, state = live_state(env, spec, node, q.reader)
    features, valid = visible_prefix(env, spec, node, limits, q.plans, q.slot, q.caps, q.seam,
                                     case.task != "femoral", public, state["probabilities"][:, 0], q.reader["position_threshold"])
    streams = TAPES if case.split == "search_test" else DEV_TAPES
    offsets = {f"{name}_{i}": offset for name, seq in FUTURE_SEEDS.items() for i, offset in enumerate(seq)}
    tapes = [trial if name == "actual" else trial.branch(node.tick, case.tape_seed + offsets[name]) for name in streams]
    earliest = len(node.reader.frames)
    region = motion.region_of(spec, node.station)
    base_old = score_branch(q, node, earliest, region, complete_scan=False)
    y = np.zeros((len(ACTIONS), len(tapes), len(TARGETS)), dtype=np.float32)
    detail = np.zeros((len(ACTIONS), len(tapes), 6, 2), dtype=np.int8)
    extra = np.zeros((len(ACTIONS), len(tapes), 8), dtype=np.float32)
    for draw, (stream, tape) in enumerate(zip(streams, tapes, strict=True)):
        if tape.layout is not trial.layout or tape.window is not trial.window:
            raise RuntimeError("Counterfactual replay altered the anatomy or acoustic condition")
        for action in np.flatnonzero(valid):
            cur, traj = review(env, spec, node, tape, action, q.plans, q.slot, q.caps)
            after = score_branch(q, cur, earliest, region, complete_scan=False) if action and not stream.startswith("search") else base_old
            complete(env, spec, cur, tape, limits)
            y[action, draw], detail[action, draw], old = score_branch(q, cur, earliest, region, complete_scan=True)
            peak = max((abs(motion._sector_delta(row[2], node.sector)) for row in traj), default=0.)
            extent = max((abs(row[1] - node.station) for row in traj), default=0.)
            extra[action, draw] = (cur.tick, cur.distance, after, old, peak, extent, len(traj), region)
    meta = {"case": asdict(case), "layout_id": trial.layout.identifier, "scope": q.scene_band,
            "complete": True, "old": base_old, "prefix_station": int(node.station), "prefix_sector": int(node.sector),
            "prefix_region": region, "review_slot_ticks": q.slot, "prefix_frame_count": earliest,
            "continuations": int(valid.sum()) * len(streams), "reader": q.reader["method"]}
    return {"features": features, "valid": valid, "outcomes": y, "detail": detail, "extra": extra}, meta


def pack_search(items):
    return {name: np.stack([row[name] for row, _ in items]) for name in items[0][0]}


def run_feedback(case):
    q = reached_prefix(case)
    env, spec, node, trial, limits = q.env, q.spec, q.node, q.trial, q.limits
    live = LiveControl(env, spec, q.reader)
    # The legal probe program is selected before either feedback arm runs.
    action = next((i for i in (2, 3, 4, 5, 6) if i in q.plans), 0)
    cut = len(node.reader.frames)
    region = motion.region_of(spec, node.station)
    before = score_branch(q, node, cut, region, complete_scan=False)
    records, tracks, terminal = {}, {}, {}
    for name, ctrl in (("mpc", None), ("probe_scores_only", None), ("probe_and_reader_feedback", live)):
        active = 0 if name == "mpc" else action
        cur, traj = review(env, spec, node, trial, active, q.plans, q.slot, q.caps, ctrl)
        mid = score_branch(q, cur, cut, region, complete_scan=False) if active else before
        advance_to_end(env, spec, cur, trial, limits, ctrl)
        values, detail, old = score_branch(q, cur, cut, region, complete_scan=True)
        records[name] = {"values": values, "detail": detail.tolist(), "old_after_probe": mid,
                         "old_terminal": old, "time_s": cur.tick * motion.CLOCK_S, "distance_mm": cur.distance}
        tracks[name] = traj
        terminal[name] = np.asarray([f[:2] for f in cur.reader.frames[cut + q.slot:]])
    left, right = (tracks[name] for name in ("probe_scores_only", "probe_and_reader_feedback"))
    a, b = (terminal[name] for name in ("probe_scores_only", "probe_and_reader_feedback"))
    delta = None if left.shape != right.shape else float(np.max(abs(left[:, 1:3] - right[:, 1:3]), initial=0.))
    changed = a.shape != b.shape or not np.array_equal(a, b)
    meta = {"case": asdict(case), "layout_id": trial.layout.identifier, "complete": True,
            "action": ACTIONS[action], "probe_pose_max_change_chart": delta, "continuation_changed": changed,
            "weights_updated_frames": live.weights_changes, "feedback_updated_frames": live.feedback_changes,
            "before": before, "arms": records}
    return {"outcomes": np.stack([r["values"] for r in records.values()])}, meta
