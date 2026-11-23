"""Persistent public track state and an information-sensitive window ranker."""

from dataclasses import dataclass

from study_data import motion, scene_links
from study_worlds import read_prefix
import numpy as np
from scipy.special import expit
from material_motion import MaterialMotion
from object_tracks import ObjectState
from readout_state import heads


@dataclass
class Beliefs:
    world_points: np.ndarray
    probability: np.ndarray
    evidence_ids: tuple
    track_ids: tuple
    unary_probability: np.ndarray


class Session:
    """Submit each acquired frame and slot once to the same persistent reader."""

    def __init__(self, env, spec, reader):
        self.env, self.spec, self.reader = env, spec, reader
        self.graph = ObjectState(scene_links(env), reader["edge_model"])
        self.frames = -1
        self.public = self.state = None
        self.previous = None

    def update(self, node):
        if len(node.reader.frames) != self.frames:
            if self.state is not None:
                self.previous = (self.public, self.state)
            public = read_prefix(self.env, self.spec, node)
            unary, embedding = heads(public["features"], self.reader["model"])
            self.state = self.graph.update(public, unary, embedding)
            self.public = public
            self.frames = len(node.reader.frames)
        return self.public, self.state

    def item(self, witness):
        if witness is None:
            return None
        m = (self.public["frames"] == witness[0]) & (self.public["slots"] == witness[1])
        ids = np.flatnonzero(m)
        return None if not len(ids) else int(ids[0])

    def feedback(self, node, frame):
        public, state = self.update(node)
        index = public["frames"] == len(node.reader.frames) - 1
        p = np.zeros(len(frame[2]), dtype=np.float32)
        p[public["slots"][index]] = state["probabilities"][index, 0]
        return MaterialMotion._feedback_shape((*frame[:6], p, *frame[7:]))

    def previous_mass(self, belief):
        if self.previous is None:
            return None
        previous_public, previous_state = self.previous
        keys = {(int(f), int(s)): i for i, (f, s) in enumerate(zip(previous_public["frames"], previous_public["slots"], strict=True))}
        masses = []
        for branch in belief.track_ids:
            shared = [keys[(int(self.public["frames"][i]), int(self.public["slots"][i]))]
                      for i in np.flatnonzero(self.state["track_ids"] == branch)
                      if (int(self.public["frames"][i]), int(self.public["slots"][i])) in keys]
            if not shared:
                return None
            last = max(shared, key=lambda k: previous_public["frames"][k])
            row = previous_state["probabilities"][last]
            masses.append(.025 + float(row[0] * (1. - row[6:].sum())))
        p = np.asarray(masses)
        return p / p.sum()


def hypotheses(public, state, station):
    """Two spatially distinct competing material trajectories near the frontier."""
    pos = np.asarray(public["stations"])
    age = abs(pos - station)
    near = np.flatnonzero(age <= 3. + 1.e-6)
    probs = state["probabilities"]
    p = probs[:, 0] * (1. - probs[:, 6:].sum(axis=1))
    score = (p + .016 * np.minimum(state["support"], 3.)) * np.exp(-age / 2.)
    ranked = near[np.argsort(-score[near], kind="stable")]
    ids = []
    for i in ranked:
        if not ids:
            ids.append(int(i))
        elif state["track_ids"][i] != state["track_ids"][ids[0]] and np.linalg.norm(public["points"][i] - public["points"][ids[0]]) >= 4.5:
            ids.append(int(i))
            break
    if len(ids) < 2:
        return None
    points = []
    for i in ids:
        xy = np.asarray(public["points"][i], dtype=float)
        support = near[(state["track_ids"][near] == state["track_ids"][i]) & (abs(pos[near] - pos[i]) >= .75)]
        if len(support):
            past = support[np.argmax(score[support])]
            slope = (xy - public["points"][past]) / (pos[i] - pos[past])
            xy = xy + np.clip((station + 1. - pos[i]) * slope, -5., 5.)
        points.append(xy)
    weight = p[ids] + .025
    u = state["unary"][ids]
    local = u[:, 0] * (1. - u[:, 6:].sum(axis=1)) + .025
    return Beliefs(np.stack(points), weight / weight.sum(),
                   tuple((int(public["frames"][i]), int(public["slots"][i])) for i in ids),
                   tuple(int(state["track_ids"][i]) for i in ids), local / local.sum())


def view_predictions(surface, station, sectors, world_points):
    """Predict what the competing public trajectories would look like from each view."""
    visible, separated = [], []
    for sector in sectors:
        echo = surface.project(float(station), int(sector), world_points)
        shadow = surface.shadow(float(station), int(sector), echo)
        visibility = (expit((25. - abs(echo[:, 0])) / 3.) * expit((echo[:, 1] - 3.) / 4.)
                      * expit((80. - echo[:, 1]) / 5.) * (1. - .92 * shadow))
        separation = np.linalg.norm((echo[0] - echo[1]) / [4.4, 7.2])
        visible.append(visibility)
        separated.append((1. - np.exp(-.5 * separation**2)) * np.min(visibility))
    return np.asarray(visible, dtype=float).T, np.asarray(separated, dtype=float)


def information_rank(base, legal, detection, separated, probability):
    """Keep visibility and discrimination as explicit terms rather than reweighting templates."""
    expected = np.asarray(probability) @ detection
    ambiguity = 1. - abs(float(probability[0]) - float(probability[1]))
    discriminability = .75 * abs(detection[0] - detection[1]) + .25 * separated
    score = .32 * base + 1.35 * expected + .65 * ambiguity * discriminability
    return np.where(legal, score, -np.inf), discriminability


def score_choices(session, node, sectors, legal, base, mode, step):
    public, state = session.update(node)
    belief = hypotheses(public, state, node.station)
    if belief is None:
        return base, {"hypotheses": 0, "mpc_slot": int(np.argmax(base)), "chosen": int(np.argmax(base))}
    current = belief.probability.copy()
    if mode in ("anchor_first", "anchor_second") and (step == 0 or abs(current[0] - current[1]) < .27):
        current = np.asarray([.82, .18] if mode == "anchor_first" else [.18, .82])
    d, s = view_predictions(session.env.scanner.surface, int(node.station) + 1, sectors, belief.world_points)
    scores, info = information_rank(base, legal, d, s, current)
    unary = information_rank(base, legal, d, s, belief.unary_probability)[0]
    previous_p = session.previous_mass(belief)
    previous_scores = None if previous_p is None else information_rank(base, legal, d, s, previous_p)[0]
    slot1 = int(np.argmax(np.where(legal, d[0], -np.inf)))
    slot2 = int(np.argmax(np.where(legal, d[1], -np.inf)))
    swap = int(np.argmax(information_rank(base, legal, d, s, current[::-1])[0]))
    def json_scores(v):
        return [float(x) if np.isfinite(x) else None for x in v]
    def margin(v):
        a = np.sort(v[np.isfinite(v)])
        return float(a[-1] - a[-2]) if len(a) > 1 else None
    record = {"hypotheses": 2, "fixed_public_ids": belief.evidence_ids, "probability": belief.probability.tolist(),
              "used_probability": current.tolist(), "world_points_mm": belief.world_points.tolist(),
              "sectors": sectors.tolist(), "legal": legal.tolist(), "ordinary_scores": json_scores(base),
              "identity_scores": json_scores(scores), "expected_detection": d.tolist(), "discrimination": info.tolist(),
              "unary_scores_same_geometry": json_scores(unary), "unary_same_geometry_slot": int(np.argmax(unary)),
              "previous_evidence_scores_same_geometry": None if previous_scores is None else json_scores(previous_scores),
              "previous_evidence_same_geometry_slot": None if previous_scores is None else int(np.argmax(previous_scores)),
              "ordinary_margin": margin(base), "identity_margin": margin(scores), "hypothesis_preference_slots": [slot1, slot2],
              "mpc_slot": int(np.argmax(base)), "chosen": int(np.argmax(scores)), "posterior_swap_slot": swap}
    return scores, record


def sensitivity_check():
    legal = np.ones(3, dtype=bool)
    base = np.array([.18, .24, .19])
    detection = np.array([[.94, .65, .12], [.10, .67, .95]])
    separated = np.asarray([.06, .68, .06])
    p = np.array([.92, .08])
    a, _ = information_rank(base, legal, detection, separated, p)
    b, _ = information_rank(base, legal, detection, separated, p[::-1])
    assert int(a.argmax()) == 0 and int(b.argmax()) == 2
    return {"only_identity_posterior_swapped": True, "initial_rank": a.tolist(), "swapped_rank": b.tolist(),
            "initial_slot": int(a.argmax()), "swapped_slot": int(b.argmax())}
