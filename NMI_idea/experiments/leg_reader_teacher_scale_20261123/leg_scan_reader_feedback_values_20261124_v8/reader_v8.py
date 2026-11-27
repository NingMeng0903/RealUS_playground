"""Public trajectory inference and independently calibrated delivery decoding."""

import study_v8  # Load the pinned reader and simulator modules.
from causal_reader import heads, inputs, likelihood, observation_features, scores
from object_tracks import infer
from readout_state import fit_risk
from reader_v7 import fuse
from study_worlds import read_prefix, scene_links
from usability_fit import predict_usability
import numpy as np


MODES = ("unary", "old_graph_refit", "new_graph_refit", "new_graph_object")


def unary_state(p):
    return {"probabilities": p, "track_ids": np.arange(len(p)), "support": np.zeros(len(p))}


def public_states(public, base, links, edge_bundle, graphs=("old", "new")):
    p, e = heads(public["features"], base["model"])
    u = predict_usability(public, base["usability"])
    result = {"unary": unary_state(p)}
    for kind in graphs:
        if kind == "old":
            graph = infer(public, p, e, links, edge_model=base["edge_model"])
        elif kind == "new":
            from edge_v8 import graph as new_graph
            graph = new_graph(public, p, e, links, edge_bundle)
        else:
            raise ValueError(f"Unknown public graph {kind}")
        result[kind] = fuse(public, p, e, links, base["edge_model"], u, graph)
        result[kind]["scored_edges"] = graph["scored_edges"]
    return result


def object_table(public, state, labels=None):
    """Summarize independent views, preserving role, physical return and spread."""
    tracks, inverse = np.unique(state["track_ids"], return_inverse=True)
    view = np.column_stack((inverse, np.rint(public["stations"]).astype(int), np.rint(public["sectors"]).astype(int) % 24))
    priority = public["quality"] * state["usable"]
    order = np.argsort(-priority, kind="stable")
    _, keep = np.unique(view[order], axis=0, return_index=True)
    reps = order[keep]
    group = inverse[reps]
    n = len(tracks)
    count = np.bincount(group, minlength=n).clip(1)
    mean = lambda values: np.bincount(group, weights=values[reps], minlength=n) / count
    peak = np.zeros(n)
    np.maximum.at(peak, group, priority[reps])
    role = state["object_role"]
    p = state["unary"]
    feature = np.column_stack((mean(role[:, 0]), mean(role[:, 0] - role[:, 1:].max(axis=1)),
        mean(p[:, 0]), mean(p[:, :6].sum(axis=1)), mean(state["usable"]), peak,
        np.minimum(count, 8) / 8., mean(public["quality"]), mean(state["support"]) / 5.))
    target = mean(np.asarray(labels) == 0) > .5 if labels is not None else None
    return feature, inverse, target


def candidate_features(public, state, object_fit=None):
    x = observation_features(public, state, object_role=True)
    if object_fit is None:
        return x
    o, indices, _ = object_table(public, state)
    return np.c_[x, likelihood(o, object_fit)[indices]]


def candidate_scores(public, state, config):
    return likelihood(candidate_features(public, state, config.get("object")), config["joint"])


def winners(public, state, config, mask=None):
    candidate = candidate_scores(public, state, config)
    keep = np.ones(len(candidate), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    indices, features = [], []
    for region in range(6):
        local = np.flatnonzero(keep & (public["regions"] == region))
        if not len(local):
            continue
        pick = int(local[np.argmax(candidate[local])])
        tracks = state["track_ids"][local]
        rival = candidate[local[tracks != state["track_ids"][pick]]].max(initial=0.)
        role = state["object_role"][pick]
        indices.append(pick)
        features.append([candidate[pick], candidate[pick] - rival, role[0] - max(role[1:]),
                         public["quality"][pick], min(state["view_counts"][pick] / 5., 1.),
                         1. - state["usable"][pick], min(len(local) / 120., 1.), min(len(np.unique(tracks)) / 40., 1.),
                         *np.eye(6)[region]])
    return np.asarray(indices, dtype=int), np.asarray(features, dtype=float).reshape(-1, 14)


def decoder(public, states, mode, config, mask=None):
    if mode == "unary":
        return scores(public, states, "unary", config, mask=mask)
    state = states[config["graph"]]
    ix, x = winners(public, state, config, mask=mask)
    rank = np.full(len(public["frames"]), -1.)
    rank[ix] = likelihood(x, config["winner"])
    return rank


def fit_delivery(rows, mode):
    """Fit identity, per-candidate delivery and winner reliability on distinct layouts."""
    kind = "old" if mode == "old_graph_refit" else "new"
    rule = {"graph": kind}
    if mode == "new_graph_object":
        obj = [object_table(r["public"], r["states"][kind], r["labels"]) for r in rows if r["fold"] == 0]
        rule["object"] = fit_risk([(x, label) for x, _, label in obj if len(label)])
    fit = [r for r in rows if r["fold"] == 1 and len(r["labels"])]
    rule["joint"] = fit_risk([(candidate_features(r["public"], r["states"][kind], rule.get("object")), r["labels"] == 0) for r in fit])
    values = []
    for row in rows:
        if row["fold"] == 2:
            ix, features = winners(row["public"], row["states"][kind], rule)
            if len(ix):
                values.append((features, row["labels"][ix] == 0))
    rule["winner"] = fit_risk(values)
    return rule


class Reader:
    """Causal view of the current observed prefix and a fixed route decoder."""

    def __init__(self, env, spec, base, edge_bundle, mode, config):
        self.env = env
        self.spec = spec
        self.base = base
        self.edge_bundle = edge_bundle
        self.mode = mode
        self.config = config
        self.links = scene_links(env)
        self.kind = config.get("graph", "new")
        self._last = None

    def update(self, node):
        frames = node.reader.frames
        key = (node.reader, len(frames), id(frames[-1]) if frames else None)
        if key != self._last:
            self.public = inputs(read_prefix(self.env, self.spec, node))
            self.states = public_states(self.public, self.base, self.links, self.edge_bundle, (self.kind,))
            self._last = key
        return self.public, self.states[self.kind]

    def confidence(self, node, mask=None):
        self.update(node)
        return decoder(self.public, self.states, self.mode, self.config, mask)

    def candidate_probabilities(self, node):
        public, state = self.update(node)
        if self.mode == "unary":
            return self.states["unary"]["probabilities"][:, 0]
        return candidate_scores(public, state, self.config)
