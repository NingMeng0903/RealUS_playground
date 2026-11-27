"""A causal prefix mapping shared by calibration, deployment and evaluation."""

from v6_data import V5 as _V5
from study_worlds import read_prefix, scene_links
from readout_state import candidates as legacy_candidates, heads
import numpy as np
from scipy.special import expit

from object_tracks import _track_outputs, _public_arrays, infer
from role_fusion import infer_roles
from usability_fit import predict_usability


PUBLIC_FIELDS = ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality")
METHODS = ("unary", "eight_class", "role_only", "objects", "eight_oracle", "role_oracle", "object_oracle")
JOINT_FIELDS = ("raw_target", "local_target_given_physical", "track_target", "track_role_margin",
                "measured_quality", "physical_return_probability", "raw_artifact", "supported_distinct_views",
                "graph_support", "station_neighborhood")


def inputs(public):
    arrays = {field: np.asarray(public[field]) for field in PUBLIC_FIELDS}
    # Frozen observations are archived at this precision. Applying the same
    # representation online makes archive and deployment maps identical.
    arrays["features"] = arrays["features"].astype(np.float16, copy=False)
    return arrays


def model_state(public, model, links, edge_model, usability, physical=None):
    """Re-evaluate all candidates against this prefix, with no recursive voting."""
    public = inputs(public)
    p, e = heads(public["features"], model)
    u = predict_usability(public, usability)
    graph = infer(public, p, e, links, edge_model=edge_model)
    split = infer_roles(public, p, e, links, edge_model, u, graph=graph)
    unary = {"probabilities": p, "track_ids": np.arange(len(p)), "support": np.zeros(len(p)),
             "edges": np.empty((0, 3))}
    graph_states = {"unary": unary, "eight_class": graph, "role_only": split, "objects": split}
    if physical is not None:
        # Oracle and split-state now use the exact same topology-constrained
        # edges and confidences. Oracle identities exist only on this path.
        oracle = infer_roles(public, p, e, links, edge_model, u, oracle_instance=physical)
        op, ot, os, _, _ = _track_outputs(_public_arrays(public), p, oracle["edges"], links)
        graph_states.update(eight_oracle={"probabilities": op, "track_ids": ot, "support": os, "edges": oracle["edges"]},
                            role_oracle=oracle, object_oracle=oracle)
    return graph_states


def observation_features(public, state, *, object_role):
    role = state["object_role" if object_role else "role"]
    p = state["unary"]
    counts = np.bincount(np.clip(np.rint(public["stations"]).astype(int), 0, 300), minlength=301)
    neighbors = counts[np.clip(np.rint(public["stations"]).astype(int), 0, 300)]
    return np.column_stack((p[:, 0], state["local_role"][:, 0], role[:, 0], role[:, 0] - role[:, 1:].max(axis=1),
                            public["quality"], state["usable"], p[:, 6:].sum(axis=1),
                            np.minimum(state["view_counts"], 6.) / 6., np.minimum(state["support"], 4.) / 4.,
                            np.minimum(neighbors, 24.) / 24.)).astype(np.float32)


def likelihood(features, fit):
    normalized = (features - np.asarray(fit["center"])) / np.asarray(fit["scale"])
    w = np.asarray(fit["weight"])
    return expit(normalized @ w[:-1] + w[-1])


def choose(public, state, name, joint, *, mask=None):
    """Pick one usable view per region, optionally selecting the object first."""
    use = np.ones(len(public["frames"]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if name in ("unary", "eight_class", "eight_oracle"):
        clipped = {k: v[use] for k, v in public.items()}
        sub = {key: state[key][use] for key in ("probabilities", "track_ids", "support")}
        best, f = legacy_candidates(clipped, sub)
        return np.flatnonzero(use)[best], sub["probabilities"][best, 0], f
    object_role = name in ("objects", "object_oracle")
    candidates = likelihood(observation_features(public, state, object_role=object_role), joint)
    tracks, chosen, features = state["track_ids"], [], []
    role = state["object_role" if object_role else "role"]
    for region in range(6):
        ix = np.flatnonzero(use & (public["regions"] == region))
        if not len(ix):
            continue
        if object_role:
            # One group competes once. Within the winning group, calibrated
            # local delivery chooses the view rather than the identity peak.
            ids, inverse = np.unique(tracks[ix], return_inverse=True)
            best_usable = np.zeros(len(ids))
            best_local = np.zeros(len(ids))
            target = np.zeros(len(ids))
            np.maximum.at(best_usable, inverse, state["usable"][ix] * (.5 + .5 * public["quality"][ix]))
            np.maximum.at(best_local, inverse, candidates[ix])
            np.maximum.at(target, inverse, role[ix, 0])
            priority = .60 * target + .28 * best_local + .12 * best_usable
            group = ix[inverse == int(np.argmax(priority))]
        else:
            group = ix
        best = int(group[np.argmax(candidates[group])])
        other = ix[tracks[ix] != tracks[best]]
        runner = np.max(candidates[other], initial=0.)
        features.append([candidates[best], candidates[best] - runner, role[best, 0] - role[best, 1:].max(),
                         public["quality"][best], min(1., float(state["view_counts"][best]) / 5.),
                         1. - state["usable"][best], min(1., len(ix) / 120.),
                         min(1., len(np.unique(tracks[ix])) / 40.), *np.eye(6)[region]])
        chosen.append(best)
    idx = np.asarray(chosen, dtype=np.int32)
    return idx, candidates[idx], np.asarray(features, dtype=np.float32).reshape(-1, 14)


def scores(public, states, name, calibration, *, mask=None):
    ix, raw, winner = choose(public, states[name], name, calibration.get("joint"), mask=mask)
    rank = np.full(len(public["frames"]), -1., dtype=np.float32)
    rank[ix] = raw if calibration.get("winner") is None else likelihood(winner, calibration["winner"])
    return rank


class PrefixReader:
    """Cache immutable complete prefixes, never stale per-candidate outputs."""

    def __init__(self, env, spec, reader):
        self.env, self.spec, self.reader = env, spec, reader
        self.links = scene_links(env)
        self.last_key, self.public, self.states = None, None, None

    def update(self, node):
        frames = node.reader.frames
        # The scanner appends immutable tuples. Including the last tuple
        # handles forks of equal length while allowing idempotent reads.
        last_frame = frames[-1] if frames else None
        key = (node.reader, len(frames))
        if key != self.last_key or getattr(self, "last_frame", None) is not last_frame:
            self.public = inputs(read_prefix(self.env, self.spec, node))
            self.states = model_state(self.public, self.reader["model"], self.links, self.reader["edge_model"],
                                      self.reader["usability"])
            self.last_key, self.last_frame = key, last_frame
        return self.public, self.states

    def confidence(self, node, mode=None, *, mask=None):
        public, states = self.update(node)
        mode = self.reader["delivery"] if mode is None else mode
        return scores(public, states, mode, self.reader["calibration"][mode], mask=mask)

    def feedback(self, node, frame, mode="objects"):
        from material_motion import MaterialMotion
        public, states = self.update(node)
        newest = public["frames"] == len(node.reader.frames) - 1
        ix = public["slots"][newest]
        p = np.zeros(len(frame[2]), dtype=np.float32)
        if mode == "unary":
            p[ix] = states[mode]["probabilities"][newest, 0]
        else:
            features = observation_features(public, states["objects"], object_role=True)[newest]
            p[ix] = likelihood(features, self.reader["calibration"]["objects"]["joint"])
        return MaterialMotion._feedback_shape((*frame[:6], p, *frame[7:]))
