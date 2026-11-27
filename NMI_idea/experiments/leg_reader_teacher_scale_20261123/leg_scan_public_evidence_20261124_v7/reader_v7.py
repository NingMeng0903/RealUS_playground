"""Public causal readout and decision-readiness fixes."""

import study_v7  # Add the pinned experiment modules to the import path.
from anchor_logic import completion
from causal_reader import PrefixReader, heads, infer, inputs, likelihood, observation_features, predict_usability
from live_programs import old_candidate
from role_fusion import _components, _path_weights, _public, _representatives, infer_roles
from study_worlds import read_prefix
import numpy as np


def fuse(public, unary, embedding, links, edge_model, usable, graph):
    """Keep local state when a component has zero credible representatives."""
    result = infer_roles(public, unary, embedding, links, edge_model, usable, graph=graph)
    q = _public(public)
    tracks, adj = _components(len(unary), result["edges"])
    corrected = 0
    for track in np.unique(tracks):
        group = np.flatnonzero(tracks == track)
        if not np.any(usable[group] * q["quality"][group] > 1.e-6):
            worth = False
        else:
            reps = _representatives(group.tolist(), q)
            reach = _path_weights(group.tolist(), adj)
            worth = any(usable[i] * q["quality"][i] * reach.get(i, 0.) > 1.e-6 for i in reps)
        if not worth:
            result["role"][group] = result["local_role"][group]
            result["object_role"][group] = result["local_role"][group]
            result["support"][group] = 0.
            result["view_counts"][group] = 0.
            corrected += 1
    result["stats"]["zero_credible_components"] = corrected
    return result


def state_at(public, reader, links, *, bridge=None, remove=None):
    p, e = heads(public["features"], reader["model"])
    u = predict_usability(public, reader["usability"])
    graph = infer(public, p, e, links, edge_model=reader["edge_model"])
    if bridge is not None:
        graph = bridge(public, p, e, links, u, graph)
    if remove is not None:
        graph = remove(graph)
    objects = fuse(public, p, e, links, reader["edge_model"], u, graph)
    unary = {"probabilities": p, "track_ids": np.arange(len(p)), "support": np.zeros(len(p)),
             "edges": np.empty((0, 3))}
    return {"unary": unary, "eight_class": graph, "objects": objects, "role_only": objects}


class PublicReader(PrefixReader):
    """Use one recomputed prefix with an optional public bridge policy."""

    def __init__(self, env, spec, reader, *, bridge=None):
        super().__init__(env, spec, reader)
        self.bridge = bridge

    def update(self, node):
        frames = node.reader.frames
        last = frames[-1] if frames else None
        key = (node.reader, len(frames))
        if key != self.last_key or getattr(self, "last_frame", None) is not last:
            self.public = inputs(read_prefix(self.env, self.spec, node))
            self.states = state_at(self.public, self.reader, self.links, bridge=self.bridge)
            self.last_key, self.last_frame = key, last
        return self.public, self.states


def stop_event(reader, node, origin, expected_gain):
    """Measure identity evidence separately from the deployed delivery event."""
    actual = old_candidate(reader, node, origin, reader.reader["delivery"])
    alternative = old_candidate(reader, node, origin, "objects")
    trace = {"deployed_decoder": reader.reader["delivery"], "deployed_old": actual, "object_old": alternative}
    if actual is None:
        return trace | {"stop": False, "reason": "no_deployed_old_candidate", "new_views": 0}
    public, states = reader.update(node); state = states["objects"]; i = actual["index"]
    event = completion(public, state, int(state["track_ids"][i]), origin["count"], origin["station"], node.station,
                       float(state["object_role"][i, 0]), actual["submitted"], float(expected_gain))
    return trace | event


def direct_features(public, state):
    return observation_features(public, state, object_role=True)


def candidate_delivery(public, state, joint):
    return likelihood(direct_features(public, state), joint)
