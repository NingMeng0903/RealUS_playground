"""Public, shared candidate readout for the three late scan regions."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, softmax
from scipy.spatial import cKDTree

V13 = Path(__file__).resolve().parent.parent / "leg_scan_student_history_20261124_v13"
if str(V13) not in sys.path:
    sys.path.insert(1, str(V13))

import trajectory_v13 as tr
from reader_v13 import _adjust, deployment
from track_state import request


FORMAT = "leg_scan_shared_identity_v15"
REGIONS = (3, 4, 5)
WIDTH = 177
CONTEXT_WIDTH = 40
CASE_WIDTH = 16
RESIDUAL = 6.
ROLE_LABELS = (0, 1, 2, 3, 4, 5, -1, -2)
PUBLIC_KEYS = ("x", "prior", "roles", "track", "frame", "slot", "region",
               "station", "sector", "xy", "quality")


def public_snapshot(read, legacy_reader):
    p = read.public
    x = tr.observation(read) if len(p["frames"]) else np.empty((0, WIDTH), dtype=np.float16)
    if x.shape != (len(p["frames"]), WIDTH):
        raise ValueError(f"Unexpected v13 observation shape {x.shape}")
    prior = _adjust(legacy_reader, x, tr.old_logits(read)) if len(x) else np.empty(0, np.float32)
    return {
        "x": x, "prior": prior, "roles": read.decision["role_posterior"],
        "track": read.decision.get("track_id", np.arange(len(x))),
        "frame": p["frames"], "slot": p["slots"], "region": p["regions"],
        "station": p["stations"], "sector": p["sectors"], "xy": p["points"],
        "quality": p["quality"],
    }


def _representatives(group, station, sector, frame, merit):
    order = np.lexsort((frame, -merit, sector, station, group))
    same = ((group[order[1:]] == group[order[:-1]]) &
            (station[order[1:]] == station[order[:-1]]) &
            (sector[order[1:]] == sector[order[:-1]]))
    return order[np.r_[True, ~same]]


def _objects(track, station, frame, xy, quality, roles):
    """Stitch mutually compatible endpoints of non-overlapping public tracks."""
    _, node = np.unique(track, return_inverse=True)
    n = int(node.max()) + 1
    order = np.lexsort((frame, -quality, station, node))
    first = order[np.r_[True, np.diff(node[order]) != 0]]
    order = np.lexsort((frame, -quality, -station, node))
    last = order[np.r_[True, np.diff(node[order]) != 0]]
    start, end = station[first], station[last]
    start_xy, end_xy = xy[first], xy[last]
    drift = (end_xy - start_xy) / np.maximum(1., end - start)[:, None]
    k = min(n, 8)
    _, near = cKDTree(np.column_stack((2.6 * end, end_xy))).query(
        np.column_stack((2.6 * start, start_xy)), k=k, distance_upper_bound=8.0)
    near = near.reshape(n, k)
    parent = np.minimum(near, n - 1)
    gap = start[:, None] - end[parent]
    displacement = start_xy[:, None] - end_xy[parent]
    distance = np.linalg.norm(displacement, axis=-1)
    projected = np.linalg.norm(displacement - drift[parent] * gap[..., None], axis=-1)
    a, b = roles[first, :6], roles[last, :6][parent]
    dissimilarity = np.abs(a[:, None] - b).sum(axis=-1)
    cost = .68 * distance + .32 * projected + .68 * gap + .68 * dissimilarity
    admissible = ((near < n) & (gap >= 1) & (gap <= 2) &
                  (distance <= 3.8 + .8 * gap) & (dissimilarity <= 1.25))
    cost = np.where(admissible, cost, np.inf)
    options = np.argmin(cost, axis=1)
    link_cost = cost[np.arange(n), options]
    predecessor = parent[np.arange(n), options]
    safe = np.flatnonzero(np.isfinite(link_cost))
    safe = safe[np.argsort(link_cost[safe], kind="stable")]
    # A near endpoint can continue to only one fragment. The original tracks
    # and these links each rule out overlapping longitudinal spans.
    _, first_choice = np.unique(predecessor[safe], return_index=True)
    selected = safe[first_choice]
    edges = np.arange(n)
    edges[selected] = predecessor[selected]
    while np.any(edges[edges] != edges):
        edges = edges[edges]
    return np.unique(edges[node], return_inverse=True)[1]


def _weighted_mean(values, group, weight, n):
    out = np.zeros((n, values.shape[1]), dtype=np.float32)
    # Small public fields and dense group IDs keep pooling linear in echoes.
    np.add.at(out, group, values * weight[:, None])
    count = np.bincount(group, weights=weight, minlength=n).astype(np.float32)
    return out / np.maximum(count[:, None], 1.e-6)


@dataclass(frozen=True)
class Graph:
    x: np.ndarray
    context: np.ndarray
    prior: np.ndarray
    roles: np.ndarray
    rows: np.ndarray
    objects: np.ndarray
    object_of_row: np.ndarray
    object_count: int
    view_rows: np.ndarray
    view_weight: np.ndarray
    group_of_row: np.ndarray
    group_region: np.ndarray
    group_rep: np.ndarray
    group_weight: np.ndarray
    scene: np.ndarray


def build_graph(snapshot):
    """Build the same de-duplicated measured graph in replay and GPU training."""
    x, prior, roles, track, frame, _, region, station, sector, xy, quality = (
        np.asarray(snapshot[k]) for k in PUBLIC_KEYS)
    rows = np.flatnonzero(region >= REGIONS[0])
    if not len(rows):
        return None
    if x.shape != (len(region), WIDTH) or roles.shape != (len(region), len(ROLE_LABELS)):
        raise ValueError("Incomplete public v15 candidate observation")
    station = np.rint(station).astype(int)
    sector = np.rint(sector).astype(int) % 24
    roles = roles.astype(np.float32, copy=False)
    quality = np.clip(quality.astype(np.float32, copy=False), .02, 1.)
    xy = xy.astype(np.float32, copy=False)
    objects = _objects(track, station, frame, xy, quality, roles)
    n = int(objects.max()) + 1
    pos = x[:, 133:135].astype(np.float32)
    signal = quality * (.30 + .70 * roles[:, :6].sum(axis=1))
    views = _representatives(objects, station, sector, frame, signal)
    unit = _representatives(objects, station, np.zeros_like(sector), frame, signal)
    count = np.bincount(objects[views], minlength=n)
    by_station = np.bincount(objects[unit], minlength=n)
    lo = np.full(n, station.max())
    hi = np.full(n, station.min())
    np.minimum.at(lo, objects[unit], station[unit])
    np.maximum.at(hi, objects[unit], station[unit])
    first = unit[np.r_[True, np.diff(objects[unit]) != 0]]
    last = unit[np.r_[np.diff(objects[unit]) != 0, True]]
    direction = (pos[last] - pos[first]) * (8. / np.maximum(1, hi - lo))[:, None]
    obj_features = _weighted_mean(np.column_stack((roles, x[:, 115:133], quality))[views],
                                  objects[views], signal[views], n)
    geom = obj_features[objects[rows]]
    obj_stats = np.column_stack((direction[objects[rows]],
                                np.log1p(count[objects[rows]])/4.,
                                np.log1p(by_station[objects[rows]])/4.))

    # The case is re-centered from measured named-role candidates. Only one
    # representative per object and station contributes to these softly
    # weighted local prototypes, so revisiting one view does not move them.
    ss = station - station.min()
    station_count = int(ss.max()) + 1
    kernel = np.asarray([1., 2., 3., 4., 5., 4., 3., 2., 1.], dtype=np.float32)

    def smooth(weights):
        histogram = np.bincount(ss[unit], weights=weights, minlength=station_count)
        return np.convolve(np.pad(histogram, 4), kernel, mode="valid")[ss[rows]]

    prototypes = []
    for j in (0, 1, 2):
        w = roles[unit, j] ** 2 * signal[unit]
        mass = smooth(w)
        center = np.column_stack([smooth(w * pos[unit, k]) for k in range(2)])
        center /= np.maximum(mass[:, None], .04)
        difference = (pos[rows] - center) * np.minimum(1., 2. * mass[:, None])
        prototypes.append(np.column_stack((difference, mass / (4. + mass))))
    context = np.column_stack((geom, obj_stats, *prototypes)).astype(np.float32)
    if context.shape[1] != CONTEXT_WIDTH:
        raise ValueError("Incorrect graph context width")

    _, object_index = np.unique(objects[rows], return_inverse=True)
    region_of_row = region[rows].astype(int) - REGIONS[0]
    group_key, group_of_row = np.unique(3 * object_index + region_of_row, return_inverse=True)
    group_region = group_key % 3
    row_merit = prior[rows] + .55 * signal[rows]
    order = np.lexsort((frame[rows], -row_merit, group_of_row))
    group_rep = order[np.r_[True, np.diff(group_of_row[order]) != 0]]
    group_weight = (.15 + .85 * expit(prior[rows[group_rep]])) * signal[rows[group_rep]]
    scene_values = np.column_stack((roles, pos, quality, expit(prior),
                                    np.log1p(count[objects])/4.,
                                    np.log1p(by_station[objects])/4.,
                                    np.abs(pos[:, 0]), np.abs(pos[:, 1])))
    scene = _weighted_mean(scene_values[rows[group_rep]], group_region, group_weight, 3)

    view = _representatives(object_index, station[rows], sector[rows], frame[rows], signal[rows])
    return Graph(x[rows], context, prior[rows].astype(np.float32, copy=False), roles[rows], rows,
                 objects, object_index, int(object_index.max())+1, view, signal[rows[view]],
                 group_of_row, group_region, group_rep, group_weight.astype(np.float32), scene)


def load_reader(path):
    path = Path(path)
    if path.is_dir():
        path /= "reader.json"
    value = json.loads(path.read_text())
    if value.get("format") != FORMAT or value.get("width") != WIDTH:
        raise ValueError(f"Unsupported shared identity reader at {path}")
    for key in ("center", "scale", "context_center", "context_scale"):
        value[key] = np.asarray(value[key], dtype=np.float32)
    value["weights"] = {k: np.asarray(v, dtype=np.float32) for k, v in value["weights"].items()}
    return value


def _dense(x, model, name):
    w = model["weights"]
    return x @ w[name + ".weight"].T + w[name + ".bias"]


def _silu(x):
    return x * expit(x)


def numpy_encode(model, graph):
    x = ((graph.x.astype(np.float32) - model["center"]) / model["scale"]).clip(-6., 6.)
    u = ((graph.context - model["context_center"]) / model["context_scale"]).clip(-6., 6.)
    local = _silu(_dense(_silu(_dense(x, model, "encode1")), model, "encode2"))
    a = graph.view_rows
    pooled = _weighted_mean(local[a], graph.object_of_row[a], graph.view_weight, graph.object_count)
    regional = _weighted_mean(local[graph.group_rep], graph.group_region, graph.group_weight, 3)
    shared = _silu(_dense(np.concatenate((regional, graph.scene), axis=1).reshape(1, -1), model, "scene"))[0]
    z = np.column_stack((local, pooled[graph.object_of_row], u, np.broadcast_to(shared, (len(local), len(shared)))))
    h = _silu(_dense(z, model, "fuse"))
    delta = _dense(h, model, "rank")[:, 0]
    role_logits = np.log(graph.roles.clip(.005, 1.)) + _dense(h, model, "role")
    return graph.prior + RESIDUAL * np.tanh(delta / 2.), role_logits, shared


def numpy_pool(model, graph, logits, shared):
    scores = np.full(len(graph.group_region), -1.e4, dtype=np.float32)
    np.maximum.at(scores, graph.group_of_row, logits)
    top = np.full(3, -1.e4, dtype=np.float32)
    np.maximum.at(top, graph.group_region, scores)
    count = np.bincount(graph.group_region, minlength=3).astype(np.float32)
    mean = np.bincount(graph.group_region, weights=scores, minlength=3).astype(np.float32)
    summary = np.column_stack((np.eye(3, dtype=np.float32), top.clip(-10., 10.)/5.,
                               mean / np.maximum(count, 1.)/5., np.log1p(count)/5.))
    u = np.column_stack((np.broadcast_to(shared, (3, len(shared))), summary))
    null = _dense(_silu(_dense(u, model, "unknown1")), model, "unknown2")[:, 0]
    return scores, null


def score_snapshot(model, snapshot, locked=None, *, graph=None):
    """Return public logits, posterior and one proposed submission per region."""
    n = len(snapshot["region"])
    identity = np.asarray(snapshot["prior"], dtype=np.float32).copy()
    roles = np.asarray(snapshot["roles"], dtype=np.float32).copy()
    candidate = np.zeros(n, dtype=np.float32)
    unknown = np.ones(6, dtype=np.float32)
    unknown[:3] = 0.
    winner = np.full(6, -1, dtype=int)
    chosen = winner.copy()
    objects = np.asarray(snapshot["track"], dtype=int).copy()
    graph = build_graph(snapshot) if graph is None else graph
    if graph is not None:
        z, role, shared = numpy_encode(model, graph)
        if locked:
            for j, i in enumerate(graph.rows):
                key = (int(snapshot["frame"][i]), int(snapshot["slot"][i]))
                if key in locked:
                    z[j] = locked[key]
        scores, null = numpy_pool(model, graph, z, shared)
        temperature = model.get("temperature", 1.)
        threshold = model.get("threshold", .30)
        identity[graph.rows] = z
        roles[graph.rows] = softmax(role, axis=1)
        objects = graph.objects
        best_rows = np.full(len(scores), len(z), dtype=int)
        best = np.flatnonzero(z == scores[graph.group_of_row])
        np.minimum.at(best_rows, graph.group_of_row[best], best)
        posterior = np.zeros(len(scores), dtype=np.float32)
        for r in range(3):
            i = np.flatnonzero(graph.group_region == r)
            if not len(i):
                continue
            p = softmax(np.r_[scores[i], null[r]] / temperature)
            posterior[i] = p[:-1]
            region = r + REGIONS[0]
            unknown[region] = p[-1]
            g = i[np.argmax(p[:-1])]
            j = graph.rows[best_rows[g]]
            winner[region] = j
            if posterior[g] >= max(unknown[region], threshold):
                chosen[region] = j
        candidate[graph.rows] = posterior[graph.group_of_row]
    return {"identity_score": identity, "role_posterior": roles,
            "candidate_probability": candidate, "unknown_probability": unknown,
            "object_id": objects, "winner_rows": winner, "selected_rows": chosen}


def predict(model, snapshot, baseline_decision, *, locked=None):
    result = score_snapshot(model, snapshot, locked)
    late = np.asarray(snapshot["region"]) >= REGIONS[0]
    score = baseline_decision["score"].copy()
    peak = baseline_decision["winner_score"].copy()
    target = baseline_decision["target"].copy()
    roles = baseline_decision["role_posterior"].copy()
    roles[late] = result["role_posterior"][late]
    score[late], peak[late] = -1., -1.
    target[late] = .50 * expit(result["identity_score"][late]) + .50 * roles[late, 0]
    for r in REGIONS:
        i, j = result["winner_rows"][r], result["selected_rows"][r]
        if i >= 0:
            peak[i] = result["candidate_probability"][i]
        if j >= 0:
            score[j] = result["candidate_probability"][j]
    return baseline_decision | result | {"score": score, "winner_score": peak, "target": target,
                                         "role_posterior": roles, "submit_score": score,
                                         "track_id": baseline_decision.get("track_id", snapshot["track"])}


def latch_scores(source, result, keys=None):
    """Keep the raw identity logits attached to their causal echo addresses."""
    frame = source["frame"] if "frame" in source else source["frames"]
    slot = source["slot"] if "slot" in source else source["slots"]
    addresses = ((int(f), int(s)) for f, s in zip(frame, slot, strict=True))
    return {key: float(value) for key, value in zip(addresses, result["identity_score"], strict=True)
            if keys is None or key in keys}


def apply(read, model, legacy_reader, legacy_mode="new", *, locked=None):
    if model is None or legacy_mode != "new" or not len(read.public["frames"]):
        baseline = deployment(read, legacy_reader, legacy_mode)
        decision = baseline if model is None else predict(model, public_snapshot(read, legacy_reader),
                                                          baseline, locked=locked)
    else:
        # v13 deployment copies these three fields from v10 before replacing
        # only the late regions. Their late entries are replaced below.
        baseline = read.decision | {k: read.old[k] for k in ("score", "winner_score", "target")}
        decision = predict(model, public_snapshot(read, legacy_reader), baseline, locked=locked)
    hypothesis_decision = decision
    if "object_id" in decision:
        hypothesis_decision = decision | {"track_id": decision["object_id"]}
    elif "track_id" not in decision and len(read.public["frames"]):
        hypothesis_decision = decision | {"track_id": read.decision["track_id"]}
    hypotheses = {r: q for r in REGIONS if (q := request(read.public, hypothesis_decision, r)) is not None}
    return replace(read, decision=decision, hypotheses=hypotheses)
