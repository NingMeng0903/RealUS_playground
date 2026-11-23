"""Calibrate graph links, including clutter and mixtures, before propagating evidence."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


EDGE_FIELDS = ("base_weight", "embedding_cosine", "separation_mm", "axial_delta", "angle_delta",
               "min_quality", "min_physical_probability", "max_artifact_probability", "role_overlap",
               "target_probability_disagreement")
LINK_THRESHOLD = .90


def edge_features(public, unary, embedding, edges):
    left, right = edges[:, :2].astype(int).T
    ds = abs(public["stations"][left] - public["stations"][right])
    da = abs((public["sectors"][left] - public["sectors"][right] + 12.) % 24. - 12.)
    world = np.linalg.norm(public["points"][left] - public["points"][right], axis=1)
    x, y = unary[left], unary[right]
    physical = np.minimum(x[:, :6].sum(axis=1), y[:, :6].sum(axis=1))
    artifact = np.maximum(x[:, 6:].sum(axis=1), y[:, 6:].sum(axis=1))
    return np.column_stack((edges[:, 2], np.sum(embedding[left] * embedding[right], axis=1), world / 16., ds / 3.2,
                            da / 2., np.minimum(public["quality"][left], public["quality"][right]), physical, artifact,
                            np.sum(x[:, :6] * y[:, :6], axis=1), abs(x[:, 0] - y[:, 0])))


def fit_edge_confidence(features, labels):
    z = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    center, scale = z.mean(axis=0), np.maximum(z.std(axis=0), .035)
    design = np.c_[(z - center) / scale, np.ones(len(z))]
    penalty = np.r_[np.full(z.shape[1], 2.e-3), 0.]

    def objective(weight):
        t = design @ weight
        data = np.mean(np.logaddexp(0, t) - target * t)
        grad = design.T @ (expit(t) - target) / len(target)
        return float(data + .5 * weight @ (penalty * weight)), grad + penalty * weight

    answer = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not answer.success:
        raise RuntimeError(f"Association confidence fitting failed: {answer.message}")
    return {"mean": center.tolist(), "scale": scale.tolist(), "weight": answer.x.tolist(), "features": EDGE_FIELDS,
            "positive_links": int(target.sum()), "negative_links_including_artifacts": int(len(target) - target.sum())}


def supported_roles(first, last, links):
    """An identity can traverse a span only when one source supports it end to end."""
    low = int(np.floor(min(first, last)))
    high = int(np.ceil(max(first, last)))
    allow = np.ones(8, dtype=bool)
    if high > low:
        allow[:6] = np.any(np.all(links[:, low:high, :6], axis=1), axis=0)
    return allow


def guard(public, graph, links, frozen):
    """Keep per-candidate beliefs and attenuate weak cross-view messages."""
    unary, embedding, edges = graph["unary"], graph["embedding"], graph["edges"]
    z = (edge_features(public, unary, embedding, edges) - np.asarray(frozen["mean"])) / np.asarray(frozen["scale"])
    coef = np.asarray(frozen["weight"])
    p = expit(z @ coef[:-1] + coef[-1])
    accept = p >= LINK_THRESHOLD
    taken = edges[accept]
    views = np.rint(np.c_[public["stations"], public["sectors"]]).astype(int)
    views[:, 1] %= 24
    best = [{} for _ in unary]
    group = np.arange(len(unary), dtype=int)

    def root(i):
        while i != group[i]:
            i = int(group[i])
        return i

    for edge, probability in zip(taken, p[accept], strict=True):
        left, right, weight = int(edge[0]), int(edge[1]), float(edge[2] * probability)
        group[root(right)] = root(left)
        allowed = supported_roles(public["stations"][left], public["stations"][right], links)
        for first, last in ((left, right), (right, left)):
            view = tuple(views[last])
            present = best[first].get(view)
            if present is None or weight > present[0]:
                best[first][view] = (weight, last, allowed)
    revised = unary.copy()
    support = np.zeros(len(unary), dtype=np.float32)
    for i, neighbors in enumerate(best):
        if not neighbors:
            continue
        weight = sum(v[0] for v in neighbors.values())
        offset = np.zeros(8, dtype=np.float32)
        for reliability, j, allowed in neighbors.values():
            neighbor = unary[j, allowed]
            if neighbor.sum() > 1.e-8:
                independent = neighbor * (unary[i, allowed].sum() / neighbor.sum())
                offset[allowed] += reliability * (independent - unary[i, allowed])
        # A low-support link has low absolute influence even when it is alone.
        mixing = min(.45, .45 * weight) * (1. - .40 * unary[i].max())
        revised[i] += mixing * offset / max(weight, 1.e-8)
        support[i] = weight
    return graph | {"probabilities": revised, "support": support,
                    "track_ids": np.array([root(i) for i in range(len(unary))], dtype=np.int32), "edges": taken}
