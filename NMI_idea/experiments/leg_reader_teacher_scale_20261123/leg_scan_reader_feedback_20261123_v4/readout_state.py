"""Public identity, physical-track association, and calibrated delivery risk."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

from public_graph import associate
from route_metrics import outcomes


METHODS = ("v3_scaled", "fixed_unary", "fixed_graph")
RISK_FIELDS = ("p_target", "track_margin", "target_vs_other_role", "quality", "view_support",
               "candidate_artifact_probability", "section_population", "available_track_groups",
               *(f"section_{n}" for n in range(6)))


def export_cpu(bundle):
    parts = bundle["model"].state_dict()
    return {"model": {k: v.detach().cpu().numpy() for k, v in parts.items()},
            "center": np.asarray(bundle["center"], dtype=np.float32),
            "scale": np.asarray(bundle["scale"], dtype=np.float32)}


def heads(features, bundle):
    x = np.clip((features.astype(np.float32, copy=False) - bundle["center"]) / bundle["scale"], -9., 9.)
    weights = bundle["model"]
    for name in ("features.0", "features.2"):
        x = x @ weights[f"{name}.weight"].T + weights[f"{name}.bias"]
        x *= expit(x)
    y = x @ weights["identity.weight"].T + weights["identity.bias"]
    embedding = x @ weights["association.weight"].T + weights["association.bias"]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1.e-12)
    return softmax(y, axis=1), embedding


def make_state(public, bundle, links, method, *, model_output=None):
    """The public state can be recomputed at any acquisition time."""
    features = public["legacy_features"] if method == "v3_scaled" and "legacy_features" in public else public["features"]
    p, e = heads(features, bundle) if model_output is None else model_output
    if method == "fixed_graph":
        linked = associate(public, p, e, links)
        return linked | {"unary": p, "embedding": e}
    return {"probabilities": p, "unary": p, "embedding": e, "track_ids": np.arange(len(p)),
            "support": np.zeros(len(p), dtype=np.float32), "edges": np.empty((0, 2), dtype=np.int32)}


def candidates(public, state, max_frame=None):
    """Select a public winner per section and construct its confidence features."""
    use = np.ones(len(public["frames"]), dtype=bool)
    if max_frame is not None:
        use &= public["frames"] < max_frame
    p = state["probabilities"]
    tracks = state["track_ids"]
    winners, counts = [], []
    for region in range(6):
        idx = np.flatnonzero(use & (public["regions"] == region))
        if not len(idx):
            continue
        best = idx[np.argmax(p[idx, 0])]
        other = idx[tracks[idx] != tracks[best]]
        runner_up = np.max(p[other, 0], initial=0.)
        competitor = np.max(p[best, 1:])
        counts.append([p[best, 0], p[best, 0] - runner_up, p[best, 0] - competitor, public["quality"][best],
                       min(1., float(state["support"][best]) / 3.), p[best, 6] + p[best, 7],
                       min(1., len(idx) / 120.), min(1., len(np.unique(tracks[idx])) / 40.), *np.eye(6)[region]])
        winners.append(best)
    return np.asarray(winners, dtype=np.int32), np.asarray(counts, dtype=np.float32).reshape(-1, len(RISK_FIELDS))


def fit_risk(samples):
    """Fit confidence of an already chosen image on a separate calibration panel."""
    x, y = (np.concatenate(v) for v in zip(*samples, strict=True))
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .05)
    features = (x - center) / scale
    features = np.c_[features, np.ones(len(features))]
    reg = np.r_[np.full(x.shape[1], 3.e-3), 0.]

    def objective(w):
        z = features @ w
        penalty = float(w @ (reg * w)) / 2.
        value = float(np.mean(np.logaddexp(0, z) - y * z)) + penalty
        gradient = features.T @ (expit(z) - y) / len(y) + reg * w
        return value, gradient

    fit = minimize(objective, np.zeros(features.shape[1]), method="L-BFGS-B", jac=True)
    if not fit.success:
        raise RuntimeError(f"Winner confidence fit failed: {fit.message}")
    return {"center": center.tolist(), "scale": scale.tolist(), "weight": fit.x.tolist(), "images": len(y),
            "correct_images": int(y.sum()), "field_order": RISK_FIELDS}


def confidence(public, state, risk, *, max_frame=None):
    winners, values = candidates(public, state, max_frame)
    scores = np.full(len(public["frames"]), -1., dtype=np.float32)
    if risk is None:
        scores[winners] = state["probabilities"][winners, 0]
    else:
        z = (values - np.asarray(risk["center"])) / np.asarray(risk["scale"])
        weights = np.asarray(risk["weight"])
        scores[winners] = expit(z @ weights[:-1] + weights[-1])
    return scores


def evaluate(public, labels, scores, threshold, *, complete, submit_count=None):
    return outcomes({"label": labels, "region": public["regions"], "complete": complete},
                    scores, threshold, submit_count=submit_count)


def earlier_image(public, state, labels, risk, threshold, frame_cut, region):
    p = confidence(public, state, risk, max_frame=frame_cut)
    indices = np.flatnonzero((public["frames"] < frame_cut) & (public["regions"] == region))
    if not len(indices):
        return -99
    best = indices[np.argmax(p[indices])]
    return int(labels[best]) if p[best] >= threshold else -99


def measured_length(public, state, labels, threshold, widths):
    """Report deduplicated correct coverage and longest connected run separately."""
    p = state["probabilities"][:, 0]
    at = np.clip(np.rint(public["stations"]).astype(int), 0, len(widths) - 1)
    maximum = np.full(len(widths), -1.)
    np.maximum.at(maximum, at, p)
    winners = np.flatnonzero((p == maximum[at]) & (p >= threshold))
    stations, positions = np.unique(at[winners], return_index=True)
    correct = np.zeros(len(widths), dtype=bool)
    correct[stations] = labels[winners[positions]] == 0
    cover = float(widths[correct].sum())
    runs = np.flatnonzero(np.diff(np.r_[False, correct, False])).reshape(-1, 2)
    longest = max((float(widths[a:b].sum()) for a, b in runs), default=0.)
    return cover, longest
