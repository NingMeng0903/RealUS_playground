"""Public-only identity beliefs, information ranking, and completion logic.

This module contains the numerical part of the v6 forward-anchor controller.  It
does not read a layout, a label, a seed, or a simulator object.  The caller
provides the anonymous candidate arrays and the public association state.  The
state keeps object identity and candidate usability separate.  A low-quality
view may therefore contribute weak evidence to an object without erasing a
clear view that belongs to that object.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


_EPS = 1.0e-12
_TARGET_ROLE = 0


def _array(mapping: Mapping[str, Any], name: str, *, length: int | None = None,
           ndim: int | None = None, dtype: Any = float) -> np.ndarray:
    if name not in mapping:
        raise KeyError(f"missing public/state field {name!r}")
    value = np.asarray(mapping[name], dtype=dtype)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name!r} must have {ndim} dimensions, got {value.shape}")
    if length is not None and len(value) != length:
        raise ValueError(f"{name!r} has length {len(value)}, expected {length}")
    return value


def _public(public: Mapping[str, Any]) -> dict[str, np.ndarray]:
    frames = _array(public, "frames", ndim=1, dtype=np.int64)
    n = len(frames)
    out = {
        "frames": frames,
        "slots": _array(public, "slots", length=n, ndim=1, dtype=np.int64),
        "stations": _array(public, "stations", length=n, ndim=1, dtype=float),
        "sectors": _array(public, "sectors", length=n, ndim=1, dtype=float),
        "points": _array(public, "points", length=n, ndim=2, dtype=float),
    }
    if out["points"].shape != (n, 2):
        raise ValueError(f"public points must have shape ({n}, 2), got {out['points'].shape}")
    out["regions"] = np.asarray(public.get("regions", np.zeros(n)), dtype=np.int64)
    if len(out["regions"]) != n:
        raise ValueError("public regions has an invalid length")
    out["quality"] = np.clip(np.asarray(public.get("quality", np.ones(n)), dtype=float), 0., 1.)
    if len(out["quality"]) != n:
        raise ValueError("public quality has an invalid length")
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("public arrays must contain finite values")
    return out


def _probability(state: Mapping[str, Any], key: str, n: int, fallback: Any = None) -> np.ndarray:
    value = state.get(key, fallback)
    if value is None:
        raise KeyError(f"state is missing {key!r}")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[0] != n:
        raise ValueError(f"state[{key!r}] must have shape ({n}, classes), got {value.shape}")
    value = np.clip(value, 0., None)
    total = value.sum(axis=1, keepdims=True)
    return value / np.maximum(total, _EPS)


def _state_arrays(state: Mapping[str, Any], n: int) -> dict[str, np.ndarray]:
    local = state.get("local_role")
    if local is None:
        local = state.get("role", state.get("probabilities"))
    local = _probability({"local": local}, "local", n)
    role = state.get("role", state.get("probabilities", local))
    role = _probability({"role": role}, "role", n)
    object_role = state.get("object_role", role)
    object_role = _probability({"object_role": object_role}, "object_role", n)
    tracks = np.asarray(state.get("track_ids", np.arange(n)), dtype=np.int64)
    if tracks.ndim != 1 or len(tracks) != n:
        raise ValueError("state track_ids must have one entry per candidate")
    usable = np.asarray(state.get("usable", np.ones(n)), dtype=float)
    if usable.ndim != 1 or len(usable) != n:
        raise ValueError("state usable must have one entry per candidate")
    usable = np.clip(np.nan_to_num(usable, nan=0.), 0., 1.)
    support = np.asarray(state.get("support", np.zeros(n)), dtype=float)
    views = np.asarray(state.get("view_counts", np.ones(n)), dtype=float)
    for name, values in (("support", support), ("view_counts", views)):
        if values.ndim != 1 or len(values) != n:
            raise ValueError(f"state {name} must have one entry per candidate")
    return {"local": local, "role": role, "object_role": object_role,
            "track_ids": tracks, "usable": usable, "support": support,
            "view_counts": views}


def _target_mass(values: np.ndarray, usable: np.ndarray | float = 1.) -> np.ndarray:
    """Target-role mass with artifact classes excluded when present."""
    mass = np.asarray(values[..., _TARGET_ROLE], dtype=float)
    if values.shape[-1] >= 8:
        mass = mass * np.clip(1. - values[..., 6:].sum(axis=-1), 0., 1.)
    return np.clip(mass * usable, 0., None)


def _conditional_target(values: np.ndarray) -> np.ndarray:
    """Return conditional physical-target mass without a usability factor."""
    mass = np.asarray(values[..., _TARGET_ROLE], dtype=float)
    if values.shape[-1] >= 8:
        mass = mass * np.clip(1. - values[..., 6:].sum(axis=-1), 0., 1.)
    return np.clip(mass, 0., None)


def _view_key(station: float, sector: float) -> tuple[int, int]:
    return int(np.rint(station)), int(np.rint(sector)) % 24


def _project_point(public: dict[str, np.ndarray], index: int, track: int, station: float,
                   tracks: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Project a track a short distance along its measured public direction."""
    point = public["points"][index].astype(float, copy=True)
    candidates = np.flatnonzero((tracks == track) & (np.abs(public["stations"] - public["stations"][index]) >= .75))
    if not len(candidates):
        return point
    prior = candidates[np.argmax(score[candidates])]
    delta_station = float(public["stations"][index] - public["stations"][prior])
    if abs(delta_station) < _EPS:
        return point
    slope = (point - public["points"][prior]) / delta_station
    return point + np.clip((station - public["stations"][index]) * slope, -6., 6.)


def beliefs(public: Mapping[str, Any], state: Mapping[str, Any], station: float,
            step_ahead: float = 1.) -> dict[str, Any] | None:
    """Return two public competing object hypotheses near ``station``.

    The returned mapping has exactly the fields consumed by the controller.
    Candidate identities are selected from public tracks and role posteriors.
    No source, instance, or hidden layout field is consulted.
    """
    p = _public(public)
    n = len(p["frames"])
    s = _state_arrays(state, n)
    if n < 2:
        return None
    # Usability screens which representatives are useful.  It is deliberately
    # absent from the hypothesis weights, which remain conditional role
    # posteriors rather than a product of identity and delivery events.
    object_target = _conditional_target(s["object_role"])
    local_target = _conditional_target(s["local"])
    screened_target = object_target * (.25 + .75 * s["usable"])
    age = np.abs(p["stations"] - float(station))
    nearby = np.flatnonzero(age <= 3.5)
    if len(nearby) < 2:
        nearby = np.argsort(age, kind="stable")[: min(n, 16)]
    order = nearby[np.argsort(-(screened_target[nearby] + .025 * np.minimum(s["support"][nearby], 4.)
                               + .01 * p["quality"][nearby]) * np.exp(-age[nearby] / 2.), kind="stable")]
    selected: list[int] = []
    for candidate in order:
        candidate = int(candidate)
        if not selected:
            selected.append(candidate)
            continue
        distinct_track = s["track_ids"][candidate] != s["track_ids"][selected[0]]
        separated = np.linalg.norm(p["points"][candidate] - p["points"][selected[0]]) >= 4.0
        if distinct_track and separated:
            selected.append(candidate)
            break
    if len(selected) != 2:
        return None
    tracks = s["track_ids"][selected]
    weights = np.asarray([max(float(object_target[i]), _EPS) for i in selected], dtype=float)
    weights /= max(weights.sum(), _EPS)
    unary_weights = np.asarray([max(float(local_target[i]), _EPS) for i in selected], dtype=float)
    unary_weights /= max(unary_weights.sum(), _EPS)
    projected = np.stack([_project_point(p, i, int(track), float(station) + float(step_ahead),
                                          s["track_ids"], screened_target) for i, track in zip(selected, tracks, strict=True)])
    return {
        "points": projected.astype(float),
        "weights": weights.astype(float),
        "unary_weights": unary_weights.astype(float),
        "ids": tuple((int(p["frames"][i]), int(p["slots"][i])) for i in selected),
        "tracks": tuple(int(v) for v in tracks),
        "stations": tuple(float(p["stations"][i]) for i in selected),
    }


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1.e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _observation_likelihood(d_a: float, d_b: float) -> np.ndarray:
    """P(A-only, B-only, no-target, ambiguous | true A/B)."""
    d_a, d_b = float(np.clip(d_a, 0., 1.)), float(np.clip(d_b, 0., 1.))
    # A small cross-talk term represents a mixed or aliased response.  The
    # residual is assigned to the explicitly ambiguous outcome so rows sum to
    # one for every public action and hypothesis.
    a_only = d_a * (1. - .50 * d_b)
    b_only = .25 * d_a * d_b
    no_target = (1. - d_a) * (1. - d_b)
    row_a = np.array([a_only, b_only, no_target, 0.], dtype=float)
    row_a[3] = max(0., 1. - row_a[:3].sum())
    row_b = np.array([.25 * d_a * d_b, d_b * (1. - .50 * d_a), no_target, 0.], dtype=float)
    row_b[3] = max(0., 1. - row_b[:3].sum())
    return np.stack((row_a, row_b))


def _expected_error(weights: np.ndarray, likelihood: np.ndarray) -> float:
    prior = np.asarray(weights, dtype=float)
    probabilities = prior @ likelihood
    post_error = []
    for probability, column in zip(probabilities, likelihood.T, strict=True):
        if probability <= _EPS:
            continue
        posterior = prior * column / probability
        post_error.append(float(probability * min(posterior[0], posterior[1])))
    return float(sum(post_error))


def rank(base: Any, legal: Any, detection: Any, separated: Any,
         weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """Rank legal views with expected public identity information value.

    ``detection`` has shape ``(2, actions)`` and contains predicted detection
    probabilities for the two public hypotheses.  The second output is the
    unweighted value-of-information diagnostic.  Neither output uses labels or
    hidden geometry.
    """
    base = np.asarray(base, dtype=float).reshape(-1)
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    detection = np.asarray(detection, dtype=float)
    separated = np.asarray(separated, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if detection.shape != (2, len(base)) or len(legal) != len(base) or len(separated) != len(base):
        raise ValueError("rank inputs have incompatible action shapes")
    if len(weights) != 2 or np.any(weights < 0.) or weights.sum() <= _EPS:
        raise ValueError("weights must contain two nonnegative masses")
    weights = weights / weights.sum()
    baseline_error = float(min(weights))
    information = np.zeros(len(base), dtype=float)
    common = np.zeros(len(base), dtype=float)
    for i in range(len(base)):
        likelihood = _observation_likelihood(detection[0, i], detection[1, i])
        information[i] = max(0., baseline_error - _expected_error(weights, likelihood))
        expected_detection = float(weights @ np.clip(detection[:, i], 0., 1.))
        discrimination = float(np.clip(separated[i], 0., 1.))
        common[i] = .70 * expected_detection + .45 * discrimination
    score = .30 * _normalise(np.nan_to_num(base, nan=-np.inf, posinf=0., neginf=0.))
    score += 1.40 * information + .35 * common
    score = np.where(legal, score, -np.inf)
    information = np.where(legal, information, 0.)
    return score, information


def _distinct_views(public: dict[str, np.ndarray], indices: np.ndarray) -> set[tuple[int, int]]:
    return {_view_key(float(public["stations"][i]), float(public["sectors"][i])) for i in indices}


def completion(public: Mapping[str, Any], state: Mapping[str, Any], target_track: int,
               frame_cut: int, prefix_station: float, current_station: float,
               identity_confidence: float, local_delivery: Any, extra_gain: float,
               *, risk_threshold: float = .18, ready_threshold: float = .75,
               gain_threshold: float = .065) -> dict[str, Any]:
    """Decide whether a public object has enough new evidence to stop seeking.

    Completion uses only newly acquired view keys, forward distance, track
    support, and current public delivery confidence.  It deliberately does not
    compare against a stale origin score or require an old-score increase.
    """
    p = _public(public)
    n = len(p["frames"])
    s = _state_arrays(state, n)
    fresh = (p["frames"] >= int(frame_cut)) & (p["stations"] > float(prefix_station) + .24)
    fresh_ids = np.flatnonzero(fresh)
    matched = fresh_ids[s["track_ids"][fresh_ids] == int(target_track)] if len(fresh_ids) else np.empty(0, dtype=int)
    new_views = _distinct_views(p, matched)
    forward_span = float(current_station) - float(prefix_station)
    two_forward = forward_span >= 2.0
    competing = fresh_ids[s["track_ids"][fresh_ids] != int(target_track)] if len(fresh_ids) else np.empty(0, dtype=int)
    # A track elsewhere in the history is not a visible competing relation.
    # Require a usable candidate close to a newly observed target candidate and
    # a separated public point before it contributes to completion.
    local_competing = []
    for candidate in competing:
        if not len(matched):
            continue
        near_target = np.min(np.abs(p["stations"][matched] - p["stations"][candidate])) <= 2.5
        visible = s["usable"][candidate] >= .35
        separated = np.min(np.linalg.norm(p["points"][matched] - p["points"][candidate], axis=1)) >= 3.5
        if near_target and visible and separated:
            local_competing.append(int(candidate))
    local_competing = np.asarray(local_competing, dtype=int)
    competing_tracks = set(int(v) for v in s["track_ids"][local_competing])
    relation_distance = 0.
    if len(matched) and len(local_competing):
        relation_distance = float(np.min(np.linalg.norm(p["points"][matched, None] - p["points"][None, local_competing], axis=-1)))
    angles = np.asarray([float(p["sectors"][i]) for i in matched], dtype=float)
    cross_window = bool(len(angles) >= 2 and np.max(
        np.minimum(np.abs(angles[:, None] - angles[None, :]), 24. - np.abs(angles[:, None] - angles[None, :]))
    ) >= 2.)
    adjacent_target_support = bool(len(matched) >= 2 and np.ptp(p["stations"][matched]) >= .75)
    # A competing relation must be visible and local.  A target's own
    # cross-window evidence can also satisfy the relation when it spans
    # adjacent stations, without treating an arbitrary old track as evidence.
    distinct_relation = bool(competing_tracks or (cross_window and adjacent_target_support))
    usable_mass = float(np.mean(s["usable"][matched])) if len(matched) else 0.
    support = float(np.max(s["support"][matched], initial=0.)) if len(matched) else 0.
    confidence = float(np.clip(identity_confidence, 0., 1.))
    risk = 1. - confidence
    if isinstance(local_delivery, (bool, np.bool_)):
        delivery_ready = bool(local_delivery)
    else:
        delivery_ready = float(np.asarray(local_delivery, dtype=float).reshape(-1)[0]) >= ready_threshold
    gain = float(np.nan_to_num(extra_gain, nan=0.))
    confidence_ready = confidence >= ready_threshold and risk <= risk_threshold
    evidence_ready = len(new_views) >= 2 and distinct_relation and usable_mass >= .35 and (support > 0. or len(matched) >= 2)
    stop = bool(two_forward and evidence_ready and confidence_ready and delivery_ready and gain <= gain_threshold)
    if not two_forward:
        reason = "needs_two_forward_stations"
    elif not evidence_ready:
        reason = "needs_distinct_forward_evidence"
    elif not confidence_ready:
        reason = "identity_risk_above_threshold"
    elif not delivery_ready:
        reason = "delivery_not_ready"
    elif gain > gain_threshold:
        reason = "expected_future_gain_still_high"
    else:
        reason = "public_identity_and_delivery_ready"
    return {
        "stop": stop,
        "reason": reason,
        "new_views": int(len(new_views)),
        "new_view_ids": sorted(new_views),
        "forward_span": forward_span,
        "matched_candidates": int(len(matched)),
        "matched_tracks": sorted(set(int(v) for v in s["track_ids"][matched])),
        "competing_tracks": sorted(competing_tracks),
        "distinct_competing_or_relation": distinct_relation,
        "relation_distance_mm": relation_distance,
        "cross_window": bool(cross_window),
        "usable_mass": usable_mass,
        "track_support": support,
        "identity_confidence": confidence,
        "identity_risk": risk,
        "local_delivery_ready": delivery_ready,
        "extra_gain": gain,
        "criteria": {
            "two_forward_stations": two_forward,
            "evidence_ready": evidence_ready,
            "confidence_ready": confidence_ready,
            "delivery_or_gain_ready": bool(delivery_ready or gain >= gain_threshold),
        },
    }


def _toy() -> None:
    public = {
        "frames": np.arange(6), "slots": np.zeros(6, dtype=int),
        "stations": np.array([0., 1., 2., 3., 4., 5.]),
        "sectors": np.array([0., 0., 0., 2., 4., 4.]),
        "points": np.array([[0., 0.], [0.2, 0.], [0.4, 0.], [5., 0.], [5.2, 0.], [5.4, 0.]]),
        "quality": np.ones(6),
    }
    role = np.full((6, 8), .01)
    role[:, 0] = np.array([.55, .60, .65, .40, .43, .46])
    role[:, 1] = 1. - role[:, 0] - role[:, 6:].sum(1)
    state = {"role": role, "local_role": role.copy(), "object_role": role.copy(),
             "track_ids": np.array([0, 0, 0, 1, 1, 1]), "usable": np.ones(6),
             "support": np.ones(6), "view_counts": np.full(6, 2.)}
    b = beliefs(public, state, 2.5)
    assert b is not None and len(b["ids"]) == 2 and np.isclose(sum(b["weights"]), 1.)
    scores, voi = rank(np.array([.2, .3, .1]), np.ones(3, dtype=bool),
                       np.array([[.95, .5, .1], [.1, .5, .95]]), np.array([.1, .8, .1]), b["weights"])
    assert int(np.argmax(scores)) in (0, 2) and np.max(voi) > 0.
    done = completion(public, state, 1, 3, 2., 4., .86, True, .01)
    assert done["stop"] and done["new_views"] >= 2
    high_gain = completion(public, state, 1, 3, 2., 4., .86, True, .20)
    assert not high_gain["stop"] and high_gain["reason"] == "expected_future_gain_still_high"
    state["object_role"][:3, 0] = .45
    refused = completion(public, state, 1, 3, 2., 4., .50, False, .01)
    assert not refused["stop"] and refused["reason"] == "identity_risk_above_threshold"


if __name__ == "__main__":
    _toy()
    print("anchor_logic CPU toy passed")
