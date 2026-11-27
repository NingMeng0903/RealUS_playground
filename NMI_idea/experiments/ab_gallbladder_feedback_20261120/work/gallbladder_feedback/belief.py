"""Public-evidence multimodal view selection for gallbladder recovery.

The scene posterior is discrete: collapsing it to its weighted mean can put the
predicted organ between two plausible locations.  This module keeps posterior
modes separate when choosing a view.  It uses only the online state, public
packets, registered geometry, and model predictions.
"""

from __future__ import annotations

import numpy as np

from view_geometry import (CONTACT, POSE_DISTANCE_SCALE, measure_view,
                           offsets_around, settled_plan)


def _posterior_modes(scene, state, fitted_parameters, history, limit=7):
    """Return weighted anatomy hypotheses; never inspect hidden case labels."""
    weights = np.asarray(scene.gall_weights(state), dtype=float).reshape(-1)
    params = np.asarray(scene.parameters, dtype=float)
    if len(weights) != len(params) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("invalid public gallbladder posterior")
    weights = np.maximum(weights, 0.)
    weights /= weights.sum()
    # gall_weights(state) is already the live public posterior. Do not apply
    # the same image packets again here (which would count native evidence
    # repeatedly at each station).
    # Retain distinct high-mass modes, then group the tail into a few explicit
    # representative modes. This bounds exact geometry work while preserving
    # posterior multimodality.
    ranked = np.argsort(-weights)
    chosen = []
    for i in ranked:
        if all(np.linalg.norm((params[i] - params[j]) / np.array([10., 10., 10., 18., 15., .20, .22])) > .55
               for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= limit:
            break
    if not chosen:
        chosen = [int(ranked[0])]
    # Each retained mode represents nearby omitted posterior mass using a
    # weighted barycenter. Keep the highest-weight member as the geometry
    # representative to avoid smoothing across separated modes.
    assignments = np.argmin(np.stack([
        np.sum(((params - params[j]) / np.array([10., 10., 10., 18., 15., .20, .22]))**2, axis=1)
        for j in chosen
    ]), axis=0)
    masses = np.array([weights[assignments == k].sum() for k in range(len(chosen))])
    masses /= masses.sum()
    return [(scene.body.anatomy(params[j]), float(masses[k]), params[j].copy())
            for k, j in enumerate(chosen) if masses[k] > 1.e-5]


def _history_shadow_evidence(history):
    """Describe observed image availability, without inferring absence/shadow."""
    visible, empty, marked = [], [], []
    for packet in history:
        reading = np.asarray(packet.reading)
        if reading.size < 3 or not np.isfinite(reading[0, 0]):
            continue
        area, center = float(reading[0, 0]), float(reading[0, 1])
        nmark = int(np.isfinite(np.asarray(packet.marks)[:, 0]).sum())
        if area >= 45.:
            visible.append((area, center))
        elif area >= 0.:
            empty.append((area, center))
        if nmark:
            marked.append(nmark)
    return {"visible_frames": len(visible), "low_area_frames": len(empty),
            "marked_frames": len(marked)}


def propose_multimodal(scene, state, search, fitted_parameters, history, visited,
                       stations, frames_left):
    """Propose a reachable stationary pose robust to discrete public posterior modes.

    Return ``(q, plan, predicted, diagnostics)``. ``predicted`` follows
    ``acquire.propose_station`` and is the metrics value for the dominant mode.
    No part of this proposer reads scene.truth, scene.case, or offline outputs.
    """
    modes = _posterior_modes(scene, state, fitted_parameters, history)
    if not modes or frames_left < 4:
        return None
    # Keep high-mass posterior modes for precise evaluation, while generating
    # seeds from each retained mode. The fitted continuous estimate and
    # successful public-history views remain independent candidate families.
    exact_order = np.argsort([-w for _, w, _ in modes])[:4]
    exact_modes = [modes[int(k)] for k in exact_order]
    family_seeds = []
    for k, (anatomy, _, _) in enumerate(modes):
        family_seeds.extend((f"mode{k}", q) for q in
                            search.proposals(anatomy, current=state.q, limit=13))
    fitted = None
    if fitted_parameters is not None:
        fp = np.asarray(fitted_parameters, dtype=float)
        if fp.shape == np.asarray(scene.parameters[0]).shape and np.isfinite(fp).all():
            fitted = scene.body.anatomy(fp)
            family_seeds.extend(("fitted", q) for q in search.proposals(fitted, current=state.q, limit=16))
    # Preserve useful historic image locations as real candidate poses.
    try:
        from acquire import revisit_candidates
        family_seeds.extend(("history", np.asarray(q)) for q in revisit_candidates(history))
    except ImportError:
        pass
    family_seeds.extend(("local", q) for q in offsets_around(state.q))
    unique, family_map = {}, {}
    for family, q in family_seeds:
        q = np.clip(np.asarray(q, dtype=float), CONTACT[0], CONTACT[1])
        key = tuple(np.round(q, 5))
        if key not in unique:
            unique[key] = q
            family_map[key] = set()
        family_map[key].add(family)
    poses = np.asarray(list(unique.values()))
    if visited:
        prev = np.asarray(visited, dtype=float)
        d = np.linalg.norm((poses[:, None] - prev[None]) * POSE_DISTANCE_SCALE, axis=2)
        repetition = np.exp(-.5 * (d / 9.)**2).sum(axis=1)
    else:
        repetition = np.zeros(len(poses))
    surface0 = scene.body.surface(state.q)
    surface_distance = np.linalg.norm(scene.body.surface(poses) - surface0, axis=1)
    motion_distance = np.linalg.norm((poses - state.q) * POSE_DISTANCE_SCALE, axis=1)
    budget = (surface_distance / 49. * 19. < frames_left) & (motion_distance >= 1.)
    ids = np.flatnonzero(budget & (repetition < 2.0))
    if not len(ids):
        return None

    # scene.project is the cheap vector public forecast (quality/support for
    # all posterior particles). Use it to rank candidate support before exact
    # mesh sections, with explicit reservations for each family.
    posterior = np.asarray(scene.gall_weights(state), dtype=float).reshape(-1)
    posterior = np.maximum(posterior, 0.)
    posterior /= posterior.sum()
    quick = {}
    for i in ids:
        _, support, quality = scene.project(poses[i])
        support = np.asarray(support, dtype=float)
        quality = np.asarray(quality, dtype=float)
        quick[int(i)] = float(quality @ posterior + .05 * support.mean())
    candidate_order = sorted(ids, key=lambda i: quick[int(i)] - .012 * surface_distance[i]
                             - .005 * motion_distance[i] - .9 * repetition[i], reverse=True)
    reserved = []
    for family in [f"mode{k}" for k in range(len(modes))] + ["fitted", "history"]:
        family_ids = [int(i) for i in candidate_order if family in family_map[tuple(np.round(poses[i], 5))]]
        reserved.extend(family_ids[:2 if family.startswith("mode") else 3])
    ids = list(dict.fromkeys(reserved + [int(i) for i in candidate_order]))[:18]
    exact_anatomies = [a for a, _, _ in exact_modes]
    exact_weights = np.array([w for _, w, _ in exact_modes], dtype=float)
    if fitted is not None:
        exact_anatomies.append(fitted)
        exact_weights = np.r_[exact_weights * .9, .1]
    exact_weights /= exact_weights.sum()
    scores = np.zeros((len(ids), len(exact_anatomies)), dtype=float)
    strict = np.zeros_like(scores)
    metric_cache = {}
    for ii, i in enumerate(ids):
        q = poses[i]
        for k, anatomy in enumerate(exact_anatomies):
            metric = measure_view(scene.body, q, anatomy, np.zeros(2))
            metric_cache[(ii, k)] = metric
            scores[ii, k] = metric.score
            strict[ii, k] = float(metric.qualifies)
    # Weight strict success and score by posterior mass. A mild lower-tail
    # term protects plausible minority modes without giving tiny modes equal
    # influence. The fitted estimate receives a bounded 10% contribution.
    expected = scores @ exact_weights
    strict_mass = strict @ exact_weights
    lower = np.array([scores[i, np.argsort(scores[i])[0]] for i in range(len(ids))])
    rankings = expected + .7 * strict_mass + .10 * lower - .012 * surface_distance[ids] - .005 * motion_distance[ids] - .9 * repetition[ids]
    shadow_diag = _history_shadow_evidence(history)
    order = np.argsort(-rankings)
    for oi in order:
        i = ids[int(oi)]
        q = poses[i]
        plan = settled_plan(scene.motor, state.q, q, frames_left)
        if plan is None:
            continue
        dominant = int(np.argmax(exact_weights))
        predicted = metric_cache[(int(oi), dominant)]
        diag = {"mode_count": len(modes), "mode_masses": [float(w) for _, w, _ in modes],
                "expected_score": float(expected[oi]), "lowest_mode_score": float(lower[oi]),
                "mode_scores": scores[oi].tolist(), "strict_mass": float(strict_mass[oi]),
                "candidate_family_count": len(family_map[tuple(np.round(q, 5))]), **shadow_diag}
        return q, plan, predicted, diag
    return None
