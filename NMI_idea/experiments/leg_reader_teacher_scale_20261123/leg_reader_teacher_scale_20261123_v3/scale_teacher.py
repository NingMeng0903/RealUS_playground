"""Action programs evaluated on real continuations with a frozen public reader."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.special import expit, softmax

from scale_worlds import (ARM, FEATURES, FUTURE_SEEDS, complete, init_worlds, motion, private_outcomes,
                          start, vectors, windows_visited)
from route_metrics import outcomes
from environment import FEATURE_NAMES as ANCHOR_FEATURES, station_bins


ACTIONS = ("continue", "equal_time_hold", "window_left", "window_right",
           "revisit_left", "revisit_center", "revisit_right")
TARGETS = ("utility", "strict", "wrong_branch", "wrong_delivery", "correct_regions", "correct_mm",
           "old_correct", "old_wrong")
ACTION_FEATURES = (*ANCHOR_FEATURES, "route_is_full", "seam_offset_stations",
                   *(f"region_{r}_{k}" for r in range(6) for k in
                     ("top", "runner_up", "threshold_margin", "views", "different_view_top")),
                   *(f"action_{a}" for a in ACTIONS), "goal_axial_change", "goal_angle_change",
                   "requested_shape", "review_slot", "planned_travel", "applied_plan_fraction",
                   "forecast_detection", "forecast_separation", "forecast_sd", "forecast_shadow",
                   "forecast_in_window", "forecast_merge", "forecast_gain")
TAPES = {"teacher_fit": ("fit_0", "fit_1"), "teacher_calibration": ("actual",),
         "teacher_test": ("actual", "proof_0", "proof_1", "search_0", "search_1")}


def cpu_bundle(bundle):
    """Serialize an inference-only bundle for CPU rollout workers."""
    return {"model": {name: tensor.detach().cpu().numpy() for name, tensor in bundle["model"].state_dict().items()},
            "center": np.asarray(bundle["center"], dtype=np.float32),
            "scale": np.asarray(bundle["scale"], dtype=np.float32)}


def probabilities(features, bundle):
    x = np.clip((features - bundle["center"]) / bundle["scale"], -9., 9.).astype(np.float32, copy=False)
    model = bundle["model"]
    for layer in ("features.0", "features.2"):
        x = x @ model[f"{layer}.weight"].T + model[f"{layer}.bias"]
        x *= expit(x)
    logits = x @ model["identity.weight"].T + model["identity.bias"]
    return softmax(logits, axis=1)[:, 0]


def probe_menu(env, spec, node, limits, reference):
    caps = limits.probe(node, spec)
    commands = [(0, 0., 0.), (0, 0., 0.), (1, -.80, 0.), (1, .80, 0.),
                (2, -.72, .50), (2, 0., .50), (2, .72, .50)]
    plans = {}
    for i, (skill, turn, axial) in enumerate(commands[2:], 2):
        if skill == 2:
            axial = float(np.clip((node.station - reference) / motion.AXIAL_RANGE, .30, 1.))
        controls = np.array([turn, axial, -.24 * turn])
        plan = env.plan(node, spec, skill, controls, caps, ARM)
        if plan is None:
            continue
        ds, da = plan.goal - plan.start
        if skill == 1 and abs(da) < .75:
            continue
        if skill == 2 and (abs(ds) < .55 or (abs(node.station - reference) <= 2 and abs(plan.goal[0] - reference) > .36)):
            continue
        plans[i] = plan
    slot = max((plan.ticks for plan in plans.values()), default=min(8, caps[0]))
    return plans, slot, caps


def review(env, spec, node, trial, plans, action, slot, caps):
    current = node.fork()
    if action in plans:
        result = env.material.run_probe(current, spec, trial, plans[action], *caps, use_dird=False, copy_node=False)
        if not result.legal:
            raise RuntimeError(f"Checked review failed in execution: {result.reason}")
        current = result.node
    if action:
        while current.tick < node.tick + slot:
            current.tick += 1
            env.scanner._observe(current, spec, trial, current.station, current.sector, current.tick)
        if (current.tick, current.station, current.sector) != (node.tick + slot, node.station, node.sector):
            raise RuntimeError("Review arms must enter continuation at the same pose and time")
    return current


def prefix_summary(obs, p, threshold, node, spec):
    summary = []
    for region in range(6):
        indices = np.flatnonzero(obs["regions"] == region)
        order = indices[np.argsort(-p[indices], kind="stable")]
        if not len(order):
            summary.extend([0.] * 5)
            continue
        frames = [node.reader.frames[int(f)] for f in obs["frames"][order]]
        s, a = frames[0][:2]
        views = {(round(f[0]), round(f[1])) for f in frames}
        alternative = next((i for i, frame in zip(order, frames)
                            if abs(frame[0] - s) >= .65 or abs(motion._sector_delta(a, frame[1])) >= .75), None)
        summary.extend((p[order[0]], p[order[1]] if len(order) > 1 else 0., p[order[0]] - threshold,
                        min(1., len(views) / max(3., (spec[2] - spec[1]) / 6.)),
                        p[alternative] if alternative is not None else 0.))
    return summary


def visible_prefix(env, spec, node, limits, plans, slot, caps, seam, full, obs, p, threshold):
    loc = node.station, node.sector
    w = node.reader.w
    current = env.model.feature(*loc, w)
    public = env.features(node, spec, limits, caps, ARM)
    shared = np.r_[public, full, np.clip((node.station - seam) / 12., -3., 3.),
                   prefix_summary(obs, p, threshold, node, spec)]
    mask = np.asarray([i < 2 or i in plans for i in range(len(ACTIONS))], dtype=bool)
    rows = np.zeros((len(ACTIONS), len(ACTION_FEATURES)), dtype=np.float32)
    for i in np.flatnonzero(mask):
        plan = plans.get(i)
        goal = loc if plan is None else plan.goal
        delta = np.asarray(goal) - loc
        detection = env.model.feature(*goal, w)
        one_hot = np.eye(len(ACTIONS), dtype=np.float32)[i]
        p_travel = (0. if plan is None else
                    sum(np.linalg.norm(np.diff(env.scanner.surface.path_world(seg[0][:, 0], seg[0][:, 1]),
                                               axis=0), axis=1).sum() for seg in plan.segments))
        facts = [*delta, 0. if plan is None else plan.controls[2], 0. if i == 0 else slot / 30.,
                 p_travel / 86., 0. if plan is None else plan.applied_fraction]
        gain = detection[0] - current[0] + .19 * (detection[1] - current[1]) - .04 * p_travel / 86.
        rows[i] = np.r_[shared, one_hot, facts, detection, gain]
    if not np.isfinite(rows).all():
        raise RuntimeError("Invalid public action features")
    return rows, mask


def select_old(scores, obs, labels, count, region, threshold):
    indices = np.flatnonzero((obs["frames"] < count) & (obs["regions"] == region))
    if not len(indices):
        return -99
    candidate = indices[np.argmax(scores[indices])]
    return int(labels[candidate]) if scores[candidate] >= threshold else -99


def grade(env, spec, current, bundle, threshold, old_count, old_region, *, terminal):
    obs = vectors(env, spec, current)
    p = probabilities(obs["features"], bundle)
    private = private_outcomes(current, obs)
    labels = private["labels"]
    old = select_old(p, obs, labels, old_count, old_region, threshold)
    if not terminal:
        return old
    delivery = outcomes({"label": labels, "region": obs["regions"], "complete": current.station == spec[2]}, p, threshold)
    # Deduplicate coverage in material station bins. Within each station,
    # the deployed reader's top acquired candidate is eligible for delivery.
    nearest = np.clip(np.rint(private["stations"]).astype(int), spec[1], spec[2]) - spec[1]
    best = np.full(spec[2] - spec[1] + 1, -np.inf)
    np.maximum.at(best, nearest, p)
    winners = np.flatnonzero((p >= threshold) & (p == best[nearest]))
    bins, firsts = np.unique(nearest[winners], return_index=True)
    widths = station_bins(spec, env.scanner.surface.y)
    correct_mm = float(widths[bins[labels[winners[firsts]] == 0]].sum())
    old_wrong = old not in (-99, 0)
    utility = (3. * delivery["strict"] + .38 * delivery["correct_regions"] + .002 * correct_mm
               - .66 * delivery["wrong_region_count"] - .22 * delivery["wrong_branch_regions"]
               + .26 * (old == 0) - .32 * old_wrong)
    targets = (utility, delivery["strict"], delivery["wrong_branch"], delivery["wrong_delivery"],
               delivery["correct_regions"], correct_mm, old == 0, old_wrong)
    detail = np.stack((delivery["selected_ids"], delivery["capture"]), axis=-1).astype(np.int8)
    return targets, detail, old


_BUNDLE = None
_THRESHOLDS = None


def init_teacher(pack, bundle, thresholds):
    global _BUNDLE, _THRESHOLDS
    init_worlds(pack)
    _BUNDLE, _THRESHOLDS = bundle, thresholds


def run_prefix(case):
    env, spec, node, trial, limits = start(case)
    anchor_mm = -750. if case.task == "femoral" else -910.
    seam = int(np.argmin(abs(env.scanner.stations - anchor_mm)))
    # Alternate clinically relevant seam checks, earlier coverage decisions,
    # and late decisions. Timing is determined by the fixed schedule, not labels.
    onset = (min(spec[2] - 1, seam + 1),
             spec[1] + (spec[2] - spec[1]) // 3,
             spec[1] + 4 * (spec[2] - spec[1]) // 5)[case.band % 3]
    complete(env, spec, node, trial, limits, onset)
    reference = seam if onset == seam + 1 else onset - 1
    plans, slot, caps = probe_menu(env, spec, node, limits, reference)
    threshold = _THRESHOLDS[case.task]
    obs = vectors(env, spec, node)
    p = probabilities(obs["features"], _BUNDLE)
    features, valid = visible_prefix(env, spec, node, limits, plans, slot, caps, seam,
                                     case.task != "femoral", obs, p, threshold)
    labels = private_outcomes(node, obs)["labels"]
    old_count = len(node.reader.frames)
    old_region = motion.region_of(spec, reference)
    old_before = select_old(p, obs, labels, old_count, old_region, threshold)
    streams = []
    tape_numbers = {f"{kind}_{i}": offset for kind, offsets in FUTURE_SEEDS.items() for i, offset in enumerate(offsets)}
    for name in TAPES[case.split]:
        streams.append(trial if name == "actual" else trial.branch(node.tick, case.tape_seed + tape_numbers[name]))
    count = len(streams)
    values = np.zeros((len(ACTIONS), count, len(TARGETS)), dtype=np.float32)
    detail = np.zeros((len(ACTIONS), count, 6, 2), dtype=np.int8)
    # Ticks, travel, earlier-region decision right after review, final old image,
    # and actual exposure to the configured ambiguous and alternate windows.
    extra = np.zeros((len(ACTIONS), count, 6), dtype=np.float32)
    for j, tape in enumerate(streams):
        if tape.layout is not trial.layout or tape.window is not trial.window:
            raise RuntimeError("A future perturbation changed the frozen layout or acoustic window")
        for a in np.flatnonzero(valid):
            current = review(env, spec, node, tape, plans, a, slot, caps)
            if a and j == 0 and case.split != "teacher_fit":
                after = grade(env, spec, current, _BUNDLE, threshold, old_count, old_region, terminal=False)
            else:
                after = old_before
            complete(env, spec, current, tape, limits)
            values[a, j], detail[a, j], old = grade(env, spec, current, _BUNDLE, threshold,
                                                      old_count, old_region, terminal=True)
            extra[a, j] = (current.tick, current.distance, after, old, *windows_visited(current, trial))
    metadata = {"case": asdict(case), "layout_id": trial.layout.identifier, "prefix_station": onset,
                "prefix_sector": int(node.sector), "anchor_station": int(reference), "old_region": old_region,
                "old_before": old_before, "review_ticks": slot, "complete": True,
                "window_generated": trial.window is not None,
                "candidate_frames": len(obs["frames"])}
    return {"features": features, "valid": valid, "outcomes": values, "detail": detail, "extra": extra,
            "route": np.asarray(case.task)}, metadata


def pack_prefixes(rows):
    return {key: np.stack([data[key] for data, _ in rows]) for key in rows[0][0]}
