"""Paired physical execution with pinned v5 and causal v6 reader-state arms."""

from dataclasses import asdict

from v6_data import init_worlds
from anchor_experiment import alternatives as time_alternatives, short_review as v5_short_review
from anchor_policy import Session as V5Session, score_choices as v5_choices, view_predictions
from active_trial import ReaderContactMotion
from forward_anchor import find_anchor, mpc_candidates, step_slot
from readout_state import confidence as v5_confidence, evaluate, measured_length
from scale_worlds import FUTURE_SEEDS
from study_data import ARM, complete, held_frames, motion, scene_links, start
from study_worlds import material_widths, truth_at
from anchor_logic import beliefs, completion, rank
from causal_reader import PrefixReader, likelihood, observation_features
import numpy as np


PROGRAMS = ("mpc", "equal_time_hold", "v5_public", "prefix_frozen", "unary_dynamic", "objects_dynamic",
            "objects_no_stop", "objects_stop")
TEACH_ACTIONS = ("mpc", "equal_time_hold", "v5_public", "objects_dynamic", "objects_stop")
REFERENCE = "v5_public"
SCORE_STREAMS = ("actual", "proof_0", "proof_1")
SEARCH_STREAM_OFFSETS = (*FUTURE_SEEDS["search"], 7_000_000_000_000, 8_000_000_000_000)
_READERS, _VALUE = None, None


def init_live(atlas, readers, value=None):
    global _READERS, _VALUE
    init_worlds(atlas)
    _READERS, _VALUE = readers, value


def old_candidate(reader, node, origin, mode):
    public, states = reader.update(node)
    pool = public["frames"] < origin["count"]
    score = reader.confidence(node, mode, mask=pool)
    use = np.flatnonzero(pool & (public["regions"] == origin["region"]) & (score >= 0.))
    if not len(use):
        return None
    i = int(use[np.argmax(score[use])]); cutoff = reader.reader["calibration"][mode]["threshold"]
    return {"index": i, "id": [int(public["frames"][i]), int(public["slots"][i])],
            "confidence": float(score[i]), "submitted": bool(score[i] >= cutoff)}


def final_result(reader, node, tape, origin, limits):
    public, states = reader.update(node)
    mode = reader.reader["delivery"]
    calibration = reader.reader["calibration"][mode]
    truth = truth_at(node, public, tape)["labels"]
    cap = node.station == reader.spec[2] and node.tick <= limits.ticks and node.distance <= limits.mm + 1.e-6
    if not cap:
        raise RuntimeError("Physical continuation did not legally complete")
    score = reader.confidence(node, mode)
    final = evaluate(public, truth, score, calibration["threshold"], complete=cap)
    old = old_candidate(reader, node, origin, mode)
    if old is not None:
        old["label"] = int(truth[old["index"]])
    old_correct = bool(old is not None and old["label"] == 0 and old["submitted"])
    keep = (public["frames"] >= origin["count"]) & (public["stations"] > origin["station"] + .24)
    suffix_score = reader.confidence(node, mode, mask=keep)
    suffix = evaluate(public, truth, suffix_score, calibration["threshold"], complete=cap)
    regions = list(range(origin["region"] + 1, 6))
    suffix_all = all(suffix["region_outcomes"][r] == 1 and final["region_outcomes"][r] == 1 for r in regions)
    joint = old_correct and suffix_all and cap
    p = states[mode].get("probabilities")
    if p is None:
        p = likelihood(observation_features(public, states[mode], object_role=mode == "objects"), calibration["joint"])[:, None]
    mm, longest = measured_length(public, {"probabilities": p}, truth, .75, material_widths(reader.env, reader.spec))
    utility = 3. * final["strict"] + .9 * joint + .36 * final["correct_regions"] - .80 * final["wrong_branch_regions"] \
        - .45 * final["artifact_regions"] + .0015 * mm
    return final | {"completed_legal": bool(cap), "old_pool_correct": old_correct, "old_pool": old,
                    "suffix_required": regions, "suffix_valid": suffix_all, "suffix_forward_delivery": suffix["region_outcomes"],
                    "joint": bool(joint), "mm": mm, "longest_mm": longest, "time_s": node.tick * motion.CLOCK_S,
                    "travel_mm": float(node.distance), "utility": float(utility), "terminal_clock": int(node.tick),
                    "layout_id": tape.layout.identifier}


def decision_belief(public, state, station, mode, initial):
    if mode == "prefix_frozen":
        return initial
    if mode == "unary_dynamic":
        n = len(public["frames"])
        state = state | {"object_role": state["local_role"], "role": state["local_role"], "track_ids": np.arange(n),
                         "support": np.zeros(n), "view_counts": np.ones(n)}
    return beliefs(public, state, station)


def rank_live(env, node, reader, mode, initial, sectors, legal, base):
    public, states = reader.update(node)
    hypothesis = decision_belief(public, states["objects"], node.station, mode, initial)
    if hypothesis is None:
        return base, {"chosen": int(np.argmax(base)), "mpc": int(np.argmax(base)), "two_hypotheses": False, "voi": 0.}
    detection, separate = view_predictions(env.scanner.surface, int(node.station) + 1, sectors, hypothesis["points"])
    result, info = rank(base, legal, detection, separate, hypothesis["weights"])
    control, _ = rank(base, legal, detection, separate, hypothesis["unary_weights"])
    slot, unary = int(np.argmax(result)), int(np.argmax(control))
    return result, {"chosen": slot, "same_geometry_unary": unary, "mpc": int(np.argmax(base)), "two_hypotheses": True,
                    "object_ids": hypothesis["ids"], "identity_weights": hypothesis["weights"].tolist(),
                    "unary_weights": hypothesis["unary_weights"].tolist(), "voi": float(np.max(info)),
                    "passive_candidate_slots": sectors.tolist(), "chosen_sector": int(sectors[slot]),
                    "view_scores": [float(x) if np.isfinite(x) else None for x in result]}


def target_station(ctx, node, reader, nominal):
    """Public target search around the atlas seam using predicted distinguishability."""
    env, spec, start_s = ctx["env"], ctx["spec"], int(node.station)
    links = scene_links(env)
    public, states = reader.update(node)
    options = []
    for end in range(start_s + 3, min(start_s + 7, nominal.station + 2, spec[2] - 1) + 1):
        if not np.any(np.all(links[:, start_s:end, 0], axis=1)):
            continue
        forecast = beliefs(public, states["objects"], start_s, step_ahead=end - start_s)
        if forecast is None:
            continue
        sectors = np.array([(int(node.sector) + i) % 24 for i in (0, -1, 1, -2, 2)])
        allow = np.array([env.scanner.surface.legal(float(end), int(a)) for a in sectors], dtype=bool)
        if not allow.any():
            continue
        reserve, _ = env.route.reserve(start_s, end)
        tick = node.tick + reserve[0, int(node.sector)]
        if tick >= ctx["limits"].ticks:
            continue
        detection, sep = view_predictions(env.scanner.surface, end, sectors, forecast["points"])
        _, voi = rank(np.zeros(len(sectors)), allow, detection, sep, forecast["weights"])
        evidence = float(np.max(voi + .12 * np.where(allow, sep, 0.)))
        options.append({"station": end, "public_evidence": evidence, "forecast_tick": int(tick),
                        "value": float(evidence - .018 * abs(end - nominal.station))})
    selected = max(options, key=lambda x: x["value"])["station"] if options else nominal.station
    return selected, options


def observe_completion(reader, node, origin, info):
    item = old_candidate(reader, node, origin, "objects")
    if item is None:
        return {"stop": False, "reason": "no_old_object_to_confirm", "new_views": 0}
    public, states = reader.update(node); state = states["objects"]; ix = item["index"]
    # Object role is conditional on physical evidence. Delivery readiness is a
    # separate calibrated joint event, and the lookahead only uses the public forecast.
    done = completion(public, state, int(state["track_ids"][ix]), origin["count"], origin["station"],
                      node.station, float(state["object_role"][ix, 0]), item["submitted"], float(info))
    return done | {"old_candidate_id": item["id"], "old_delivery_probability": item["confidence"]}


def review_in_place(ctx, node, tape, reader, *, allow_stop):
    env, spec, limits = (ctx[k] for k in ("env", "spec", "limits"))
    caps = limits.probe(node, spec)
    for turn in (-.63, .63):
        plan = env.plan(node, spec, 2, np.array([turn, .55, -.12 * turn]), caps, ARM)
        if plan is None or abs(plan.goal[0] - node.station) < .55:
            continue
        if not isinstance(env.material, ReaderContactMotion):
            env.material = ReaderContactMotion(env.scanner)

        def stop(probe, _exit):
            # Compare the foreseeable next public re-observation under the
            # current acoustic view at the actual outbound sample.
            public, states = reader.update(probe)
            b = beliefs(public, states["objects"], probe.station, step_ahead=0.)
            if b is None:
                return False
            d, s = view_predictions(env.scanner.surface, int(round(probe.station)), [int(round(probe.sector)) % 24], b["points"])
            _, voi = rank(np.zeros(1), np.ones(1, dtype=bool), d, s, b["weights"])
            return observe_completion(reader, probe, ctx["origin"], float(np.max(voi)))["stop"]

        env.material._feedback_hook = lambda frame: reader.feedback(node, frame)
        try:
            result = env.material.run_probe(node, spec, tape, plan, *caps, stop=stop if allow_stop else None,
                                            use_dird=False, copy_node=False)
        finally:
            env.material._feedback_hook = None
        if not result.legal:
            raise RuntimeError(f"Infeasible admitted review {result.reason}")
        return result.node, {"ticks": result.ticks, "early": bool(result.stopped), "distance_mm": result.distance,
                             "goal": plan.goal.tolist(), "path": result.trajectory.tolist()}
    return node, None


def execute(ctx, tape, mode):
    env, spec, limits, nominal, origin = (ctx[k] for k in ("env", "spec", "limits", "anchor", "origin"))
    node = ctx["node"].fork(); reader = PrefixReader(env, spec, ctx["reader"])
    active = mode not in ("mpc", "equal_time_hold")
    real_stop = mode == "objects_stop"
    dynamic_landmark = mode in ("objects_no_stop", "objects_stop")
    matched = mode != "mpc" and not dynamic_landmark
    use_v5 = mode == "v5_public"
    old = V5Session(env, spec, ctx["reader"]["v5"]) if use_v5 else None
    end, options = target_station(ctx, node, reader, nominal) if dynamic_landmark else (nominal.station, [])
    if mode == "equal_time_hold":
        held_frames(env, spec, node, tape, 7)
    prior = ctx["initial"]
    trace, path, stop_info, reviewed = [], [], None, None
    v5_early = False
    while node.station < end:
        sectors, legal, base = (time_alternatives(env, spec, node, limits, nominal, ctx["clock"], True)
                                 if matched else mpc_candidates(env, spec, node, limits))
        current = base
        before = int(node.station)
        if active and use_v5 and not v5_early:
            current, info = v5_choices(old, node, sectors, legal, base, "anchor_information", len(trace))
            relative = old.item(ctx["v5_origin"]["witness"])
            p = None if relative is None else float(old.state["probabilities"][relative, 0]); q = ctx["v5_origin"]["witness_probability"]
            if node.station - origin["station"] >= 2 and p is not None and q is not None and p >= .89 and p - q >= .10:
                v5_early, current = True, base
            info = info | {"chosen": int(np.argmax(current)), "station": before, "tick": int(node.tick)}
            trace.append(info)
        elif active and not use_v5:
            current, info = rank_live(env, node, reader, mode, prior, sectors, legal, base)
            if dynamic_landmark:
                done = observe_completion(reader, node, origin, info["voi"])
                info["completion"] = done
                if done["stop"] and real_stop:
                    stop_info = done | {"station": before, "tick": int(node.tick), "sector": int(node.sector)}
                    trace.append(info | {"station": before, "tick": int(node.tick), "executed": False})
                    break
            trace.append(info | {"station": before, "tick": int(node.tick), "executed": True})
        _, step = step_slot(env, spec, node, tape, limits, int(np.argmax(current)))
        path.append([int(node.station), int(node.sector), int(node.tick), float(step["distance_mm"])])
        if use_v5:
            old.update(node)
    if use_v5 and not v5_early:
        node, reviewed = v5_short_review(env, spec, node, tape, limits, ctx["clock"], old)
    if dynamic_landmark and stop_info is None and node.station < spec[2]:
        sectors, legal, base = mpc_candidates(env, spec, node, limits)
        _, final_info = rank_live(env, node, reader, mode, prior, sectors, legal, base)
        endpoint_evidence = observe_completion(reader, node, origin, final_info["voi"])
        if not endpoint_evidence["stop"]:
            node, reviewed = review_in_place(ctx, node, tape, reader, allow_stop=real_stop)
    else:
        endpoint_evidence = None
    if matched:
        if node.station != nominal.station or node.tick > ctx["clock"] or (
                ctx["clock"] + limits.reserve_ticks[nominal.station-spec[1], int(node.sector)] > limits.ticks):
            raise RuntimeError("Programme did not reach the shared comparison boundary")
        held_frames(env, spec, node, tape, ctx["clock"] - node.tick)
    at_exit = [int(node.station), int(node.sector), int(node.tick)]
    after = old_candidate(reader, node, origin, reader.reader["delivery"])
    if after is not None:
        after["label"] = int(truth_at(node, reader.public, tape)["labels"][after["index"]])
    rejoin = []
    if dynamic_landmark:
        # Sparse identity-aware re-acquisition immediately after the actual exit.
        for _ in range(min(2, spec[2] - int(node.station))):
            sectors, legal, base = mpc_candidates(env, spec, node, limits)
            guided, info = rank_live(env, node, reader, mode, prior, sectors, legal, base)
            rejoin.append(info | {"station": int(node.station), "tick": int(node.tick)})
            step_slot(env, spec, node, tape, limits, int(np.argmax(guided)))
    complete(env, spec, node, tape, limits)
    return final_result(reader, node, tape, origin, limits) | {
        "program": mode, "at_exit": at_exit, "forward_path": path, "trace": trace, "rejoin_trace": rejoin,
        "requested_target_station": end, "public_target_options": options, "after_review": reviewed,
        "old_at_exit": after, "completion_stop": stop_info, "endpoint_evidence": endpoint_evidence}


def prefix_setup(case):
    env, spec, node, tape, limits = start(case)
    anchor = find_anchor(env, spec, route=case.task, prefix_stations=3 + case.band % 4)
    complete(env, spec, node, tape, limits, anchor.prefix_station)
    cut = len(node.reader.frames)
    origin = {"count": cut, "station": int(node.station), "region": int(motion.region_of(spec, node.station))}
    reserve, _ = env.route.reserve(anchor.prefix_station, anchor.station)
    clock = int(node.tick + reserve[0, int(node.sector)] + 28)
    reader = _READERS[case.task]
    original = V5Session(env, spec, reader["v5"])
    pub, old = original.update(node)
    v5_conf = v5_confidence(pub, old, reader["v5"]["risk"])
    pool = np.flatnonzero(pub["regions"] == origin["region"])
    i = int(pool[np.argmax(v5_conf[pool])]) if len(pool) else None
    old_origin = origin | {"witness": [int(pub["frames"][i]), int(pub["slots"][i])] if i is not None else None,
                           "witness_probability": float(old["probabilities"][i, 0]) if i is not None else None}
    sectors, legal, base = time_alternatives(env, spec, node, limits, anchor, clock, True)
    _, forecast = v5_choices(original, node, sectors, legal, base, "anchor_information", 0)
    trigger = (forecast["hypotheses"] == 2 and len(set(forecast["hypothesis_preference_slots"])) == 2 and
               max(forecast["discrimination"]) > .10)
    live = PrefixReader(env, spec, reader)
    public, states = live.update(node)
    b = beliefs(public, states["objects"], node.station)
    p = states["objects"]; mask = public["regions"] == origin["region"]
    newest = public["frames"] >= cut - 4
    _, info = rank_live(env, node, live, "objects_dynamic", b, sectors, legal, base)
    gap = float(abs(np.diff(b["weights"])[0])) if b is not None else 0.
    stat = [(node.station - spec[1]) / max(1., spec[2]-spec[1]), (anchor.station-node.station)/7.,
            node.tick / limits.ticks, node.distance / limits.mm, float(trigger), np.max(p["role"][mask,0], initial=0.),
            np.max(p["usable"][newest], initial=0.), float(mask.sum()) / 120., float(newest.sum()) / 24.,
            gap, float(info["voi"]), float(np.mean(legal)), float(np.max(p["support"][mask], initial=0.)) / 4.,
            float(spec[2] - spec[1]) / 150.]
    return {"case": case, "env": env, "spec": spec, "node": node, "tape": tape, "limits": limits,
            "anchor": anchor, "origin": origin, "v5_origin": old_origin, "initial": b, "clock": clock,
            "reader": reader, "trigger": bool(trigger), "features": [float(x) for x in stat]}


def run_programs(ctx, tape, requested):
    by_action = {}
    for name in requested:
        physical = name if ctx["trigger"] or name == "equal_time_hold" else "mpc"
        if physical not in by_action:
            by_action[physical] = execute(ctx, tape, physical)
        by_action[name] = by_action[physical]
    return {name: by_action[name] for name in requested}


def teacher_search(ctx):
    from paired_value import recheck, select_teacher
    case, tape, node = (ctx[k] for k in ("case", "tape", "node"))
    paired_samples = {name: [] for name in TEACH_ACTIONS}
    shorts = []

    def evaluate_search(actions, offset):
        disturbed = tape.branch(node.tick, case.tape_seed + offset)
        if disturbed.layout is not tape.layout:
            raise RuntimeError("Paired future modified anatomy")
        values = run_programs(ctx, disturbed, actions)
        v = values[REFERENCE]
        for name, current in values.items():
            paired_samples[name].append([current["utility"] - v["utility"], int(current["strict"]) - int(v["strict"]),
                                         int(current["wrong_branch"]) - int(v["wrong_branch"])])

    for seed in SEARCH_STREAM_OFFSETS[:2]:
        evaluate_search(TEACH_ACTIONS, seed)
    if ctx["trigger"]:
        shorts = recheck(paired_samples, REFERENCE)
        if len(shorts) > 1:
            for seed in SEARCH_STREAM_OFFSETS[2:]:
                evaluate_search(shorts, seed)
    choice, trace = select_teacher(paired_samples, REFERENCE)
    return choice, paired_samples, trace, shorts


def case_rollout(case):
    from paired_value import student_action
    ctx = prefix_setup(case)
    tape, node = ctx["tape"], ctx["node"]
    tune = case.split == "closed_dev"
    ordinal = int(case.identifier.rsplit("_", 1)[1]); teacher_test = (ordinal // 2) % 3 == 0
    if tune or teacher_test:
        preferred, samples, estimate, shortlist = teacher_search(ctx)
    else:
        preferred, samples, estimate, shortlist = None, None, None, None
    choice = student_action(ctx["features"], _VALUE, REFERENCE) if _VALUE is not None else REFERENCE
    outcomes = {}
    if not tune:
        for name, salt in zip(SCORE_STREAMS, (None, *FUTURE_SEEDS["proof"]), strict=True):
            future = tape if salt is None else tape.branch(node.tick, case.tape_seed + salt)
            if future.layout is not tape.layout:
                raise RuntimeError("Student or proof future modified anatomy")
            # The student commands the actual episode first. The remaining
            # action arms run as same-prefix branches on the frozen tape.
            ordering = (choice, *(arm for arm in PROGRAMS if arm != choice)) if name == "actual" else PROGRAMS
            completed = run_programs(ctx, future, ordering)
            outcomes[name] = {arm: completed[arm] for arm in PROGRAMS}
    else:
        # One actual exploratory student-controlled episode from the common
        # physical prefix, before the supervised value update.
        rng = np.random.default_rng(case.action_seed)
        choice = TEACH_ACTIONS[int(rng.integers(len(TEACH_ACTIONS)))]
        outcomes["student_exploration"] = run_programs(ctx, tape, (choice,))
    strict = np.asarray([[int(value["strict"]) for value in values.values()] for values in outcomes.values()], dtype=np.int8)
    num_executed = sum(len({a["program"] for a in v.values()}) for v in outcomes.values())
    result = {"case": asdict(case), "layout_id": tape.layout.identifier, "complete": True, "trigger": ctx["trigger"],
              "features": ctx["features"], "origin": ctx["origin"], "anchor": ctx["anchor"].as_dict(),
              "teacher": preferred, "paired_samples": samples, "teacher_estimate": estimate, "adaptive_actions": shortlist,
              "student": choice, "outcomes": outcomes, "scoring_executions": num_executed}
    return {"strict": strict}, result


def pack_rollouts(rows):
    return {"strict": np.stack([data["strict"] for data, _ in rows])}
