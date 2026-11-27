"""Integrated, time-matched action programs with one fixed public object readout."""

from __future__ import annotations

from dataclasses import asdict

from study_data import (ARM, complete, held_frames, init_worlds, motion, start)
import numpy as np
from active_trial import ReaderContactMotion
from anchor_policy import Session, hypotheses, score_choices
from forward_anchor import find_anchor, mpc_candidates, step_slot
from readout_state import confidence, evaluate, measured_length
from study_worlds import material_widths, truth_at
from scale_worlds import FUTURE_SEEDS


PROGRAMS = ("mpc", "equal_time_hold", "anchor_information", "anchor_first", "anchor_second")
DEV_STREAMS = ("search_0", "search_1", "proof_0", "proof_1")
TEST_STREAMS = ("actual", *DEV_STREAMS)
_READERS = None


def init_integrated(pack, readers):
    global _READERS
    init_worlds(pack)
    _READERS = readers


def public_best(public, state, score, mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return None
    j = int(ids[np.argmax(score[ids])])
    return {"index": j, "frame_slot": [int(public["frames"][j]), int(public["slots"][j])],
            "target_probability": float(state["probabilities"][j, 0]), "confidence": float(score[j])}


def read_old(session, node, trial, origin, frozen):
    public, state = session.update(node)
    scores = confidence(public, state, frozen["risk"], max_frame=origin["count"])
    old = public_best(public, state, scores, (public["frames"] < origin["count"]) & (public["regions"] == origin["region"]))
    lab = truth_at(node, public, trial)["labels"]
    if old is not None:
        old["truth"] = int(lab[old["index"]])
        old["submitted"] = old["confidence"] >= frozen["threshold"]
    fixed = session.item(origin.get("witness"))
    if fixed is not None:
        linked = (state["track_ids"] == state["track_ids"][fixed]) & (public["frames"] >= origin["count"]) \
            & (public["stations"] >= origin["station"] + 2.)
        pair = state["edges"][:, :2].astype(int)
        sources = pair[(pair == fixed).any(axis=1)]
        direct = np.where(sources[:, 0] == fixed, sources[:, 1], sources[:, 0])
        nonlocal_evidence = np.flatnonzero(linked & ~np.isin(np.arange(len(linked)), direct))
        witness = {"frame_slot": origin["witness"], "truth": int(lab[fixed]),
                   "target_probability": float(state["probabilities"][fixed, 0]),
                   "unary_target_probability": float(state["unary"][fixed, 0]),
                   "far_linked_frames": int(len(np.unique(public["frames"][linked]))),
                   "far_non_neighbor_frames": int(len(np.unique(public["frames"][nonlocal_evidence])))}
    else:
        witness = None
    return old, witness


def terminal(session, node, trial, origin, frozen, limits):
    public, state = session.update(node)
    lab = truth_at(node, public, trial)["labels"]
    spec, env = session.spec, session.env
    budget_ok = node.station == spec[2] and node.tick <= limits.ticks and node.distance <= limits.mm + 1.e-6
    final = evaluate(public, lab, confidence(public, state, frozen["risk"]), frozen["threshold"], complete=budget_ok)
    old, witness = read_old(session, node, trial, origin, frozen)
    # Only physically forward frames acquired since starting the program can
    # satisfy its suffix requirement. Earlier views still inform identities.
    required = list(range(origin["region"] + 1, 6))
    after = (public["frames"] >= origin["count"]) & (public["stations"] > origin["station"] + .24)
    clip = {k: v[after] for k, v in public.items() if isinstance(v, np.ndarray) and len(v) == len(after)}
    belief = {k: state[k][after] for k in ("probabilities", "track_ids", "support")}
    suffix_scores = confidence(clip, belief, frozen["risk"])
    prefix_removed = evaluate(clip, lab[after], suffix_scores, frozen["threshold"], complete=budget_ok)
    old_correct = bool(old is not None and old["submitted"] and old["truth"] == 0)
    suffix_ok = all(prefix_removed["region_outcomes"][r] == 1 for r in required)
    joint = bool(old_correct and suffix_ok and budget_ok)
    width = material_widths(env, spec)
    total, contiguous = measured_length(public, state, lab, .75, width)
    utility = float(3. * final["strict"] + 1. * joint + .36 * final["correct_regions"]
                    - .8 * final["wrong_branch_regions"] - .45 * final["artifact_regions"] + .0015 * total)
    return final | {"joint": joint, "old_correct": old_correct, "old_terminal": old, "fixed_old_terminal": witness,
                    "suffix_regions": required, "suffix_valid": bool(suffix_ok),
                    "suffix_outcomes": prefix_removed["region_outcomes"], "complete_budget": bool(budget_ok),
                    "correct_mm": total, "contiguous_mm": contiguous, "utility": utility,
                    "time_s": node.tick * motion.CLOCK_S, "travel_mm": node.distance}


def alternatives(env, spec, node, limits, anchor, clock, arrival):
    sectors, legal, score = mpc_candidates(env, spec, node, limits)
    if arrival:
        s = int(node.station)
        ticks, _ = env.route.reserve(anchor.prefix_station, anchor.station)
        future = ticks[s + 1 - anchor.prefix_station, sectors]
        legal &= node.tick + env.route.clock[s, int(node.sector)] + future <= clock
        score = np.where(legal, score, -np.inf)
        if not legal.any():
            raise RuntimeError("Review could not legally reach the shared comparison boundary")
    return sectors, legal, score


def short_review(env, spec, node, tape, limits, clock, session):
    public, state = session.update(node)
    belief = hypotheses(public, state, node.station)
    if belief is None or abs(belief.probability[0] - belief.probability[1]) > .44:
        return node, {"requested": False, "actual_ticks": 0}
    ticks, mm = limits.probe(node, spec)
    caps = (min(ticks, clock - node.tick), mm)
    if caps[0] < 8:
        return node, {"requested": True, "actual_ticks": 0}
    for turn in (-.63, .63):
        plan = env.plan(node, spec, 2, np.array([turn, .55, -.12 * turn]), caps, ARM)
        if plan is not None and abs(plan.goal[0] - node.station) >= .55:
            if not isinstance(env.material, ReaderContactMotion):
                env.material = ReaderContactMotion(env.scanner)
            outcome = env.material.run_with_reader(node, spec, tape, plan, caps, session)
            if not outcome.legal:
                raise RuntimeError(f"Admitted after-anchor return failed {outcome.reason}")
            return outcome.node, {"requested": True, "actual_ticks": outcome.ticks, "distance_mm": outcome.distance,
                                  "observed_trajectory": outcome.trajectory.tolist(), "goal": plan.goal.tolist()}
    return node, {"requested": True, "actual_ticks": 0}


def execute(env, spec, first, tape, limits, anchor, clock, origin, reader, name):
    node = first.fork()
    session = Session(env, spec, reader)
    session.update(node)
    guided = name.startswith("anchor_")
    active = guided
    matched = name != "mpc"
    trace = []
    forward = []
    early = None
    after_visit = None
    if name == "equal_time_hold":
        held_frames(env, spec, node, tape, 7)
        session.update(node)
    while node.station < anchor.station:
        sectors, legal, base = alternatives(env, spec, node, limits, anchor, clock, matched)
        if active:
            scores, row = score_choices(session, node, sectors, legal, base, name, len(trace))
            index = session.item(origin["witness"])
            p = None if index is None else float(session.state["probabilities"][index, 0])
            if (node.station - first.station >= 2 and p is not None and origin["witness_probability"] is not None
                    and p >= .89 and p - origin["witness_probability"] >= .10):
                early = {"station": int(node.station), "sector": int(node.sector), "tick": int(node.tick),
                         "reason": "fixed_public_candidate_confirmed", "new_target_probability": p}
                active = False
                scores = base
            row.update({"station": int(node.station), "sector": int(node.sector), "tick": int(node.tick),
                        "chosen": int(np.argmax(scores)), "fixed_old_probability": p})
            trace.append(row)
        else:
            scores = base
        _, travelled = step_slot(env, spec, node, tape, limits, int(np.argmax(scores)))
        forward.append({"station": int(node.station), "sector": int(node.sector), "tick": int(node.tick),
                        "distance_step_mm": travelled["distance_mm"],
                        "world_mm": env.scanner.surface.world(node.station, node.sector).tolist()})
        session.update(node)
    if guided and active:
        node, after_visit = short_review(env, spec, node, tape, limits, clock, session)
    if matched:
        if node.tick > clock or clock + limits.reserve_ticks[anchor.station - spec[1], int(node.sector)] > limits.ticks:
            raise RuntimeError("The compared arms cannot pad to their common legal boundary")
        held_frames(env, spec, node, tape, clock - node.tick)
        if node.tick != clock or node.station != anchor.station:
            raise RuntimeError("Invalid equal-clock, equal-station continuation boundary")
    boundary = {"station": int(node.station), "sector": int(node.sector), "tick": int(node.tick), "distance_mm": node.distance,
                "signed_forward_travel_mm": float(abs(env.scanner.surface.y[int(node.station)] - env.scanner.surface.y[int(first.station)]))}
    mid, fixed_after = read_old(session, node, tape, origin, reader)
    # Exact common continuation. The readout state is updated at subsequently
    # acquired section boundaries, never by replaying old evidence as new.
    while node.station < spec[2]:
        previous = motion.region_of(spec, node.station)
        env.route.advance(node, spec, tape, limits)
        if motion.region_of(spec, node.station) != previous:
            session.update(node)
    result = terminal(session, node, tape, origin, reader, limits)
    result.update({"boundary": boundary, "old_at_boundary": mid, "fixed_at_boundary": fixed_after,
                   "ranking": trace, "forward_path": forward, "evidence_stop": early, "after_anchor": after_visit,
                   "layout_id": tape.layout.identifier, "executed_program": name})
    return result


def anchor_case(case):
    env, spec, node, tape, limits = start(case)
    anchor = find_anchor(env, spec, route=case.task, prefix_stations=3 + case.band % 4)
    complete(env, spec, node, tape, limits, anchor.prefix_station)
    reader = _READERS[case.task]
    session = Session(env, spec, reader)
    pub, graph = session.update(node)
    cut = len(node.reader.frames)
    region = motion.region_of(spec, node.station)
    first = public_best(pub, graph, confidence(pub, graph, reader["risk"]), pub["regions"] == region)
    origin = {"count": cut, "region": region, "station": int(node.station), "witness": first["frame_slot"] if first else None,
              "witness_probability": first["target_probability"] if first else None}
    reserve, _ = env.route.reserve(anchor.prefix_station, anchor.station)
    deadline = int(node.tick + reserve[0, int(node.sector)] + 28)
    sectors, legal, score = alternatives(env, spec, node, limits, anchor, deadline, True)
    _, forecast = score_choices(session, node, sectors, legal, score, "anchor_information", 0)
    triggered = (forecast["hypotheses"] == 2 and len(set(forecast["hypothesis_preference_slots"])) == 2
                 and max(forecast["discrimination"]) > .10)
    policy = "anchor_information" if triggered else "mpc"
    old, value = read_old(session, node, tape, origin, reader)
    roster = TEST_STREAMS if case.split == "anchor_test" else DEV_STREAMS
    offsets = {f"{kind}_{i}": number for kind, values in FUTURE_SEEDS.items() for i, number in enumerate(values)}
    records = {}
    for label in roster:
        future = tape if label == "actual" else tape.branch(node.tick, case.tape_seed + offsets[label])
        if future.layout is not tape.layout:
            raise RuntimeError("Independent noise changed the matched anatomy")
        records[label] = {name: execute(env, spec, node, future, limits, anchor, deadline, origin, reader, name) for name in PROGRAMS}
    if any(not a["complete_budget"] for r in records.values() for a in r.values()):
        raise RuntimeError("An integrated probe exceeded the shared budget")
    measures = np.asarray([[p[name]["strict"] for name in PROGRAMS] for p in records.values()], dtype=bool)
    meta = {"case": asdict(case), "layout_id": tape.layout.identifier, "complete": True, "anchor": anchor.as_dict(),
            "origin": origin, "origin_old": old, "origin_fixed": value, "threshold": reader["threshold"], "public_policy": policy,
            "elapsed_comparison_boundary_tick": deadline, "programs": records, "prefix_forecast": forecast}
    return {"strict": measures}, meta


def pack_anchors(rows):
    return {"strict": np.stack([row["strict"] for row, _ in rows])}
