"""Anchor evidence revises an old fixed candidate and changes view selection."""

from study_data import DEFAULT_PACK, ROOT, atomic_json, complete, init_worlds, motion, read_json, roster, start
import numpy as np
from anchor_experiment import execute, public_best
from anchor_policy import Session, information_rank, score_choices
from fit_public import load_public
from forward_anchor import find_anchor, mpc_candidates, step_slot
from object_tracks import ObjectState
from readout_state import confidence, export_cpu


def chain():
    # Each station has two simultaneous candidates. No direct first-to-last
    # link is allowed by station distance. Later evidence is the only change.
    obs = {"frames": np.repeat(np.arange(3), 2), "slots": np.tile(np.arange(2), 3),
           "stations": np.repeat([0., 2., 4.], 2), "sectors": np.full(6, 8.),
           "points": np.tile([[-5., 0.], [5., 0.]], (3, 1)), "quality": np.ones(6), "regions": np.zeros(6)}
    embedding = np.zeros((6, 8))
    embedding[:, 0] = [1., -1.] * 3
    links = np.ones((1, 6, 6), dtype=bool)
    deck = []
    for future in ((.94, .12), (.12, .94)):
        p = np.full((6, 8), .01)
        p[:, 0] = [.50, .50, .50, .50, *future]
        p[:, 1] = 1. - p[:, 0] - .06
        reader = ObjectState(links)
        before = reader.update({k: a[:4] for k, a in obs.items()}, p[:4], embedding[:4])
        past = before["probabilities"][:2, 0].copy()
        revised = reader.update(obs, p, embedding)
        current = revised["probabilities"][:2, 0]
        assert not np.any((revised["edges"][:, 0] < 2) & (revised["edges"][:, 1] >= 4))
        assert revised["track_ids"][0] == revised["track_ids"][4] and revised["track_ids"][1] == revised["track_ids"][5]
        assert revised["track_ids"][0] != revised["track_ids"][1]
        scores, _ = information_rank(np.array([.2, .23, .2]), np.ones(3, dtype=bool),
                                     np.asarray([[.95, .48, .10], [.10, .48, .95]]), np.array([.07, .15, .07]),
                                     current / current.sum())
        repeated = reader.update(obs, p, embedding)
        np.testing.assert_array_equal(repeated["probabilities"], revised["probabilities"])
        deck.append({"old_before": past.tolist(), "old_with_late_evidence": current.tolist(),
                     "later_frame_probabilities": list(future), "scores": scores.tolist(), "slot": int(scores.argmax()),
                     "edges": revised["edges"].tolist(), "direct_first_to_last_edge": False,
                     "distinct_trajectory_ids": revised["track_ids"][:2].tolist(), "ledger_candidates": reader.admitted_count})
    np.testing.assert_allclose(deck[0]["old_before"], deck[1]["old_before"], rtol=0., atol=1.e-8)
    assert deck[0]["slot"] == 0 and deck[1]["slot"] == 2
    return deck


def public_contract(out):
    frozen = read_json(out / "reader_freeze.json")
    weights = export_cpu(load_public(frozen["fit"]["weights"]))
    init_worlds(DEFAULT_PACK)
    ledger = []
    controlled = {}
    for case in list(roster("anchor_dev"))[:8]:
        task = case.task
        reader = {"model": weights, "edge_model": frozen["edge_models"]["prematch"],
                  "risk": frozen["risk"][task]["prematch_track"],
                  "threshold": frozen["thresholds"][task]["methods"]["prematch_track"]["threshold"]}
        env, spec, node, tape, limits = start(case)
        anchor = find_anchor(env, spec, route=task, prefix_stations=4)
        complete(env, spec, node, tape, limits, anchor.prefix_station)
        session, swapped = Session(env, spec, reader), Session(env, spec, reader)
        disguised = node.fork()
        disguised.reader.frames = [(*f[:3], np.full_like(f[3], 99), *f[4:]) for f in node.reader.frames]
        public, state = session.update(node)
        other_public, other_state = swapped.update(disguised)
        np.testing.assert_array_equal(public["features"], other_public["features"])
        np.testing.assert_array_equal(state["probabilities"], other_state["probabilities"])
        sectors, legal, base = mpc_candidates(env, spec, node, limits)
        a, _ = score_choices(session, node, sectors, legal, base, "anchor_information", 0)
        b, _ = score_choices(swapped, disguised, sectors, legal, base, "anchor_information", 0)
        np.testing.assert_array_equal(a, b)
        assert session.update(node)[1] is state
        first, second = node.fork(), node.fork()
        step_slot(env, spec, first, tape, limits, int(np.argmax(base)))
        env.route.advance(second, spec, tape, limits)
        assert (first.station, first.sector, first.tick) == (second.station, second.sector, second.tick)
        np.testing.assert_allclose(first.distance, second.distance, rtol=0., atol=1.e-10)
        ledger.append({"route": task, "pose_invariance_with_same_slot": True,
                       "label_swap_actor_invariant": True, "idempotent_update": True})
        if task in controlled:
            continue
        region = motion.region_of(spec, node.station)
        best = public_best(public, state, confidence(public, state, reader["risk"]), public["regions"] == region)
        if best is None or best["target_probability"] < .92:
            continue
        reserve, _ = env.route.reserve(anchor.prefix_station, anchor.station)
        clock = int(node.tick + reserve[0, int(node.sector)] + 28)
        # Drive the stop predicate with a controlled baseline. The contact
        # observations, transition and remaining route are executed normally.
        origin = {"count": len(node.reader.frames), "region": region, "station": node.station,
                  "witness": best["frame_slot"], "witness_probability": .70}
        ran = execute(env, spec, node, tape, limits, anchor, clock, origin, reader, "anchor_information")
        if ran["evidence_stop"] is not None:
            assert ran["evidence_stop"]["station"] > node.station and ran["complete_budget"]
            assert ran["boundary"]["station"] == anchor.station and not ran["after_anchor"]
            controlled[task] = {"initial_probability_fixture": origin["witness_probability"],
                                "stop": ran["evidence_stop"], "boundary": ran["boundary"],
                                "complete_budget": ran["complete_budget"]}
    if set(controlled) != {r["route"] for r in ledger}:
        raise RuntimeError("Did not exercise the physical stop and resume path for both routes")
    atomic_json(out / "public_contract.json", ledger)
    atomic_json(out / "controlled_stop_resume.json", controlled)
    print(f"Checked {len(ledger)} label-invariant prefixes and legal stop/resume on both routes", flush=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--public-contract", action="store_true")
    args = parser.parse_args()
    result = chain()
    atomic_json(ROOT / "run" / "anchor_identity_mechanism.json", {"scope": "controlled two-track mechanism test", "outcomes": result})
    print("Checked two-track evidence and posterior-driven selection", flush=True)
    if args.public_contract:
        public_contract(ROOT / "run")
