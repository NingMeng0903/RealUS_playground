"""Verify archived physical costs, Bayesian retractions and endpoint outcomes."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from adaptive_controller import BINS, CENTER_MM, END, IDENTITY_ACCEPT, MAX_TRAVEL, QUALITY_ACCEPT, station_bin
from run_adaptive_recovery import WORKERS, init_worker
from recovery_world import Scene


def audit_visible_evidence(controller, prefix, suffix, packets, truth, metrics, arrived):
    evidence = [[] for _ in range(BINS)]
    belief = controller.world.prior.copy()
    previously_retained = [None] * BINS
    confirmed = []
    revoked = []
    mistakes = set()
    root_frame = prefix["public"]["packets"][-1]["frame"]
    for frame, packet in sorted(packets.items()):
        mass, associations = controller.world.likelihood(
            np.asarray(packet["y"], dtype=float).reshape(-1, 2), packet["s"], packet["x"], packet["q"])
        belief *= mass
        belief /= belief.sum()
        np.testing.assert_allclose(belief, packet["belief"], atol=1e-12, rtol=0.)
        association = belief @ associations
        np.testing.assert_allclose(association, packet["association"], atol=1e-12, rtol=0.)
        selected = packet["accepted_slot"]
        assert selected == (int(np.argmax(association)) if association.max(initial=0.) >= IDENTITY_ACCEPT else -1)
        if frame > root_frame and selected >= 0 and truth[frame][0][selected] != 0:
            mistakes.add((frame, selected))
        j = station_bin(packet["s"])
        if j >= 0:
            evidence[j].extend((frame, slot, weight) for slot, weight in enumerate(associations.T))
        retained = []
        for j, candidates in enumerate(evidence):
            scores = np.asarray([w @ belief for _, _, w in candidates])
            match = None
            if len(scores) and scores.max() >= IDENTITY_ACCEPT:
                index = int(np.flatnonzero(scores >= max(IDENTITY_ACCEPT, scores.max() - 1e-9))[0])
                match = list(candidates[index][:2])
            old = previously_retained[j]
            if match != old:
                if old is not None:
                    revoked.append((frame, j, *old))
                if match is not None:
                    acquired, slot = match
                    roles, target = truth[acquired]
                    correct = roles[slot] == 0
                    centered = abs(target[0] - packets[acquired]["x"]) <= CENTER_MM
                    confirmed.append((frame, j, acquired, slot, correct, centered))
                    if frame > root_frame and not correct:
                        mistakes.add((acquired, slot))
            retained.append(match)
        assert retained == packet["visible_ledger"]
        previously_retained = retained
    observed = prefix["evaluator"]["certificates"] + suffix["evaluator"]["certificates"]
    retractions = prefix["public"]["retractions"] + suffix["public"]["retractions"]
    assert confirmed == [(a["accepted_at"], a["bin"], a["frame"], a["slot"], a["identity_ok"], a["center_ok"])
                         for a in observed]
    assert revoked == [(a["frame"], a["bin"], a["witness"], a["slot"]) for a in retractions]
    assert len(revoked) == metrics["retractions"]
    assert metrics["new_identity_mistake"] == bool(mistakes)
    assert metrics["wrong_certificates"] == sum(not row[4] for row in confirmed)
    final = [pair for pair in retained if pair is not None]
    good_errors = [abs(truth[f][1][0] - packets[f]["x"]) for f, slot in final if truth[f][0][slot] == 0]
    centered = sum(error <= CENTER_MM for error in good_errors)
    assert metrics["final_wrong"] == (len(good_errors) < len(final))
    assert metrics["corrected_complete"] == (arrived and len(good_errors) == BINS)
    assert metrics["centered_complete"] == (arrived and centered == BINS)
    assert metrics["correct_visible_bins"] == len(good_errors)
    assert metrics["centered_correct_bins"] == centered
    assert metrics["covered_mask"] == sum(1 << j for j, pair in enumerate(retained) if pair is not None)
    assert np.isclose(metrics["abs_center_error_sum_mm"], sum(good_errors))
    return len(retractions), len(mistakes)


def audit_trial(controller, prefix, suffix, metrics):
    left, right = prefix["public"], suffix["public"]
    motion = np.asarray(left["executed_motion"] + right["executed_motion"][1:])
    packets = {p["frame"]: p for p in left["packets"] + right["packets"]}
    truth = dict((frame, (roles, target)) for frame, roles, target in
                 prefix["evaluator"]["packet_roles"] + suffix["evaluator"]["packet_roles"])
    assert len(packets) == len(truth) == metrics["frames"]
    assert np.allclose(np.diff(motion[:, 0]), .02)
    assert np.max(abs(motion[:, 3])) <= 16. + 1e-7
    assert np.max(abs(motion[:, 4])) <= 18. + 1e-7
    assert np.max(abs(np.diff(motion[:, 3]))) <= 75. * .02 + 1e-7
    assert np.max(abs(np.diff(motion[:, 4]))) <= 100. * .02 + 1e-7
    distance, turn = controller.geometry.cost(motion[:, 1], motion[:, 2])
    assert distance <= MAX_TRAVEL + 1e-7 and motion[-1, 0] <= 8. + 1e-7
    assert np.isclose(distance, metrics["tcp_mm"]) and np.isclose(turn, metrics["turn_degs"])
    assert np.isclose(motion[-1, 0], metrics["seconds"])
    assert metrics["max_handoff_error"] < 1e-7
    if "visible_ledger" in packets[max(packets)]:
        return audit_visible_evidence(controller, prefix, suffix, packets, truth, metrics, motion[-1, 1] >= END - .28)

    certificates = prefix["evaluator"]["certificates"] + suffix["evaluator"]["certificates"]
    certificate_by_frame = {(a["frame"], a["bin"]): a for a in certificates}
    retractions = left["retractions"] + right["retractions"]
    revoked = {(r["witness"], r["bin"]): r for r in retractions}
    assert len(revoked) == len(retractions) == metrics["retractions"]
    final = {a["bin"]: a for a in certificates if (a["frame"], a["bin"]) not in revoked}
    final_packet = packets[max(packets)]
    coverage = (1 << BINS) - 1
    if "qualified_coverage" in final_packet:
        by_frame = {}
        for certificate in certificates:
            packet = packets[certificate["frame"]]
            _, resp = controller.world.likelihood(np.asarray(packet["y"]), packet["s"], packet["x"], packet["q"])
            target = controller.world.target_grid(packet["s"], packet["q"])[:, 0]
            by_frame[certificate["frame"]] = (certificate["bin"],
                                              resp[:, certificate["slot"]] * (abs(target - packet["x"]) <= CENTER_MM))
            assert certificate["bin"] == station_bin(packet["s"])
        retired = {}
        for event in retractions:
            retired.setdefault(event["frame"], []).append(event["bin"])
        witnesses = {}
        for frame, packet in sorted(packets.items()):
            for index in retired.get(frame, []):
                witnesses.pop(index)
            if frame in by_frame:
                index, weight = by_frame[frame]
                assert index not in witnesses
                witnesses[index] = weight
            belief = np.asarray(packet["belief"])
            coverage = sum(1 << index for index, weight in witnesses.items() if belief @ weight >= QUALITY_ACCEPT)
            assert coverage == packet["qualified_coverage"]
        assert coverage == metrics["covered_mask"]

    def validate(a):
        frame, slot = a["frame"], a["slot"]
        roles, target = truth[frame]
        assert packets[frame]["accepted_slot"] == slot >= 0
        assert abs(packets[frame]["y"][slot][0]) <= CENTER_MM
        assert a["identity_ok"] == (roles[slot] == 0)
        assert a["center_ok"] == (abs(target[0] - packets[frame]["x"]) <= CENTER_MM)

    for a in certificates:
        validate(a)
        key = a["frame"], a["bin"]
        if key in revoked:
            event, packet = revoked[key], packets[a["frame"]]
            assert event["frame"] > a["frame"]
            _, resp = controller.world.likelihood(np.asarray(packet["y"]), packet["s"], packet["x"], packet["q"])
            belief = np.asarray(packets[event["frame"]]["belief"])
            if event.get("reason") == "higher_centered_probability":
                new_packet = packets[event["frame"]]
                replacement = certificate_by_frame[(event["frame"], a["bin"])]
                _, new_resp = controller.world.likelihood(np.asarray(new_packet["y"]), new_packet["s"],
                                                         new_packet["x"], new_packet["q"])
                old_target = controller.world.target_grid(packet["s"], packet["q"])[:, 0]
                new_target = controller.world.target_grid(new_packet["s"], new_packet["q"])[:, 0]
                previous = float(belief @ (resp[:, a["slot"]] * (abs(old_target - packet["x"]) <= CENTER_MM)))
                current = float(belief @ (new_resp[:, replacement["slot"]] *
                                         (abs(new_target - new_packet["x"]) <= CENTER_MM)))
                assert current > previous + 1e-9
                assert np.isclose(previous, event["old_quality"]) and np.isclose(current, event["new_quality"])
            else:
                p = belief @ resp[:, a["slot"]]
                assert p < IDENTITY_ACCEPT and np.isclose(p, event["posterior"])
    assert metrics["final_wrong"] == any(not a["identity_ok"] for a in final.values())
    assert metrics["wrong_certificates"] == sum(not a["identity_ok"] for a in certificates)
    arrived = motion[-1, 1] >= END - .28
    assert metrics["corrected_complete"] == (
        arrived and coverage == (1 << BINS) - 1 and len(final) == BINS
        and all(a["identity_ok"] and a["center_ok"] for a in final.values()))

    legacy = prefix["evaluator"]["legacy_accepts"] + suffix["evaluator"]["legacy_accepts"]
    for a in legacy:
        validate(a)
    mistakes = [frame for frame, packet in packets.items() if packet["accepted_slot"] >= 0
                and truth[frame][0][packet["accepted_slot"]] != 0
                and abs(packet["y"][packet["accepted_slot"]][0]) <= CENTER_MM]
    assert metrics["ever_wrong"] == bool(mistakes or any(not a["identity_ok"] for a in legacy))
    assert metrics["old_strict_complete"] == (
        arrived and not mistakes and len({a["bin"] for a in legacy if a["identity_ok"] and a["center_ok"]}) == BINS)
    root_frame = max(p["frame"] for p in left["packets"])
    assert metrics["new_centered_mistake"] == any(t > root_frame for t in mistakes)
    return len(retractions), len(mistakes)


def audit_alias():
    world = WORKERS["alias"].world
    tape = world.draw(7100).tape
    half = len(world.prior) // 2
    s = np.linspace(2., 31.4, 81)
    x = -14. - 7. * np.sin(s)
    belief = world.prior.copy()
    for frame in range(len(s)):
        i = frame % half
        a = world.observe(Scene(i, tape), frame, s[frame], x[frame])[0]
        b = world.observe(Scene(i + half, tape), frame, s[frame], x[frame])[0]
        np.testing.assert_array_equal(a, b)
        belief, association = world.update(belief, a, s[frame], x[frame], 15.)
        np.testing.assert_allclose(belief[:half], belief[half:], atol=1e-13)
        assert np.all(association <= .5 + 1e-13)
    return len(s)


def audit(path):
    counts = Counter()
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            controller = WORKERS[row["family"]]
            counts["worlds"] += 1
            if isinstance(row["verification"], dict):
                trials = ((name, runs[0]["trace"], runs[0]["metrics"]) for name, runs in row["verification"].items())
            else:
                first = row["verification"][0]
                trials = ((name, logs, first["outcomes"][name]) for name, logs in first["traces"].items())
            for name, log, metrics in trials:
                prefix = row.get("prefix_by_arm", {}).get(name, row["prefix"])
                revocations, mistakes = audit_trial(controller, prefix, log, metrics)
                counts.update(trajectories=1, retractions=revocations)
                key = "identity_mistakes" if "new_identity_mistake" in metrics else "centered_mistakes"
                counts[key] += mistakes
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    config = parser.parse_args()
    init_worker(None)
    report = {"alias_packet_checks": audit_alias(),
              "archives": {str(file): audit(file) for file in config.archives}}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
