"""Verify archived physical costs, Bayesian retractions and endpoint outcomes."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from adaptive_controller import BINS, CENTER_MM, END, IDENTITY_ACCEPT, MAX_TRAVEL
from run_adaptive_recovery import WORKERS, init_worker
from recovery_world import Scene


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

    certificates = prefix["evaluator"]["certificates"] + suffix["evaluator"]["certificates"]
    retractions = left["retractions"] + right["retractions"]
    revoked = {(r["witness"], r["bin"]): r for r in retractions}
    assert len(revoked) == len(retractions) == metrics["retractions"]
    final = {a["bin"]: a for a in certificates if (a["frame"], a["bin"]) not in revoked}

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
            p = np.asarray(packets[event["frame"]]["belief"]) @ resp[:, a["slot"]]
            assert p < IDENTITY_ACCEPT and np.isclose(p, event["posterior"])
    assert metrics["final_wrong"] == any(not a["identity_ok"] for a in final.values())
    assert metrics["wrong_certificates"] == sum(not a["identity_ok"] for a in certificates)
    arrived = motion[-1, 1] >= END - .28
    assert metrics["corrected_complete"] == (
        arrived and len(final) == BINS and all(a["identity_ok"] and a["center_ok"] for a in final.values()))

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
                trials = ((runs[0]["trace"], runs[0]["metrics"]) for runs in row["verification"].values())
            else:
                first = row["verification"][0]
                trials = ((logs, first["outcomes"][name]) for name, logs in first["traces"].items())
            for log, metrics in trials:
                revocations, mistakes = audit_trial(controller, row["prefix"], log, metrics)
                counts.update(trajectories=1, retractions=revocations, centered_mistakes=mistakes)
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
