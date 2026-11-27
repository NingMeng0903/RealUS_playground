"""Enumerate the information and executable-continuation ablation on a finite graph."""

import json
import math
from fractions import Fraction as F
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGETS = ("L", "R")
ABSTENTION = "missing"
COST_PER_SECOND = F(1, 20)
PROBE_TIME = {"direct": F(0), "restricted": F(4, 5), "branching": F(1)}
TERMINAL_TIME = {
    "direct": {"L": F(1), "R": F(1)},
    "restricted": {"L": F(1), "R": F(2)},
    "branching": {"L": F(1), "R": F(1)},
}


def observation_distribution(probe, use_new_observation, visible, accuracy):
    """Return weighted hypotheses indexed by the public event used for selection."""
    if probe == "direct" or not use_new_observation:
        return {ABSTENTION: {target: F(1, 2) for target in TARGETS}}
    return {
        "L": {"L": visible * accuracy / 2, "R": visible * (1 - accuracy) / 2},
        "R": {"L": visible * (1 - accuracy) / 2, "R": visible * accuracy / 2},
        ABSTENTION: {target: (1 - visible) / 2 for target in TARGETS},
    }


def continuation(probe, outcomes, terminal_time, deadline):
    duration = PROBE_TIME[probe]
    available = [target for target in TARGETS if duration + terminal_time[target] <= deadline]
    probability = F(0)
    elapsed = duration
    selected = {}
    for event, weights in outcomes.items():
        mass = sum(weights.values())
        if not mass:
            continue
        if not available:
            selected[event] = None
            continue
        target = max(available, key=lambda action: weights[action] - COST_PER_SECOND * mass * terminal_time[action])
        selected[event] = target
        probability += weights[target]
        elapsed += mass * terminal_time[target]
    return {
        "selection_by_event": selected, "successful_by_deadline": probability, "mean_seconds": elapsed,
        "objective": probability - COST_PER_SECOND * elapsed,
    }


def evaluate(deadline, visible, accuracy, restricted_return_time):
    actual_time = TERMINAL_TIME | {"restricted": {"L": F(1), "R": restricted_return_time}}
    methods = {}
    for use_new_observation in (False, True):
        for consider_actual_exit in (False, True):
            planned = {}
            realized = {}
            for probe in PROBE_TIME:
                outcomes = observation_distribution(probe, use_new_observation, visible, accuracy)
                actual = actual_time[probe]
                anticipated = actual if consider_actual_exit else actual_time["branching"]
                planned[probe] = continuation(probe, outcomes, anticipated, deadline)
                realized[probe] = continuation(probe, outcomes, actual, deadline)
            first = max(planned, key=lambda probe: planned[probe]["objective"])
            methods[f"I{int(use_new_observation)}_F{int(consider_actual_exit)}"] = {
                "first": first, "predicted": planned[first], "realized": realized[first],
            }
    baselines = {
        probe: continuation(probe, observation_distribution(probe, True, visible, accuracy), actual_time[probe], deadline)
        for probe in PROBE_TIME
    }
    increments = {
        probe: baselines[probe]["successful_by_deadline"] -
        continuation(probe, observation_distribution(probe, False, visible, accuracy), actual_time[probe], deadline)[
            "successful_by_deadline"
        ]
        for probe in ("restricted", "branching")
    }
    entropy = -sum(float(x) * math.log2(float(x)) for x in (accuracy, 1 - accuracy) if x)
    return {
        "deadline_seconds": deadline, "visibility": visible, "accuracy_when_visible": accuracy,
        "information_bits_for_each_probe": float(visible) * (1 - entropy),
        "methods": methods, "fixed_first_baselines": baselines,
        "fixed_first_success_increment_from_observation": increments,
    }



def joint_timing_diagnostic():
    """Equal marginals can conceal different usable evidence before the deadline."""
    # Per-target weights: correct-short, correct-long, wrong-short, wrong-long.
    kernels = {
        "correct_frames_can_arrive_early": (F(1, 2), F(1, 4), F(0), F(1, 4)),
        "correct_frames_arrive_late": (F(1, 4), F(1, 2), F(1, 4), F(0)),
    }
    rows = {}
    for name, (correct_short, correct_long, wrong_short, wrong_long) in kernels.items():
        assert correct_short + correct_long + wrong_short + wrong_long == 1
        accuracy = correct_short + correct_long
        short_probability = correct_short + wrong_short
        rows[name] = {
            "weights": [correct_short, correct_long, wrong_short, wrong_long],
            "image_accuracy": accuracy,
            "probability_of_timely_exit": short_probability,
            "probability_each_label_and_timely_exit": short_probability / 2,
            "correct_completion_using_joint": max(correct_short, wrong_short),
            "predicted_completion_if_factorized": short_probability * max(accuracy, 1 - accuracy),
        }
    a, b = rows.values()
    assert a["correct_completion_using_joint"] == F(1, 2)
    assert b["correct_completion_using_joint"] == F(1, 4)
    assert a["predicted_completion_if_factorized"] == b["predicted_completion_if_factorized"] == F(3, 8)
    return rows


def main():
    results = {
        "constrained": evaluate(F(2), F(4, 5), F(4, 5), F(2)),
        "open_exit": evaluate(F(2), F(4, 5), F(4, 5), F(1)),
        "longer_deadline": evaluate(F(3), F(4, 5), F(4, 5), F(2)),
        "no_information": evaluate(F(2), F(4, 5), F(1, 2), F(2)),
        "always_missing": evaluate(F(2), F(0), F(4, 5), F(2)),
    }
    constrained = results["constrained"]
    methods = constrained["methods"]
    assert [row["first"] for row in methods.values()] == ["direct", "direct", "restricted", "branching"]
    assert [row["realized"]["objective"] for row in methods.values()] == [F(45, 100), F(45, 100), F(41, 100), F(64, 100)]
    assert constrained["fixed_first_success_increment_from_observation"] == {"restricted": F(0), "branching": F(24, 100)}
    opened = results["open_exit"]["methods"]
    assert opened["I1_F0"] == opened["I1_F1"]
    assert opened["I1_F1"]["realized"]["objective"] == F(65, 100)
    long = results["longer_deadline"]["fixed_first_baselines"]
    assert long["restricted"]["successful_by_deadline"] == long["branching"]["successful_by_deadline"] == F(74, 100)
    for case in ("no_information", "always_missing"):
        assert all(row["first"] == "direct" for row in results[case]["methods"].values())
    payload = {
        "scope": "Exactly enumerated abstract timing and observation model. No robot or acoustic evidence.",
        "objective": "P(fresh correct acquisition before deadline) - 0.05 * mean elapsed seconds",
        "results": results,
        "joint_timing_diagnostic": joint_timing_diagnostic(),
    }
    rendered = json.dumps(payload, indent=2, default=float) + "\n"
    (HERE / "CONTINUATION_RESULTS.json").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

