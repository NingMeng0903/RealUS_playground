#!/usr/bin/env python3
"""Independent exhaustive checks for the binary decision mechanism."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import run_mechanism as exp


ROOT = Path(__file__).resolve().parent


def hand_branch_masses(p: float, r: float):
    return {
        1: {1: p * r, -1: (1.0 - p) * (1.0 - r)},
        -1: {1: p * (1.0 - r), -1: (1.0 - p) * r},
    }


def hand_terminal(action: int, mass, delta: float) -> float:
    total = sum(mass.values())
    wrong = mass[-1] if action == 1 else mass[1]
    return wrong + (delta * total if action == 1 else 0.0)


def hand_probe(p: float, r: float, delta: float, c: float):
    branches = hand_branch_masses(p, r)
    total = c
    actions = []
    for y in (1, -1):
        mass = branches[y]
        risk_minus = hand_terminal(-1, mass, delta)
        risk_plus = hand_terminal(1, mass, delta)
        if risk_plus < risk_minus - exp.TIE_TOL:
            actions.append(1)
            total += risk_plus
        else:
            actions.append(-1)
            total += risk_minus
    return total, actions


def hand_direct(p: float, delta: float):
    mass = {1: p, -1: 1.0 - p}
    rm = hand_terminal(-1, mass, delta)
    rp = hand_terminal(1, mass, delta)
    return (rp if rp < rm - exp.TIE_TOL else rm), (1 if rp < rm - exp.TIE_TOL else -1)


def check() -> None:
    results = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    raw = np.load(ROOT / "RAW_RESULTS.npz")
    assert results["status"] == "exact_binary_decision_mechanism"
    assert results["protocol"]["n_mc"] == 4096

    # Recompute every default-grid branch and both policy risks without calling
    # the implementation's branch/risk functions.
    for rec in results["default_grid"]["analytic"]:
        pars = rec["parameters"]
        p, r, c, delta = pars["p"], pars["r"], pars["c"], pars["delta"]
        direct_risk, direct_action = hand_direct(p, delta)
        probe_risk, probe_actions = hand_probe(p, r, delta, c)
        assert np.isclose(direct_risk, rec["direct_myopic"]["risk"], atol=1e-12)
        assert np.isclose(probe_risk, rec["always_informative"]["risk"], atol=1e-12)
        assert probe_actions == rec["always_informative"]["branch_actions"]
        expected_choice = "probe" if probe_risk < direct_risk - exp.TIE_TOL else "skip"
        assert expected_choice == rec["cheap_exact_threshold"]["choice"]
        assert expected_choice == rec["two_step_bellman"]["choice"]
        assert np.isclose(
            min(direct_risk, probe_risk), rec["two_step_bellman"]["risk"], atol=1e-12
        )

    assert results["default_grid"]["cheap_equals_bellman"]
    assert results["default_grid"]["threshold_grid_matches"]
    assert results["default_grid"]["cheap_choices"] == [
        "probe",
        "probe",
        "probe",
        "skip",
        "skip",
        "skip",
        "skip",
    ]

    # Exact branch tables contain both public signals and both hidden-state
    # joint masses; no state has been silently omitted.
    for table in results["branch_tables"]:
        assert len(table["branches"]) == 2
        for branch in table["branches"]:
            assert branch["joint_mass_h_plus"] >= 0
            assert branch["joint_mass_h_minus"] >= 0
            assert np.isclose(
                branch["branch_probability"],
                branch["joint_mass_h_plus"] + branch["joint_mass_h_minus"],
            )
        assert np.isclose(sum(b["branch_probability"] for b in table["branches"]), 1.0)

    # Recheck the standard threshold in the three points that straddle it.
    near = {rec["parameters"]["delta"]: rec for rec in results["default_grid"]["analytic"]}
    assert near[0.39]["cheap_exact_threshold"]["choice"] == "probe"
    assert near[0.4]["cheap_exact_threshold"]["choice"] == "skip"
    assert near[0.41]["cheap_exact_threshold"]["choice"] == "skip"
    assert np.isclose(near[0.4]["cheap_exact_threshold"]["risk"], 0.5)

    pressures = {rec["name"]: rec for rec in results["pressure_groups"]}
    skew = pressures["skewed_prior"]
    assert skew["always_informative"]["branch_actions"] == [1, 1]
    assert np.isclose(skew["cheap_exact_threshold"]["probe"]["decision_information_before_probe_cost"], 0.0)
    uninformative = pressures["uninformative_signal"]
    assert uninformative["always_informative"]["branch_actions"] == [-1, -1]
    assert np.isclose(uninformative["always_informative"]["probe"]["decision_information_before_probe_cost"], 0.0)
    high_cost = pressures["high_observation_cost"]
    assert high_cost["cheap_exact_threshold"]["choice"] == "skip"
    high_delta = pressures["all_branches_choose_minus"]
    assert high_delta["always_informative"]["branch_actions"] == [-1, -1]
    assert np.isclose(high_delta["always_informative"]["probe"]["decision_information_before_probe_cost"], 0.0)

    # MI is checked independently from the stored numerical values.
    def hb(x):
        return 0.0 if x in (0.0, 1.0) else -x * math.log2(x) - (1 - x) * math.log2(1 - x)

    expected_equal = 1 - hb(0.8)
    expected_high = 1 - hb(0.95)
    assert np.isclose(results["mi"]["equal_mi_bits"], expected_equal, atol=1e-14)
    assert np.isclose(results["mi"]["u_high_bits"], expected_high, atol=1e-14)
    assert results["mi"]["u_high_strictly_greater"]
    # Same MI has two explicit tie-break records; no universal error claim.
    for rec in results["default_grid"]["analytic"]:
        h = rec["entropy_equal_mi_favor_H"]
        u = rec["entropy_equal_mi_favor_U"]
        assert np.isclose(h["mi_h_bits"], u["mi_u_bits"], atol=1e-14)
        assert h["selected_sensor"] == "H"
        assert u["selected_sensor"] == "U"

    # Independently verify one paired MC row from its saved per-case costs.
    target_name = "default_delta_0.39"
    key = f"mc_cost_{target_name}_cheap_exact_threshold"
    costs = raw[key]
    row = next(x for x in results["mc"] if x["name"] == target_name)["policies"]["cheap_exact_threshold"]
    assert len(costs) == 4096
    assert np.isclose(costs.mean(), row["mc_mean"], atol=1e-14)
    assert np.isclose(costs.std(ddof=1), row["mc_sd"], atol=1e-14)
    assert row["ci95"][0] <= row["mc_mean"] <= row["ci95"][1]

    # The two exact policies have identical analytic and MC risks for every
    # scenario, rather than merely agreeing on the default threshold labels.
    for rec in results["mc"]:
        cheap = rec["policies"]["cheap_exact_threshold"]
        bellman = rec["policies"]["two_step_bellman"]
        assert np.isclose(cheap["analytic_risk"], bellman["analytic_risk"], atol=1e-12)
        assert np.isclose(cheap["mc_mean"], bellman["mc_mean"], atol=1e-14)

    print("independent B mechanism checks: PASS")


if __name__ == "__main__":
    check()
