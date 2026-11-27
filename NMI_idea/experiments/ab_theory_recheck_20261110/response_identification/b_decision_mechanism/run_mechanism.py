#!/usr/bin/env python3
"""Exact binary Bayes decision-mechanism experiment for B.

The implementation is deliberately a toy: one hidden binary state, one public
binary signal, one state-dependent terminal execution cost, and at most one
probe.  It tests a decision threshold, not a robot or ultrasound system.  See
PROTOCOL.txt for the frozen constants and interpretation limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
L = 1.0
P_DEFAULT = 0.5
R_DEFAULT = 0.8
C_DEFAULT = 0.1
DELTA_GRID = np.asarray([0.0, 0.2, 0.39, 0.4, 0.41, 0.6, 1.0], dtype=float)
MC_SEED = 720942
N_MC = 4096
Z95 = 1.959963984540054
TIE_TOL = 1e-12


def h_binary(prob: float) -> float:
    """Binary entropy in bits, with exact handling at 0 and 1."""

    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return float(-prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob))


def binary_sensor_mi(accuracy: float) -> float:
    """MI of a symmetric sensor with a uniform binary target, in bits."""

    return 1.0 - h_binary(accuracy)


def joint_masses(p: float, r: float) -> Dict[int, Dict[int, float]]:
    """Return public-signal branch masses mass[y][h], h/y in {-1,+1}."""

    # y=+1 is correct for h=+1 with r and false for h=-1 with 1-r.
    return {
        1: {1: p * r, -1: (1.0 - p) * (1.0 - r)},
        -1: {1: p * (1.0 - r), -1: (1.0 - p) * r},
    }


def total_mass(mass: Mapping[int, float]) -> float:
    return float(sum(mass.values()))


def terminal_unnormalized_risk(
    action: int, mass: Mapping[int, float], delta: float, loss: float = L
) -> float:
    """Expected terminal cost multiplied by the public branch mass."""

    branch_probability = total_mass(mass)
    wrong_mass = mass[-1] if action == 1 else mass[1]
    execution = delta * branch_probability if action == 1 else 0.0
    return float(loss * wrong_mass + execution)


def choose_terminal_action(
    mass: Mapping[int, float], delta: float, loss: float = L
) -> Tuple[int, float, float, float]:
    """Choose using only public branch masses; ties deterministically choose '-'."""

    risk_minus = terminal_unnormalized_risk(-1, mass, delta, loss)
    risk_plus = terminal_unnormalized_risk(1, mass, delta, loss)
    if risk_plus < risk_minus - TIE_TOL:
        return 1, risk_plus, risk_minus, risk_plus - risk_minus
    return -1, risk_minus, risk_minus, risk_plus - risk_minus


def direct_decision(p: float, delta: float, loss: float = L) -> Dict[str, object]:
    mass = {1: p, -1: 1.0 - p}
    action, chosen, risk_minus, margin = choose_terminal_action(mass, delta, loss)
    # choose_terminal_action returns chosen risk as its second result; expose
    # both action risks explicitly for audit.
    risk_plus = terminal_unnormalized_risk(1, mass, delta, loss)
    risk_minus = terminal_unnormalized_risk(-1, mass, delta, loss)
    return {
        "action": int(action),
        "risk": float(min(risk_plus, risk_minus)),
        "risk_plus": float(risk_plus),
        "risk_minus": float(risk_minus),
        "margin_plus_minus": float(risk_plus - risk_minus),
    }


def probe_decision(
    p: float, r: float, delta: float, c: float, loss: float = L
) -> Dict[str, object]:
    """Enumerate both public branches and aggregate before minimizing."""

    branches = joint_masses(p, r)
    branch_records = []
    terminal_sum = 0.0
    for y in (1, -1):
        mass = branches[y]
        branch_prob = total_mass(mass)
        posterior_plus = mass[1] / branch_prob if branch_prob > 0 else p
        action, chosen, risk_minus, margin = choose_terminal_action(mass, delta, loss)
        risk_plus = terminal_unnormalized_risk(1, mass, delta, loss)
        risk_minus = terminal_unnormalized_risk(-1, mass, delta, loss)
        terminal_sum += min(risk_plus, risk_minus)
        branch_records.append(
            {
                "y": int(y),
                "joint_mass_h_plus": float(mass[1]),
                "joint_mass_h_minus": float(mass[-1]),
                "branch_probability": float(branch_prob),
                "posterior_p_plus": float(posterior_plus),
                "action": int(action),
                "action_label": "+" if action == 1 else "-",
                "unnormalized_risk_plus": float(risk_plus),
                "unnormalized_risk_minus": float(risk_minus),
                "chosen_unnormalized_risk": float(min(risk_plus, risk_minus)),
                "action_risk_margin_plus_minus": float(risk_plus - risk_minus),
            }
        )
    total = c + terminal_sum
    return {
        "risk": float(total),
        "risk_without_probe_cost": float(terminal_sum),
        "probe_cost": float(c),
        "branches": branch_records,
        "branch_actions": [int(row["action"]) for row in branch_records],
        "branch_actions_same": bool(
            branch_records[0]["action"] == branch_records[1]["action"]
        ),
        "decision_information_before_probe_cost": float(
            direct_decision(p, delta, loss)["risk"] - terminal_sum
        ),
    }


def cheap_exact_threshold(
    p: float, r: float, delta: float, c: float, loss: float = L
) -> Dict[str, object]:
    direct = direct_decision(p, delta, loss)
    probe = probe_decision(p, r, delta, c, loss)
    # Strict improvement is required; ties skip to avoid charging c for no gain.
    observe = bool(probe["risk"] < direct["risk"] - TIE_TOL)
    return {
        "choice": "probe" if observe else "skip",
        "risk": float(probe["risk"] if observe else direct["risk"]),
        "direct": direct,
        "probe": probe,
        "net_gain_of_probe": float(direct["risk"] - probe["risk"]),
    }


def two_step_bellman(
    p: float, r: float, delta: float, c: float, loss: float = L
) -> Dict[str, object]:
    """Exact one-probe/one-terminal Bellman backup by enumeration."""

    direct = direct_decision(p, delta, loss)
    probe = probe_decision(p, r, delta, c, loss)
    if probe["risk"] < direct["risk"] - TIE_TOL:
        choice = "probe"
        risk = probe["risk"]
    else:
        choice = "skip"
        risk = direct["risk"]
    return {
        "choice": choice,
        "risk": float(risk),
        "direct_risk": float(direct["risk"]),
        "probe_risk": float(probe["risk"]),
        "action_if_skip": int(direct["action"]),
        "probe_branch_actions": probe["branch_actions"],
    }


def always_informative(
    p: float, r: float, delta: float, c: float, loss: float = L
) -> Dict[str, object]:
    probe = probe_decision(p, r, delta, c, loss)
    return {
        "choice": "probe",
        "risk": float(probe["risk"]),
        "branch_actions": probe["branch_actions"],
        "probe": probe,
    }


def entropy_sensor_choice(
    p: float,
    r_h: float,
    r_u: float,
    delta: float,
    c: float,
    tie_break: str,
    loss: float = L,
) -> Dict[str, object]:
    """Always pay for the sensor selected by full-state entropy."""

    mi_h = binary_sensor_mi(r_h)
    mi_u = binary_sensor_mi(r_u)
    if mi_h > mi_u + TIE_TOL:
        selected = "H"
    elif mi_u > mi_h + TIE_TOL:
        selected = "U"
    else:
        selected = tie_break

    h_probe = probe_decision(p, r_h, delta, c, loss)
    direct = direct_decision(p, delta, loss)
    # U is independent of H; after U the posterior over H remains the prior.
    # An entropy policy still pays c and then chooses the same direct action.
    u_risk = c + direct["risk"]
    return {
        "selected_sensor": selected,
        "mi_h_bits": float(mi_h),
        "mi_u_bits": float(mi_u),
        "risk": float(h_probe["risk"] if selected == "H" else u_risk),
        "branch_actions": h_probe["branch_actions"] if selected == "H" else [direct["action"], direct["action"]],
        "h_probe": h_probe,
        "u_risk": float(u_risk),
        "tie_break": tie_break,
    }


def analytic_scenario(
    name: str, p: float, r: float, c: float, delta: float, loss: float = L
) -> Dict[str, object]:
    direct = direct_decision(p, delta, loss)
    probe = always_informative(p, r, delta, c, loss)
    cheap = cheap_exact_threshold(p, r, delta, c, loss)
    bellman = two_step_bellman(p, r, delta, c, loss)
    entropy_equal_h = entropy_sensor_choice(p, r, r, delta, c, "H", loss)
    entropy_equal_u = entropy_sensor_choice(p, r, r, delta, c, "U", loss)
    entropy_high_u = entropy_sensor_choice(p, r, 0.95, delta, c, "U", loss)
    return {
        "name": name,
        "parameters": {"p": p, "r": r, "c": c, "delta": delta, "L": loss},
        "direct_myopic": direct,
        "always_informative": probe,
        "cheap_exact_threshold": cheap,
        "two_step_bellman": bellman,
        "entropy_equal_mi_favor_H": entropy_equal_h,
        "entropy_equal_mi_favor_U": entropy_equal_u,
        "entropy_strictly_higher_mi_U": entropy_high_u,
        "mi_equal_bits": float(binary_sensor_mi(r)),
        "mi_h_bits_high_u": float(binary_sensor_mi(r)),
        "mi_u_bits_high_u": float(binary_sensor_mi(0.95)),
    }


def sample_binary_signal(
    rng: np.random.Generator, hidden: np.ndarray, accuracy: float
) -> np.ndarray:
    correct = rng.random(hidden.shape[0]) < accuracy
    return np.where(correct, hidden, -hidden).astype(np.int8)


def balanced_hidden(rng: np.random.Generator, p: float, n: int) -> np.ndarray:
    """Fixed-count stratification for the registered paired MC check."""

    n_plus = int(round(p * n))
    hidden = np.full(n, -1, dtype=np.int8)
    hidden[:n_plus] = 1
    rng.shuffle(hidden)
    return hidden


def action_cost(hidden: np.ndarray, action: np.ndarray, delta: float, loss: float = L) -> np.ndarray:
    wrong = (action != hidden).astype(float) * loss
    extra = (action == 1).astype(float) * delta
    return wrong + extra


def public_action_from_branch(
    p: float, r: float, delta: float, signal: np.ndarray, loss: float = L
) -> np.ndarray:
    """Map public H-signal to terminal action without receiving hidden H."""

    branches = joint_masses(p, r)
    branch_action = {
        y: choose_terminal_action(branches[int(y)], delta, loss)[0] for y in (1, -1)
    }
    return np.asarray([branch_action[int(y)] for y in signal], dtype=np.int8)


def direct_public_action(p: float, delta: float, n: int, loss: float = L) -> np.ndarray:
    a = direct_decision(p, delta, loss)["action"]
    return np.full(n, int(a), dtype=np.int8)


def mc_policy_scores(
    p: float,
    r: float,
    c: float,
    delta: float,
    hidden: np.ndarray,
    h_signal: np.ndarray,
    u_signal_equal: np.ndarray,
    u_signal_high: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    """Score public-signal policies with hidden state used only here."""

    n = hidden.shape[0]
    direct_action = direct_public_action(p, delta, n)
    direct_cost = action_cost(hidden, direct_action, delta)
    h_action = public_action_from_branch(p, r, delta, h_signal)
    h_cost = c + action_cost(hidden, h_action, delta)
    # Exact cheap and Bellman choose the same public choice, but the scorer
    # retains both rows to verify equality numerically rather than aliasing them.
    cheap = cheap_exact_threshold(p, r, delta, c)
    bellman = two_step_bellman(p, r, delta, c)
    cheap_cost = h_cost if cheap["choice"] == "probe" else direct_cost
    bellman_cost = h_cost if bellman["choice"] == "probe" else direct_cost

    # Equal-MI entropy choices are two legitimate tie-breaks.
    entropy_h_cost = h_cost
    entropy_u_cost = c + direct_cost
    # Strictly higher-MI U is selected by entropy; U does not update H.
    entropy_high_u_cost = c + direct_cost

    def row(cost: np.ndarray) -> Dict[str, object]:
        mean = float(cost.mean())
        sd = float(cost.std(ddof=1))
        half = Z95 * sd / math.sqrt(n)
        return {
            "n": int(n),
            "mc_mean": mean,
            "mc_sd": sd,
            "ci95": [mean - half, mean + half],
            "per_case_cost": cost,
        }

    return {
        "myopic_no_observation": row(direct_cost),
        "always_informative_H": row(h_cost),
        "cheap_exact_threshold": row(cheap_cost),
        "two_step_bellman": row(bellman_cost),
        "entropy_equal_mi_favor_H": row(entropy_h_cost),
        "entropy_equal_mi_favor_U": row(entropy_u_cost),
        "entropy_strictly_higher_mi_U": row(entropy_high_u_cost),
        "public_actions": {
            "direct": direct_action,
            "h_signal": h_action,
            "u_signal_equal_not_used_for_h": u_signal_equal,
            "u_signal_high_not_used_for_h": u_signal_high,
        },
    }


def make_pressure_scenarios() -> List[Dict[str, object]]:
    return [
        {
            "name": "skewed_prior",
            "p": 0.9,
            "r": 0.8,
            "c": 0.1,
            "delta": 0.2,
            "group": "zero_decision_gain",
        },
        {
            "name": "uninformative_signal",
            "p": 0.5,
            "r": 0.5,
            "c": 0.1,
            "delta": 0.0,
            "group": "zero_decision_gain",
        },
        {
            "name": "high_observation_cost",
            "p": 0.5,
            "r": 0.8,
            "c": 0.35,
            "delta": 0.0,
            "group": "zero_net_gain",
        },
        {
            "name": "all_branches_choose_minus",
            "p": 0.5,
            "r": 0.8,
            "c": 0.1,
            "delta": 0.6,
            "group": "zero_decision_gain",
        },
    ]


def strip_arrays(value: object) -> object:
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {k: strip_arrays(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, list):
        return [strip_arrays(v) for v in value]
    return value


def run() -> Dict[str, object]:
    scenarios: List[Dict[str, object]] = []
    for delta in DELTA_GRID:
        scenarios.append(
            {
                "name": f"default_delta_{delta:g}",
                "p": P_DEFAULT,
                "r": R_DEFAULT,
                "c": C_DEFAULT,
                "delta": float(delta),
                "group": "default_grid",
            }
        )
    scenarios.extend(make_pressure_scenarios())

    # Fixed paired latent states/signals shared across all scenario rows.
    rng = np.random.default_rng(MC_SEED)
    hidden = balanced_hidden(rng, P_DEFAULT, N_MC)
    h_signal_default = sample_binary_signal(rng, hidden, R_DEFAULT)
    u_signal_equal = sample_binary_signal(rng, balanced_hidden(rng, 0.5, N_MC), R_DEFAULT)
    u_signal_high = sample_binary_signal(rng, balanced_hidden(rng, 0.5, N_MC), 0.95)

    analytic_records = []
    mc_records = []
    branch_tables = []
    raw_arrays: Dict[str, np.ndarray] = {
        "mc_hidden_default_p05": hidden,
        "mc_h_signal_default_r08": h_signal_default,
        "mc_u_signal_equal_mi_r08": u_signal_equal,
        "mc_u_signal_high_mi_r095": u_signal_high,
        "delta_grid": DELTA_GRID,
    }

    for scenario in scenarios:
        p = float(scenario["p"])
        r = float(scenario["r"])
        c = float(scenario["c"])
        delta = float(scenario["delta"])
        name = str(scenario["name"])
        analytic = analytic_scenario(name, p, r, c, delta)
        analytic["group"] = scenario["group"]
        analytic_records.append(analytic)
        branch_tables.append(
            {
                "name": name,
                "p": p,
                "r": r,
                "c": c,
                "delta": delta,
                "branches": probe_decision(p, r, delta, c)["branches"],
            }
        )

        # For pressure cases, generate a paired hidden/signal realization with
        # the scenario's p/r while retaining the fixed MC seed and no selection.
        if p == P_DEFAULT and r == R_DEFAULT:
            h_signal = h_signal_default
            # The hidden draws are p=.5 only for default cases.  Pressure
            # scenarios are scored with a fresh deterministic transform from a
            # fixed auxiliary stream so their declared p is respected.
            if p != P_DEFAULT:
                raise AssertionError("unreachable")
            hidden_scenario = hidden
        else:
            aux_seed = MC_SEED + int(round(1000 * (p + r + c + delta))) + 17
            aux = np.random.default_rng(aux_seed)
            hidden_scenario = balanced_hidden(aux, p, N_MC)
            h_signal = sample_binary_signal(aux, hidden_scenario, r)

        # Default rows use one shared H/U realization; pressure rows use a
        # deterministic per-row stream to keep their own p/r explicit.
        if p == P_DEFAULT and r == R_DEFAULT and c == C_DEFAULT:
            u_equal = u_signal_equal
            u_high = u_signal_high
        else:
            aux_u = np.random.default_rng(
                MC_SEED + 100000 + int(round(1000 * (p + r + c + delta))) + 17
            )
            u_equal = sample_binary_signal(aux_u, balanced_hidden(aux_u, 0.5, N_MC), r)
            u_high = sample_binary_signal(aux_u, balanced_hidden(aux_u, 0.5, N_MC), 0.95)

        mc = mc_policy_scores(p, r, c, delta, hidden_scenario, h_signal, u_equal, u_high)
        analytic_lookup = {
            "myopic_no_observation": analytic["direct_myopic"]["risk"],
            "always_informative_H": analytic["always_informative"]["risk"],
            "cheap_exact_threshold": analytic["cheap_exact_threshold"]["risk"],
            "two_step_bellman": analytic["two_step_bellman"]["risk"],
            "entropy_equal_mi_favor_H": analytic["entropy_equal_mi_favor_H"]["risk"],
            "entropy_equal_mi_favor_U": analytic["entropy_equal_mi_favor_U"]["risk"],
            "entropy_strictly_higher_mi_U": analytic["entropy_strictly_higher_mi_U"]["risk"],
        }
        mc_summary = {}
        for policy, row in mc.items():
            if policy == "public_actions":
                continue
            mc_summary[policy] = {
                "analytic_risk": float(analytic_lookup[policy]),
                "mc_mean": row["mc_mean"],
                "mc_sd": row["mc_sd"],
                "ci95": row["ci95"],
                "analytic_minus_mc": float(analytic_lookup[policy] - row["mc_mean"]),
            }
            raw_arrays[f"mc_cost_{name}_{policy}"] = row["per_case_cost"]
        mc_records.append({"name": name, "policies": mc_summary})
        raw_arrays[f"mc_hidden_{name}"] = hidden_scenario
        raw_arrays[f"mc_h_signal_{name}"] = h_signal
        raw_arrays[f"mc_u_signal_equal_{name}"] = u_equal
        raw_arrays[f"mc_u_signal_high_{name}"] = u_high

    # Exact threshold checks for the default grid.
    default = [x for x in analytic_records if x["name"].startswith("default_delta_")]
    cheap_choices = [x["cheap_exact_threshold"]["choice"] for x in default]
    bellman_choices = [x["two_step_bellman"]["choice"] for x in default]
    expected_default_choices = ["probe", "probe", "probe", "skip", "skip", "skip", "skip"]

    result = {
        "status": "exact_binary_decision_mechanism",
        "protocol": {
            "L": L,
            "p_default": P_DEFAULT,
            "r_default": R_DEFAULT,
            "c_default": C_DEFAULT,
            "delta_grid": DELTA_GRID.tolist(),
            "mc_seed": MC_SEED,
            "n_mc": N_MC,
            "u_equal_accuracy": R_DEFAULT,
            "u_high_accuracy": 0.95,
            "tie_rule": "terminal ties choose minus; probe ties choose skip",
        },
        "default_grid": {
            "analytic": [strip_arrays(x) for x in default],
            "cheap_choices": cheap_choices,
            "bellman_choices": bellman_choices,
            "expected_threshold_choices": expected_default_choices,
            "cheap_equals_bellman": cheap_choices == bellman_choices,
            "threshold_grid_matches": cheap_choices == expected_default_choices,
        },
        "pressure_groups": [strip_arrays(x) for x in analytic_records if x["group"] != "default_grid"],
        "branch_tables": branch_tables,
        "mc": mc_records,
        "mi": {
            "equal_mi_bits": binary_sensor_mi(R_DEFAULT),
            "h_default_bits": binary_sensor_mi(R_DEFAULT),
            "u_high_bits": binary_sensor_mi(0.95),
            "u_high_strictly_greater": bool(binary_sensor_mi(0.95) > binary_sensor_mi(R_DEFAULT)),
        },
        "interpretation": {
            "failure_to_success": "The unconditioned always-probe/entropy choices can pay for an observation outside its decision-value threshold; cheap-exact and two-step Bellman switch at Delta=.4 on the default grid.",
            "same_mi_caveat": "Equal-MI H/U is a tie; favoring U is an allowed but unfavorable tie-break, not a universal entropy failure.",
            "scope": "This is a standard binary Bayes threshold mechanism, not a claim of novel VOI theory, complete B, or real robot performance.",
        },
    }

    np.savez_compressed(ROOT / "RAW_RESULTS.npz", **raw_arrays)
    (ROOT / "RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ROOT / "REPORT.txt").write_text(make_report(result), encoding="utf-8")
    return result


def fmt(value: float) -> str:
    return f"{float(value):.8g}"


def make_report(result: Dict[str, object]) -> str:
    lines = [
        "B binary decision mechanism report",
        "===================================",
        "",
        "Scope: exact toy Bayes decision threshold; not ultrasound/robot validation and not a new VOI theorem.",
        "",
        "Default p=.5, r=.8, c=.1, L=1",
        "--------------------------------",
        "delta | direct | always-probe | cheap choice/risk | Bellman choice/risk | entropy H-tie | entropy U-tie | entropy higher-MI U",
    ]
    for rec in result["default_grid"]["analytic"]:
        lines.append(
            f"{rec['parameters']['delta']:>4g} | "
            f"{fmt(rec['direct_myopic']['risk']):>8} | "
            f"{fmt(rec['always_informative']['risk']):>11} | "
            f"{rec['cheap_exact_threshold']['choice']:>5}/{fmt(rec['cheap_exact_threshold']['risk']):<9} | "
            f"{rec['two_step_bellman']['choice']:>5}/{fmt(rec['two_step_bellman']['risk']):<9} | "
            f"{fmt(rec['entropy_equal_mi_favor_H']['risk']):>10} | "
            f"{fmt(rec['entropy_equal_mi_favor_U']['risk']):>10} | "
            f"{fmt(rec['entropy_strictly_higher_mi_U']['risk']):>15}"
        )
    lines.extend(
        [
            "",
            "Threshold check",
            "---------------",
            f"cheap choices:  {result['default_grid']['cheap_choices']}",
            f"Bellman choices:{result['default_grid']['bellman_choices']}",
            f"cheap == Bellman: {result['default_grid']['cheap_equals_bellman']}; grid threshold matches: {result['default_grid']['threshold_grid_matches']}",
            "",
            "Pressure groups",
            "---------------",
        ]
    )
    for rec in result["pressure_groups"]:
        lines.append(
            f"{rec['name']}: direct={fmt(rec['direct_myopic']['risk'])}, "
            f"probe={fmt(rec['always_informative']['risk'])}, "
            f"cheap={rec['cheap_exact_threshold']['choice']}/{fmt(rec['cheap_exact_threshold']['risk'])}, "
            f"branch_actions={rec['always_informative']['branch_actions']}, "
            f"branch_same={rec['always_informative']['probe']['branch_actions_same']}"
        )
    lines.extend(["", "Paired Monte Carlo (N=4096)", "----------------------------"])
    for rec in result["mc"]:
        lines.append(f"{rec['name']}:")
        for policy, row in rec["policies"].items():
            lines.append(
                f"  {policy}: analytic={fmt(row['analytic_risk'])}, mc={fmt(row['mc_mean'])}, "
                f"CI95=[{fmt(row['ci95'][0])},{fmt(row['ci95'][1])}], "
                f"analytic-mc={fmt(row['analytic_minus_mc'])}"
            )
    lines.extend(
        [
            "",
            "Mutual information",
            "-------------------",
            f"equal-MI H/U: {fmt(result['mi']['equal_mi_bits'])} bits; higher-MI U: {fmt(result['mi']['u_high_bits'])} bits; strict={result['mi']['u_high_strictly_greater']}.",
            "Equal-MI H/U is a tie, so the two tie-break results are both reported.  The higher-MI U result is intentionally task-irrelevant and is not an equal-MI claim.",
            "",
            "Interpretation",
            "--------------",
            "The exact two-step decision changes from probe to skip at Delta=.4 because the observation's decision benefit falls below its cost. Always probing and entropy selection do not perform this task-value threshold. The result is a standard binary Bayes mechanism and only supports the narrow claim that a reliable observation can be worth or not worth its cost depending on downstream executable cost.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_manifest() -> None:
    names = [
        "PROTOCOL.txt",
        "run_mechanism.py",
        "test_mechanism.py",
        "RAW_RESULTS.npz",
        "RESULTS.json",
        "REPORT.txt",
    ]
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "files_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
    write_manifest()
