#!/usr/bin/env python3
"""Exact, finite-state adaptive-horizon versus memory comparison.

This is deliberately a small symbolic model.  It is not a real acoustic or
robotics fixture: the only latent variables are two independent signs and the
only observations are the two stated noisy signs.  The script computes every
expectation by enumerating the four latent states and all binary observation
branches.  No Monte-Carlo estimate is used for the reported values.

Run from any directory with::

    python run_exact.py

The default output is ``RESULTS.json`` next to this file.  The model and all
policy legality checks live here so the JSON can be regenerated without any
other repository files.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


State = Tuple[int, int]  # (S, R), each in {-1,+1}
History = Tuple[Tuple[str, int], ...]

SIGNS: Tuple[int, int] = (-1, 1)
STATES: Tuple[State, ...] = tuple((s, r) for s in SIGNS for r in SIGNS)
DEFAULT_PRIOR: Mapping[State, float] = {state: 0.25 for state in STATES}
EPS = 1e-12


@dataclass(frozen=True)
class Model:
    """Public action, timing, and reward contract.

    ``h`` below is a *macro decision depth* (a terminal action counts as one
    decision).  Physical legality still uses these explicit costs and the
    common deadline/budget.  This keeps the requested h=1/2/3/4 comparison
    meaningful even though a terminal action consumes three unit-time ticks.
    """

    accuracy: float = 0.96
    budget: int = 5
    probe_cost: int = 1
    terminal_cost: int = 3
    fallback_success: float = 0.67
    max_terminal_actions: int = 1
    probes: Tuple[str, ...] = ("probe_R", "probe_P")
    terminal_sides: Tuple[int, ...] = (-1, 1)
    geometry_summary: str = "left_right_equivalent_public_summary"
    image_id: str = "shared_symbolic_image_v1"


@dataclass
class Node:
    """A deterministic policy tree rooted at one public belief state."""

    action: str
    value: float
    expected_time: float
    terminal_choice: Optional[str] = None
    children: Dict[int, "Node"] = field(default_factory=dict)

    def to_json(self, depth: int = 0, max_depth: int = 8) -> dict:
        """Compact serialisation used for audit/debug output."""

        row = {
            "action": self.action,
            "value": self.value,
            "expected_time": self.expected_time,
        }
        if self.terminal_choice is not None:
            row["terminal_choice"] = self.terminal_choice
        if self.children and depth < max_depth:
            row["children"] = {
                str(obs): child.to_json(depth + 1, max_depth)
                for obs, child in sorted(self.children.items())
            }
        elif self.children:
            row["children"] = {"...": "truncated"}
        return row


def _normalise(weights: Mapping[State, float]) -> Dict[State, float]:
    z = sum(weights.values())
    if z <= 0.0:
        raise ValueError("posterior has zero mass")
    return {state: weight / z for state, weight in weights.items()}


def true_signal(state: State, action: str) -> int:
    """Latent sign returned by a probe before independent flip noise."""

    s, r = state
    if action == "probe_R":
        return r
    if action == "probe_P":
        return s * r
    raise ValueError(f"not a signal probe: {action}")


def likelihood(state: State, action: str, observation: int, accuracy: float) -> float:
    if observation not in SIGNS:
        raise ValueError(f"binary sign expected, got {observation!r}")
    return accuracy if observation == true_signal(state, action) else 1.0 - accuracy


def posterior(
    history: History,
    accuracy: float,
    prior: Mapping[State, float] = DEFAULT_PRIOR,
) -> Dict[State, float]:
    """Bayes update over the persistent hidden (S,R) state.

    Unknown/non-probe events (for example forced middle steps in the prefix
    study) are deliberately ignored because they are independent of (S,R).
    """

    weights: Dict[State, float] = {state: float(prior[state]) for state in STATES}
    for action, observation in history:
        if action not in ("probe_R", "probe_P"):
            continue
        for state in STATES:
            weights[state] *= likelihood(state, action, observation, accuracy)
    return _normalise(weights)


def predictive_outcomes(
    history: History,
    action: str,
    accuracy: float,
    prior: Mapping[State, float] = DEFAULT_PRIOR,
) -> Dict[int, float]:
    post = posterior(history, accuracy, prior)
    return {
        observation: sum(
            post[state] * likelihood(state, action, observation, accuracy)
            for state in STATES
        )
        for observation in SIGNS
    }


def side_probability(post: Mapping[State, float], side: int) -> float:
    return sum(weight for state, weight in post.items() if state[0] == side)


def best_terminal(
    post: Mapping[State, float],
    model: Model,
    remaining_budget: int,
    *,
    terminal_sides: Optional[Sequence[int]] = None,
    fallback_enabled: bool = True,
) -> Optional[Node]:
    """Choose the best legal terminal action, with shortest-time tie-break."""

    if remaining_budget < model.terminal_cost:
        return None

    sides = tuple(model.terminal_sides if terminal_sides is None else terminal_sides)
    options = []
    for side in sides:
        options.append((side_probability(post, side), f"terminal:{side}"))
    if fallback_enabled:
        options.append((model.fallback_success, "fallback"))
    if not options:
        return None

    # All terminal actions have the same duration in this fixture.  The action
    # name is only a deterministic final tie-break for reproducible JSON.
    value, choice = max(options, key=lambda row: (row[0], row[1] == "fallback", row[1]))
    return Node(
        action="terminal",
        terminal_choice=choice,
        value=float(value),
        expected_time=float(model.terminal_cost),
    )


def _better(candidate: Node, incumbent: Optional[Node]) -> bool:
    if incumbent is None:
        return True
    if candidate.value > incumbent.value + EPS:
        return True
    if abs(candidate.value - incumbent.value) <= EPS:
        if candidate.expected_time < incumbent.expected_time - EPS:
            return True
        if abs(candidate.expected_time - incumbent.expected_time) <= EPS:
            # Stable action order only affects exact ties, never the value.
            order = {"terminal": 0, "probe_R": 1, "probe_P": 2, "stop": 3}
            return order.get(candidate.action, 9) < order.get(incumbent.action, 9)
    return False


def adaptive_plan(
    h: int,
    model: Model,
    *,
    assumed_accuracy: Optional[float] = None,
    prior: Mapping[State, float] = DEFAULT_PRIOR,
    allowed_probes: Optional[Sequence[str]] = None,
    terminal_sides: Optional[Sequence[int]] = None,
    fallback_enabled: bool = True,
) -> Node:
    """Exact finite-horizon Bayes-optimal adaptive policy.

    The planner can branch on each probe result.  A terminal action ends the
    episode, so the max-terminal-action constraint is structural.  A probe is
    used at most once, matching the two legal windows in the protocol.
    """

    if h < 0:
        raise ValueError("h must be non-negative")
    acc = model.accuracy if assumed_accuracy is None else float(assumed_accuracy)
    probes = tuple(model.probes if allowed_probes is None else allowed_probes)
    if any(action not in model.probes for action in probes):
        raise ValueError("allowed_probes contains an unknown action")

    @lru_cache(maxsize=None)
    def solve(
        history: History,
        budget_left: int,
        depth_left: int,
        remaining_probes: Tuple[str, ...],
    ) -> Node:
        post = posterior(history, acc, prior)
        terminal = best_terminal(
            post,
            model,
            budget_left,
            terminal_sides=terminal_sides,
            fallback_enabled=fallback_enabled,
        )
        candidates = []
        if terminal is not None and depth_left >= 1:
            candidates.append(terminal)

        # A probe at depth 1 cannot reach a terminal action, and therefore has
        # value zero.  Omitting that dominated branch also keeps the tree small.
        if depth_left >= 2:
            for action in remaining_probes:
                if budget_left < model.probe_cost:
                    continue
                future_remaining = tuple(a for a in remaining_probes if a != action)
                branch_probs = predictive_outcomes(history, action, acc, prior)
                children = {}
                value = 0.0
                expected_time = float(model.probe_cost)
                for observation, probability in branch_probs.items():
                    child_history = history + ((action, observation),)
                    child = solve(
                        child_history,
                        budget_left - model.probe_cost,
                        depth_left - 1,
                        future_remaining,
                    )
                    children[observation] = child
                    value += probability * child.value
                    expected_time += probability * child.expected_time
                candidates.append(
                    Node(
                        action=action,
                        value=float(value),
                        expected_time=float(expected_time),
                        children=children,
                    )
                )

        if not candidates:
            return Node(action="stop", value=0.0, expected_time=0.0)
        best = candidates[0]
        for candidate in candidates[1:]:
            if _better(candidate, best):
                best = candidate
        return best

    return solve(tuple(), model.budget, int(h), tuple(probes))


def fixed_open_loop_plan(
    n: int,
    model: Model,
    *,
    assumed_accuracy: Optional[float] = None,
    prior: Mapping[State, float] = DEFAULT_PRIOR,
    probe_sequence: Optional[Sequence[str]] = None,
    terminal_sides: Optional[Sequence[int]] = None,
    fallback_enabled: bool = True,
) -> Node:
    """Commit to exactly ``n`` probes before seeing any result.

    Terminal-side selection may use all retained probe outputs after the fixed
    window.  The action sequence itself cannot branch.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    acc = model.accuracy if assumed_accuracy is None else float(assumed_accuracy)
    sequence = tuple(model.probes if probe_sequence is None else probe_sequence)
    if n > len(sequence):
        raise ValueError("n exceeds the available legal probe sequence")

    def build(index: int, history: History, budget_left: int) -> Node:
        if index >= n:
            post = posterior(history, acc, prior)
            terminal = best_terminal(
                post,
                model,
                budget_left,
                terminal_sides=terminal_sides,
                fallback_enabled=fallback_enabled,
            )
            return terminal if terminal is not None else Node("stop", 0.0, 0.0)

        action = sequence[index]
        if budget_left < model.probe_cost:
            return Node("stop", 0.0, 0.0)
        branch_probs = predictive_outcomes(history, action, acc, prior)
        children: Dict[int, Node] = {}
        value = 0.0
        expected_time = float(model.probe_cost)
        for observation, probability in branch_probs.items():
            child = build(
                index + 1,
                history + ((action, observation),),
                budget_left - model.probe_cost,
            )
            children[observation] = child
            value += probability * child.value
            expected_time += probability * child.expected_time
        return Node(
            action=action,
            value=float(value),
            expected_time=float(expected_time),
            children=children,
        )

    return build(0, tuple(), model.budget)


def _node_cost(node: Node, model: Model) -> int:
    if node.action in ("probe_R", "probe_P"):
        return model.probe_cost
    if node.action == "terminal":
        return model.terminal_cost
    return 0


def max_elapsed(node: Node, model: Model) -> int:
    """Maximum physical cost over any branch of a policy tree."""

    if not node.children:
        return _node_cost(node, model)
    return _node_cost(node, model) + max(
        max_elapsed(child, model) for child in node.children.values()
    )


def terminal_count(node: Node) -> int:
    if node.action == "terminal":
        return 1
    return sum(terminal_count(child) for child in node.children.values())


def max_terminals_on_path(node: Node) -> int:
    """Maximum number of terminal actions on any single policy branch."""

    own = 1 if node.action == "terminal" else 0
    if not node.children:
        return own
    return own + max(max_terminals_on_path(child) for child in node.children.values())


def evaluate_policy_exact(
    node: Node,
    model: Model,
    *,
    true_accuracy: Optional[float] = None,
    prior: Mapping[State, float] = DEFAULT_PRIOR,
) -> Tuple[float, float]:
    """Independent hidden-state/observation enumeration of a policy tree."""

    acc = model.accuracy if true_accuracy is None else float(true_accuracy)

    def recurse(current: Node, state: State) -> Tuple[float, float]:
        if current.action == "terminal":
            if current.terminal_choice == "fallback":
                return model.fallback_success, float(model.terminal_cost)
            if current.terminal_choice == "terminal:-1":
                return float(state[0] == -1), float(model.terminal_cost)
            if current.terminal_choice == "terminal:1":
                return float(state[0] == 1), float(model.terminal_cost)
            raise ValueError(f"unknown terminal choice {current.terminal_choice}")
        if current.action == "stop":
            return 0.0, 0.0
        if current.action not in ("probe_R", "probe_P"):
            raise ValueError(f"unknown policy action {current.action}")
        value = 0.0
        expected_time = float(model.probe_cost)
        for observation, child in current.children.items():
            probability = likelihood(state, current.action, observation, acc)
            child_value, child_time = recurse(child, state)
            value += probability * child_value
            expected_time += probability * child_time
        return value, expected_time

    value = 0.0
    expected_time = 0.0
    for state in STATES:
        state_value, state_time = recurse(node, state)
        value += float(prior[state]) * state_value
        expected_time += float(prior[state]) * state_time
    return value, expected_time


def posterior_table(history: History, accuracy: float) -> Dict[str, float]:
    return {
        f"S{s},R{r}": posterior(history, accuracy)[(s, r)]
        for s, r in STATES
    }


def summarise_policy(
    node: Node,
    model: Model,
    *,
    true_accuracy: Optional[float] = None,
) -> dict:
    actual_value, actual_time = evaluate_policy_exact(
        node, model, true_accuracy=true_accuracy
    )
    return {
        "planner_value": node.value,
        "exact_enumerated_value": actual_value,
        "expected_time": actual_time,
        "root_action": (
            f"{node.action}:{node.terminal_choice}"
            if node.action == "terminal"
            else node.action
        ),
        "max_elapsed": max_elapsed(node, model),
        "terminal_nodes": terminal_count(node),
        "max_terminals_on_any_path": max_terminals_on_path(node),
        "policy_tree": node.to_json(),
    }


def _assert_close(a: float, b: float, label: str) -> None:
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=2e-12):
        raise AssertionError(f"{label}: {a} != {b}")


def main_results() -> dict:
    model = Model()
    formula_two_probe = model.accuracy**2 + (1.0 - model.accuracy) ** 2

    horizon_rows = []
    independent_checks = []
    for h in (1, 2, 3, 4):
        full = adaptive_plan(h, model, assumed_accuracy=model.accuracy)
        ignored = adaptive_plan(h, model, assumed_accuracy=0.5)
        fixed_candidates = []
        for n in range(min(2, max(0, h - 1)) + 1):
            fixed = fixed_open_loop_plan(
                n, model, assumed_accuracy=model.accuracy
            )
            fixed_actual, fixed_time = evaluate_policy_exact(fixed, model)
            fixed_candidates.append(
                {
                    "n_probes": n,
                    "planner_value": fixed.value,
                    "exact_enumerated_value": fixed_actual,
                    "expected_time": fixed_time,
                    "root_action": (
                        f"{fixed.action}:{fixed.terminal_choice}"
                        if fixed.action == "terminal"
                        else fixed.action
                    ),
                    "max_elapsed": max_elapsed(fixed, model),
                }
            )
        best_fixed = fixed_candidates[0]
        for candidate in fixed_candidates[1:]:
            # The rows are summaries rather than Node objects, so apply the
            # same max-value/shortest-time rule explicitly.
            if candidate["exact_enumerated_value"] > best_fixed[
                "exact_enumerated_value"
            ] + EPS or (
                abs(
                    candidate["exact_enumerated_value"]
                    - best_fixed["exact_enumerated_value"]
                )
                <= EPS
                and candidate["expected_time"] < best_fixed["expected_time"] - EPS
            ):
                best_fixed = candidate

        full_summary = summarise_policy(full, model)
        ignored_summary = summarise_policy(ignored, model, true_accuracy=model.accuracy)
        horizon_rows.append(
            {
                "h": h,
                "adaptive_full_bayes": full_summary,
                "adaptive_ignore_diagnostic": ignored_summary,
                "fixed_open_loop_candidates": fixed_candidates,
                "fixed_open_loop_selected": best_fixed,
            }
        )
        independent_checks.append(
            {
                "h": h,
                "adaptive_dp_vs_enumeration_abs_error": abs(
                    full.value - full_summary["exact_enumerated_value"]
                ),
                "ignore_dp_vs_enumeration_abs_error": abs(
                    ignored.value - ignored_summary["exact_enumerated_value"]
                ),
            }
        )

    fixed_windows = []
    for n in (0, 1, 2):
        full = fixed_open_loop_plan(n, model, assumed_accuracy=model.accuracy)
        ignored = fixed_open_loop_plan(n, model, assumed_accuracy=0.5)
        full_summary = summarise_policy(full, model)
        ignored_summary = summarise_policy(
            ignored, model, true_accuracy=model.accuracy
        )
        fixed_windows.append(
            {
                "n_probes": n,
                "full_bayes": full_summary,
                "ignore_diagnostic": ignored_summary,
            }
        )

    # Pure-geometry A never reads the diagnostic channel.  Its geometry-only
    # public summary is symmetric, so the legal fallback is its optimum.
    pure_geometry = fixed_open_loop_plan(0, model, assumed_accuracy=0.5)
    strong_a = adaptive_plan(3, model, assumed_accuracy=model.accuracy)
    method_classification = {
        "pure_geometry_A": {
            "reads": ["geometry_equivalent_public_summary"],
            "diagnostic_channel": "ignored",
            "policy": summarise_policy(pure_geometry, model),
        },
        "strong_A_full_history_planning": {
            "reads": ["same_geometry_summary", "all_persistent_probe_history"],
            "diagnostic_channel": "full_bayes",
            "policy": summarise_policy(strong_a, model),
            "fairness_note": "same model, legal actions, image, time, and deadline as B",
        },
        "B_adaptive_full_bayes": {
            "reads": ["same_geometry_summary", "all_persistent_probe_history"],
            "diagnostic_channel": "full_bayes",
            "policy": summarise_policy(strong_a, model),
            "fairness_note": "identical public action contract; only replanning/memory policy differs",
        },
    }

    prefix = prefix_results(model)
    counterexamples = counterexample_results(model)

    # Independent checks are intentionally duplicated in this script rather
    # than inferred from the DP recurrence.
    _assert_close(formula_two_probe, 0.9232, "two-probe formula")
    _assert_close(
        fixed_windows[2]["full_bayes"]["exact_enumerated_value"],
        formula_two_probe,
        "two-probe enumeration",
    )
    for row in horizon_rows:
        _assert_close(
            row["adaptive_full_bayes"]["planner_value"],
            row["adaptive_full_bayes"]["exact_enumerated_value"],
            f"adaptive h={row['h']}",
        )
        if row["adaptive_full_bayes"]["max_elapsed"] > model.budget:
            raise AssertionError("adaptive policy exceeded the common deadline")

    return {
        "schema": "ab_core_memory_exact_v1",
        "generated_by": "mechanism/run_exact.py",
        "scope": "symbolic exact finite-state mechanism only; no real acoustic fixture",
        "model": {
            "hidden_state": "S,R independent uniform signs in {-1,+1}",
            "probe_R_signal": "R",
            "probe_P_signal": "S*R",
            "probe_accuracy": model.accuracy,
            "probe_cost": model.probe_cost,
            "terminal_cost": model.terminal_cost,
            "budget": model.budget,
            "max_terminal_actions": model.max_terminal_actions,
            "fallback_success": model.fallback_success,
            "terminal_sides": list(model.terminal_sides),
            "image_id_shared_by_all_methods": model.image_id,
            "geometry_summary": model.geometry_summary,
            "h_definition": "macro decision depth; costs enforce the common physical deadline",
            "exact_expectation": "enumerate all four hidden states and every binary probe branch",
        },
        "shared_action_catalog": {
            "probe_R": {
                "legal": True,
                "cost": model.probe_cost,
                "direct_reward": 0.0,
                "image_id": model.image_id,
                "observable": "noisy R",
            },
            "probe_P": {
                "legal": True,
                "cost": model.probe_cost,
                "direct_reward": 0.0,
                "image_id": model.image_id,
                "observable": "noisy S*R",
            },
            "terminal:-1": {
                "legal": True,
                "cost": model.terminal_cost,
                "image_id": model.image_id,
                "success_condition": "S=-1",
                "max_terminal_actions": 1,
            },
            "terminal:1": {
                "legal": True,
                "cost": model.terminal_cost,
                "image_id": model.image_id,
                "success_condition": "S=+1",
                "max_terminal_actions": 1,
            },
            "fallback": {
                "legal": True,
                "cost": model.terminal_cost,
                "image_id": model.image_id,
                "independent_success": model.fallback_success,
                "max_terminal_actions": 1,
            },
        },
        "formula_checks": {
            "one_probe_P_independent_of_S": True,
            "two_probe_success": formula_two_probe,
            "two_probe_success_formula": "a^2 + (1-a)^2",
            "fallback_gap_at_a_0.96": formula_two_probe - model.fallback_success,
            "two_probe_beats_fallback_if_accuracy_gt": 0.5 + math.sqrt(
                (model.fallback_success - 0.5) / 2.0
            ),
        },
        "horizon_rows": horizon_rows,
        "fixed_windows": fixed_windows,
        "method_classification": method_classification,
        "fixed_prefix": prefix,
        "counterexamples": counterexamples,
        "independent_checks": independent_checks,
        "verification": {
            "all_policy_branches_within_budget": all(
                row["adaptive_full_bayes"]["max_elapsed"] <= model.budget
                for row in horizon_rows
            ),
            "all_policies_max_one_terminal": all(
                row["adaptive_full_bayes"]["max_terminals_on_any_path"] <= 1
                for row in horizon_rows
            ),
            "exact_two_probe_value": formula_two_probe,
        },
    }


def prefix_results(model: Model) -> dict:
    """Exact forced-prefix study with explicit memory retention policies."""

    prefix_cost = 1 + 3  # R observation + three forced uninformative steps
    target_probe_cost = model.probe_cost
    deadline = prefix_cost + target_probe_cost + model.terminal_cost
    middle_history = (("middle_1", 0), ("middle_2", 0), ("middle_3", 0))

    def retained_history(y_r: int, y_p: Optional[int], mode: str) -> History:
        full: History = (("probe_R", y_r),) + middle_history
        if y_p is not None:
            full += (("probe_P", y_p),)
        if mode in ("full_history", "replay", "address_cache"):
            return full
        if mode == "window1":
            return full[-1:]
        if mode == "window2":
            return full[-2:]
        if mode == "ignore_diagnostic":
            return tuple(event for event in full if event[0].startswith("middle"))
        if mode == "pure_geometry_A":
            return tuple(event for event in full if event[0].startswith("middle"))
        raise ValueError(mode)

    def target_decision(y_r: int, mode: str) -> Tuple[str, float, float]:
        # At target local, only P is legal; R cannot be re-read.  Compare an
        # immediate terminal with probe_P followed by the best terminal.
        pre = retained_history(y_r, None, mode)
        assumed_accuracy = 0.5 if mode in ("ignore_diagnostic", "pure_geometry_A") else model.accuracy
        post = posterior(pre, assumed_accuracy)
        direct = best_terminal(post, model, deadline - prefix_cost)
        if direct is None:
            raise AssertionError("prefix direct terminal should be affordable")

        probe_probs = predictive_outcomes(pre, "probe_P", assumed_accuracy)
        future_value = 0.0
        future_time = float(target_probe_cost)
        children = {}
        for observation, probability in probe_probs.items():
            after = retained_history(y_r, observation, mode)
            leaf_post = posterior(after, assumed_accuracy)
            leaf = best_terminal(
                leaf_post,
                model,
                deadline - prefix_cost - target_probe_cost,
            )
            if leaf is None:
                raise AssertionError("prefix probe terminal should be affordable")
            children[observation] = leaf
            future_value += probability * leaf.value
            future_time += probability * leaf.expected_time
        probe_node = Node(
            action="probe_P",
            value=future_value,
            expected_time=future_time,
            children=children,
        )
        if _better(probe_node, direct):
            return "probe_P", probe_node.value, probe_node.expected_time
        return (
            f"{direct.action}:{direct.terminal_choice}",
            direct.value,
            direct.expected_time,
        )

    modes = (
        "full_history",
        "window1",
        "window2",
        "replay",
        "address_cache",
        "ignore_diagnostic",
        "pure_geometry_A",
    )
    rows = []
    for mode in modes:
        value = 0.0
        elapsed = 0.0
        # Enumerate the prefix R branch, then the target P branch if the policy
        # actually chooses it.  Middle steps are fixed and carry no mass.
        for state in STATES:
            state_weight = 0.25
            for y_r in SIGNS:
                p_r = likelihood(state, "probe_R", y_r, model.accuracy)
                target_action, assumed_target_value, assumed_target_time = target_decision(
                    y_r, mode
                )
                if target_action == "probe_P":
                    for y_p in SIGNS:
                        p_p = likelihood(state, "probe_P", y_p, model.accuracy)
                        after = retained_history(y_r, y_p, mode)
                        # The selected policy's leaf is recomputed from its
                        # retained state under the assumed model, while reward
                        # uses the true hidden S and the true sensor branch.
                        assumed_accuracy = (
                            0.5
                            if mode in ("ignore_diagnostic", "pure_geometry_A")
                            else model.accuracy
                        )
                        leaf = best_terminal(
                            posterior(after, assumed_accuracy),
                            model,
                            deadline - prefix_cost - target_probe_cost,
                        )
                        if leaf is None:
                            raise AssertionError("prefix leaf not affordable")
                        if leaf.terminal_choice == "fallback":
                            reward = model.fallback_success
                        else:
                            chosen_side = int(leaf.terminal_choice.split(":", 1)[1])
                            reward = float(chosen_side == state[0])
                        probability = state_weight * p_r * p_p
                        value += probability * reward
                        elapsed += probability * (
                            prefix_cost + target_probe_cost + model.terminal_cost
                        )
                else:
                    # Direct terminal after the fixed prefix.
                    assumed_accuracy = (
                        0.5
                        if mode in ("ignore_diagnostic", "pure_geometry_A")
                        else model.accuracy
                    )
                    direct_post = posterior(
                        retained_history(y_r, None, mode), assumed_accuracy
                    )
                    direct = best_terminal(
                        direct_post, model, deadline - prefix_cost
                    )
                    if direct is None:
                        raise AssertionError("prefix direct action unavailable")
                    if direct.terminal_choice == "fallback":
                        reward = model.fallback_success
                    else:
                        chosen_side = int(direct.terminal_choice.split(":", 1)[1])
                        reward = float(chosen_side == state[0])
                    probability = state_weight * p_r
                    value += probability * reward
                    elapsed += probability * (prefix_cost + model.terminal_cost)

        retained_after = {
            "full_history": "R,middle_1,middle_2,middle_3,P",
            "window1": "P",
            "window2": "middle_3,P",
            "replay": "R,middle_1,middle_2,middle_3,P",
            "address_cache": "R,middle_1,middle_2,middle_3,P",
            "ignore_diagnostic": "middle_1,middle_2,middle_3,P (diagnostics ignored)",
            "pure_geometry_A": "middle_1,middle_2,middle_3,P (diagnostics ignored)",
        }[mode]
        # All y_R branches have the same action in this symmetric model.
        target_action = target_decision(1, mode)[0]
        rows.append(
            {
                "memory_policy": mode,
                "retained_after_target": retained_after,
                "drops_old_R": mode in ("window1", "window2", "ignore_diagnostic", "pure_geometry_A"),
                "target_action": target_action,
                "exact_enumerated_success": value,
                "expected_elapsed": elapsed,
                "prefix_cost": prefix_cost,
                "deadline": deadline,
                "public_robot_exit": "same_target_local_after_three_forced_steps",
                "old_patch_revisit": "forbidden",
                "image_id": model.image_id,
            }
        )

    values = {row["memory_policy"]: row["exact_enumerated_success"] for row in rows}
    _assert_close(values["full_history"], 0.9232, "prefix full history")
    _assert_close(values["replay"], values["full_history"], "prefix replay identity")
    _assert_close(values["address_cache"], values["full_history"], "prefix cache identity")
    _assert_close(values["window1"], 0.67, "prefix window1")
    _assert_close(values["window2"], 0.67, "prefix window2")
    return {
        "protocol": {
            "forced_prefix": [
                {"action": "probe_R", "cost": 1, "observable": "noisy R"},
                {"action": "middle_1", "cost": 1, "observable": "independent uninformative"},
                {"action": "middle_2", "cost": 1, "observable": "independent uninformative"},
                {"action": "middle_3", "cost": 1, "observable": "independent uninformative"},
            ],
            "target_local_legal_probe": "probe_P only",
            "target_local_terminal_cost": model.terminal_cost,
            "budget_equal_to_deadline": deadline,
            "shared_public_exit_and_time": True,
            "no_old_patch_revisit": True,
            "full_history_replay_and_address_cache_same": True,
        },
        "rows": rows,
        "separation": {
            "full_history_minus_window1": values["full_history"] - values["window1"],
            "full_history_minus_window2": values["full_history"] - values["window2"],
            "information_sufficient_only_when_R_and_P_coexist": True,
        },
    }


def stale_memory_value(model: Model) -> float:
    """Exact value of multiplying current P by an independent old-R cache."""

    value = 0.0
    for s in SIGNS:
        for r_new in SIGNS:
            for r_old in SIGNS:
                state_weight = 1.0 / 8.0
                for y_old in SIGNS:
                    p_old = likelihood((s, r_old), "probe_R", y_old, model.accuracy)
                    for y_p in SIGNS:
                        p_p = likelihood((s, r_new), "probe_P", y_p, model.accuracy)
                        guessed_s = y_old * y_p
                        value += state_weight * p_old * p_p * float(guessed_s == s)
    return value


def counterexample_results(model: Model) -> dict:
    # No information: both probe outcomes are fair coins.  The correct Bayes
    # planner therefore chooses the .67 fallback immediately.
    no_info_model = Model(
        accuracy=0.5,
        budget=model.budget,
        probe_cost=model.probe_cost,
        terminal_cost=model.terminal_cost,
        fallback_success=model.fallback_success,
        max_terminal_actions=model.max_terminal_actions,
        probes=model.probes,
        terminal_sides=model.terminal_sides,
        geometry_summary=model.geometry_summary,
        image_id=model.image_id,
    )
    no_info_plan = adaptive_plan(3, no_info_model, assumed_accuracy=0.5)
    no_info_value, no_info_time = evaluate_policy_exact(no_info_plan, no_info_model)

    # Known geometry: a public, correct S label removes the need for either
    # probe.  This is a counterexample, not the symmetric main geometry.
    known_geometry = {
        "public_geometry_reveals_S": True,
        "best_terminal_success": 1.0,
        "expected_elapsed": model.terminal_cost,
        "probe_needed": False,
    }

    stale = stale_memory_value(model)
    mismatch_plan = adaptive_plan(3, model, assumed_accuracy=model.accuracy)
    mismatch_assumed = mismatch_plan.value
    mismatch_actual, mismatch_time = evaluate_policy_exact(
        mismatch_plan, model, true_accuracy=0.55
    )

    # Minimal negative control: one legal L endpoint, no fallback, and a
    # single R probe.  S remains independent of that one observation.
    negative_model = Model(
        accuracy=model.accuracy,
        budget=3,
        probe_cost=1,
        terminal_cost=2,
        fallback_success=0.0,
        max_terminal_actions=1,
        probes=("probe_R",),
        terminal_sides=(-1,),
        geometry_summary=model.geometry_summary,
        image_id=model.image_id,
    )
    negative_plan = adaptive_plan(
        3,
        negative_model,
        assumed_accuracy=negative_model.accuracy,
        allowed_probes=("probe_R",),
        terminal_sides=(-1,),
        fallback_enabled=False,
    )
    negative_value, negative_time = evaluate_policy_exact(
        negative_plan, negative_model
    )
    negative_fixed = fixed_open_loop_plan(
        1,
        negative_model,
        assumed_accuracy=negative_model.accuracy,
        probe_sequence=("probe_R",),
        terminal_sides=(-1,),
        fallback_enabled=False,
    )
    negative_fixed_value, negative_fixed_time = evaluate_policy_exact(
        negative_fixed, negative_model
    )

    _assert_close(no_info_value, 0.67, "no-information fallback")
    _assert_close(stale, 0.5, "stale independent epoch memory")
    _assert_close(mismatch_actual, 0.55**2 + 0.45**2, "accuracy mismatch")
    _assert_close(negative_value, 0.5, "negative control adaptive")
    _assert_close(negative_fixed_value, 0.5, "negative control fixed")
    return {
        "no_information_accuracy_0.5": {
            "adaptive_h3_exact_success": no_info_value,
            "expected_elapsed": no_info_time,
            "root_action": (
                f"{no_info_plan.action}:{no_info_plan.terminal_choice}"
                if no_info_plan.action == "terminal"
                else no_info_plan.action
            ),
            "interpretation": "diagnostics carry no S information; fallback dominates",
        },
        "known_geometry_sufficient": known_geometry,
        "memory_epoch_reset_stale_R": {
            "correct_current_R_plus_P_success": model.accuracy**2
            + (1.0 - model.accuracy) ** 2,
            "stale_old_R_times_current_P_success": stale,
            "old_R_and_current_R_independent": True,
            "interpretation": "stale address cache destroys the S correlation",
        },
        "model_mismatch_actual_accuracy_0.55_assumed_0.96": {
            "planner_assumed_success": mismatch_assumed,
            "selected_root_action": mismatch_plan.action,
            "actual_exact_success": mismatch_actual,
            "actual_expected_elapsed": mismatch_time,
            "fallback_if_model_were_correctly_known": model.fallback_success,
            "interpretation": "two probes are chosen under the wrong model, but .505 is below .67",
        },
        "minimal_single_probe_negative_control": {
            "terminal_actions": ["terminal:-1"],
            "fallback_enabled": False,
            "budget": negative_model.budget,
            "probe_cost": negative_model.probe_cost,
            "terminal_cost": negative_model.terminal_cost,
            "adaptive_exact_success": negative_value,
            "adaptive_expected_elapsed": negative_time,
            "fixed_one_probe_exact_success": negative_fixed_value,
            "fixed_one_probe_expected_elapsed": negative_fixed_time,
            "A_and_B_upper_bound": 0.5,
            "interpretation": "only L is executable; one probe cannot identify S",
        },
    }


def write_results(path: Path) -> dict:
    results = main_results()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().with_name("RESULTS.json"),
        help="JSON output path (default: sibling RESULTS.json)",
    )
    args = parser.parse_args()
    results = write_results(args.out)
    print(json.dumps({
        "out": str(args.out),
        "schema": results["schema"],
        "two_probe_success": results["formula_checks"]["two_probe_success"],
        "h3_adaptive_success": results["horizon_rows"][2]["adaptive_full_bayes"]["exact_enumerated_value"],
        "prefix_full_history_success": results["fixed_prefix"]["separation"]["full_history_minus_window1"] + 0.67,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

