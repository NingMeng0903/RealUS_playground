"""Exact finite-horizon acquisition planning on a public model catalogue."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHODS = (
    "A_geometry", "A_task", "A_iid", "A_hard",
    "B_two", "B_fixed", "B_no_future_cap",
)


@dataclass(frozen=True)
class Assessment:
    probability: np.ndarray
    correct_mass: np.ndarray
    accepted: np.ndarray


@dataclass(frozen=True)
class Plan:
    action: int | None
    scores: np.ndarray
    immediate_scores: np.ndarray
    continuations: np.ndarray | None = None
    continuation_scores: np.ndarray | None = None
    fixed_action: int | None = None


def initial_belief(world_count: int, stationary: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(stationary, dtype=float) / world_count, (world_count, 1))


def _accepted(probability, correct_mass, threshold):
    return (probability > 0.0) & (correct_mass >= threshold * probability)


def _choose(scores, costs, legal):
    candidates = np.flatnonzero(legal)
    if candidates.size == 0:
        return None
    order = np.lexsort((candidates, costs[candidates], -scores[candidates]))
    return int(candidates[order[0]])


class Planner:
    """All distributions here are from the public prior, never the renderer.

    Belief axes are [world, mode], with the mode prediction for the next
    acquisition block. Future matrix products use exactly that flattening order.
    """

    def __init__(
        self, likelihood, correct_joint, joint, public_gate, transition, threshold=0.65,
    ):
        self.likelihood = np.asarray(likelihood, dtype=float)
        self.correct_joint = np.asarray(correct_joint, dtype=float)
        self.joint = np.asarray(joint, dtype=float)
        self.gate = np.asarray(public_gate, dtype=bool)
        self.transition = np.asarray(transition, dtype=float)
        self.threshold = float(threshold)

        self.world_count, modes, self.action_count, self.outcome_count = self.likelihood.shape
        self.emissions = np.ascontiguousarray(
            self.likelihood.transpose(2, 3, 0, 1)
        ).reshape(self.action_count, self.outcome_count, self.world_count * modes)
        self.public_columns = np.flatnonzero(self.gate)
        gate_probability = np.ascontiguousarray(
            self.likelihood[..., self.public_columns]
        )
        gate_correct = np.ascontiguousarray(
            self.correct_joint[..., self.public_columns]
        )
        public_shape = (self.world_count * modes, -1)
        self.immediate_operator = np.concatenate(
            [gate_probability.reshape(public_shape), gate_correct.reshape(public_shape)],
            axis=1,
        )
        # For a branch b*L(first action, outcome), put the HMM prediction
        # into the second-action observation operator once.
        future_operator = np.einsum(
            "ij,wjk->wik",
            self.transition,
            self.immediate_operator.reshape(self.world_count, modes, -1),
        )
        self.future_operator = np.ascontiguousarray(
            future_operator.reshape(self.world_count * modes, -1)
        )
        self.geometry_operator = (
            gate_probability.sum(axis=-1) + 0.03 * (1.0 - self.likelihood[..., 0])
        ).reshape(self.world_count * modes, self.action_count)

    def assess(self, belief, action):
        """Predict the public outcome and true-qualified-stop mass."""
        flat = np.asarray(belief, dtype=float).reshape(-1)
        probability = self.emissions[action] @ flat
        correct = np.zeros(self.outcome_count)
        public_probability, public_correct = self._gate_masses(flat @ self.immediate_operator)
        probability[self.public_columns] = public_probability[action]
        correct[self.public_columns] = public_correct[action]
        accepted = self.gate & _accepted(probability, correct, self.threshold)
        return Assessment(probability, correct, accepted)

    def advance(self, belief, action, outcome, *, hard=False):
        """Condition on the observed category, then predict the next mode.

        The hard ablation chooses a source using its posterior mass across
        worlds and modes, then conditions on the selected source as well.
        """
        if hard:
            source_likelihood = self.joint[:, :, action, :, outcome]
            source_mass = np.einsum("wh,whs->s", belief, source_likelihood)
            weighted = belief * source_likelihood[:, :, source_mass.argmax()]
        else:
            weighted = belief * self.likelihood[:, :, action, outcome]
        evidence = float(weighted.sum())
        if evidence <= 0.0 or not np.isfinite(evidence):
            raise ValueError("observed category has zero model support")
        return (weighted / evidence) @ self.transition

    def _gate_masses(self, values):
        shape = values.shape[:-1] + (self.action_count, len(self.public_columns))
        probability, correct = np.split(values, 2, axis=-1)
        return probability.reshape(shape), correct.reshape(shape)

    def plan(
        self, method, belief, current, remaining, remaining_actions,
        available, actual_cost, free_cost, *, pending=None,
    ):
        """Return a current choice and forecasts for replay from the public inputs."""
        if method not in METHODS:
            raise ValueError(f"unknown acquisition policy {method}")
        cost = actual_cost[current]
        available = np.asarray(available, dtype=bool)
        legal = available & np.isfinite(cost) & (cost <= remaining)
        if remaining_actions < 1 or not legal.any():
            scores = np.full(self.action_count, np.nan)
            return Plan(None, scores, scores.copy())

        flat = np.asarray(belief, dtype=float).reshape(-1)
        probability, correct = self._gate_masses(flat @ self.immediate_operator)
        accepted = _accepted(probability, correct, self.threshold)
        current_saved = np.maximum(0.0, remaining - cost)
        immediate = np.where(
            legal, current_saved * np.where(accepted, correct, 0.0).sum(axis=-1), np.nan,
        )
        if pending is not None:
            action = int(pending) if legal[pending] else None
            return Plan(action, immediate.copy(), immediate)
        if method == "A_geometry":
            scores = np.where(legal, flat @ self.geometry_operator, np.nan)
            return Plan(_choose(scores, cost, legal), scores, immediate)
        if method.startswith("A_") or remaining_actions == 1:
            return Plan(_choose(immediate, cost, legal), immediate.copy(), immediate)

        first_actions = np.flatnonzero(legal)
        branch_weights = self.emissions[first_actions] * flat
        future_masses = branch_weights.reshape(-1, flat.size) @ self.future_operator
        future_masses = future_masses.reshape(first_actions.size, self.outcome_count, -1)
        future_probability, future_correct = self._gate_masses(future_masses)
        future_accepted = _accepted(future_probability, future_correct, self.threshold)
        continuation = np.where(future_accepted, future_correct, 0.0).sum(axis=-1)

        first_accepted = np.zeros((first_actions.size, self.outcome_count), dtype=bool)
        first_accepted[:, self.public_columns] = accepted[first_actions]
        continuation[first_accepted] = 0.0

        # Every likelihood factor above is unnormalized. Thus continuation
        # rewards already include first-outcome probability and sum directly.
        future_cost = (
            free_cost[first_actions] if method == "B_no_future_cap"
            else actual_cost[first_actions]
        )
        future_legal = (
            np.isfinite(future_cost)
            & (future_cost <= current_saved[first_actions, None])
        )
        if method != "B_no_future_cap":
            future_legal &= available[None, :]
        time_saved = np.where(
            future_legal, current_saved[first_actions, None] - future_cost, 0.0,
        )
        continuation *= time_saved[:, None, :]
        common_actions = None
        if method == "B_fixed":
            totals = continuation.sum(axis=1)
            future_score = totals.max(axis=-1)
            common_actions = [
                _choose(row, row_cost, row_legal)
                for row, row_cost, row_legal in zip(totals, future_cost, future_legal)
            ]
        else:
            future_score = continuation.max(axis=-1).sum(axis=-1)

        scores = immediate.copy()
        scores[first_actions] += future_score
        action = _choose(scores, cost, legal)
        chosen_row = int(np.searchsorted(first_actions, action))
        branch_scores = continuation[chosen_row]
        nonstopping = ~first_accepted[chosen_row] & (branch_weights[chosen_row].sum(axis=-1) > 0.0)
        fixed = common_actions[chosen_row] if common_actions is not None else None
        branches = np.full(self.outcome_count, -1, dtype=int)
        if common_actions is not None:
            if fixed is not None:
                branches[nonstopping] = fixed
        else:
            row_legal = future_legal[chosen_row]
            for outcome in np.flatnonzero(nonstopping):
                continuation_action = _choose(branch_scores[outcome], future_cost[chosen_row], row_legal)
                if continuation_action is not None:
                    branches[outcome] = continuation_action
        branch_values = np.zeros(self.outcome_count)
        active = np.flatnonzero(branches >= 0)
        branch_values[active] = branch_scores[active, branches[active]]
        return Plan(action, scores, immediate, branches, branch_values, fixed)
