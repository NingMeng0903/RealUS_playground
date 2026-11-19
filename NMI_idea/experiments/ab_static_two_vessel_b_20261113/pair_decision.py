"""Exact two-fresh-frame decisions from complete public packets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TOL = 1.e-12


@dataclass(frozen=True)
class PairDecision:
    """Next pose and forecast value, with -inf for excluded pose scores."""

    action: int | None
    kind: str
    value: float
    continuation: tuple[int | None, ...]
    action_values: tuple[float, ...]


class FreshPairProblem:
    """Choose diagnostic views, execute a far view, and select two artery slots.

    ``likelihood[p, h, o]`` and ``artery[p, h, o, i]`` are packet probability
    and joint probability of that packet and artery identity at public slot i.
    Given the static hypothesis, fresh frame innovations are independent.
    ``costs[p, q, d]`` is the checked resource debit for moving and acquiring.
    A far view requires an immediately preceding, newly acquired near frame.
    Both predicted and live declarations require ``min_pair_probability``.
    """

    def __init__(self, likelihood, artery, costs, near, far, diagnostics, *,
                 wrong_penalty=0., prices=None, min_pair_probability=0.):
        likelihood = np.array(likelihood, dtype=float, copy=True)
        artery = np.array(artery, dtype=float, copy=True)
        costs = np.array(costs, dtype=float, copy=True)
        if likelihood.ndim != 3 or min(likelihood.shape) < 1:
            raise ValueError("likelihood must have shape (poses, hypotheses, packets)")
        poses, hypotheses, _ = likelihood.shape
        if artery.shape != (*likelihood.shape, 2):
            raise ValueError("artery must contain two slot masses per packet")
        if costs.ndim != 3 or costs.shape[:2] != (poses, poses) or not costs.shape[2]:
            raise ValueError("costs must have shape (poses, poses, resources)")
        if (np.any(~np.isfinite(likelihood)) or np.any(likelihood < 0.) or
                not np.allclose(likelihood.sum(axis=2), 1., rtol=0., atol=_TOL)):
            raise ValueError("likelihood rows must be probability distributions")
        if (np.any(~np.isfinite(artery)) or np.any(artery < 0.) or
                np.any(artery.sum(axis=3) > likelihood + _TOL) or
                np.any((likelihood == 0.) & (artery.sum(axis=3) > 0.))):
            raise ValueError("artery slot masses must fit within packet probabilities")
        if np.any(np.isnan(costs)) or np.any(costs < 0.):
            raise ValueError("resource costs must be nonnegative or +inf")
        if not np.isfinite(wrong_penalty) or wrong_penalty < 0.:
            raise ValueError("wrong_penalty must be finite and nonnegative")
        if not 0. <= min_pair_probability <= 1.:
            raise ValueError("min_pair_probability must be in [0, 1]")
        if prices is None:
            prices = np.zeros(costs.shape[2])
        prices = np.array(prices, dtype=float, copy=True)
        if prices.shape != costs.shape[2:] or np.any(~np.isfinite(prices)) or np.any(prices < 0.):
            raise ValueError("prices must be finite and nonnegative on each resource")

        self.near = self._pose_list(near, poses)
        self.far = self._pose_list(far, poses)
        self.diagnostics = self._pose_list(diagnostics, poses)
        if not self.near or not self.far or not set(self.near).issubset(self.diagnostics):
            raise ValueError("near poses must be diagnostic and near and far must be nonempty")
        if set(self.far).intersection(self.diagnostics):
            raise ValueError("far and diagnostic poses must be disjoint")

        self.likelihood = likelihood
        self.artery = artery
        self.costs = costs
        self.prices = prices
        self.wrong_penalty = float(wrong_penalty)
        self.min_pair_probability = float(min_pair_probability)
        self._available = np.isfinite(costs).all(axis=2)
        self._price = np.where(np.isfinite(costs), costs, 0.) @ prices
        self._ratio = np.divide(artery, likelihood[..., None], out=np.zeros_like(artery),
                                where=likelihood[..., None] > 0.)
        self._near_kernel = artery.transpose(0, 2, 3, 1).copy()
        # An output with no artery association under any hypothesis has zero
        # declaration utility. Keep its public packet index for actual updates.
        self._pair_support = (artery > 0.).any(axis=(1, 3))
        self._far_kernel = {}
        self._far_likelihood = {}
        for far_pose in self.far:
            support = self._pair_support[far_pose]
            self._far_kernel[far_pose] = np.ascontiguousarray(
                artery[far_pose][:, support, :]).reshape(hypotheses, -1)
            self._far_likelihood[far_pose] = np.ascontiguousarray(likelihood[far_pose][:, support])
        for array in (likelihood, artery, costs, prices):
            array.setflags(write=False)

    @staticmethod
    def _pose_list(indices, poses):
        indices = tuple(indices)
        if any(not isinstance(i, (int, np.integer)) or not 0 <= i < poses for i in indices):
            raise ValueError("invalid pose index")
        return tuple(sorted(set(map(int, indices))))

    def _belief(self, belief):
        belief = np.asarray(belief, dtype=float)
        if belief.shape != self.likelihood.shape[1:2] or np.any(~np.isfinite(belief)) or np.any(belief < 0.):
            raise ValueError("belief must be a nonnegative finite vector over hypotheses")
        total = belief.sum()
        if total <= 0. or not np.isfinite(total):
            raise ValueError("belief must have positive finite mass")
        return belief / total

    def _packet(self, packet):
        if not isinstance(packet, (int, np.integer)) or not 0 <= packet < self.likelihood.shape[2]:
            raise ValueError("invalid packet index")

    def _start(self, remaining, at, near_packet):
        self._pose_list((at,), len(self.costs))
        remaining = np.asarray(remaining, dtype=float)
        if remaining.shape != self.prices.shape or np.any(np.isnan(remaining)) or np.any(remaining < -_TOL):
            raise ValueError("remaining resources must be a nonnegative vector")
        if near_packet is not None:
            if at not in self.near:
                raise ValueError("the latest fresh near packet must be from the current pose")
            self._packet(near_packet)
        return np.maximum(0., remaining)

    def posterior(self, belief, action, outcome):
        """Update the static hypothesis belief using the realized full packet."""
        self._pose_list((action,), len(self.costs))
        self._packet(outcome)
        joint = self._belief(belief) * self.likelihood[action, :, outcome]
        mass = joint.sum()
        if mass == 0.:
            raise ValueError("observed packet has zero probability under the public belief")
        return joint / mass

    def declare(self, belief, near, near_packet, far, far_packet):
        """Return the best two zero-based slots from the belief after both frames."""
        if near not in self.near or far not in self.far:
            raise ValueError("declaration requires fresh near and far poses")
        self._packet(near_packet)
        self._packet(far_packet)
        belief = self._belief(belief)
        probability = (self._ratio[near, :, near_packet, :] * belief[:, None]).T @ (
            self._ratio[far, :, far_packet, :])
        best = np.unravel_index(probability.argmax(), probability.shape)
        correct = probability[best]
        utility = (1. + self.wrong_penalty) * correct - self.wrong_penalty
        return tuple(map(int, best)) if self._eligible(correct, 1.) and utility > 0. else None

    def _legal(self, at, remaining, candidates):
        return [p for p in candidates if self._available[at, p] and
                np.all(self.costs[at, p] <= remaining + _TOL)]

    def _eligible(self, joint, packet_mass):
        return joint >= self.min_pair_probability * packet_mass

    def _far_value(self, weights, identified, masses, at, far):
        """Expected paid far-view utility for batched unnormalized near packets."""
        price = masses * self._price[at, far]
        kernel = self._far_kernel[far]
        if not kernel.shape[1] or not identified.size:
            return -price
        flat = identified.reshape(-1, self.likelihood.shape[1])
        joint = (flat @ kernel).reshape(*weights.shape[:-1], 2, -1, 2)
        best = np.maximum(joint[..., 0, :, 0], joint[..., 0, :, 1])
        np.maximum(best, joint[..., 1, :, 0], out=best)
        np.maximum(best, joint[..., 1, :, 1], out=best)
        if self.wrong_penalty or self.min_pair_probability:
            packet_mass = weights @ self._far_likelihood[far]
            eligible = self._eligible(best, packet_mass)
            best = np.where(eligible, (1. + self.wrong_penalty) * best - self.wrong_penalty * packet_mass, 0.)
        return np.maximum(0., best).sum(axis=-1) - price

    def _near_then_far(self, belief, masses, at, near, remaining):
        """Score a forced near view, followed by a far choice on its real packet."""
        reserve = remaining - self.costs[at, near]
        fars = self._legal(near, reserve, self.far)
        price = masses * self._price[at, near]
        if not fars:
            return -price
        supported = masses > 0.
        active_hypotheses = (belief > 0.).any(axis=0)
        possible_near = (self._near_kernel[near][..., active_hypotheses] > 0.).any(axis=(1, 2))
        if not possible_near.any():
            return -price
        weights = belief[supported, None, :] * self.likelihood[near, :, possible_near]
        identified = belief[supported, None, None, :] * self._near_kernel[near, possible_near]
        near_masses = weights.sum(axis=-1)
        value = np.zeros_like(near_masses)
        for far in fars:
            candidate = self._far_value(weights, identified, near_masses, near, far)
            value = np.maximum(value, candidate)
        answer = -price
        answer[supported] += value.sum(axis=1)
        return answer

    def _diagnose(self, belief, at, remaining, first, horizon, mode, second_fixed):
        reserve = remaining - self.costs[at, first]
        weights = self.likelihood[first].T * belief
        masses = weights.sum(axis=1)
        values = np.zeros_like(masses)
        commands = np.full(len(masses), -1, dtype=int)
        if first in self.near:
            can_declare = np.flatnonzero(self._pair_support[first] & (masses > 0.))
            identified = self._near_kernel[first, can_declare] * belief
            for far in self._legal(first, reserve, self.far):
                candidate = self._far_value(weights[can_declare], identified, masses[can_declare], first, far)
                better = candidate > values[can_declare] + _TOL * masses[can_declare]
                changed = can_declare[better]
                values[changed] = candidate[better]
                commands[changed] = far
        if horizon == 2:
            legal = self._legal(first, reserve, self.near)
            if second_fixed is not None:
                if second_fixed not in legal:
                    raise ValueError("fixed second diagnostic requires a checked near view")
                values = self._near_then_far(weights, masses, first, second_fixed, reserve)
                commands[:] = second_fixed
            elif mode == "adaptive":
                for near in legal:
                    candidate = self._near_then_far(weights, masses, first, near, reserve)
                    better = candidate > values + _TOL * masses
                    values[better] = candidate[better]
                    commands[better] = near
            elif mode == "mpc":
                for near in legal:
                    candidate = self._near_then_far(weights, masses, first, near, reserve)
                    if candidate.sum() > values.sum() + _TOL:
                        values = candidate
                        commands[:] = near
        commands[masses == 0.] = -1
        forecast = tuple(int(cmd) if cmd >= 0 else None for cmd in commands)
        return float(values.sum() - self._price[at, first]), forecast

    def decide(self, belief, remaining, *, at, horizon=2, mode="adaptive", near_packet=None, fixed=()):
        """Choose one paid view or abstain, then replan on the realized packet.

        Horizon counts remaining diagnostics. The far view has its own physical
        debit and is available at horizon zero if the latest packet was near.
        MPC forecasts one fixed second diagnostic or the shared far policy.
        Adaptive search allows that second diagnostic decision to use the first
        packet. Fixed mode forces the supplied diagnostics and forecasts live
        replanning if the sequence leaves a diagnostic slot. All modes use the
        same near-conditioned far choice and slot rule.
        """
        if mode not in {"adaptive", "mpc", "fixed", "greedy"} or horizon not in (0, 1, 2):
            raise ValueError("invalid planning mode or horizon")
        if mode == "greedy":
            horizon = min(horizon, 1)
        fixed = tuple(fixed)
        if (fixed and mode != "fixed") or len(fixed) > horizon:
            raise ValueError("invalid fixed diagnostic sequence")
        self._pose_list(fixed, len(self.costs))
        belief = self._belief(belief)
        remaining = self._start(remaining, at, near_packet)

        scores = np.full(len(self.costs), -np.inf)
        action, kind, value, continuation = None, "abstain", 0., ()
        if near_packet is not None:
            identified = self._ratio[at, :, near_packet, :].T * belief
            for far in self._legal(at, remaining, self.far):
                candidate = float(self._far_value(belief, identified, 1., at, far))
                scores[far] = candidate
                if candidate > value + _TOL:
                    action, kind, value = far, "far", candidate
        if horizon:
            candidates = self._legal(at, remaining, self.diagnostics if horizon == 2 else self.near)
            if mode == "fixed":
                if fixed and fixed[0] not in candidates:
                    raise ValueError("fixed first diagnostic requires a checked pose")
                candidates = fixed[:1]
            for first in candidates:
                second_fixed = fixed[1] if len(fixed) == 2 else None
                tail_mode = "adaptive" if mode == "fixed" else mode
                candidate, forecast = self._diagnose(belief, at, remaining, first, horizon, tail_mode, second_fixed)
                scores[first] = candidate
                if mode == "fixed" or candidate > value + _TOL:
                    action, kind, value, continuation = first, "diagnostic", candidate, forecast
        return PairDecision(action, kind, value, continuation, tuple(scores))

    def followup(self, belief, remaining, action, *, at, near_packet=None):
        """Value of a forced root command followed by realized one-step replanning."""
        if action is None:
            return 0.
        if action in self.far:
            result = self.decide(belief, remaining, at=at, horizon=0, near_packet=near_packet)
            if not np.isfinite(result.action_values[action]):
                raise ValueError("forced far view requires a checked edge and a fresh near frame")
            return result.action_values[action]
        belief = self._belief(belief)
        remaining = self._start(remaining, at, near_packet)
        if action not in self._legal(at, remaining, self.diagnostics):
            raise ValueError("forced diagnostic requires a checked edge")
        value, _ = self._diagnose(belief, at, remaining, action, 2, "adaptive", None)
        return value
