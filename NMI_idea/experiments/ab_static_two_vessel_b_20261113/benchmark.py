"""Enumerate the static artery–vein mechanism and replay paired acquisitions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr

from pair_decision import FreshPairProblem


START = 3
ANCHOR = (0, 1, 2)
NEAR = (3, 4, 5)
FAR = (6, 7, 8)
CAP = np.array([65., 90., 250., 4.5])
MARGINS = np.array([2., 2., 5., .10])
PREFIX_PRICE = np.array([.001, .0005, .0001, 0.])
MIN_PAIR = .87
DETECT = .84
IMAGE_SD = np.array([1., .68])
NOISE_CUTOFF = 6.
X_EDGES = np.array([-25., -15., -5., 5., 15., 25.])
DEPTH_EDGES = np.array([3., 13., 23., 33., 43., 52.])
TOL = 1.e-12
RESOURCES = ("tcp_mm", "turn_deg", "rail_mm", "motion_s", "frames")
POLICIES = ("mpc", "b", "two_images", "fixed_center", "fixed_minus12", "fixed_plus12")


def make_sensor(geometry):
    centers = np.asarray(geometry["centers_mm"])
    norm = ndtr(NOISE_CUTOFF) - ndtr(-NOISE_CUTOFF)
    x = np.diff(ndtr(np.clip((X_EDGES - centers[..., 0, None]) / IMAGE_SD[0],
                            -NOISE_CUTOFF, NOISE_CUTOFF)), axis=-1) / norm
    z = np.diff(ndtr(np.clip((DEPTH_EDGES - centers[..., 1, None]) / IMAGE_SD[1],
                            -NOISE_CUTOFF, NOISE_CUTOFF)), axis=-1) / norm
    seen = DETECT * (x[..., :, None] * z[..., None, :]).reshape(*x.shape[:-1], -1)
    individual = np.concatenate((1. - seen.sum(axis=-1, keepdims=True), seen), axis=-1)
    bins = individual.shape[-1]
    packets = np.asarray(list(combinations_with_replacement(range(bins), 2)))
    lookup = np.empty((bins, bins), dtype=int)
    total = np.zeros((*individual.shape[:2], len(packets)))
    artery = np.zeros((*total.shape, 2))
    for o, (i, j) in enumerate(packets):
        lookup[i, j] = lookup[j, i] = o
        order = individual[:, :, 0, i] * individual[:, :, 1, j]
        if i == j:
            total[:, :, o] = order
            if i:
                artery[:, :, o, :] = order[..., None] / 2.
        else:
            swapped = individual[:, :, 0, j] * individual[:, :, 1, i]
            total[:, :, o] = order + swapped
            if i:
                artery[:, :, o, 0] = order
            artery[:, :, o, 1] = swapped
    assert np.allclose(total.sum(axis=-1), 1., rtol=0., atol=TOL)
    assert np.allclose(artery.sum(axis=(2, 3)), seen[:, :, 0].sum(axis=-1), rtol=0., atol=TOL)
    # All-zero packets carry no probability in any pose or hypothesis.
    supported = np.any(total > 0., axis=(0, 1))
    compact = np.full(len(packets), -1, dtype=int)
    compact[supported] = np.arange(supported.sum())
    return total[..., supported], artery[:, :, supported], individual, packets[supported], compact[lookup]


class Benchmark:
    def __init__(self, geometry):
        costs = np.asarray([[row if row is not None else [np.inf] * 4 for row in rows]
                            for rows in geometry["costs"]])
        likelihood, artery, individual, self.packets, self.lookup = make_sensor(geometry)
        self.problem = FreshPairProblem(likelihood, artery, costs, NEAR, FAR, (*ANCHOR, *NEAR),
                                        min_pair_probability=MIN_PAIR)
        self.prior = np.asarray(geometry["prior"])
        self.cdf = np.cumsum(individual, axis=-1)
        self.cdf[..., -1] = 1.
        self.edges = costs
        self.ok = np.isfinite(costs).all(axis=-1)
        self.likelihood = likelihood
        self.costs = np.where(np.isfinite(costs), costs, 0.)
        self.ratio = np.divide(artery, likelihood[..., None], out=np.zeros_like(artery),
                               where=likelihood[..., None] > 0.)
        assert self.ok[START, list(NEAR)].all() and self.ok[np.ix_(NEAR, ANCHOR)].all()
        assert self.ok[np.ix_(ANCHOR, NEAR)].all()
        self.states = []
        self._prepare_prefix()

    def prefix_value(self, weights):
        mass = weights.sum(axis=-1)
        return np.divide((weights * weights).sum(axis=-1), mass,
                         out=np.zeros_like(mass), where=mass > 0.).sum(axis=-1)

    def _prepare_prefix(self):
        best, self.initial = -np.inf, None
        for a in NEAR:
            first = self.likelihood[a].T * self.prior
            for b in ANCHOR:
                second = first[:, None, :] * self.likelihood[b].T
                price = (self.costs[START, a] + self.costs[a, b]) @ PREFIX_PRICE
                score = self.prefix_value(second).sum() - price
                if score > best + TOL:
                    best, self.initial = score, a
        assert self.initial is not None
        first = self.likelihood[self.initial].T * self.prior
        self.prefix_policy, self.prefix_lookup = {}, {}
        for u, row in enumerate(first):
            if row.sum() == 0.:
                continue
            b = max(ANCHOR, key=lambda q: self.prefix_value(row * self.likelihood[q].T)
                    - row.sum() * (self.costs[self.initial, q] @ PREFIX_PRICE))
            self.prefix_policy[u] = b
            for v, weights in enumerate(row * self.likelihood[b].T):
                mass = float(weights.sum())
                if mass > 0.:
                    self.prefix_lookup[u, v] = len(self.states)
                    self.states.append((u, v, b, weights / mass, mass))
        assert abs(sum(s[-1] for s in self.states) - 1.) < TOL

    def draw(self, h, pose, tape):
        a = int(np.searchsorted(self.cdf[pose, h, 0], tape[0], side="right"))
        v = int(np.searchsorted(self.cdf[pose, h, 1], tape[1], side="right"))
        packet = int(self.lookup[a, v])
        assert packet >= 0
        artery_slot = -1 if a == 0 else int(a > v or (a == v and tape[2] >= .5))
        return packet, artery_slot

    def best_slots(self, belief, a, o, b, w):
        """Independent batch evaluation of the live terminal decision."""
        assert np.isfinite(self.edges[a, b]).all()
        next_weights = self.likelihood[b].T * belief
        masses = next_weights.sum(axis=-1)
        joint = np.einsum("hi,oh,ohj->oij", self.ratio[a, :, o], next_weights,
                          self.ratio[b].transpose(1, 0, 2), optimize=True).reshape(-1, 4)
        best = joint.argmax(axis=1)
        correct = joint[np.arange(len(best)), best]
        accepted = (correct > 0.) & (correct >= MIN_PAIR * masses)
        if w is not None:
            return (int(best[w] // 2), int(best[w] % 2)) if accepted[w] else None
        return float(correct[accepted].sum()), float((masses - correct)[accepted].sum())

    def integrate_far(self, weights, a, remaining):
        """Integrate best legal far action and declaration across new near packets."""
        occupied = weights.sum(axis=-1) > 0.
        weights = weights[occupied]
        selected = np.full(len(weights), -1, dtype=int)
        correct = np.zeros(len(weights))
        wrong = np.zeros(len(weights))
        mass = weights.sum(axis=-1)
        identified = weights[:, None, :] * self.ratio[a].transpose(1, 2, 0)[occupied]
        flat = identified.reshape(-1, weights.shape[-1])
        for b in FAR:
            if not self.ok[a, b] or np.any(self.costs[a, b] > remaining + TOL):
                continue
            numerator = (flat @ self.problem.artery[b].reshape(weights.shape[-1], -1)).reshape(len(weights), 2, -1, 2)
            maximum = numerator.max(axis=(1, 3))
            packet_mass = weights @ self.likelihood[b]
            accepted = (maximum > 0.) & (maximum >= MIN_PAIR * packet_mass)
            success = np.where(accepted, maximum, 0.).sum(axis=1)
            improved = success > correct + mass * TOL
            wrong[improved] = np.where(accepted, packet_mass - maximum, 0.).sum(axis=1)[improved]
            correct[improved] = success[improved]
            selected[improved] = b
        acquired = selected >= 0
        cost = mass[acquired] @ self.costs[a, selected[acquired]]
        return np.r_[correct.sum(), wrong.sum(), cost, mass[acquired].sum()]

    def suffix_metrics(self, state, command, *, horizon):
        _, _, at, belief, _ = state
        if command is None:
            return np.zeros(2 + len(RESOURCES))
        paid = self.costs[at, command].copy()
        output = np.r_[0., 0., paid, 1.]
        weights = self.likelihood[command].T * belief
        for o, branch in enumerate(weights):
            mass = branch.sum()
            if mass == 0.:
                continue
            tail = self.problem.decide(branch, CAP - paid, at=command, horizon=horizon - 1,
                                       mode="mpc", near_packet=o if command in NEAR else None)
            if tail.kind == "abstain":
                continue
            b = tail.action
            second = self.costs[command, b]
            output[2:] += mass * np.r_[second, 1.]
            if tail.kind == "far":
                output[:2] += mass * np.asarray(self.best_slots(branch / mass, command, o, b, None))
                continue
            second_weights = self.likelihood[b].T * branch
            output += self.integrate_far(second_weights, b, CAP - paid - second)
        return output


def as_metrics(values):
    return dict(zip(("success", "wrong", *RESOURCES), map(float, values), strict=True))


def summarize_exact(bench, state_path):
    stats = {arm: np.zeros(2 + len(RESOURCES)) for arm in POLICIES}
    prefix_cost = np.zeros(len(RESOURCES))
    prefix_correct = 0.
    root_disagreement = second_option_changes = 0.
    gain_mass = defaultdict(float)
    avg_b = avg_m = avg_forecast = avg_tie_safe = 0.
    root_choices = []
    omitted_mass = 0.
    with state_path.open("w", encoding="utf-8") as stream:
        for state in bench.states:
            u, v, at, belief, mass = state
            prefix_cost += mass * np.r_[bench.costs[START, bench.initial] + bench.costs[bench.initial, at], 2.]
            prefix_correct += mass * float(belief.max())
            if mass < TOL:
                omitted_mass += mass
                root_choices.append(None)
                stream.write(json.dumps({"prescan": [bench.initial, u, at, v], "mass": mass,
                                         "below_numerical_cutoff": True}) + "\n")
                continue
            problem = bench.problem
            m = problem.decide(belief, CAP, at=at, mode="mpc")
            b = problem.decide(belief, CAP, at=at, mode="adaptive")
            two = problem.decide(belief, CAP, at=at, mode="greedy")
            executed = b.action_values[m.action] if m.action is not None else 0.
            near_ties = [i for i, val in enumerate(m.action_values) if abs(val - m.value) <= TOL]
            best_tied = max((b.action_values[i] for i in near_ties), default=0.)
            avg_m += mass * executed
            avg_b += mass * b.value
            avg_forecast += mass * m.value
            avg_tie_safe += mass * max(0., b.value - best_tied)
            root_disagreement += mass * (b.action != m.action)
            if b.action is not None:
                choices = {"extra_near" if a in NEAR else "far" for a, prob in
                           zip(b.continuation, belief @ bench.likelihood[b.action], strict=True)
                           if prob > 0. and a is not None}
                second_option_changes += mass * (len(choices) > 1)
            gain = b.value - executed
            for level in (1.e-10, .001, .01, .05):
                gain_mass[str(level)] += mass * (gain > level)
            controls = (m.action, b.action, two.action, *NEAR)
            root_choices.append(controls)
            memo = {}
            for arm, command in zip(POLICIES, controls, strict=True):
                horizon = 1 if arm == "two_images" else 2
                key = (command, horizon)
                if key not in memo:
                    memo[key] = bench.suffix_metrics(state, command, horizon=horizon)
                stats[arm] += mass * memo[key]
            def finite(values):
                return [float(v) if np.isfinite(v) else None for v in values]
            stream.write(json.dumps({"prescan": [bench.initial, u, at, v], "mass": mass,
                                     "actions": controls, "b": b.value, "mpc_forecast": m.value,
                                     "mpc_executed": executed, "b_root_scores": finite(b.action_values),
                                     "mpc_root_scores": finite(m.action_values)}, allow_nan=False) + "\n")
    np.testing.assert_allclose([avg_m, avg_b], [stats["mpc"][0], stats["b"][0]], atol=2.e-10, rtol=0.)
    fixed = max((name for name in stats if name.startswith("fixed")), key=lambda name: stats[name][0])
    total = {name: as_metrics(v + np.r_[0., 0., prefix_cost]) for name, v in stats.items()}
    deltas = {}
    for control in ("mpc", fixed, "two_images"):
        delta = stats["b"] - stats[control]
        deltas[control] = {**as_metrics(delta), "within_modeled_cost_bands": bool(np.all(np.abs(delta[2:6]) <= MARGINS))}
    return {
        "prescan_states": len(bench.states), "prescan_first": bench.initial,
        "omitted_prescan_probability": omitted_mass,
        "prescan_exit_probabilities": {i: sum(state[4] for state in bench.states if state[2] == i) for i in ANCHOR},
        "prescan_registration_map_accuracy": prefix_correct,
        "prescan_mean_resources": dict(zip(RESOURCES, map(float, prefix_cost), strict=True)),
        "mpc_forecast": avg_forecast, "mpc_executed": avg_m, "adaptive_public_ceiling": avg_b,
        "adaptive_over_mpc": avg_b - avg_m, "tie_safe_adaptive_over_mpc": avg_tie_safe,
        "root_disagreement_mass": root_disagreement, "second_diagnostic_or_far_switch_mass": second_option_changes,
        "mass_of_gain_above": dict(gain_mass), "best_fixed": fixed,
        "total": total, "suffix": {name: as_metrics(v) for name, v in stats.items()},
        "paired_expectation_differences_b": deltas,
    }, root_choices


def simulate(bench, state, command, h, tapes, *, horizon):
    _, _, at, belief, _ = state
    paid = np.zeros(4)
    near_packet = near_slot = None
    observations = []
    success = wrong = False
    kind = "diagnostic" if command is not None else "abstain"
    while command is not None:
        assert bench.ok[at, command]
        paid += bench.costs[at, command]
        assert np.all(paid <= CAP + TOL)
        packet, slot = bench.draw(h, command, tapes[len(observations), command // 3])
        observations.append((command, packet, slot))
        belief = bench.problem.posterior(belief, command, packet)
        if kind == "far":
            selected = bench.problem.declare(belief, at, near_packet, command, packet)
            success = selected is not None and selected == (near_slot, slot)
            wrong = selected is not None and not success
            break
        at = command
        near_packet, near_slot = (packet, slot) if at in NEAR else (None, None)
        horizon -= 1
        next_move = bench.problem.decide(belief, CAP - paid, at=at, horizon=horizon,
                                         mode="mpc", near_packet=near_packet)
        command, kind = next_move.action, next_move.kind
    return np.r_[int(success), int(wrong), paid, len(observations)], observations


def replay(bench, root_choices, n, seed, out):
    rng = np.random.default_rng(seed)
    hiddens = rng.choice(len(bench.prior), size=n, p=bench.prior)
    tapes = rng.random((n, 5, 3, 3))
    stats = np.zeros((n, len(POLICIES), 2 + len(RESOURCES)))
    identical = 0
    with out.open("w", encoding="utf-8") as stream:
        for k, (h, tape) in enumerate(zip(hiddens, tapes, strict=True)):
            u, _ = bench.draw(h, bench.initial, tape[0, 1])
            at = bench.prefix_policy[u]
            v, _ = bench.draw(h, at, tape[1, 0])
            index = bench.prefix_lookup[u, v]
            state = bench.states[index]
            prefix = np.r_[0., 0., bench.costs[START, bench.initial] + bench.costs[bench.initial, at], 2.]
            recorded = {}
            controls = root_choices[index]
            if controls is None:
                controls = (*(bench.problem.decide(state[3], CAP, at=at, mode=mode).action
                              for mode in ("mpc", "adaptive", "greedy")), *NEAR)
            for i, (arm, command) in enumerate(zip(POLICIES, controls, strict=True)):
                metrics, path = simulate(bench, state, command, h, tape[2:], horizon=1 if arm == "two_images" else 2)
                stats[k, i] = metrics + prefix
                recorded[arm] = {**as_metrics(stats[k, i]), "trace": path}
            identical += recorded["mpc"] == recorded["b"]
            stream.write(json.dumps({"case": k, "shift_id": int(h), "prescan": [bench.initial, u, at, v],
                                     "arms": recorded}, allow_nan=False) + "\n")
    paired = stats[:, 1] - stats[:, 0]
    return {
        "n": n, "seed": seed, "identical_mpc_b_traces": identical,
        "b_vs_mpc_wins": int((paired[:, 0] > 0.).sum()), "b_vs_mpc_losses": int((paired[:, 0] < 0.).sum()),
        "total": {name: as_metrics(stats[:, i].mean(axis=0)) for i, name in enumerate(POLICIES)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=int, default=512)
    args = parser.parse_args()
    if args.replay <= 0:
        parser.error("--replay must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    bench = Benchmark(json.loads(args.geometry.read_text()))
    exact, root_choices = summarize_exact(bench, args.out / "prescan_states.jsonl")
    report = {
        "hashes": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
                   [args.geometry, *[Path(__file__).with_name(name) for name in
                    ("benchmark.py", "pair_decision.py", "PROTOCOL.md")]]},
        "shape": list(bench.likelihood.shape), "exact": exact,
        "replay": replay(bench, root_choices, args.replay, 2026111311, args.out / "paired.jsonl"),
    }
    report["runtime_s"] = time.perf_counter() - start
    (args.out / "SUMMARY.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: exact[k] for k in ("mpc_executed", "adaptive_public_ceiling", "adaptive_over_mpc",
                                           "root_disagreement_mass", "best_fixed", "total")}, indent=2))


if __name__ == "__main__":
    main()
