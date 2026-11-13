"""Reexecute the two-vessel decision screen with live ProDMP motion synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from motion import ProDMPSynthesizer


PARENT = Path("NMI_idea/experiments/ab_static_two_vessel_b_20261113")
HERE = Path(__file__).resolve().parent
SEED = 2026111311


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Executor:
    def __init__(self, synthesizer, costs, *, uncached=False):
        self.synthesizer = synthesizer
        self.costs = costs
        self.uncached = uncached
        self.reused = 0
        self.traces = {}
        self.built = 0
        self.maximum_cost_residual = np.zeros(4)

    def move(self, at, action, current_q):
        key = at, action, current_q.tobytes()
        if self.uncached or key not in self.traces:
            if not np.isfinite(self.costs[at, action]).all():
                raise RuntimeError(f"planner selected an unavailable ProDMP transition {at}->{action}")
            trace = self.synthesizer.execute(at, action, current_q)
            self.built += 1
            if not trace.legal:
                raise RuntimeError(f"new synthesis for {at}->{action} rejected: {trace.reason}")
            if not np.array_equal(trace.q[0], current_q):
                raise RuntimeError("ProDMP did not start at the actual preceding joint state")
            if not np.array_equal(trace.q[-1], self.synthesizer.endpoint_q[action]):
                raise RuntimeError("realized joint exit differs from the finite planning state")
            residual = np.abs(trace.cost - self.costs[at, action])
            self.maximum_cost_residual = np.maximum(self.maximum_cost_residual, residual)
            np.testing.assert_allclose(trace.cost, self.costs[at, action], atol=1.e-8, rtol=0.)
            self.traces[key] = trace
        else:
            self.reused += 1
        return self.traces[key]

    def report(self):
        return {"generated_and_checked": self.built, "identical_state_action_reuses": self.reused,
                "unique_start_action_pairs": len(self.traces),
                "maximum_forecast_execution_cost_difference": self.maximum_cost_residual.tolist()}

    def archive(self, path):
        arrays = {}
        for (a, b, _), trace in self.traces.items():
            stem = f"edge_{a}_{b}"
            if f"{stem}_q" in arrays:
                raise RuntimeError("expected a single rest state at each of the nine pose indices")
            for name in ("q", "t_s", "target", "fk", "cost"):
                arrays[f"{stem}_{name}"] = np.asarray(getattr(trace, name))
        np.savez_compressed(path, **arrays)


def finite_benchmark(prior_module, geometry, cost_matrix):
    from pair_decision import FreshPairProblem

    class RuntimeBenchmark(prior_module.Benchmark):
        """The old decision model, with the current checked motion candidates."""

        def __init__(self):
            likelihood, artery, individual, self.packets, self.lookup = prior_module.make_sensor(geometry)
            self.problem = FreshPairProblem(likelihood, artery, cost_matrix, prior_module.NEAR,
                                            prior_module.FAR, (*prior_module.ANCHOR, *prior_module.NEAR),
                                            min_pair_probability=prior_module.MIN_PAIR)
            self.prior = np.asarray(geometry["prior"])
            self.cdf = np.cumsum(individual, axis=-1)
            self.cdf[..., -1] = 1.
            self.edges = cost_matrix
            self.ok = np.isfinite(cost_matrix).all(axis=-1)
            self.likelihood = likelihood
            self.costs = np.where(np.isfinite(cost_matrix), cost_matrix, 0.)
            self.ratio = np.divide(artery, likelihood[..., None], out=np.zeros_like(artery),
                                   where=likelihood[..., None] > 0.)
            self.states = []
            self._prepare_prefix()

        def _prepare_prefix(self):
            p = prior_module
            if self.ok[p.START, list(p.NEAR)].all() and self.ok[np.ix_(p.NEAR, p.ANCHOR)].all():
                super()._prepare_prefix()
                return
            best = -np.inf
            self.initial = None
            for a in p.NEAR:
                if not self.ok[p.START, a]:
                    continue
                first = self.likelihood[a].T * self.prior
                for b in p.ANCHOR:
                    if not self.ok[a, b]:
                        continue
                    second = first[:, None, :] * self.likelihood[b].T
                    price = (self.costs[p.START, a] + self.costs[a, b]) @ p.PREFIX_PRICE
                    score = self.prefix_value(second).sum() - price
                    if score > best + p.TOL:
                        best, self.initial = score, a
            if self.initial is None:
                raise RuntimeError("there is no checked two-view ProDMP prescan")
            first = self.likelihood[self.initial].T * self.prior
            anchors = [a for a in p.ANCHOR if self.ok[self.initial, a]]
            self.prefix_policy, self.prefix_lookup = {}, {}
            for u, row in enumerate(first):
                if row.sum() == 0.:
                    continue
                at = max(anchors, key=lambda a: self.prefix_value(row * self.likelihood[a].T)
                         - row.sum() * (self.costs[self.initial, a] @ p.PREFIX_PRICE))
                self.prefix_policy[u] = at
                for v, weights in enumerate(row * self.likelihood[at].T):
                    mass = float(weights.sum())
                    if mass > 0.:
                        self.prefix_lookup[u, v] = len(self.states)
                        self.states.append((u, v, at, weights / mass, mass))
            np.testing.assert_allclose(sum(state[4] for state in self.states), 1., atol=p.TOL, rtol=0.)

        def suffix_metrics(self, state, command, *, horizon):
            if command is not None and not self.ok[state[2], command]:
                return np.zeros(2 + len(prior_module.RESOURCES))
            return super().suffix_metrics(state, command, horizon=horizon)

    return RuntimeBenchmark()


def execute_suffix(p, bench, executor, at, belief, command, q, h, tapes, horizon):
    paid = np.zeros(4)
    near_packet = near_slot = None
    observations = []
    success = wrong = False
    selected = None
    if command is not None and not bench.ok[at, command]:
        command = None
    kind = "diagnostic"
    while command is not None:
        trace = executor.move(at, command, q)
        q = trace.q[-1]
        paid += trace.cost
        if np.any(paid > p.CAP + p.TOL):
            raise RuntimeError("executed ProDMP exceeded the shared suffix resource caps")
        packet, slot = bench.draw(h, command, tapes[len(observations), command // 3])
        observations.append([int(command), int(packet), int(slot)])
        belief = bench.problem.posterior(belief, command, packet)
        if kind == "far":
            selected = bench.problem.declare(belief, at, near_packet, command, packet)
            success = selected is not None and selected == (near_slot, slot)
            wrong = selected is not None and not success
            break
        at = command
        near_packet, near_slot = (packet, slot) if at in p.NEAR else (None, None)
        horizon -= 1
        nxt = bench.problem.decide(belief, p.CAP - paid, at=at, horizon=horizon,
                                   mode="mpc", near_packet=near_packet)
        command, kind = nxt.action, nxt.kind
    return np.r_[int(success), int(wrong), paid, len(observations)], observations, selected


def replay(p, bench, executor, uncached, root_choices, *, count, uncached_count, path):
    rng = np.random.default_rng(SEED)
    hiddens = rng.choice(len(bench.prior), size=count, p=bench.prior)
    tapes = rng.random((count, 5, 3, 3))
    paired = np.empty((count, len(p.POLICIES), 2 + len(p.RESOURCES)))
    with path.open("w", encoding="utf-8") as stream:
        for k, (h, tape) in enumerate(zip(hiddens, tapes, strict=True)):
            active = uncached if k < uncached_count else executor
            a = bench.initial
            first = active.move(p.START, a, active.synthesizer.endpoint_q[p.START])
            u, _ = bench.draw(h, a, tape[0, 1])
            at = bench.prefix_policy[u]
            second = active.move(a, at, first.q[-1])
            v, _ = bench.draw(h, at, tape[1, 0])
            index = bench.prefix_lookup[u, v]
            belief = bench.problem.posterior(bench.problem.posterior(bench.prior, a, u), at, v)
            np.testing.assert_allclose(belief, bench.states[index][3], atol=p.TOL, rtol=0.)
            controls = root_choices[index]
            if controls is None:
                controls = (*(bench.problem.decide(belief, p.CAP, at=at, mode=mode).action
                              for mode in ("mpc", "adaptive", "greedy")), *p.NEAR)
            prefix = np.r_[0., 0., first.cost + second.cost, 2.]
            record = {}
            for i, (arm, command) in enumerate(zip(p.POLICIES, controls, strict=True)):
                horizon = 1 if arm == "two_images" else 2
                metrics, trace, selected = execute_suffix(p, bench, active, at, belief, command,
                                                          second.q[-1], h, tape[2:], horizon)
                paired[k, i] = prefix + metrics
                record[arm] = {**p.as_metrics(paired[k, i]), "trace": trace, "selected": selected}
            stream.write(json.dumps({"case": k, "shift_id": int(h), "prescan": [a, u, at, v],
                                     "uncached": k < uncached_count, "arms": record}, allow_nan=False) + "\n")
    best = p.POLICIES.index("fixed_center")
    contrasts = {}
    for other in (0, best):
        delta = paired[:, 1] - paired[:, other]
        contrasts[p.POLICIES[other]] = {"wins": int((delta[:, 0] > 0).sum()),
                                      "losses": int((delta[:, 0] < 0).sum()),
                                      "mean": p.as_metrics(delta.mean(axis=0))}
    return {"n": count, "uncached_cases": uncached_count, "seed": SEED,
            "contrast_b": contrasts, "cached_replay": executor.report(), "uncached_replay": uncached.report(),
            "total": {name: p.as_metrics(paired[:, i].mean(axis=0)) for i, name in enumerate(p.POLICIES)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "robot-repo", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replay", type=int, default=512)
    parser.add_argument("--uncached", type=int, default=32)
    args = parser.parse_args()
    if args.replay < 1 or not 0 <= args.uncached <= args.replay:
        parser.error("provide a positive replay size and an uncached subset")
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    repo = args.repo.resolve()
    inherited = repo / PARENT
    sys.path.insert(0, str(inherited))
    import benchmark as previous

    geometry_file = inherited / "GEOMETRY.json"
    geometry = json.loads(geometry_file.read_text())
    synthesizer = ProDMPSynthesizer(repo, args.robot_repo.resolve(), geometry)
    costs, motion_report = synthesizer.compile()
    write_json(args.out / "MOTION.json", motion_report)
    bench = finite_benchmark(previous, geometry, costs)
    exact, choices = previous.summarize_exact(bench, args.out / "prescan_states.jsonl")
    print(json.dumps({"adaptive_public_ceiling": exact["adaptive_public_ceiling"],
                      "mpc_executed": exact["mpc_executed"], "best_fixed": exact["best_fixed"],
                      "total": exact["total"]}), flush=True)
    executor = Executor(synthesizer, costs)
    uncached = Executor(synthesizer, costs, uncached=True)
    observed = replay(previous, bench, executor, uncached, choices, count=args.replay,
                      uncached_count=args.uncached, path=args.out / "paired.jsonl")
    executor.traces.update(uncached.traces)
    executor.archive(args.out / "EXECUTED_MOTION.npz")
    summary = {
        "scope": "same static artery/vein and sensor; runtime learned ProDMP with sampled ideal kinematics",
        "inherited_study": str(PARENT),
        "source_sha256": {file.name: sha256(file) for file in
                          [*(HERE / n for n in ("run.py", "motion.py", "PROTOCOL.md")),
                           geometry_file, inherited / "benchmark.py", inherited / "pair_decision.py"]},
        "exact": exact, "replay": observed, "wall_seconds": time.perf_counter() - began,
    }
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(observed), flush=True)


if __name__ == "__main__":
    main()
