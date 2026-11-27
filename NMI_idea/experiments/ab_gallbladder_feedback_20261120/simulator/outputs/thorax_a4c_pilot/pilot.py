"""Paired phase-aware four-chamber acquisition pilot using chart ProDMP and PSDP."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
from itertools import product
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingRegressor

from geometry import Anatomy, BEAMS, N_PHASE, PHASE, REFERENCE


HERE = Path(__file__).resolve().parent
from gallbladder_runtime import REPO_ROOT

REPO = REPO_ROOT
sys.path.insert(0, str(REPO))
from NMI_idea.experiments.ab_streaming_prodmp_two_vessel_20261113.primitive import LateralPrimitive


CHILD_CASES, ROOT_CASES, TEST_CASES, TEST_TAPES = 64, 96, 64, 2
MAX_MOVES = 3
RATE, CLIP_FRAMES = 10, 12
PHASE_SD = .012
NOISE = np.array([.085, 1.1, 1.35, 1.55])
STD = np.array([.23, 3.0, 3.0, 3.5])
REWARD = np.array([1., -.32, -.002, -.001, -.016])
ARMS = ("fixed", "mpc_phase_blind", "mpc", "single_b", "psdp_b", "same_first_direct")


@dataclass(frozen=True)
class Move:
    end: int
    middle: int
    travel: float
    turn: float


@dataclass(frozen=True)
class Cine:
    node: int
    phase: np.ndarray
    measurement: np.ndarray
    usable: bool


@dataclass(frozen=True)
class State:
    current: int
    tick: int
    belief: np.ndarray
    clips: tuple[Cine, ...]
    route: tuple[int, ...]
    travel: float = 0.
    turn: float = 0.


@dataclass(frozen=True)
class Tape:
    phase: np.ndarray
    measured_phase: np.ndarray
    error: np.ndarray
    uniform: np.ndarray
    gap: np.ndarray


def detection_probability(area):
    return .012 + .945 * (1. - np.exp(-area / 120.))


def phase_slice(values, phase):
    index = phase * N_PHASE
    left = int(np.floor(index)) % N_PHASE
    relative = index - np.floor(index)
    return (1. - relative) * values[..., left, :, :] + relative * values[..., (left + 1) % N_PHASE, :, :]


def tape(case, replica, bpm):
    rng = np.random.default_rng([351517, case, replica])
    times = np.arange(130) / RATE
    phase = (rng.random() + times * bpm / 60.) % 1.
    measured = (phase + rng.normal(0., PHASE_SD, len(times))) % 1.
    gap = rng.random(len(times)) < .024
    gap[1:] |= gap[:-1]
    return Tape(phase, measured, rng.standard_normal((len(times), 4, 4)) * NOISE,
                rng.random((len(times), 4)), gap)


def make_cases():
    def populate(n, seed):
        rng = np.random.default_rng(seed)
        params = np.column_stack((rng.uniform(-6., 6., n), rng.uniform(-3.5, 3.5, n), rng.uniform(.78, 1.24, n),
                                  rng.uniform(.967, 1.033, n), rng.uniform(-2.25, 2.25, n), rng.normal(0., .006, (n, 4))))
        return params, rng.uniform(62., 84., n)

    child, root, test = (populate(count, seed) for count, seed in
                         [(CHILD_CASES, 499118), (ROOT_CASES, 499213), (TEST_CASES, 658302)])
    return np.vstack((child[0], root[0], test[0])), np.concatenate((child[1], root[1], test[1]))


def make_hypotheses():
    return np.asarray([(dx, dz, gain, scale, dy, 0., 0., 0., 0.) for dx, dz, gain, (scale, dy) in
                       product([-6., -3., 0., 3., 6.], [-3.5, 0., 3.5], [.8, 1., 1.2],
                               [(1., 0.), (.975, -1.5), (.975, 1.5), (1.025, -1.5), (1.025, 1.5)])])


def public_nodes():
    stations = [-164., -158., -152., -146., -140.]
    angles = [-46., -43., -40., -37., -34.]
    rotations = [(-5., 0.), (0., 0.), (5., 0.), (0., -3.), (0., 3.)]
    return np.asarray([(y, theta, roll, rock) for y, theta, (roll, rock) in product(stations, angles, rotations)])


ROOT_CHART = np.array([[-146., -40., 0., 0.], [-146., -37., 0., 0.], [-152., -37., 0., 0.]])
CHART_STEP = np.array([6., 3., 5., 3.])


def motion_reference():
    primitive = LateralPrimitive()
    t = np.linspace(0., 1., 51)
    demos = t ** 3 * (10. - 15. * t + 6. * t ** 2)
    deriv = 30. * t ** 2 * (1. - t) ** 2
    shape = primitive.fit(demos, deriv)
    reference = primitive.condition(0, 50, 0., 0., 1., 0., shape)
    assert np.max(np.abs(reference.x[[0, -1]] - [0., 1.])) < 1.e-8
    assert np.max(np.abs(reference.v[[0, -1]])) < 1.e-8
    return reference


class Pilot:
    def __init__(self, *, fixed_only=False):
        self.anatomy = Anatomy()
        self.params, self.bpm = make_cases()
        self.hypotheses = make_hypotheses()
        self.k = len(self.hypotheses)
        # A single local skin patch, with short longitudinal, circumferential
        # and orientational moves, shared by every policy.
        self.nodes = public_nodes()
        self.views = [self.anatomy.view(node) for node in self.nodes]
        self.roots = [self.find(chart) for chart in ROOT_CHART]
        self.edges = self.moves()
        self.fixed_goals = self.fixed_candidates()
        fixed_views = None
        if fixed_only:
            fixed_views = set(self.roots)
            for root, goal in product(self.roots, self.fixed_goals):
                for edge in self.fixed_route(root, goal):
                    fixed_views.update((edge.end, edge.middle))
        self.table, self.correct, self.quality = self.forward_table(fixed_views)
        self.visit_plans = {} if fixed_only else self.plans()
        self.models = {}
        print(f"Shared catalog {len(self.nodes)} endpoint views, {len(self.views)} with motion midpoints", flush=True)

    def find(self, node):
        return int(np.flatnonzero(np.all(self.nodes == node, axis=1))[0])

    def moves(self):
        reference = motion_reference()
        atlas = self.anatomy.atlas
        middle_indices = {}
        edges = []
        for i, node in enumerate(self.nodes):
            to = np.flatnonzero(np.isclose(np.abs((self.nodes - node) / CHART_STEP).sum(axis=1), 1.))
            actions = []
            for j in to:
                target = self.nodes[j]
                path = node + reference.x[:, None] * (target - node)
                h = (path[:, 0] - atlas.y0) / (atlas.y1 - atlas.y0)
                xyz = atlas.to_world(h, np.deg2rad(path[:, 1]), np.ones(len(path)))
                ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
                if ds.max() / .02 > 35.:
                    continue
                p, q = self.views[i], self.views[j]
                trace = p.forward @ q.forward + p.right @ q.right + p.normal @ q.normal
                angle = np.rad2deg(np.arccos(np.clip((trace - 1.) / 2., -1., 1.)))
                midpoint = tuple(np.round(path[25], 3))
                if midpoint not in middle_indices:
                    middle_indices[midpoint] = len(self.views)
                    self.views.append(self.anatomy.view(midpoint))
                actions.append(Move(int(j), middle_indices[midpoint], float(ds.sum()), float(angle)))
            edges.append(tuple(actions))
        return tuple(edges)

    def forward_table(self, view_indices=None):
        n = self.k + len(self.params)
        centers, axes, _ = self.anatomy.trajectories(np.vstack((self.hypotheses, self.params)))
        features = np.empty((len(self.views), n, N_PHASE, 4, 4), dtype=np.float32)
        correct = np.empty((len(self.nodes), n), dtype=bool)
        quality = np.empty((len(self.nodes), n, 6), dtype=np.float32)
        indices = range(len(self.views)) if view_indices is None else sorted(view_indices)
        for count, i in enumerate(indices, 1):
            view = self.views[i]
            if i < len(self.nodes):
                features[i], correct[i], quality[i] = self.anatomy.quality(view, centers, axes)
            else:
                features[i] = self.anatomy.predict(view, centers, axes)[0]
            if count % 100 == 0:
                print(f"Simulated anatomical and bone visibility for {count}/{len(indices)} views", flush=True)
        return features, correct, quality

    def plans(self):
        result = {}
        for first_node in range(len(self.nodes)):
            options = []

            def visit(node, seq, travel):
                if seq:
                    options.append((tuple(seq), travel))
                if len(seq) < MAX_MOVES:
                    for edge in self.edges[node]:
                        if edge.end != first_node and edge.end not in seq:
                            visit(edge.end, (*seq, edge.end), travel + edge.travel)

            visit(first_node, (), 0.)
            walks = np.full((len(options), MAX_MOVES), -1, dtype=int)
            costs = np.empty(len(options))
            mask = np.empty((len(options), self.k), dtype=bool)
            best = np.empty((len(options), self.k), dtype=float)
            for i, (seq, travel) in enumerate(options):
                walks[i, :len(seq)] = seq
                costs[i] = .0018 * travel + .016 * len(seq)
                mask[i] = np.any(self.correct[list(seq), :self.k], axis=0)
                best[i] = self.quality[list(seq), :self.k, 4].max(axis=0)
            result[first_node] = (walks, costs, mask, best)
        return result

    def update(self, prior, view, phase, z, phase_aware):
        raw = self.table[view, :self.k]
        if phase_aware:
            raw = phase_slice(raw, phase)[:, None]
        raw = raw.astype(float)
        present = np.isfinite(z[:, 0])
        if not present.any():
            weight = np.log1p(-detection_probability(raw[..., 0])).sum(axis=-1)
        else:
            prob = detection_probability(raw[..., 0])
            weight = np.where(present, np.log(prob), np.log1p(-prob)).sum(axis=-1)
            encoded = raw[..., present, :].copy()
            encoded[..., 0] = np.log1p(encoded[..., 0] / 75.)
            residual = (encoded - z[None, None, present]) / STD
            # Heavy tails absorb approximate segmentation and small phase error.
            weight -= (2. * np.log1p(residual ** 2 / 3.)).sum(axis=(-1, -2))
        ll = logsumexp(weight, axis=-1) - np.log(weight.shape[-1])
        p = np.log(np.maximum(prior, 1.e-15)) + .55 * ll
        return np.exp(p - logsumexp(p))

    def observation(self, belief, node, case, ticks, clock, phase_aware):
        recorded = []
        for tick in ticks:
            actual = phase_slice(self.table[node, self.k + case], clock.phase[tick]).copy()
            p = detection_probability(actual[:, 0])
            mask = (clock.uniform[tick] < p) & (not clock.gap[tick])
            actual[:, 0] = np.log1p(actual[:, 0] / 75.)
            z = np.where(mask[:, None], actual + clock.error[tick], np.nan)
            phase = clock.measured_phase[tick]
            belief = self.update(belief, node, phase, z, phase_aware)
            recorded.append(z)
        return belief, np.stack(recorded)

    def cine(self, belief, node, case, tick, clock, aware):
        ticks = np.arange(tick, tick + CLIP_FRAMES)
        belief, recorded = self.observation(belief, node, case, ticks, clock, aware)
        phase = clock.measured_phase[ticks]
        present = np.isfinite(recorded[..., 0])
        # Require repeatable evidence on the same new clip in both phase bands.
        good = True
        for reference in PHASE[list(REFERENCE)]:
            close = np.abs((phase - reference + .5) % 1. - .5) <= .15
            good &= np.any(close) and bool(present[close].any(axis=0).all())
        return belief, Cine(node, phase, recorded, bool(good))

    def root(self, case, clock, aware):
        node = self.roots[case % len(self.roots)]
        belief, clip = self.cine(np.full(self.k, 1. / self.k), node, case, 0, clock, aware)
        return State(node, CLIP_FRAMES, belief, (clip,), (node,))

    def execute(self, state, edge, case, clock, aware):
        belief, _ = self.observation(state.belief, edge.middle, case, [state.tick + RATE // 2], clock, aware)
        tick = state.tick + RATE
        belief, clip = self.cine(belief, edge.end, case, tick, clock, aware)
        return State(edge.end, tick + CLIP_FRAMES, belief, (*state.clips, clip), (*state.route, edge.end),
                     state.travel + edge.travel, state.turn + edge.turn)

    def eligible(self, state):
        return tuple(edge for edge in self.edges[state.current] if edge.end not in state.route)

    def acceptance(self, state):
        eligible = [clip.node for clip in state.clips if clip.usable]
        best = max(eligible, key=lambda n: (self.correct[n, :self.k] @ state.belief,
                                            self.quality[n, :self.k, 4] @ state.belief), default=-1)
        score = self.correct[best, :self.k] @ state.belief if best >= 0 else 0.
        return best if score >= .58 else -1, float(score)

    def mpc_move(self, state, steps):
        if not steps:
            return None
        chosen, score = self.acceptance(state)
        if chosen >= 0 and score >= .90:
            return None
        walks, costs, accept, quality = self.visit_plans[state.current]
        avoided = np.isin(walks, np.asarray(state.route))
        valid = ~(avoided.any(axis=1)) & ((walks >= 0).sum(axis=1) <= steps)
        if not valid.any():
            return None
        indices = np.flatnonzero(valid)
        if chosen >= 0:
            accept = accept | self.correct[chosen, :self.k]
        ranks = accept[indices] @ state.belief + .14 * (quality[indices] @ state.belief) - costs[indices]
        next_node = walks[indices[np.argmax(ranks)], 0]
        return next(edge for edge in self.edges[state.current] if edge.end == next_node)

    def tail(self, state, steps, case, clock, aware=True):
        for remaining in range(steps, 0, -1):
            action = self.mpc_move(state, remaining)
            if action is None:
                break
            state = self.execute(state, action, case, clock, aware)
        return state

    def fixed_route(self, root, goal):
        current = root
        delta = self.nodes - self.nodes[goal]
        distance = np.abs(delta / CHART_STEP).sum(axis=1)
        for _ in range(MAX_MOVES):
            candidates = [e for e in self.edges[current] if distance[e.end] < distance[current]]
            if current == goal or not candidates:
                break
            edge = min(candidates, key=lambda e: (distance[e.end], e.travel, e.turn))
            yield edge
            current = edge.end

    def fixed_candidates(self):
        return tuple(goal for goal in range(len(self.nodes)) if all(
            (root, *(e.end for e in self.fixed_route(root, goal)))[-1] == goal for root in self.roots))

    def fixed(self, state, case, clock, goal):
        for edge in self.fixed_route(state.current, goal):
            state = self.execute(state, edge, case, clock, True)
        return state

    def fit_fixed(self):
        # Use simulated task completion under the same motion budget and all
        # training initial positions to pick one fixed pose and route rule.
        totals = np.zeros((len(self.fixed_goals), len(REWARD)))
        routes = {(root, goal): tuple(self.fixed_route(root, goal)) for root, goal in product(self.roots, self.fixed_goals)}
        for case in range(CHILD_CASES + ROOT_CASES):
            clock = tape(case, 6, self.bpm[case])
            start = self.root(case, clock, True)
            cache = {start.route: start}
            for i, goal in enumerate(self.fixed_goals):
                state = start
                for edge in routes[start.current, goal]:
                    key = (*state.route, edge.end)
                    if key not in cache:
                        cache[key] = self.execute(state, edge, case, clock, True)
                    state = cache[key]
                totals[i] += self.outcome(state, case)
        best = max(range(len(totals)), key=lambda i: tuple(totals[i] * [1., -1., -1., -1., -1.]))
        print(f"Fixed target chosen from {len(self.fixed_goals)} reachable goals", flush=True)
        return self.fixed_goals[best]

    def outcome(self, state, case):
        chosen, _ = self.acceptance(state)
        correct = bool(chosen >= 0 and self.correct[chosen, self.k + case])
        return np.array([correct, chosen >= 0 and not correct, state.travel, state.turn, state.tick / RATE])

    def result_row(self, end, case, replica, arm, oracle):
        result = self.outcome(end, case)
        chosen, confidence = self.acceptance(end)
        return {"case": case, "tape": replica, "arm": arm, "static_geometry_reachable": oracle,
                "success": bool(result[0]), "wrong_accept": bool(result[1]), "travel_mm": result[2],
                "turn_deg": result[3], "time_s": result[4], "accepted": chosen,
                "confidence": confidence, "route": end.route, "contraction_gain": end.belief @ self.hypotheses[:, 2],
                "true_gain": self.params[case, 2],
                "visited_valid_view": any(self.correct[n, self.k + case] for n in end.route),
                "recorded_valid_cine": any(c.usable and self.correct[c.node, self.k + case] for c in end.clips),
                "accepted_geometry": None if chosen < 0 else self.quality[chosen, self.k + case].tolist()}

    def feature_rows(self, state, edges, steps):
        b = state.belief
        mean = b @ self.hypotheses[:, :5]
        std = np.sqrt(b @ (self.hypotheses[:, :5] - mean) ** 2)
        node, probability = self.acceptance(state)
        clip = state.clips[-1]
        visibility = np.isfinite(clip.measurement[..., 0]).mean(axis=0)
        base = np.r_[self.views[state.current].address, self.nodes[state.current, 2:] / [5., 3.],
                     mean, std, -np.sum(b * np.log(np.maximum(b, 1.e-15))), visibility, probability, node >= 0, steps]
        rows = []
        for edge in edges:
            dest = edge.end
            raw = self.table[dest, :self.k][:, list(REFERENCE)]
            expected = np.einsum("h,hkc->kc", b, detection_probability(raw[..., 0]))
            target = self.correct[dest, :self.k] @ b
            quality = b @ self.quality[dest, :self.k][:, [0, 1, 2, 3, 5]]
            # Angular sector support is measured on the actual physical geometry.
            sector = [np.mean(self.views[dest].bone_distance[part] > 115.) for part in np.array_split(np.arange(len(BEAMS)), 3)]
            rows.append(np.r_[base, self.views[dest].address, self.views[edge.middle].address,
                              self.nodes[dest, 2:] / [5., 3.], edge.travel, edge.turn, target, quality,
                              expected.ravel(), sector, b])
        return np.stack(rows)

    def child(self, state, steps, case, clock, return_first=False):
        edges = self.eligible(state)
        if not edges or steps == 0:
            return (state, None) if return_first else state
        x = self.feature_rows(state, edges, steps)
        delta = predict_models(self.models["child"], x) @ REWARD
        best = int(np.argmax(delta))
        if delta[best] <= 0.:
            end = self.tail(state, steps, case, clock)
            movement = None
        else:
            movement = edges[best].end
            end = self.tail(self.execute(state, edges[best], case, clock, True), steps - 1, case, clock)
        return (end, movement) if return_first else end

    def train(self, out):
        rng = np.random.default_rng(2302)
        x_child, y_child = [], []
        for case in range(CHILD_CASES):
            clock = tape(case, 0, self.bpm[case])
            start = self.root(case, clock, True)
            seeds = self.eligible(start)
            for i in rng.choice(len(seeds), size=min(3, len(seeds)), replace=False):
                successor = self.execute(start, seeds[i], case, clock, True)
                edges = self.eligible(successor)
                base = self.outcome(self.tail(successor, 2, case, clock), case)
                x_child.append(self.feature_rows(successor, edges, 2))
                y_child.append(np.stack([self.outcome(self.tail(self.execute(successor, e, case, clock, True), 1, case, clock),
                                                     case) - base for e in edges]))
        self.models["child"] = fit_models(np.vstack(x_child), np.vstack(y_child))

        x_root, direct, residual = [], [], []
        for case in range(CHILD_CASES, CHILD_CASES + ROOT_CASES):
            clock = tape(case, 0, self.bpm[case])
            start = self.root(case, clock, True)
            base = self.outcome(self.tail(start, MAX_MOVES, case, clock), case)
            edges = self.eligible(start)
            d_out, a_out = [], []
            for edge in edges:
                successor = self.execute(start, edge, case, clock, True)
                direct_end = self.outcome(self.tail(successor, MAX_MOVES - 1, case, clock), case)
                adaptive_end = self.outcome(self.child(successor, MAX_MOVES - 1, case, clock), case)
                d_out.append(direct_end - base)
                a_out.append(adaptive_end - direct_end)
            x_root.append(self.feature_rows(start, edges, MAX_MOVES))
            direct.append(np.stack(d_out))
            residual.append(np.stack(a_out))
        y_child, y_root, y_delta = np.vstack(y_child), np.vstack(direct), np.vstack(residual)
        self.models["direct"] = fit_models(np.vstack(x_root), y_root)
        self.models["residual"] = fit_models(np.vstack(x_root), y_delta)
        joblib.dump(self.models, out / "psdp_models.joblib", compress=3)
        np.savez_compressed(out / "paired_training.npz", child_features=np.vstack(x_child), child_deltas=y_child,
                            root_features=np.vstack(x_root), root_direct_deltas=y_root, continuation_deltas=y_delta)
        return {"child_records": len(y_child), "root_pairs": len(y_root), "feature_count": x_root[0].shape[-1],
                "root_pairs_with_different_completion": int(np.count_nonzero(y_delta[:, 0]))}

    def learned(self, start, case, clock, arm):
        edges = self.eligible(start)
        x = self.feature_rows(start, edges, MAX_MOVES)
        direct = predict_models(self.models["direct"], x)
        if arm == "single_b":
            delta = direct
        else:
            delta = np.concatenate((direct, direct + predict_models(self.models["residual"], x)))
        score = delta @ REWARD
        index = int(np.argmax(score))
        if score[index] <= 0.:
            return self.tail(start, MAX_MOVES, case, clock), {"used_child": False, "first_override": None}
        first = edges[index % len(edges)]
        child = index >= len(edges) and arm != "same_first_direct"
        state = self.execute(start, first, case, clock, True)
        if child:
            end, override = self.child(state, MAX_MOVES - 1, case, clock, return_first=True)
        else:
            end, override = self.tail(state, MAX_MOVES - 1, case, clock), None
        return end, {"used_child": child, "first_override": first.end, "child_override": override}

    def phase_fit(self, case, clock, node):
        """Out-of-cycle prediction of apparent area from noisy observed masks."""
        _, first = self.cine(np.full(self.k, 1. / self.k), node, case, 0, clock, True)
        future = np.arange(15, 27)
        actual = np.stack([phase_slice(self.table[node, self.k + case], p)[:, 0] for p in clock.phase[future]])
        fit, constant = np.full_like(actual, np.nan), np.full_like(actual, np.nan)
        basis = harmonic_basis(first.phase)
        target = harmonic_basis(clock.measured_phase[future])
        for chamber in range(4):
            y = first.measurement[:, chamber, 0]
            available = np.isfinite(y)
            if available.sum() < 7:
                continue
            z = basis[available]
            fix = np.linalg.solve(z.T @ z + np.diag([1.e-7, .25, .25, .55, .55]), z.T @ y[available])
            fit[:, chamber] = np.maximum(0., np.expm1(target @ fix) * 75.)
            constant[:, chamber] = np.expm1(y[available]).mean() * 75.
        return first, clock.measured_phase[future], actual, fit, constant

    def evaluate(self, out, training):
        start_test = CHILD_CASES + ROOT_CASES
        goal = self.fit_fixed()
        rows, gallery, squared = [], None, []
        for case in range(start_test, len(self.params)):
            for replica in range(TEST_TAPES):
                clock = tape(case, replica + 101, self.bpm[case])
                start = self.root(case, clock, True)
                distances = np.abs((self.nodes - self.nodes[start.current]) / CHART_STEP).sum(axis=1)
                oracle = bool(np.any(self.correct[distances <= MAX_MOVES, self.k + case]))
                for arm in ARMS:
                    info = {}
                    if arm == "fixed":
                        end = self.fixed(start, case, clock, goal)
                    elif arm == "mpc_phase_blind":
                        end = self.tail(self.root(case, clock, False), MAX_MOVES, case, clock, False)
                    elif arm == "mpc":
                        end = self.tail(start, MAX_MOVES, case, clock)
                    else:
                        end, info = self.learned(start, case, clock, arm)
                    row = self.result_row(end, case, replica, arm, oracle)
                    rows.append(row | info)
                    chosen = row["accepted"]
                    if (arm == "psdp_b" and row["success"] and gallery is None and chosen != start.current
                            and not self.correct[start.current, self.k + case]):
                        gallery = (case, replica, start.current, end, chosen, clock)
                _, _, actual, learned, flat = self.phase_fit(case, clock, goal)
                valid = np.isfinite(learned)
                squared.append([np.sum((x[valid] - actual[valid]) ** 2) for x in (flat, learned)] + [valid.sum()])

        if gallery is not None:
            self.save_example(out, gallery)
        return summarize(rows, training, squared, self.nodes[goal], out, len(self.nodes),
                         None if gallery is None else gallery[0], len(self.fixed_goals))

    def recheck_fixed(self, out):
        with gzip.open(out / "heldout.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        old = json.loads((out / "summary.json").read_text())
        goal = self.fit_fixed()
        squared = []
        for i, row in enumerate(rows):
            if row["arm"] != "fixed":
                continue
            case, replica = row["case"], row["tape"]
            clock = tape(case, replica + 101, self.bpm[case])
            start = self.root(case, clock, True)
            end = self.fixed(start, case, clock, goal)
            rows[i] = self.result_row(end, case, replica, "fixed", row["static_geometry_reachable"])
            _, _, actual, learned, flat = self.phase_fit(case, clock, goal)
            valid = np.isfinite(learned)
            squared.append([np.sum((x[valid] - actual[valid]) ** 2) for x in (flat, learned)] + [valid.sum()])
        return summarize(rows, old["training"], squared, self.nodes[goal], out, len(self.nodes),
                         old["example_case"], len(self.fixed_goals))

    def save_example(self, out, gallery):
        case, replica, first, end, final, clock = gallery
        example_phase = PHASE[list(REFERENCE)]
        centers, axes, _ = self.anatomy.trajectories(self.params[case:case + 1], example_phase)
        positions = [first, final]
        masks, shadows = [], []
        for node in positions:
            view = self.views[node]
            a, b = [], []
            for k in range(len(example_phase)):
                _, _, _, shadow, _, observed = self.anatomy.raster(view, centers[0, k], axes[0, k])
                a.append(observed)
                b.append(shadow)
            masks.append(a)
            shadows.append(b)
        cine, phase, actual, fit, flat = self.phase_fit(case, clock, final)
        np.savez_compressed(out / "example.npz", case=case, tape=replica, initial=first, final=final,
                            acquired_chart=self.nodes[list(end.route)], displayed_views=self.nodes[positions],
                            frames_phase=example_phase, visible_masks=np.array(masks), bone_shadow=np.array(shadows),
                            observed_phase=cine.phase, observed_mask_features=cine.measurement, future_phase=phase,
                            true_area_mm2=actual, periodic_area_mm2=fit, phase_blind_area_mm2=flat,
                            accepted_probability=self.acceptance(end)[1], quality=self.quality[positions, self.k + case])


def harmonic_basis(phase):
    theta = 2. * np.pi * phase
    return np.column_stack((np.ones(len(phase)), np.sin(theta), np.cos(theta), np.sin(2. * theta), np.cos(2. * theta)))


def fit_models(features, targets):
    return [HistGradientBoostingRegressor(max_iter=85, max_leaf_nodes=11, min_samples_leaf=18,
                                          learning_rate=.06, l2_regularization=3., random_state=629).fit(features, y)
            for y in targets.T]


def predict_models(models, features):
    predictions = np.column_stack([model.predict(features) for model in models])
    predictions[:, :2] = np.clip(predictions[:, :2], -1., 1.)
    return predictions


def summarize(rows, training, squared, fixed, out, catalog_size, example_case, fixed_goals):
    with gzip.open(out / "heldout.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    grouped = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    stats = {}
    for arm, cases in grouped.items():
        stats[arm] = {"correct": sum(r["success"] for r in cases), "wrong_confirmations": sum(r["wrong_accept"] for r in cases),
                      "visited_valid_view": sum(r["visited_valid_view"] for r in cases),
                      "recorded_valid_cine": sum(r["recorded_valid_cine"] for r in cases),
                      **{f"mean_{k}": float(np.mean([r[k] for r in cases])) for k in ("travel_mm", "turn_deg", "time_s")},
                      "contraction_gain_rmse": float(np.sqrt(np.mean([(r["contraction_gain"] - r["true_gain"]) ** 2 for r in cases]))),
                      "selected_conditional_child": sum(bool(r.get("used_child")) for r in cases)}
    paired = {}
    rng = np.random.default_rng(814)
    indices = rng.integers(0, TEST_CASES, (6000, TEST_CASES))
    for left, right in [("mpc", "mpc_phase_blind"), ("psdp_b", "mpc"), ("psdp_b", "fixed"),
                        ("psdp_b", "single_b"), ("psdp_b", "same_first_direct")]:
        a = np.array([int(r["success"]) for r in grouped[left]])
        b = np.array([int(r["success"]) for r in grouped[right]])
        d = (a - b).reshape(TEST_CASES, TEST_TAPES).mean(axis=1)
        key = f"{left}_minus_{right}"
        paired[key] = {"percentage_points": float(100 * d.mean()),
                       "bootstrap_95": (100 * np.quantile(d[indices].mean(axis=1), [.025, .975])).tolist(),
                       "paired_wins": int(np.sum(a > b)), "paired_losses": int(np.sum(a < b))}
    total = np.sum(squared, axis=0)
    report = {
        "training": training, "test_anatomies": TEST_CASES, "tapes_per_anatomy": TEST_TAPES,
        "public_candidate_views": catalog_size, "oracle_geometry_reachable_anatomies": sum(
            r["static_geometry_reachable"] for r in grouped["mpc"][::TEST_TAPES]),
        "phase_error_sd_cycles": PHASE_SD, "reference_phases": PHASE[list(REFERENCE)].tolist(),
        "prior_mean_contraction_gain_rmse": float(np.sqrt(np.mean([(1. - r["true_gain"]) ** 2 for r in grouped["mpc"]]))),
        "best_fixed_pose_train_only": fixed.tolist(), "fixed_targets_compared_on_training": fixed_goals,
        "scores": stats, "paired_completion": paired,
        "next_cycle_visible_area_prediction": {"phase_blind_rmse_mm2": float(np.sqrt(total[0] / total[2])),
                                              "phase_conditioned_rmse_mm2": float(np.sqrt(total[1] / total[2])),
                                              "matched_chamber_frame_predictions": int(total[2])},
        "example_case": example_case,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--recheck-fixed", action="store_true")
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    pilot = Pilot(fixed_only=options.recheck_fixed)
    if options.recheck_fixed:
        report = pilot.recheck_fixed(options.out)
    else:
        training = pilot.train(options.out)
        print(json.dumps(training), flush=True)
        report = pilot.evaluate(options.out, training)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
