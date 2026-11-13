"""Reconstruct public roots and paired execution without importing either planner."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations_with_replacement
import json
import lzma
from pathlib import Path

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parent
ANCHOR = range(3)
NEAR = range(3, 6)
FAR = range(6, 9)
CAP = np.array([65., 90., 250., 4.5])
PREFIX_PRICE = np.array([.001, .0005, .0001, 0.])
TOL = 1.e-12
THRESHOLD = .87
POLICIES = ('mpc', 'b', 'two_images', 'fixed_center', 'fixed_minus12', 'fixed_plus12')
METRICS = ('success', 'wrong', 'tcp_mm', 'turn_deg', 'rail_mm', 'motion_s', 'frames')


class Reconstruction:
    def __init__(self, geometry):
        self.prior = np.array(geometry['prior'])
        self.cost = np.array([[e if e is not None else [np.inf] * 4 for e in row]
                              for row in geometry['costs']])
        centers = np.array(geometry['centers_mm'])
        weights = []
        norm = ndtr(6.) - ndtr(-6.)
        for dim, (edges, sigma) in enumerate((([-25., -15., -5., 5., 15., 25.], 1.),
                                               ([3., 13., 23., 33., 43., 52.], .68))):
            boundaries = (np.array(edges) - centers[..., dim, None]) / sigma
            weights.append(np.diff(ndtr(np.clip(boundaries, -6., 6.))) / norm)
        seen = .84 * (weights[0][..., :, None] * weights[1][..., None, :]).reshape(*centers.shape[:-1], 25)
        individual = np.concatenate((1. - seen.sum(axis=-1, keepdims=True), seen), axis=-1)
        self.cdf = np.cumsum(individual, axis=-1)
        self.cdf[..., -1] = 1.

        # Accumulate from ordered, labeled returns before removing the labels.
        keys = tuple(combinations_with_replacement(range(26), 2))
        indices = {key: i for i, key in enumerate(keys)}
        likelihood = np.zeros((*individual.shape[:2], len(keys)))
        target = np.zeros((*likelihood.shape, 2))
        for artery in range(26):
            for vein in range(26):
                o = indices[tuple(sorted((artery, vein)))]
                mass = individual[:, :, 0, artery] * individual[:, :, 1, vein]
                likelihood[:, :, o] += mass
                if artery:
                    if artery == vein:
                        target[:, :, o, :] += mass[..., None] / 2.
                    else:
                        target[:, :, o, int(artery > vein)] += mass
        support = (likelihood > 0.).any(axis=(0, 1))
        self.index = {key: i for i, key in enumerate(k for i, k in enumerate(keys) if support[i])}
        self.like = likelihood[:, :, support].transpose(0, 2, 1)
        self.target = target[:, :, support, :].transpose(0, 2, 1, 3)
        self.ratio = np.divide(self.target, self.like[..., None], out=np.zeros_like(self.target),
                               where=self.like[..., None] > 0.)
        self.support = [(rows > 0.).any(axis=1) for rows in self.like]
        self.like_compact = [rows[s] for rows, s in zip(self.like, self.support, strict=True)]
        self.target_compact = [rows[s] for rows, s in zip(self.target, self.support, strict=True)]

        score = -np.inf
        for first in NEAR:
            w = self.like[first] * self.prior
            for second in ANCHOR:
                candidate = self.concentration(w[:, None, :] * self.like[second]).sum()
                candidate -= (self.cost[3, first] + self.cost[first, second]) @ PREFIX_PRICE
                if candidate > score + TOL:
                    self.initial, score = first, candidate

    @staticmethod
    def concentration(w):
        total = w.sum(axis=-1)
        return np.divide((w * w).sum(axis=-1), total, where=total > 0., out=np.zeros_like(total))

    def anchor(self, u):
        w = self.prior * self.like[self.initial, u]
        return max(ANCHOR, key=lambda a: self.concentration(w * self.like[a]).sum()
                   - w.sum() * (self.cost[self.initial, a] @ PREFIX_PRICE))

    def root_weight(self, prefix):
        initial, u, anchor, v = prefix
        assert initial == self.initial and anchor == self.anchor(u)
        return self.prior * self.like[initial, u] * self.like[anchor, v]

    def legal(self, at, remaining, actions):
        return [p for p in actions if np.all(self.cost[at, p] <= remaining + TOL)]

    def draw(self, h, p, u):
        a = int(np.count_nonzero(self.cdf[p, h, 0] <= u[0]))
        v = int(np.count_nonzero(self.cdf[p, h, 1] <= u[1]))
        return self.index[tuple(sorted((a, v)))], (-1 if not a else int(a > v or (a == v and u[2] >= .5)))

    def finish(self, w, identified, far):
        """Integrate joint declaration mass over fresh far packets."""
        mass = w @ self.like_compact[far].T
        joint = np.einsum('...hi,phj->...pij', identified, self.target_compact[far], optimize=True)
        correct = joint.max(axis=(-2, -1))
        accepted = (correct > 0.) & (correct >= THRESHOLD * mass)
        return np.where(accepted, correct, 0.).sum(axis=-1), np.where(accepted, mass - correct, 0.).sum(axis=-1)

    def direct(self, w, near, packet, reserve):
        shape = w.shape[:-1]
        value, wrong = np.zeros(shape), np.zeros(shape)
        selected = np.full(shape, -1, dtype=int)
        identified = w[..., None] * self.ratio[near, packet]
        for far in self.legal(near, reserve, FAR):
            correct, incorrect = self.finish(w, identified, far)
            changed = correct > value + w.sum(axis=-1) * TOL
            value, wrong = np.where(changed, correct, value), np.where(changed, incorrect, wrong)
            selected = np.where(changed, far, selected)
        return value, wrong, selected

    def second_near(self, w, near, reserve):
        support = np.flatnonzero(self.support[near])
        after = w[..., None, :] * self.like[near, support]
        correct, wrong, far = self.direct(after, near, support, reserve)
        return correct.sum(axis=-1), wrong, far, after.sum(axis=-1)

    def options(self, w, first, reserve, horizon):
        support = np.flatnonzero(self.support[first])
        after = w * self.like[first, support]
        columns = [self.direct(after, first, support, reserve)[0] if first in NEAR else np.zeros(len(after))]
        for near in self.legal(first, reserve, NEAR) if horizon == 2 else ():
            columns.append(self.second_near(after, near, reserve - self.cost[first, near])[0])
        return np.stack(columns, axis=1)

    @staticmethod
    def choose(values):
        best, value = None, 0.
        for action, candidate in enumerate(values):
            if candidate > value + TOL:
                best, value = action, float(candidate)
        return best, value

    def score_roots(self, w, at):
        f, g, two = np.full((3, len(self.cost)), -np.inf)
        for first in self.legal(at, CAP, (*ANCHOR, *NEAR)):
            options = self.options(w, first, CAP - self.cost[at, first], 2)
            f[first] = options.sum(axis=0).max()
            g[first] = options.max(axis=1).sum()
            two[first] = options[:, 0].sum()
        return f, g, two

    def next_view(self, w, at, packet, reserve, horizon):
        best, value = None, 0.
        if at in NEAR:
            good, _, far = self.direct(w, at, packet, reserve)
            if far >= 0:
                best, value = int(far), float(good)
        if horizon:
            for near in self.legal(at, reserve, NEAR):
                good, _, _, _ = self.second_near(w, near, reserve - self.cost[at, near])
                if good > value + w.sum() * TOL:
                    best, value = near, float(good)
        return best

    def expected_suffix(self, w, at, first, horizon):
        """Integrate the realized, common replanner and all acquisition debits."""
        if first is None:
            return np.zeros(len(METRICS))
        paid = self.cost[at, first]
        reserve = CAP - paid
        support = np.flatnonzero(self.support[first])
        after = w * self.like[first, support]
        mass = after.sum(axis=-1)
        rows = np.zeros((len(support), len(METRICS)))
        if first in NEAR:
            correct, wrong, far = self.direct(after, first, support, reserve)
            selected = far >= 0
            rows[:, :2] = np.column_stack((correct, wrong))
            rows[selected, 2:6] = mass[selected, None] * self.cost[first, far[selected]]
            rows[selected, 6] = mass[selected]
        for near in self.legal(first, reserve, NEAR) if horizon == 2 else ():
            move = self.cost[first, near]
            correct, wrong, far, near_mass = self.second_near(after, near, reserve - move)
            improved = correct > rows[:, 0] + mass * TOL
            if not improved.any():
                continue
            rows[improved, 0] = correct[improved]
            rows[improved, 1] = wrong[improved].sum(axis=1)
            acquired = far[improved] >= 0
            far_mass = np.where(acquired, near_mass[improved], 0.)
            edge = np.vstack((np.zeros(4), self.cost[near, list(FAR)]))
            index = np.where(acquired, far[improved] - min(FAR) + 1, 0)
            rows[improved, 2:6] = mass[improved, None] * move + np.einsum('bp,bpd->bd', far_mass, edge[index])
            rows[improved, 6] = mass[improved] + far_mass.sum(axis=1)
        return rows.sum(axis=0) + np.r_[0., 0., paid, 1.] * w.sum()

    def run_trace(self, prefix, command, h, tape, horizon):
        w = self.root_weight(prefix)
        at = prefix[2]
        paid = np.zeros(4)
        observed = []
        success = wrong = 0
        while command is not None:
            assert command in self.legal(at, CAP - paid, (*ANCHOR, *NEAR, *FAR))
            paid += self.cost[at, command]
            packet, slot = self.draw(h, command, tape[len(observed), command // 3])
            observed.append([command, packet, slot])
            if command in FAR:
                assert at in NEAR and len(observed) in (2, 3)
                near_packet, near_slot = observed[-2][1:]
                joint = (w[:, None] * self.ratio[at, near_packet]).T @ self.target[command, packet]
                pair = np.unravel_index(joint.argmax(), joint.shape)
                if joint[pair] > 0. and joint[pair] >= THRESHOLD * (w @ self.like[command, packet]):
                    success = int(pair == (near_slot, slot))
                    wrong = 1 - success
                break
            w *= self.like[command, packet]
            at = command
            horizon -= 1
            command = self.next_view(w, at, packet, CAP - paid, horizon)
        prefix_cost = self.cost[3, prefix[0]] + self.cost[prefix[0], prefix[2]]
        return np.r_[success, wrong, paid + prefix_cost, len(observed) + 2], observed


def read_jsonl(path):
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]
    with lzma.open(path.with_suffix(path.suffix + '.xz'), 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def audit(directory):
    summary = json.loads((directory / 'SUMMARY.json').read_text())
    for name, expected in summary['hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    model = Reconstruction(json.loads((ROOT / 'GEOMETRY.json').read_text()))
    roots = read_jsonl(directory / 'prescan_states.jsonl')
    active = [row for row in roots if 'actions' in row]
    high = sorted(active, key=lambda r: r['mass'], reverse=True)[:12]
    differences = sorted((r for r in active if r['actions'][0] != r['actions'][1]),
                         key=lambda r: r['mass'], reverse=True)[:8]
    contributions = sorted(active, key=lambda r: r['mass'] * (r['b'] - r['mpc_executed']), reverse=True)[:8]
    selected = {tuple(r['prescan']): r for r in high + differences + contributions}
    root_audit = []
    for prefix, row in selected.items():
        joint = model.root_weight(prefix)
        f, g, two = model.score_roots(joint / joint.sum(), prefix[2])
        expected = lambda name: [np.nan if x is None else x for x in row[name]]
        np.testing.assert_allclose(np.where(np.isfinite(f), f, np.nan), expected('mpc_root_scores'), atol=1.e-12, rtol=0.)
        np.testing.assert_allclose(np.where(np.isfinite(g), g, np.nan), expected('b_root_scores'), atol=1.e-12, rtol=0.)
        m, mv = model.choose(f)
        b, bv = model.choose(g)
        assert (m, b, model.choose(two)[0]) == tuple(row['actions'][:3])
        ev = float(g[m]) if m is not None else 0.
        root_audit.append({'prescan': list(prefix), 'mass': row['mass'], 'mpc': m, 'b': b, 'f_mpc': mv,
                           'g_mpc': ev, 'g_b': bv, 'b_over_mpc': bv - ev,
                           'b_over_fixed_center': bv - float(g[min(NEAR)])})

    for row in roots:
        np.testing.assert_allclose(model.root_weight(row['prescan']).sum(), row['mass'], atol=1.e-14, rtol=0.)
    expected = summary['exact']
    aggregates = {name: sum(r['mass'] * r[key] for r in active) for name, key in (
        ('adaptive_public_ceiling', 'b'), ('mpc_executed', 'mpc_executed'), ('mpc_forecast', 'mpc_forecast'))}
    aggregates['adaptive_over_mpc'] = aggregates['adaptive_public_ceiling'] - aggregates['mpc_executed']
    aggregates['root_disagreement_mass'] = sum(r['mass'] for r in active if r['actions'][0] != r['actions'][1])
    aggregates['omitted_prescan_probability'] = sum(r['mass'] for r in roots if 'actions' not in r)
    for name, value in aggregates.items():
        np.testing.assert_allclose(value, expected[name], atol=2.e-13, rtol=0.)
    np.testing.assert_allclose(sum(r['mass'] for r in roots), 1., atol=2.e-13, rtol=0.)

    exact_total = np.zeros((len(POLICIES), len(METRICS)))
    for row in roots:
        prefix = row['prescan']
        initial, _, at, _ = prefix
        w = model.root_weight(prefix)
        exact_total += w.sum() * np.r_[0., 0., model.cost[3, initial] + model.cost[initial, at], 2.]
        if 'actions' not in row:
            continue
        memo = {}
        for i, (policy, action) in enumerate(zip(POLICIES, row['actions'], strict=True)):
            horizon = 1 if policy == 'two_images' else 2
            key = (action, horizon)
            if key not in memo:
                memo[key] = model.expected_suffix(w, at, action, horizon)
            exact_total[i] += memo[key]
    for i, policy in enumerate(POLICIES):
        np.testing.assert_allclose(exact_total[i], [expected['total'][policy][k] for k in METRICS],
                                   atol=3.e-10, rtol=0., err_msg=policy)

    records = read_jsonl(directory / 'paired.jsonl')
    rng = np.random.default_rng(summary['replay']['seed'])
    assert len(records) == summary['replay']['n']
    truth = rng.choice(len(model.prior), size=len(records), p=model.prior)
    tapes = rng.random((len(records), 5, 3, 3))
    root_map = {tuple(r['prescan']): r for r in active}
    replay = []
    identical = 0
    for k, (row, h, tape) in enumerate(zip(records, truth, tapes, strict=True)):
        assert k == row['case'] and h == row['shift_id']
        u, _ = model.draw(h, model.initial, tape[0, 1])
        anchor = model.anchor(u)
        v, _ = model.draw(h, anchor, tape[1, 0])
        prefix = (model.initial, u, anchor, v)
        assert prefix == tuple(row['prescan'])
        commands = root_map[prefix]['actions']
        actual = []
        for policy, action in zip(POLICIES, commands, strict=True):
            measured, trace = model.run_trace(prefix, action, h, tape[2:], 1 if policy == 'two_images' else 2)
            published = row['arms'][policy]
            assert trace == published['trace'], (k, policy, trace, published['trace'])
            np.testing.assert_allclose(measured, [published[key] for key in METRICS], atol=1.e-12, rtol=0.)
            actual.append(measured)
        identical += row['arms']['mpc'] == row['arms']['b']
        replay.append(actual)
    replay = np.asarray(replay)
    for i, policy in enumerate(POLICIES):
        np.testing.assert_allclose(replay[:, i].mean(axis=0),
                                   [summary['replay']['total'][policy][k] for k in METRICS], atol=1.e-12, rtol=0.)
    diff = replay[:, 1] - replay[:, 0]
    assert identical == summary['replay']['identical_mpc_b_traces']
    assert int((diff[:, 0] > 0).sum()) == summary['replay']['b_vs_mpc_wins']
    assert int((diff[:, 0] < 0).sum()) == summary['replay']['b_vs_mpc_losses']
    return {'source_hashes_match': True, 'exact_aggregates': aggregates, 'root_cross_checks': root_audit,
            'exact_totals': {p: dict(zip(METRICS, row.tolist(), strict=True))
                             for p, row in zip(POLICIES, exact_total, strict=True)},
            'exact_b_vs_fixed_center': dict(zip(METRICS, (exact_total[1] - exact_total[3]).tolist(), strict=True)),
            'prescan_states_checked': len(roots), 'replay_cases_checked': len(records),
            'replay_arm_traces_checked': len(records) * len(POLICIES),
            'replay_identical_mpc_b_traces': identical,
            'replay_b_vs_mpc_mean_difference': dict(zip(METRICS, diff.mean(axis=0).tolist(), strict=True)),
            'replay_b_vs_mpc_paired_se': dict(zip(METRICS, (diff.std(axis=0, ddof=1) / np.sqrt(len(diff))).tolist(), strict=True))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'verification')
    args = parser.parse_args()
    outcome = audit(args.results)
    (args.results / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(outcome, indent=2, allow_nan=False) + '\n')
