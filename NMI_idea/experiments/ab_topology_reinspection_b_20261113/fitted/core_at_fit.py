"""Shared public belief, checked motion and fresh-evidence endpoint."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

from numba import njit
import numpy as np

import world


NODES = 30
ABSENT, ABSTAIN, ACTIONS = 30, 35, 36
PREFIX = 5
SUFFIX = 8
FINAL_CLOCK = PREFIX + SUFFIX
CAPS = np.array([65., 90., 250.])
PRESCAN_CAPS = np.array([142., 114., 401.])
PRICES = np.array([.001, .0005, .0001])
TIME_PRICE = .012
WRONG_PRICE = 1.5
FORK_TOLERANCE = 5.
TARGET_SITES = np.array([[4, 3], [5, 4]], dtype=np.int64)


def seed_for(namespace: int, index: int) -> int:
    return int(np.random.SeedSequence([namespace, index]).generate_state(1)[0])


@njit(cache=True)
def legal_move(cost, at, goal, reserve):
    for i in range(3):
        if cost[at, goal, i] > reserve[i] + 1e-8:
            return False
    return True


@njit(cache=True)
def packet_draw(detect, cdf, mapping, lookup, tape):
    bins = np.zeros(3, np.int64)
    kinds = np.full(3, -1, np.int64)
    for kind in range(3):
        if tape[kind] < detect[kind]:
            fine = min(np.searchsorted(cdf[kind], tape[kind + 3]), 48)
            bins[kind] = mapping[fine]
            kinds[kind] = kind
    for i in range(1, 3):
        for j in range(i, 0, -1):
            if bins[j] >= bins[j-1]:
                break
            bins[j], bins[j-1] = bins[j-1], bins[j]
            kinds[j], kinds[j-1] = kinds[j-1], kinds[j]
    if bins[0] > 0 and bins[0] == bins[2]:
        # Six equiprobable private permutations of indistinguishable returns.
        first = min(2, int(3 * tape[6]))
        swap = int(6 * tape[6]) % 2
        order = (first, (first+1+swap) % 3, (first+2-swap) % 3)
        kinds = np.array([kinds[order[0]], kinds[order[1]], kinds[order[2]]])
    elif tape[6] >= .5:
        for j in range(2):
            if bins[j] > 0 and bins[j] == bins[j+1]:
                kinds[j], kinds[j+1] = kinds[j+1], kinds[j]
                break
    return lookup[bins[0], bins[1], bins[2]], kinds


@njit(cache=True)
def update(b, probability):
    result = b * probability
    normalizer = result.sum()
    if normalizer <= 1e-100:
        return b.copy(), False
    return result / normalizer, True


@njit(cache=True)
def label_mass(b, labels):
    mass = np.zeros(5)
    for i in range(len(b)):
        mass[labels[i]] += b[i]
    return mass


@njit(cache=True)
def append_pair(pairs, last, now, sites):
    result = pairs.copy()
    if last[0] >= 0 and last[2] > 0 and now[2] > 0:
        for group in range(2):
            if sites[last[1]] == TARGET_SITES[group, 0] and sites[now[1]] == TARGET_SITES[group, 1]:
                result[group, :3] = last
                result[group, 3:] = now
    return result


@njit(cache=True)
def terminal_evidence(b, pairs, conditional, addresses, labels, forks, bins):
    """Marginalize the joint two-frame identity event before committing."""
    mass = label_mass(b, labels)
    probability = np.zeros(5)
    probability[0] = mass[0]
    positions = np.zeros(5)
    chosen = np.full((5, 2), -1, np.int64)
    for group in range(2):
        t0, a0, p0, t1, a1, p1 = pairs[group]
        if t0 < 0:
            continue
        left = conditional[t0, addresses[a0], p0]
        right = conditional[t1, addresses[a1], p1]
        for label in range(1 + 2*group, 3 + 2*group):
            best = -1.
            for i in range(3):
                if bins[p0, i] == 0:
                    continue
                for j in range(3):
                    if bins[p1, j] == 0:
                        continue
                    # Particles are stored in sorted fork-coordinate order.
                    # The best fixed-width interval is the Bayes point action
                    # for the scored localization tolerance.
                    position, value, running = 0., 0., 0.
                    lo = 0
                    for h in range(len(b)):
                        if labels[h] != label:
                            continue
                        running += b[h] * left[i, h] * right[j, h]
                        while forks[h]-forks[lo] > 2.*FORK_TOLERANCE:
                            if labels[lo] == label:
                                running -= b[lo] * left[i, lo] * right[j, lo]
                            lo += 1
                        if running > value:
                            value = running
                            position = .5*(forks[lo]+forks[h])
                    if value > best:
                        best = value
                        chosen[label, 0], chosen[label, 1] = i, j
                        probability[label], positions[label] = value, position
    return probability, positions, chosen, mass


@njit(cache=True)
def best_stop(probabilities, chosen):
    value, action = 0., ABSTAIN
    for label, probability in enumerate(probabilities):
        if label > 0 and chosen[label, 0] < 0:
            continue
        utility = probability - WRONG_PRICE * (1. - probability)
        if utility > value + 1e-12:
            value, action = utility, NODES + label
    return value, action


def _paths(cost: np.ndarray, sites: np.ndarray):
    """Build a small catalog of complete physical suffix witnesses."""
    n = len(sites)
    adjacency = [np.flatnonzero(np.isfinite(row[:, 0])) for row in cost]
    # [start, group, option, (steps,TCP,turn,rail)]
    rows = [[[] for _ in TARGET_SITES] for _ in sites]
    for start in range(n):
        for group, (near, far) in enumerate(TARGET_SITES):
            for next_node in adjacency[start]:
                if sites[next_node] == far:
                    rows[start][group].append((1., *cost[start, next_node], 1.))
            # The longest shortest site-to-site path needs four moves. With
            # alternatives at each station, enumerate checked monotone paths
            # to the near plane, then the adjacent far plane.
            def visit(at, moves, expense):
                if moves and sites[at] == near:
                    for end in adjacency[at]:
                        if sites[end] == far:
                            rows[start][group].append((moves + 1., *(expense + cost[at, end]), 0.))
                    return
                if moves >= 4:
                    return
                for end in adjacency[at]:
                    approaching = abs(sites[end]-near) < abs(sites[at]-near)
                    if approaching or (moves == 0 and sites[at] == near and sites[end] == near):
                        visit(end, moves+1, expense+cost[at, end])
            visit(start, 0, np.zeros(3))
            # Keep the nondominated concrete witnesses in each path class.
            candidates = np.asarray(rows[start][group], dtype=float)
            survivors = []
            for i, row in enumerate(candidates):
                dominant = (candidates[:, -1] == row[-1]) & np.all(candidates[:, :4] <= row[:4]+1e-9, axis=1)
                better = np.any(candidates[:, :4] < row[:4]-1e-9, axis=1)
                duplicate = np.all(np.abs(candidates[:, :4]-row[:4]) <= 1e-9, axis=1)
                if not np.any(dominant & (better | (duplicate & (np.arange(len(candidates)) < i)))):
                    survivors.append(row)
            rows[start][group] = survivors
    width = max(len(x) for row in rows for x in row)
    result = np.full((n, 2, width, 5), np.inf)
    for i, row in enumerate(rows):
        for j, paths in enumerate(row):
            result[i, j, :len(paths)] = np.asarray(paths).reshape(-1, 5)
    return result


@njit(cache=True)
def potential(at, clock, reserve, last, pair_prob, mass, paths, sites):
    """Shared approximate tail using only actual joint-cap route witnesses."""
    best = 0.
    slots = FINAL_CLOCK-clock
    if slots <= 0:
        return best
    for group in range(2):
        for path in paths[at, group]:
            if path[0] > slots:
                continue
            if path[-1] == 1 and (last[0] < 0 or last[2] <= 0 or sites[at] != TARGET_SITES[group, 0]):
                continue
            if np.any(path[1:4] > reserve+1e-8):
                continue
            expense = np.dot(path[1:4], PRICES) + path[0]*TIME_PRICE
            for label in range(1+2*group, 3+2*group):
                chance = (.62 if path[-1] == 0 else .72) * mass[label]**2
                best = max(best, chance - expense)
        for label in range(1+2*group, 3+2*group):
            best = max(best, .80*pair_prob[label]**2 - TIME_PRICE)
    return best


@njit(cache=True)
def leaf(b, pairs, last, at, clock, reserve, model):
    _, conditional, _, _, _, bins, addresses, labels, forks, sites, _, paths = model
    p, _, chosen, mass = terminal_evidence(b, pairs, conditional, addresses, labels, forks, bins)
    stop, _ = best_stop(p, chosen)
    return max(stop, potential(at, clock, reserve, last, p, mass, paths, sites))


@dataclass(frozen=True)
class Model:
    geometry: world.Geometry
    likelihood: np.ndarray
    conditional: np.ndarray
    detect: np.ndarray
    cdf: np.ndarray
    mapping: np.ndarray
    lookup: np.ndarray
    bins: np.ndarray
    coordinates: np.ndarray
    addresses: np.ndarray
    labels: np.ndarray
    families: np.ndarray
    forks: np.ndarray
    paths: np.ndarray
    fisher: np.ndarray
    packet_map: np.ndarray
    nearest_slot: np.ndarray

    def arguments(self):
        return (self.likelihood, self.conditional, self.detect, self.cdf,
                self.mapping, self.bins, self.addresses, self.labels, self.forks,
                self.geometry.sites, self.geometry.cost, self.paths)


def make_model(repo: Path, geometry: world.Geometry, prior: world.CaseBatch, grid=5):
    sys.path.insert(0, str(repo / "NMI_idea/experiments/ab_relational_b_20261113"))
    from return_sets import build_tables

    _, unique, addresses = np.unique(np.column_stack([geometry.sites, geometry.views]), axis=0,
                                     return_index=True, return_inverse=True)
    order = np.argsort(np.nan_to_num(prior.fork_mm,nan=0.),kind="stable")
    labels, families = prior.labels[order], prior.families[order]
    detect = np.ascontiguousarray(prior.detect[:, unique][:, :, order])
    cdf = np.ascontiguousarray(prior.cdf[:, unique][:, :, order])
    tables = build_tables(detect, cdf, grid=grid)
    likelihood = tables.total.astype(np.float32)
    conditional = tables.joint.astype(np.float32)
    np.divide(conditional, likelihood[:, :, :, None], out=conditional,
              where=likelihood[:, :, :, None] > 0)
    conditional *= likelihood[:, :, :, None] > 0
    forks = np.nan_to_num(prior.fork_mm[order], nan=0.)
    info = np.zeros((len(geometry.clock_s), len(unique), 5))
    for label in range(1, 5):
        selected = labels == label
        x = forks[selected]
        if len(x) < 2 or x.var() < 1e-8:
            continue
        centers = prior.centers[:, unique][:, :, order[selected], 0].astype(float)
        derivative = np.mean((x-x.mean())[None,None,:,None]*centers, axis=2) / x.var()
        residual = centers - derivative[:, :, None] * (x-x.mean())[None,None,:,None]
        variance = residual.var(axis=2) + world.IMAGE_SD**2
        info[:, :, label] = (derivative**2 / variance).sum(axis=-1)*detect[:, :, selected, 0].mean(axis=-1)
    return Model(geometry, likelihood, conditional, detect, cdf,
                 tables.mapping, tables.lookup, tables.bins, tables.coordinates,
                 addresses.astype(np.int64), labels, families, forks,
                 _paths(geometry.cost, geometry.sites), info, tables.nearest_packet, tables.nearest_slot), tables


def nearest_model(full, tables):
    """Nested post-prescan compression of the identical simulated packets."""
    like = tables.nearest_total.astype(np.float32)
    correct = np.zeros((*like.shape[:3], 3, like.shape[-1]), np.float32)
    np.divide(tables.nearest_correct, like, out=correct[..., 0, :], where=like > 0)
    bins = np.zeros((like.shape[2], 3), np.int64)
    bins[1:, 0] = 1 + np.arange(like.shape[2]-1) % len(full.coordinates)
    return Model(full.geometry, like, correct, full.detect, full.cdf, full.mapping,
                 full.lookup, bins, full.coordinates, full.addresses, full.labels,
                 full.families, full.forks, full.paths, full.fisher,
                 full.packet_map, full.nearest_slot)


@njit(cache=True)
def arrival_batch(detect, cdf, indices, clock, actions, tapes, sites, mapping, lookup, packet_map, nearest_slot, nearest):
    packets = np.zeros(len(indices), np.int64)
    kinds = np.full((len(indices), 3), -1, np.int64)
    for i in range(len(indices)):
        a = actions[i]
        if a < 0:
            continue
        packet, labels = packet_draw(detect[clock[i], a, indices[i]], cdf[clock[i], a, indices[i]],
                                     mapping, lookup, tapes[i, clock[i], sites[a]-3])
        if nearest:
            slot = nearest_slot[packet]
            if slot >= 0:
                kinds[i, 0] = labels[slot]
            packet = packet_map[packet]
        else:
            kinds[i] = labels
        packets[i] = packet
    return packets, kinds


class BatchEnv:
    """Identical initial MPC scan with disjoint simulator and inference draws."""
    action_count = ACTIONS

    def __init__(self, model, cases, namespace, *, stochastic=False, full_model=None):
        self.model, self.cases = model, cases
        self.full_model = full_model if full_model is not None else model
        self.nearest = self.full_model is not model
        self.geometry = model.geometry
        self.namespace = namespace
        self.stochastic = stochastic
        self.rng = np.random.default_rng(namespace+177)
        self.n_cases = len(cases.labels)
        self.feature_dim = len(model.labels) + 5 + 4 + 5 + NODES + 4 + 9 + 2
        self.prescan = None

    def tape(self, indices):
        out = np.empty((len(indices), len(self.geometry.clock_s), 5, 7))
        for k, index in enumerate(indices):
            fixed = np.random.default_rng(seed_for(self.namespace, index))
            out[k] = fixed.random(out[k].shape)
        return out

    def arrivals(self, indices, clocks, actions, tapes, nearest=False):
        m = self.model
        return arrival_batch(self.cases.detect, self.cases.cdf, indices, clocks, actions,
                             tapes, self.geometry.sites, m.mapping, m.lookup, m.packet_map,
                             m.nearest_slot, nearest)

    def prepare_prefix(self):
        from prescan import coverage_step

        if self.prescan is not None:
            return
        m = self.full_model
        n, h = self.n_cases, len(m.labels)
        belief = np.full((n, PREFIX+1, h), 1./h)
        nodes = np.full((n, PREFIX+1), self.geometry.start, np.int64)
        packets = np.zeros((n, PREFIX+1), np.int64)
        costs = np.zeros((n, 3))
        indices = np.arange(n, dtype=np.int64)
        tapes = self.tape(indices)
        since = time.perf_counter()
        for t in range(PREFIX):
            goal = coverage_step(belief[:,t], nodes[:,t], t, m.likelihood, m.addresses,
                                 m.labels, m.geometry.sites, m.geometry.cost, PRESCAN_CAPS-costs)
            if np.any(goal < 0):
                raise RuntimeError("common coverage scan has an unreachable station")
            obs, _ = self.arrivals(indices, np.full(n,t+1), goal, tapes)
            like = m.likelihood[t+1,m.addresses[goal],obs]
            posterior = belief[:,t]*like
            total = posterior.sum(axis=1,keepdims=True)
            belief[:,t+1] = np.divide(posterior,total,out=belief[:,t].copy(),where=total>1e-100)
            costs += m.geometry.cost[nodes[:,t],goal]
            nodes[:,t+1], packets[:,t+1] = goal,obs
        self.prescan = belief, nodes, packets, costs
        self.prefix_cpu_s = time.perf_counter()-since

    def reset(self, indices):
        self.prepare_prefix()
        return Episode(self, np.asarray(indices, dtype=np.int64))


class Episode:
    def __init__(self, env, indices):
        self.env, self.indices = env, indices
        self.model = env.model
        self.clock = np.full(len(indices), PREFIX, np.int64)
        self.reserve = np.tile(CAPS, (len(indices),1))
        beliefs, nodes, packets, self.prefix_cost = (v[indices].copy() for v in env.prescan)
        self.prefix = beliefs, nodes, packets
        self.b = beliefs[:, -1].copy()
        self.at = nodes[:, -1].copy()
        self.last_packet = packets[:, -1].copy()
        self.last = np.full((len(indices),3), -1, np.int64)
        self.pairs = np.full((len(indices),2,6), -1, np.int64)
        self.private = np.full((len(indices), len(env.geometry.clock_s), 3), -1, np.int64)
        self.done = np.zeros(len(indices), bool)
        self.records = [None]*len(indices)
        self.traces = [[] for _ in indices]
        self.tapes = env.tape(indices)
        if env.stochastic:
            self.tapes[:,PREFIX+1:] = env.rng.random(self.tapes[:,PREFIX+1:].shape)

    def _features(self, b, at, packet, clock, reserve, p, pairs, last, model):
        m = model
        mass = np.column_stack([b[:,m.labels==label].sum(1) for label in range(5)])
        family = np.column_stack([b[:,m.families==f].sum(1) for f in range(4)])
        view = np.eye(NODES,dtype=np.float32)[at]
        relative = np.column_stack([reserve/CAPS, (FINAL_CLOCK-clock)/SUFFIX])
        bins = m.bins[packet]
        coords = np.vstack([np.zeros((1,2)),m.coordinates])[bins]/10.
        returns = np.concatenate([coords, (bins>0)[...,None]],axis=2).reshape(len(b),9)
        witnesses = pairs[...,0]>=0
        return np.column_stack([np.sqrt(b),mass,family,p,view,relative,returns,witnesses]).astype(np.float32)

    def prefix_features(self):
        b, n, o = self.prefix
        empty = np.full((len(b),2,6),-1)
        vals = np.zeros((len(b),5))
        return np.stack([self._features(b[:,t],n[:,t],o[:,t],np.full(len(b),t),self.reserve,
                                      vals,empty,self.last,self.env.full_model) for t in range(PREFIX)],axis=1)

    def evidence(self, i):
        m = self.model
        return terminal_evidence(self.b[i],self.pairs[i],m.conditional,m.addresses,m.labels,m.forks,m.bins)

    def features(self):
        m = self.model
        probs = np.zeros((len(self.indices),5))
        mask = np.zeros((len(self.indices),ACTIONS),bool)
        mask[:, ABSTAIN] = True
        for i in np.flatnonzero(~self.done):
            p, _, slots, _ = self.evidence(i)
            probs[i] = p
            mask[i,ABSENT] = True
            mask[i,ABSENT+1:ABSTAIN] = slots[1:,0]>=0
            if self.clock[i] < FINAL_CLOCK:
                mask[i,:NODES] = np.all(m.geometry.cost[self.at[i]]<=self.reserve[i]+1e-8,axis=1)
        packet = self.last_packet.copy()
        if self.env.nearest:
            prefix_exit = self.clock==PREFIX
            packet[prefix_exit] = m.packet_map[packet[prefix_exit]]
        x = self._features(self.b,self.at,packet,self.clock,self.reserve,probs,self.pairs,self.last,m)
        return x,mask

    def step(self, actions):
        m = self.model
        actions = np.asarray(actions,dtype=np.int64)
        moving = (~self.done)&(actions<NODES)
        scan = np.where(moving,actions,-1)
        arrival = self.clock+moving
        packets,kinds = self.env.arrivals(self.indices,arrival,scan,self.tapes,self.env.nearest)
        rewards = np.zeros(len(actions),np.float32)
        for i in np.flatnonzero(~self.done):
            a = actions[i]
            if not 0 <= a < ACTIONS:
                raise ValueError(f"invalid action {a}")
            if a < NODES:
                if self.clock[i]>=FINAL_CLOCK or not legal_move(m.geometry.cost,self.at[i],a,self.reserve[i]):
                    raise ValueError(f"unavailable physical move {self.at[i]} to {a}")
                before = leaf(self.b[i],self.pairs[i],self.last[i],self.at[i],self.clock[i],self.reserve[i],m.arguments())
                expense = m.geometry.cost[self.at[i],a]
                self.reserve[i] -= expense
                self.clock[i] += 1
                t = self.clock[i]
                obs = packets[i]
                self.b[i], supported = update(self.b[i],m.likelihood[t,m.addresses[a],obs])
                now = np.array([t,a,obs if supported else 0],np.int64)
                self.pairs[i] = append_pair(self.pairs[i],self.last[i],now,m.geometry.sites)
                self.last[i], self.at[i], self.last_packet[i] = now,a,obs
                self.private[i,t] = kinds[i]
                after = leaf(self.b[i],self.pairs[i],self.last[i],a,t,self.reserve[i],m.arguments())
                rewards[i] = after-before-(expense@PRICES+TIME_PRICE)
                self.traces[i].append([int(t),int(a),int(obs),kinds[i].tolist(),bool(supported)])
            else:
                rewards[i] = self._finish(i,a)
        return rewards, self.done.copy()

    def _finish(self, i, action):
        p, positions, slots, mass = self.evidence(i)
        prediction = action-NODES if action != ABSTAIN else -1
        index = self.indices[i]
        truth = int(self.env.cases.labels[index])
        truth_family = int(self.env.cases.families[index])
        choice = None
        error = None
        correct_geometry = prediction==truth
        correct_images = False
        if prediction>0:
            if slots[prediction,0] < 0:
                raise ValueError("positive declaration without fresh observed pair")
            group = (prediction-1)//2
            t0,a0,k0,t1,a1,k1 = map(int,self.pairs[i,group])
            i0,i1 = map(int,slots[prediction])
            correct_images = self.private[i,t0,i0]==0 and self.private[i,t1,i1]==0
            if truth>0:
                error = abs(float(positions[prediction])-float(self.env.cases.fork_mm[index]))
            choice = dict(frames=[t0,t1],nodes=[a0,a1],packets=[k0,k1],slots=[i0,i1],
                          fork_hat_mm=float(positions[prediction]))
        success = prediction==truth and (truth==0 or (correct_images and error is not None and error<=FORK_TOLERANCE))
        wrong = prediction>=0 and not success
        suffix = CAPS-self.reserve[i]
        total = suffix+self.prefix_cost[i]
        prefix = np.column_stack([self.prefix[1][i,1:],self.prefix[2][i,1:]]).astype(int)
        self.records[i] = dict(index=int(index),success=int(success),wrong=int(wrong),positive=int(truth>0),
            truth=truth,family=truth_family,prediction=int(prediction),correct_declaration=int(correct_geometry),
            correct_images=int(correct_images),affirmative_error=int(prediction>0 and wrong),
            fork_error_mm=error,chosen=choice,posterior=mass.tolist(),
            probability=float(p[prediction]) if prediction>=0 else None,
            tcp_mm=float(total[0]),turn_deg=float(total[1]),rail_mm=float(total[2]),
            prefix_cost=self.prefix_cost[i].tolist(),suffix_cost=suffix.tolist(),
            seconds=float(self.env.geometry.clock_s[self.clock[i]]),captures=int(self.clock[i]),
            prefix=prefix.tolist(),prefix_hash=hashlib.sha256(prefix.tobytes()).hexdigest(),trace=self.traces[i])
        self.done[i] = True
        value = leaf(self.b[i],self.pairs[i],self.last[i],self.at[i],self.clock[i],self.reserve[i],self.model.arguments())
        return float(success)-WRONG_PRICE*float(wrong)-value

    def outcomes(self):
        if not self.done.all():
            raise RuntimeError("some episodes have not terminated")
        return self.records

    def actions(self, method="greedy"):
        from suffix import commands
        return commands(self,method)
