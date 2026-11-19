"""Public-only coverage and suffix planners for the topology screen.

The planners use only an :class:`core.Episode`'s public belief, packet history,
checked graph and public likelihood tables.  Imagined packets are generated
with fixed, independent public tapes; simulator truth is never consulted.
"""
from __future__ import annotations

import numpy as np
from numba import njit

try:  # Direct imports are used by the experiment scripts.
    import core
except ImportError:  # pragma: no cover - package import fallback.
    from . import core


_COVER_SAMPLES = 10
_MPC_SAMPLES = 8
_ROLLOUT_SAMPLES = 6
_TAIL_SAMPLES = 3
_MAX_FIRST = 10
_MAX_SECOND = 8
_MAX_ROLLOUT_FIRST = 8
_EPS = 1.e-10
_PUBLIC_RNG = np.random.default_rng(24681357)
_COVER_H = _PUBLIC_RNG.random(_COVER_SAMPLES)
_COVER_P = _PUBLIC_RNG.random(_COVER_SAMPLES)
_SIM_H = _PUBLIC_RNG.random(64)
_SIM_TAPE = _PUBLIC_RNG.random((64, 16, 7))
_PRICE = np.asarray(core.PRICES, dtype=np.float64)
_COVER_BUDGET = np.asarray([142., 114., 401.], dtype=np.float64)


@njit(cache=True)
def _sample_weight(probability, uniform):
    total = 0.0
    for value in probability:
        total += value
    if total <= 1.e-15:
        return 0
    target = uniform * total
    cumulative = 0.0
    for index in range(len(probability)):
        cumulative += probability[index]
        if target <= cumulative:
            return index
    return len(probability) - 1


@njit(cache=True)
def _coverage_row(belief, node, clock, likelihood, addresses, labels, sites, cost,
                  budget, h_uniforms, packet_uniforms):
    """Score one public prescan move by expected branch-label concentration."""
    target_site = 7 - clock
    selected, best = -1, -1.e300
    for goal in range(len(sites)):
        if sites[goal] != target_site:
            continue
        if not np.isfinite(cost[node, goal, 0]):
            continue
        if np.any(cost[node, goal, :] > budget + 1.e-8):
            continue
        price = core.TIME_PRICE
        for k in range(3):
            price += cost[node, goal, k] * _PRICE[k]
        score = -price
        address = addresses[goal]
        next_clock = clock + 1
        for sample in range(len(h_uniforms)):
            hidden = _sample_weight(belief, h_uniforms[sample])
            packet = _sample_weight(likelihood[next_clock, address, :, hidden],
                                    packet_uniforms[sample])
            ordinary = likelihood[next_clock, address, packet]
            normalizer = 0.0
            for h in range(len(belief)):
                normalizer += belief[h] * ordinary[h]
            if normalizer <= 1.e-100:
                continue
            masses = np.zeros(5)
            for h in range(len(belief)):
                masses[labels[h]] += belief[h] * ordinary[h] / normalizer
            concentration = 0.0
            for label in range(1, 5):
                concentration += masses[label] * masses[label]
            score += concentration / len(h_uniforms)
        if score > best + 1.e-12:
            best, selected = score, goal
    if selected >= 0:
        return selected
    return -1


@njit(cache=True)
def _coverage_batch(beliefs, nodes, clock, likelihood, addresses, labels, sites, cost,
                    budgets, h_uniforms, packet_uniforms):
    result = np.full(len(nodes), -1, np.int64)
    for row in range(len(nodes)):
        result[row] = _coverage_row(beliefs[row], nodes[row], clock, likelihood,
                                    addresses, labels, sites, cost,
                                    budgets[row],
                                    h_uniforms, packet_uniforms)
    return result


def coverage_step(b, nodes, clock, like, addresses, labels, sites, cost, remaining=None):
    """Choose the common prescan view at fixed target site ``7-clock``.

    Ten fixed public hidden-hypothesis/packet draws estimate expected positive
    label squared mass after the packet, less ``PRICES`` and ``TIME_PRICE``.
    ``remaining`` optionally supplies a per-row physical budget; the default
    prescan route envelope is ``[142, 114, 401]`` (the common scan is charged
    separately from the suffix caps).  The simulated packet uses the public
    ``like`` table because this compact interface deliberately does not expose
    detector CDFs or the packet lookup.
    """
    beliefs = np.asarray(b, dtype=np.float64)
    if beliefs.ndim == 1:
        beliefs = beliefs[None]
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if beliefs.shape[0] != len(nodes):
        raise ValueError("belief and node batch lengths differ")
    if int(clock) < 0 or int(clock) + 1 >= np.asarray(like).shape[0]:
        raise ValueError("coverage clock outside likelihood table")
    if remaining is None:
        budgets = np.tile(_COVER_BUDGET, (len(nodes), 1))
    else:
        budgets = np.asarray(remaining, dtype=np.float64)
        if budgets.ndim == 1:
            budgets = np.broadcast_to(budgets, (len(nodes), 3)).copy()
        if budgets.shape != (len(nodes), 3):
            raise ValueError("remaining must have shape [B,3]")
    return _coverage_batch(np.ascontiguousarray(beliefs), nodes, int(clock),
                           np.asarray(like), np.asarray(addresses, dtype=np.int64),
                           np.asarray(labels, dtype=np.int64), np.asarray(sites, dtype=np.int64),
                           np.asarray(cost, dtype=np.float64), budgets,
                           _COVER_H, _COVER_P)


def _price(cost, at, goal):
    return float(np.dot(np.asarray(cost[at, goal, :3], dtype=float), _PRICE)
                 + core.TIME_PRICE)


def _legal_actions(model, at, clock, reserve):
    if int(clock) >= core.FINAL_CLOCK:
        return []
    cost = model.geometry.cost
    reserve = np.asarray(reserve, dtype=float)
    return [goal for goal in range(len(model.geometry.sites))
            if core.legal_move(cost, int(at), goal, reserve)]


def _stop(model, belief, pairs, last):
    probability, _, chosen, _ = core.terminal_evidence(
        belief, pairs, model.conditional, model.addresses, model.labels,
        model.forks, model.bins)
    value, action = core.best_stop(probability, chosen)
    return float(value), int(action)


def _sample_hidden(belief, scenario, depth):
    cumulative = np.cumsum(np.asarray(belief, dtype=float))
    if cumulative[-1] <= _EPS:
        return 0
    uniform = _SIM_H[(scenario * 11 + depth * 17) % len(_SIM_H)]
    return int(min(len(cumulative) - 1, np.searchsorted(cumulative, uniform * cumulative[-1])))


def _draw_packet(model, time, goal, belief, scenario, depth, nearest):
    hidden = _sample_hidden(belief, scenario, depth)
    address = int(model.addresses[goal])
    tape = _SIM_TAPE[(scenario * 7 + depth * 13 + time) % len(_SIM_TAPE),
                     time % _SIM_TAPE.shape[1]]
    packet, _ = core.packet_draw(model.detect[time, address, hidden],
                                 model.cdf[time, address, hidden],
                                 model.mapping, model.lookup, tape)
    packet = int(packet)
    if nearest:
        packet = int(model.packet_map[packet])
    return packet


def _advance(model, state, goal, packet):
    belief, pairs, last, at, clock, reserve = state
    time = int(clock) + 1
    ordinary = model.likelihood[time, model.addresses[goal], packet]
    posterior, supported = core.update(belief, ordinary)
    now = np.asarray([time, goal, packet if supported else 0], dtype=np.int64)
    new_pairs = core.append_pair(pairs, last, now, model.geometry.sites)
    return (np.asarray(posterior, dtype=float), new_pairs, now, int(goal), time,
            np.asarray(reserve, dtype=float) - model.geometry.cost[at, goal])


def _leaf(model, state):
    belief, pairs, last, at, clock, reserve = state
    return float(core.leaf(belief, pairs, last, at, clock, reserve, model.arguments()))


def _one_step_value(model, state, goal, samples, nearest, use_leaf=False, fisher=False):
    belief, _, _, at, clock, _ = state
    if clock >= core.FINAL_CLOCK:
        return -np.inf
    value = 0.0
    info = 0.0
    for scenario in range(int(samples)):
        packet = _draw_packet(model, clock + 1, goal, belief, scenario, 0, nearest)
        successor = _advance(model, state, goal, packet)
        if use_leaf:
            outcome = _leaf(model, successor)
        else:
            outcome, _ = _stop(model, successor[0], successor[1], successor[2])
        value += outcome
        if fisher:
            masses = core.label_mass(successor[0], model.labels)
            row = model.fisher[successor[4], model.addresses[goal]]
            info += float(np.dot(masses[1:], row[1:]))
    value /= max(1, int(samples))
    if fisher:
        info /= max(1, int(samples))
        value = .45 * value + .35 * (_leaf(model, state)) + .20 * (8. * info)
    return value - _price(model.geometry.cost, at, goal)


def _rank_actions(model, state, candidates, nearest, limit, samples=3, use_leaf=False):
    scored = [(_one_step_value(model, state, goal, samples, nearest, use_leaf), goal)
              for goal in candidates]
    scored.sort(reverse=True)
    return [goal for _, goal in scored[:max(1, int(limit))]]


def _greedy_action(model, state, nearest, samples=_MPC_SAMPLES, use_leaf=False):
    _, _, _, at, clock, reserve = state
    candidates = _legal_actions(model, at, clock, reserve)
    if not candidates:
        return -1, -np.inf
    best_value, selected = -np.inf, -1
    for goal in candidates:
        value = _one_step_value(model, state, goal, samples, nearest, use_leaf=use_leaf)
        if value > best_value + 1.e-12:
            best_value, selected = value, goal
    return selected, best_value


def _rank_common_second(model, state, candidates, nearest, limit):
    """Rank legal second moves from one common, packet-independent public row."""
    if not candidates:
        return []
    belief, _, _, at, clock, _ = state
    masses = core.label_mass(belief, model.labels)
    scored = []
    for goal in candidates:
        fisher = model.fisher[min(clock + 1, len(model.fisher) - 1), model.addresses[goal]]
        score = 8. * float(np.dot(masses[1:], fisher[1:])) - _price(model.geometry.cost, at, goal)
        scored.append((score, goal))
    scored.sort(reverse=True)
    return [goal for _, goal in scored[:max(1, int(limit))]]


def _mpc_action(model, state, nearest):
    """Evaluate a fixed two-capture suffix and execute only its first move."""
    _, _, _, at, clock, reserve = state
    candidates = _legal_actions(model, at, clock, reserve)
    if not candidates:
        return -1, -np.inf
    firsts = _rank_actions(model, state, candidates, nearest, _MAX_FIRST, 3, True)
    best_value, selected = -np.inf, -1
    for first in firsts:
        first_states = [
            _advance(model, state, first,
                     _draw_packet(model, clock + 1, first, state[0], s, 0, nearest))
            for s in range(_MPC_SAMPLES)
        ]
        first_value = np.mean([_leaf(model, successor) for successor in first_states]) - _price(
            model.geometry.cost, at, first)
        if first_value > best_value:
            best_value, selected = first_value, first
        representative = first_states[0]
        second_candidates = _legal_actions(model, representative[3], representative[4], representative[5])
        # All first-packet branches have identical physical state/capacity.
        # Rank second actions from the common prior-weighted public Fisher row,
        # rather than pruning from one arbitrary imagined packet branch.
        seconds = _rank_common_second(model, state, second_candidates, nearest,
                                      _MAX_SECOND) if second_candidates else []
        for second in seconds:
            total = 0.0
            for scenario, successor in enumerate(first_states):
                packet = _draw_packet(model, successor[4] + 1, second, successor[0], scenario, 1, nearest)
                total += _leaf(model, _advance(model, successor, second, packet))
            value = total / len(first_states) - _price(model.geometry.cost, at, first)
            value -= _price(model.geometry.cost, representative[3], second)
            if value > best_value + 1.e-12:
                best_value, selected = value, first
    return selected, best_value


def _fisher_action(model, state, nearest):
    _, _, _, at, clock, reserve = state
    candidates = _legal_actions(model, at, clock, reserve)
    if not candidates:
        return -1, -np.inf
    best_value, selected = -np.inf, -1
    for goal in candidates:
        value = _one_step_value(model, state, goal, _MPC_SAMPLES, nearest,
                                use_leaf=True, fisher=True)
        if value > best_value + 1.e-12:
            best_value, selected = value, goal
    return selected, best_value


def _greedy_tail(model, state, nearest, scenario):
    """Sample a public greedy continuation until stop or the eight-step cap."""
    paid = 0.0
    depth = 2
    while state[4] < core.FINAL_CLOCK:
        stop_value, _ = _stop(model, state[0], state[1], state[2])
        goal, value = _greedy_action(model, state, nearest, _TAIL_SAMPLES, use_leaf=True)
        if goal < 0 or value <= stop_value + 1.e-10:
            return stop_value - paid
        packet = _draw_packet(model, state[4] + 1, goal, state[0], scenario, depth, nearest)
        paid += _price(model.geometry.cost, state[3], goal)
        state = _advance(model, state, goal, packet)
        depth += 1
    stop_value, _ = _stop(model, state[0], state[1], state[2])
    return stop_value - paid


def _rollout_action(model, state, nearest):
    _, _, _, at, clock, reserve = state
    candidates = _legal_actions(model, at, clock, reserve)
    if not candidates:
        return -1, -np.inf
    firsts = _rank_actions(model, state, candidates, nearest,
                           _MAX_ROLLOUT_FIRST, 2, False)
    best_value, selected = -np.inf, -1
    for first in firsts:
        value = 0.0
        for scenario in range(_ROLLOUT_SAMPLES):
            packet = _draw_packet(model, clock + 1, first, state[0], scenario, 0, nearest)
            successor = _advance(model, state, first, packet)
            value += _greedy_tail(model, successor, nearest, scenario)
        value = value / _ROLLOUT_SAMPLES - _price(model.geometry.cost, at, first)
        if value > best_value + 1.e-12:
            best_value, selected = value, first
    return selected, best_value


def commands(episode, method="greedy"):
    """Return one legal public action per episode row and replan each call."""
    method = str(method).lower()
    if method not in {"greedy", "mpc", "fisher", "rollout"}:
        raise ValueError(f"unknown planning method: {method}")
    model = episode.model
    nearest = bool(getattr(episode.env, "nearest", False))
    out = np.full(len(episode.indices), core.ABSTAIN, dtype=np.int64)
    for row in range(len(out)):
        if bool(episode.done[row]):
            continue
        state = (np.asarray(episode.b[row], dtype=float).copy(),
                 np.asarray(episode.pairs[row], dtype=np.int64).copy(),
                 np.asarray(episode.last[row], dtype=np.int64).copy(),
                 int(episode.at[row]), int(episode.clock[row]),
                 np.asarray(episode.reserve[row], dtype=float).copy())
        stop_value, stop_action = _stop(model, state[0], state[1], state[2])
        if method == "greedy":
            goal, value = _greedy_action(model, state, nearest)
        elif method == "mpc":
            goal, value = _mpc_action(model, state, nearest)
        elif method == "fisher":
            goal, value = _fisher_action(model, state, nearest)
        else:
            goal, value = _rollout_action(model, state, nearest)
        out[row] = int(goal) if goal >= 0 and value > stop_value + 1.e-10 else int(stop_action)
    return out


__all__ = ["commands", "coverage_step"]
