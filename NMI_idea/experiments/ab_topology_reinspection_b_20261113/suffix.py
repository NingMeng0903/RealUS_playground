"""Sampled belief planners with identical real observation updates and stops."""
from __future__ import annotations

import numpy as np
from numba import njit

from core import (ABSTAIN, FINAL_CLOCK, PRICES, TIME_PRICE, append_pair, best_stop,
                  label_mass, leaf, legal_move, packet_draw, terminal_evidence, update)


_rng = np.random.default_rng(723301)
_hypothesis_draws = _rng.random((16, 64))
_observation_draws = _rng.random((16, 64, 7))


@njit(cache=True)
def price(cost, at, goal):
    return np.dot(cost[at, goal], PRICES) + TIME_PRICE


@njit(cache=True)
def sample_h(belief, u):
    c = 0.
    for h in range(len(belief)):
        c += belief[h]
        if c >= u:
            return h
    return len(belief)-1


@njit(cache=True)
def draw(model, h, clock, goal, tape):
    args, lookup, packet_map, nearest, _ = model
    _, _, detect, cdf, mapping, _, addresses, _, _, _, _, _ = args
    address = addresses[goal]
    packet, _ = packet_draw(detect[clock, address, h],cdf[clock, address, h],mapping,lookup,tape)
    return packet_map[packet] if nearest else packet


@njit(cache=True)
def propagate(model, state, goal, packet):
    args = model[0]
    like, _, _, _, _, _, addresses, _, _, sites, cost, _ = args
    b, pairs, last, at, clock, reserve = state
    time = clock+1
    posterior, supported = update(b,like[time,addresses[goal],packet])
    now = np.array([time,goal,packet if supported else 0],np.int64)
    return posterior,append_pair(pairs,last,now,sites),now,goal,time,reserve-cost[at,goal]


@njit(cache=True)
def stop(model, state):
    _, conditional, _, _, _, bins, addresses, labels, forks, _, _, _ = model[0]
    b,pairs,_,_,_,_ = state
    probs,_,slots,_ = terminal_evidence(b,pairs,conditional,addresses,labels,forks,bins)
    return best_stop(probs,slots)


@njit(cache=True)
def tail_value(model,state):
    b,pairs,last,at,clock,reserve = state
    return leaf(b,pairs,last,at,clock,reserve,model[0])


@njit(cache=True)
def moves(model,state):
    _,_,_,at,clock,reserve = state
    cost = model[0][10]
    n = cost.shape[0]
    legal = np.zeros(n,np.bool_)
    if clock < FINAL_CLOCK:
        for a in range(n):
            legal[a] = legal_move(cost,at,a,reserve)
    return legal


@njit(cache=True)
def greedy(model,state,samples,offset=0,fisher=False):
    b,_,_,at,clock,_ = state
    args,_,_,_,info = model
    labels,addresses,cost = args[7],args[6],args[10]
    best,selected = stop(model,state)
    if clock >= FINAL_CLOCK:
        return selected,best
    mask = moves(model,state)
    for a in np.flatnonzero(mask):
        estimate = 0.
        for s in range(samples):
            j = (offset+s)%64
            h = sample_h(b,_hypothesis_draws[clock+1,j])
            packet = draw(model,h,clock+1,a,_observation_draws[clock+1,j])
            after = propagate(model,state,a,packet)
            estimate += tail_value(model,after)/samples
        if fisher:
            # Conditional geometry-only bonus, averaged over possible branch labels.
            mass = label_mass(b,labels)
            estimate += .5*np.dot(mass,info[clock+1,addresses[a]])
        value = estimate - price(cost,at,a)
        if value > best+1e-12:
            best,selected = value,a
    return selected,best


@njit(cache=True)
def rank_first(model,state,samples,width):
    """Prioritize feasible physical moves by the shared one-step tail."""
    b,_,_,at,clock,_ = state
    cost = model[0][10]
    mask = moves(model,state)
    values = np.full(len(mask),-np.inf)
    for a in np.flatnonzero(mask):
        acc = 0.
        for j in range(samples):
            h = sample_h(b,_hypothesis_draws[clock+1,j+32])
            obs = draw(model,h,clock+1,a,_observation_draws[clock+1,j+32])
            acc += tail_value(model,propagate(model,state,a,obs))/samples
        values[a] = acc-price(cost,at,a)
    selected = np.full(width,-1,np.int64)
    for i in range(width):
        a = np.argmax(values)
        if not np.isfinite(values[a]):
            break
        selected[i] = a
        values[a] = -np.inf
    return selected


@njit(cache=True)
def mpc(model,state,samples):
    """Two captures, a fixed second action, and posterior update on both."""
    b,_,_,at,clock,reserve = state
    cost = model[0][10]
    best,selected = stop(model,state)
    mask = moves(model,state)
    for first in np.flatnonzero(mask):
        remaining = reserve - cost[at,first]
        fixed_second = np.zeros(len(mask),np.bool_)
        if clock+1 < FINAL_CLOCK:
            for second in range(len(mask)):
                fixed_second[second] = legal_move(cost,first,second,remaining)
        second_values = np.zeros(len(mask))
        one = 0.
        for j in range(samples):
            h = sample_h(b,_hypothesis_draws[clock+1,j])
            p = draw(model,h,clock+1,first,_observation_draws[clock+1,j])
            first_state = propagate(model,state,first,p)
            one += tail_value(model,first_state)/samples
            for second in np.flatnonzero(fixed_second):
                p2 = draw(model,h,clock+2,second,_observation_draws[clock+2,j])
                next_state = propagate(model,first_state,second,p2)
                second_values[second] += (tail_value(model,next_state)-price(cost,first,second))/samples
        plan_value = one
        for second in np.flatnonzero(fixed_second):
            plan_value = max(plan_value,second_values[second])
        plan_value -= price(cost,at,first)
        if plan_value > best+1e-12:
            best,selected = plan_value,first
    return selected,best


@njit(cache=True)
def rollout(model,state,samples,tail_samples,width):
    """Execute each imagined continuation with real posterior updates to stop."""
    b,_,_,at,clock,_ = state
    cost = model[0][10]
    best,selected = stop(model,state)
    for first in rank_first(model,state,4,width):
        if first < 0:
            continue
        value = 0.
        for j in range(samples):
            h = sample_h(b,_hypothesis_draws[clock+1,j+16])
            obs = draw(model,h,clock+1,first,_observation_draws[clock+1,j+16])
            future = propagate(model,state,first,obs)
            paid = price(cost,at,first)
            while True:
                a,_ = greedy(model,future,tail_samples,48)
                if a >= len(cost):
                    terminal,_ = stop(model,future)
                    value += (terminal-paid)/samples
                    break
                next_clock = future[4]+1
                observation = draw(model,h,next_clock,a,_observation_draws[next_clock,j+16])
                paid += price(cost,future[3],a)
                future = propagate(model,future,a,observation)
        if value > best+1e-12:
            best,selected = value,first
    return selected,best


@njit(cache=True)
def _batch(model, beliefs, pairs, last, at, clocks, reserves, done, mode):
    out = np.full(len(at),ABSTAIN,np.int64)
    for i in range(len(at)):
        if done[i]:
            continue
        state = beliefs[i],pairs[i],last[i],at[i],clocks[i],reserves[i]
        if mode == 0:
            out[i],_ = mpc(model,state,12)
        elif mode == 1:
            out[i],_ = greedy(model,state,10)
        elif mode == 2:
            out[i],_ = greedy(model,state,10,fisher=True)
        elif mode == 3:
            out[i],_ = rollout(model,state,12,4,10)
        else:
            _,out[i] = stop(model,state)
    return out


def commands(episode,method="greedy"):
    modes = {"mpc":0,"greedy":1,"fisher":2,"rollout":3,"prescan_only":4}
    m = episode.model
    model = (m.arguments(),m.lookup,m.packet_map,episode.env.nearest,m.fisher)
    return _batch(model,episode.b,episode.pairs,episode.last,episode.at,episode.clock,
                  episode.reserve,episode.done,modes[method])

