"""Two-step receding view selection over the shared ordered coverage pass."""
from __future__ import annotations

import numpy as np
from numba import njit

from core import PRESCAN_CAPS, PREFIX, PRICES, TIME_PRICE, label_mass, legal_move, update


_rng = np.random.default_rng(723300)
_draws = _rng.random((12,3))


@njit(cache=True)
def sample(mass, u):
    cumulative = 0.
    for index in range(len(mass)):
        cumulative += mass[index]
        if cumulative >= u:
            return index
    return len(mass)-1


@njit(cache=True)
def coverage_value(belief, labels):
    mass = label_mass(belief,labels)
    return np.dot(mass,mass)


@njit(cache=True)
def choose(b, at, clock, like, addresses, labels, sites, cost, remaining):
    target = 7-clock
    best, selected = -np.inf,-1
    for first in range(len(sites)):
        if sites[first]!=target or not legal_move(cost,at,first,remaining):
            continue
        first_cost=cost[at,first]
        # All complete ordered routes fit the inherited envelope, verified by
        # enumeration from the common start before benchmarking.
        seconds = np.zeros(len(sites),np.bool_)
        if clock+1<PREFIX:
            for second in range(len(sites)):
                seconds[second] = (sites[second]==target-1 and
                                   legal_move(cost,first,second,remaining-first_cost))
            if not np.any(seconds):
                continue
        one = 0.
        two = np.zeros(len(sites))
        for draws in _draws:
            h = sample(b,draws[0])
            packet = sample(like[clock+1,addresses[first],:,h],draws[1])
            updated,_ = update(b,like[clock+1,addresses[first],packet])
            one += coverage_value(updated,labels)/len(_draws)
            for second in np.flatnonzero(seconds):
                observation = sample(like[clock+2,addresses[second],:,h],draws[2])
                posterior,_ = update(updated,like[clock+2,addresses[second],observation])
                two[second] += coverage_value(posterior,labels)/len(_draws)
        value = one-np.dot(first_cost,PRICES)-TIME_PRICE
        if clock+1<PREFIX:
            value = -np.inf
            for second in np.flatnonzero(seconds):
                score = two[second]-np.dot(first_cost+cost[first,second],PRICES)-2.*TIME_PRICE
                value = max(value,score)
        if value>best+1.e-12:
            best,selected=value,first
    return selected


@njit(cache=True)
def _batch(beliefs,nodes,clock,like,addresses,labels,sites,cost,budgets):
    out=np.full(len(nodes),-1,np.int64)
    for i in range(len(nodes)):
        out[i]=choose(beliefs[i],nodes[i],clock,like,addresses,labels,sites,cost,budgets[i])
    return out


def coverage_step(b,nodes,clock,like,addresses,labels,sites,cost,remaining=None):
    if remaining is None:
        remaining=np.tile(PRESCAN_CAPS,(len(nodes),1))
    return _batch(b,nodes,clock,like,addresses,labels,sites,cost,remaining)
