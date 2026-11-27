"""Plan beyond two captures using complete checked fresh-pair route witnesses."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from core import (ABSTAIN, FINAL_CLOCK, NODES, TARGET_SITES, TIME_PRICE, PRICES,
                  append_pair, best_stop, label_mass, legal_move, terminal_evidence)
from suffix import (draw, moves, price, propagate, sample_h, stop, tail_value,
                    _hypothesis_draws, _observation_draws)


@lru_cache(maxsize=1)
def catalog(cost_bytes, site_bytes):
    cost = np.frombuffer(cost_bytes,dtype=float).reshape(NODES,NODES,3)
    sites = np.frombuffer(site_bytes,dtype=np.int64)
    adjacency = [np.flatnonzero(np.isfinite(row[:,0])) for row in cost]
    entries = [[] for _ in sites]
    for start in range(NODES):
        for group,(near,far) in enumerate(TARGET_SITES):
            def visit(at,path,paid):
                if path and sites[at]==near:
                    for end in adjacency[at]:
                        if sites[end]==far:
                            entries[start].append((group,(*path,end),paid+cost[at,end]))
                    return
                if len(path)>=4:
                    return
                for end in adjacency[at]:
                    if abs(sites[end]-near)<abs(sites[at]-near) or (not path and sites[end]==sites[at]==near):
                        visit(end,(*path,end),paid+cost[at,end])
            visit(start,(),np.zeros(3))
    # Keep the cheapest mechanical realization for each distinct sequence
    # of acoustic sites and views, first node and end pose.
    unique = []
    for routes in entries:
        selected = {}
        for group,path,paid in routes:
            key = group,path[0],path[-1],tuple(node//2 for node in path)
            previous = selected.get(key)
            if previous is None or paid@PRICES < previous[2]@PRICES:
                selected[key]=(group,path,paid)
        unique.append(list(selected.values()))
    n = max(map(len,unique))
    paths = np.full((NODES,n,5),-1,np.int64)
    expenses = np.zeros((NODES,n,3))
    lengths = np.zeros((NODES,n),np.int64)
    groups = np.full((NODES,n),-1,np.int64)
    for start,routes in enumerate(unique):
        for i,(group,path,cost) in enumerate(routes):
            groups[start,i] = group
            lengths[start,i] = len(path)
            paths[start,i,:len(path)] = path
            expenses[start,i] = cost
    return paths,expenses,lengths,groups


@njit(cache=True)
def relaxed_pair_score(model,state,route,length,group):
    """Rank complete suffixes using revealed-return detection as a proxy."""
    b,_,_,_,clock,_ = state
    args = model[0]
    detect,addresses,labels = args[2],args[6],args[7]
    near,far = route[length-2],route[length-1]
    expected = 0.
    for h in range(len(b)):
        if labels[h]>0 and (labels[h]-1)//2==group:
            expected += b[h]*detect[clock+length-1,addresses[near],h,0]*detect[clock+length,addresses[far],h,0]
    return expected


@njit(cache=True)
def score_route(model,state,route,length,paid,samples):
    b,_,_,_,clock,_ = state
    total = 0.
    for j in range(samples):
        h = sample_h(b,_hypothesis_draws[clock+1,j])
        future = state
        for k in range(length):
            a = route[k]
            t = future[4]+1
            packet = draw(model,h,t,a,_observation_draws[t,j])
            future = propagate(model,future,a,packet)
        stop_utility,_ = stop(model,future)
        total += stop_utility/samples
    return total - paid


@njit(cache=True)
def select(model,state,bank):
    b,_,_,at,clock,reserve = state
    best, selected = stop(model,state)
    if clock>=FINAL_CLOCK:
        return selected
    # Offer all legal first moves, including a possible final capture of an
    # already acquired near plane. One-step scores use the common tail.
    mask = moves(model,state)
    cost = model[0][10]
    for first in np.flatnonzero(mask):
        acc = 0.
        for j in range(12):
            h = sample_h(b,_hypothesis_draws[clock+1,j])
            p = draw(model,h,clock+1,first,_observation_draws[clock+1,j])
            acc += tail_value(model,propagate(model,state,first,p))/12.
        score = acc-price(cost,at,first)
        if score>best+1e-12:
            best,selected=score,first

    paths, expenses, lengths, groups = bank
    scores = np.full((2,NODES,3),-np.inf)
    top = np.full((2,NODES,3),-1,np.int64)
    for i in range(len(lengths[at])):
        n, group = lengths[at,i],groups[at,i]
        if group<0 or clock+n>FINAL_CLOCK or np.any(expenses[at,i]>reserve+1e-8):
            continue
        route = paths[at,i]
        first = route[0]
        paid = expenses[at,i]@PRICES+n*TIME_PRICE
        proxy = relaxed_pair_score(model,state,route,n,group)-paid
        for k in range(3):
            if proxy>scores[group,first,k]:
                for q in range(2,k,-1):
                    top[group,first,q]=top[group,first,q-1]
                    scores[group,first,q]=scores[group,first,q-1]
                top[group,first,k]=i
                scores[group,first,k]=proxy
                break
    for group in range(2):
        for first in np.flatnonzero(mask):
            for i in top[group,first]:
                if i<0:
                    continue
                n = lengths[at,i]
                paid = expenses[at,i]@PRICES+n*TIME_PRICE
                value = score_route(model,state,paths[at,i],n,paid,16)
                if value>best+1e-12:
                    best,selected=value,first
    return selected


@njit(cache=True)
def _batch(model,b,pairs,last,at,clock,reserve,done,bank):
    actions=np.full(len(at),ABSTAIN,np.int64)
    for i in range(len(at)):
        if not done[i]:
            state = b[i],pairs[i],last[i],at[i],clock[i],reserve[i]
            actions[i] = select(model,state,bank)
    return actions


def commands(episode):
    m=episode.model
    model=(m.arguments(),m.lookup,m.packet_map,episode.env.nearest,m.fisher)
    bank=catalog(m.geometry.cost.tobytes(),m.geometry.sites.tobytes())
    return _batch(model,episode.b,episode.pairs,episode.last,episode.at,
                  episode.clock,episode.reserve,episode.done,bank)
