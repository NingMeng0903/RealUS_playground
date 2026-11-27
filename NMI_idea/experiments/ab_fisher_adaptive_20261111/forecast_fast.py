"""Vectorized terminal/retry forecast for the corrected public association model."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from acoustics import (
    P_CLUTTER, QUALITY_THRESHOLD, detection_probability, geometry_value,
    response_design, sensor_variance, terminal_geometry_probability,
)


@lru_cache(maxsize=4096)
def params(action):
    return (
        response_design(action), geometry_value(action), sensor_variance(action),
        detection_probability(action), 0.80 * terminal_geometry_probability(action),
    )


@lru_cache(maxsize=10)
def probability_nodes(order):
    x, w = np.polynomial.legendre.leggauss(order)
    return (x+1.)/2., w/2.


def components(memory):
    posterior = [component.posterior() for component in memory.components]
    return np.asarray(memory.weights), np.array([v[0] for v in posterior]), np.array([v[1] for v in posterior])


def normal_logpdf(y, mu, var):
    return -0.5*(np.log(2*np.pi*var)+(y-mu)**2/var)


def action_value(action, duration_s, retry_options, memory, time_cost, *, order=16):
    """Same response and terminal acceptance forecasts, integrating below the public stop gate.

    Each retry option is (public ProbeAction, checked native duration_s).
    Separate an absent packet, and integrate the candidate branch y<quality_gate
    in component-CDF space. No private labels enter; target/clutter are
    posterior responsibilities conditional on the candidate's score.
    """
    h, g, r, p_t, ok = params(action)
    weight, mean, cov = components(memory)
    hc = cov @ h
    mu = g + mean @ h
    var = np.maximum(r + hc @ h, 1e-12)
    p_accept = float(ok * (weight @ ndtr((mu-QUALITY_THRESHOLD)/np.sqrt(var))))
    base = p_accept - time_cost * duration_s
    if not retry_options:
        return base
    c_f = (1-p_t)*P_CLUTTER
    p_miss = (1-p_t)*(1-P_CLUTTER)
    x, w = probability_nodes(order)
    lo_target = ndtr((QUALITY_THRESHOLD-mu)/np.sqrt(var))
    lo_clutter = float(ndtr((QUALITY_THRESHOLD-.12)/.08))
    y = np.r_[(
        mu[:, None]+np.sqrt(var)[:, None]*
        ndtri(np.clip(lo_target[:, None]*x[None,:], 1e-300, 1-1e-15))
    ).ravel(), .12+.08*ndtri(lo_clutter*x)]
    weights_y = np.r_[((p_t*weight*lo_target)[:,None]*w[None,:]).ravel(),
                      c_f*lo_clutter*w]
    active = weights_y > 1e-16
    y, weights_y = y[active], weights_y[active]
    a = len(weight)
    if len(y):
        log_weights = np.log(np.maximum(weight, 1e-300))
        log_t = log_weights[None,:]+math.log(max(p_t, 1e-300))+normal_logpdf(y[:,None], mu, var)
        log_f = log_weights[None,:]+math.log(max(c_f, 1e-300))+normal_logpdf(y[:,None], .12, .08**2)
        log_den = logsumexp(np.c_[log_t, log_f], axis=1)[:,None]
        wt, wf = np.exp(log_t-log_den), np.exp(log_f-log_den)
    else:
        wt, wf = np.empty((0,a)), np.empty((0,a))
    candidate_values = []
    missing_values = []
    for action2, seconds2 in retry_options:
        h2, g2, r2, _p_t2, ok2 = params(action2)
        mu2 = g2+mean@h2
        var2 = np.maximum(r2+(cov@h2)@h2, 1e-12)
        corr = hc@h2
        pred_clutter = ndtr((mu2-QUALITY_THRESHOLD)/np.sqrt(var2))
        pred_real = ndtr((
            mu2+(y[:,None]-mu[None,:])*(corr/var)-QUALITY_THRESHOLD
        )/np.sqrt(np.maximum(var2-corr**2/var, 1e-12)))
        u2 = ok2*(np.sum(wt*pred_real, axis=1)+wf@pred_clutter) - time_cost*seconds2
        candidate_values.append(u2)
        missing_values.append(float(ok2*(weight@pred_clutter) - time_cost*seconds2))
    continuation = (float(weights_y @ np.max(candidate_values, axis=0)) if len(y) else 0.) + p_miss*max(missing_values)
    return float(base+continuation)
