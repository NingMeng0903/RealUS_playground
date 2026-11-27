"""Synthetic calibrated 2D centroids for the Fisher demonstration comparison."""
from __future__ import annotations

from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr

PRIOR = np.diag([4.**2,4.**2,3.**2,3.**2])
PRECISION = np.linalg.inv(PRIOR)
SD = np.array([.9,.9])
LOWER = np.array([-25.,0.])
UPPER = np.array([25.,80.])
_NODES, _WEIGHTS = hermgauss(5)
_GRID = np.array(list(product(range(len(_NODES)),repeat=4)))
NORMAL_NODES = np.sqrt(2.)*_NODES[_GRID]
NORMAL_WEIGHTS = (_WEIGHTS[_GRID]/np.sqrt(np.pi)).prod(axis=1)
PRIOR_SAMPLES = NORMAL_NODES @ np.linalg.cholesky(PRIOR).T


def projection(contacts, frames, vessel, reference):
    """Actual posed image plane projection, with unknown shared image biases."""
    image = frames[...,1:3]
    nominal = np.einsum("ki,kij->kj",1000.*(vessel-contacts),image)
    anatomical = np.einsum("kij,il->kjl",image,reference[:,1:3])
    H = np.concatenate([anatomical,np.broadcast_to(np.eye(2),anatomical.shape)],axis=2)
    inward = frames[:,:,2]
    tilt = np.rad2deg(np.arctan2(inward@reference[:,1],inward@reference[:,2]))
    return nominal,H,tilt


def detect_probability(tilt, preferred):
    """A declared synthetic directional echo window at the same 50 mm aperture."""
    return .25+.70*np.exp(-.5*((np.asarray(tilt)-np.asarray(preferred))/8.)**2)


def _pdf(x):
    return np.exp(-x*x/2.)/np.sqrt(2.*np.pi)


def gate_mass(mu):
    return (ndtr((UPPER-mu)/SD)-ndtr((LOWER-mu)/SD)).prod(axis=-1)


def capture_probabilities(nominal, H, p_detect):
    """Predict retained frames, including both directional echo and image bounds."""
    mu = nominal[:, None, :] + np.einsum("nj,kij->kni", PRIOR_SAMPLES, H)
    return p_detect * (gate_mass(mu) @ NORMAL_WEIGHTS)


def conditional_task_risks(H, p_detect, pairs):
    """Expected task covariance over four hit/miss patterns, assuming full FOV."""
    patterns = np.array([[False, False], [True, False], [False, True], [True, True]])
    information = ((H.swapaxes(-2, -1) / SD**2) @ H)[pairs]
    precisions = PRECISION + np.einsum("mi,pijk->pmjk", patterns, information)
    covariance = np.linalg.inv(precisions)[..., :2, :2]
    probability = p_detect[pairs][:, None, :]
    weights = np.where(patterns, probability, 1. - probability).prod(axis=2)
    return np.sum(weights * np.trace(covariance, axis1=-2, axis2=-1), axis=1)


def view_information(nominal, H, p_detect, flat=False):
    """Prior-averaged Fisher of a 2D observed centroid or a censored miss."""
    if flat:
        return p_detect*(H.T/(SD**2))@H
    mu = nominal + PRIOR_SAMPLES @ H.T
    a,b = (LOWER-mu)/SD,(UPPER-mu)/SD
    fa,fb = _pdf(a),_pdf(b)
    mass = ndtr(b)-ndtr(a)
    first = (fa-fb)/SD
    second = (mass+a*fa-b*fb)/(SD**2)
    hit = np.empty((len(mu),2,2))
    hit[:,0,0] = second[:,0]*mass[:,1]
    hit[:,1,1] = second[:,1]*mass[:,0]
    hit[:,0,1] = hit[:,1,0] = first.prod(axis=1)
    dmiss = -p_detect*first*mass[:,::-1]
    miss = 1.-p_detect*mass.prod(axis=1)
    total = p_detect*hit+np.einsum("ni,nj->nij",dmiss,dmiss)/miss[:,None,None]
    return H.T@np.einsum("n,nij->ij",NORMAL_WEIGHTS,total)@H


def posterior(nominal,H,p_detect,y,seen,flat=False):
    """Batch Bayesian update; integrate censored likelihood over a Gaussian base."""
    n = len(y)
    states = np.empty((n,4))
    task_cov = np.empty((n,2,2))
    pattern = seen@np.array([1,2])
    for pattern_id in np.unique(pattern):
        take = np.flatnonzero(pattern==pattern_id)
        active = np.array([(pattern_id&1)!=0,(pattern_id&2)!=0])
        rows = H[active].reshape(-1,4)
        inv_noise = np.tile(1./(SD**2),int(active.sum()))
        C = np.linalg.inv(PRECISION+(rows.T*inv_noise)@rows)
        residual = (y[take][:,active]-nominal[active]).reshape(len(take),-1)
        mean = (residual*inv_noise)@rows@C
        if flat or active.all():
            states[take] = mean
            task_cov[take] = C[:2,:2]
            continue
        samples = mean[:,None]+(NORMAL_NODES@np.linalg.cholesky(C).T)[None]
        weights = np.broadcast_to(NORMAL_WEIGHTS,(len(take),len(NORMAL_WEIGHTS))).copy()
        for k in np.flatnonzero(~active):
            weights *= 1.-p_detect[k]*gate_mass(nominal[k]+samples@H[k].T)
        weights /= weights.sum(axis=1,keepdims=True)
        mean = np.einsum("np,npi->ni",weights,samples)
        diff = samples[:,:,:2]-mean[:,None,:2]
        states[take] = mean
        task_cov[take] = np.einsum("np,npi,npj->nij",weights,diff,diff)
    return states,task_cov


def replay(nominal,H,p_model,p_actual,truth,noise,uniform,flat=False):
    """Generate observations and evaluate both anatomical axes, not the biases."""
    y = nominal[None]+np.einsum("kij,nj->nki",H,truth)+noise
    seen = uniform<p_actual
    if not flat:
        seen &= ((y>=LOWER)&(y<=UPPER)).all(axis=2)
    estimate,C = posterior(nominal,H,p_model,y,seen,flat)
    error = estimate[:,:2]-truth[:,:2]
    d2 = np.sum(error*error,axis=1)
    distance = np.einsum("ni,nij,nj->n",error,np.linalg.inv(C),error)
    return {"error_sq_mm2":d2,"qualified_3mm":d2<=9.,
            "ellipse_95_covered":distance<=5.9914645471,"captures":seen.sum(axis=1),
            "predicted_variance_mm2":np.trace(C,axis1=1,axis2=2)}
