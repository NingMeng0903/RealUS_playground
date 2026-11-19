"""Independently recompute saved association and residual posteriors."""

import sys
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

root = Path('..') / sys.argv[1]
cohorts = sys.argv[2:] or ['reference', 'minus-plus', 'plus-minus', 'beta2', 'pose25', 'transient']
if (root / 'SUMMARY.json').is_file():
    cohorts = ['.']

offsets = np.stack(np.meshgrid(np.arange(-6.,7.,2.), np.arange(-2.,3.,2.), indexing='ij'),axis=-1).reshape(-1,2)
z = np.arange(-6.,6.01,.5)
visibility = np.array([[False,False],[False,True],[True,False],[True,True]])
P = np.kron(np.array([[.72,.28],[.06,.94]]),np.array([[.72,.28],[.06,.94]]))
clear_prior = np.kron(np.array([.06,.28])/.34,np.array([.06,.28])/.34)
cdf_prior = lambda axis,sd: np.diff(ndtr(np.r_[-np.inf, .5*(axis[:-1]+axis[1:]), np.inf]/sd))
static_prior = np.outer(cdf_prior(np.arange(-6.,7.,2.),3.5),cdf_prior(np.arange(-2.,3.,2.),1.5)).ravel()
edges = np.r_[-np.inf,.5*(z[1:]+z[:-1]),np.inf]
rho = np.exp(-.1/.8)
innov = 1.5*np.sqrt(1.-rho*rho)
def integrate_joint(val):
    return np.exp(-.5*(val/1.5)**2)/(np.sqrt(2*np.pi)*1.5)*np.diff(ndtr((edges-rho*val)/innov))
joint = np.stack([quad_vec(integrate_joint,lo,hi,epsabs=1e-13)[0] for lo,hi in zip(edges[:-1],edges[1:],strict=True)])
z_mass = joint.sum(axis=-1)
z_mass /= z_mass.sum()
T = joint / joint.sum(axis=-1,keepdims=True)
horizons = np.array([0,1,3,5,10], dtype=int)
mode_state = {'mpc_stream': (np.zeros(1), np.ones(1), np.ones((1,1))),
              'mpc_iid': (z, z_mass, np.tile(z_mass,(len(z),1))),
              'mpc_ou': (z, z_mass, T)}
floor, ceiling, sig = np.array([-25.,3.]), np.array([25.,52.]), np.array([1.,.68])
cohort_output = {}

def means_from_geometry(vessels, axes, pose, shifts):
    shift = shifts @ axes.T * .001
    coordinates = ((vessels[None] + shift[:,None,None] - pose[:3,3]) @ pose[:3,:3])*1000.
    first, second = coordinates[:,:-1], coordinates[:,1:]
    a,b = first[...,0], second[...,0]
    cross = ((a<=0.)&(b>0.))|((a>=0.)&(b<0.))
    t = np.divide(-a,b-a,out=np.zeros_like(a),where=cross)[...,None]
    yz = first[...,1:]*(1.-t)+second[...,1:]*t
    depth = np.where(cross&(yz[...,1]>0),yz[...,1],np.inf)
    best = depth.argmin(axis=1)
    o = np.arange(len(shifts))[:,None]
    v = np.arange(2)[None,:]
    return np.where(np.isfinite(depth[o,best,v])[...,None],yz[o,best,v],np.nan)


def likelihood_and_roles(packet, means):
    geometry_ok = np.isfinite(means).all(axis=-1)
    means = np.where(geometry_ok[...,None],means,0.)
    in_fov = np.prod(ndtr((ceiling-means)/sig)-ndtr((floor-means)/sig),axis=-1)
    misses = 1.-in_fov[:,None,:]*visibility[None,:,:]*geometry_ok[:,None,:]
    if len(packet)==0:
        return np.prod(misses,axis=-1), np.empty((len(means),4,0))
    density = np.prod(np.exp(-.5*((packet[None,:,None,:]-means[:,None,:,:])/sig)**2)/(np.sqrt(2*np.pi)*sig),axis=-1)
    detected = density[:,None,:,:]*visibility[None,:,None,:]*geometry_ok[:,None,None,:]
    a = detected[:,:,0,0]
    v = detected[:,:,0,1]
    if len(packet)==1:
        artery_slot_0 = a*misses[:,:,1]
        return artery_slot_0+v*misses[:,:,0], artery_slot_0[...,None]
    artery_slot_0 = a*detected[:,:,1,1]
    artery_slot_1 = v*detected[:,:,1,0]
    return artery_slot_0+artery_slot_1, np.stack([artery_slot_0,artery_slot_1],axis=-1)

def verify(cases, traces, nominal, axes, arm, model):
    support,mass,kernel = model
    all_offsets = offsets[:,None,:]+np.column_stack([support,np.zeros_like(support)])
    shifts, inds = np.unique(all_offsets.reshape(-1,2),axis=0,return_inverse=True)
    inds = inds.reshape(all_offsets.shape[:-1])
    max_delta=np.zeros(3)
    frames_checked=0
    for case in cases:
        i = case['case']
        frames = case['arms'][arm]['frames']
        path = traces[f'{i}_{arm}_fk_m'][::5][:len(frames)]
        belief = static_prior[:,None,None]*mass[None,:,None]*clear_prior[None,None,:]
        for row, pose in zip(frames,path,strict=True):
            packet = np.array(row['packet_mm']).reshape(-1,2)
            means = means_from_geometry(nominal,axes,pose,shifts)
            l,a = likelihood_and_roles(packet,means)
            predictive = np.einsum('iza,zy,ab->iyb',belief,kernel,P,optimize=True)
            total = np.sum(predictive*l[inds])
            prob = np.sum(predictive[...,None]*a[inds],axis=(0,1,2))/total
            belief = predictive*l[inds]/total
            distribution = np.bincount(inds.ravel(),weights=belief.sum(axis=-1).ravel(),minlength=len(shifts))
            lo,hi = shifts[np.searchsorted(np.cumsum(distribution),[.05,.95]),0]
            estimated = distribution @ shifts
            z_mean = belief.sum(axis=(0,2)) @ support
            if len(prob):
                max_delta[0] = max(max_delta[0],np.max(np.abs(prob-row['artery_probability'])))
            max_delta[1] = max(max_delta[1],np.max(np.abs(estimated-row['offset_mean_mm'])))
            max_delta[2] = max(max_delta[2],abs(z_mean-row['dynamic_lateral_mean_mm']))
            np.testing.assert_array_equal(row['offset_interval90_mm'],[lo,hi])
            frames_checked += 1
    assert np.max(max_delta)<1e-7, (arm,max_delta)
    return frames_checked,max_delta

for name in cohorts:
    folder = root / name
    if not (folder / 'SUMMARY.json').is_file() or not (folder / 'TRACES.npz').is_file():
        continue
    report = json.loads((folder / 'SUMMARY.json').read_text())
    cohort_output[name] = {}
    with np.load(folder/'TRACES.npz',allow_pickle=False) as traces:
        for arm,model in mode_state.items():
            count, delta = verify(report['cases'],traces,traces['nominal_vessels_m'],traces['offset_axes'],arm,model)
            cohort_output[name][arm] = {'exposures': count,
                                        'max_association_difference': float(delta[0]),
                                        'max_effective_offset_difference_mm': float(delta[1]),
                                        'max_residual_mean_difference_mm': float(delta[2])}
            print(name,arm,count,'max independent posterior differences (association/offset/z)',delta,flush=True)
if result_path := os.environ.get('ATLAS_SOL_POSTERIOR_JSON'):
    Path(result_path).write_text(json.dumps(cohort_output, indent=2, allow_nan=False) + '\n')
