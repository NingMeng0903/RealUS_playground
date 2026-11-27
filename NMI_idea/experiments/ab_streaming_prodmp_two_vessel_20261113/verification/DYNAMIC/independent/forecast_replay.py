"""Independently recompute common-history effective-offset and packet forecasts."""

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
common_values = {}

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

def likelihood(packet, means):
    geometry_ok = np.isfinite(means).all(axis=-1)
    means = np.where(geometry_ok[...,None],means,0.)
    in_fov = np.prod(ndtr((ceiling-means)/sig)-ndtr((floor-means)/sig),axis=-1)
    misses = 1.-in_fov[:,None,:]*visibility[None,:,:]*geometry_ok[:,None,:]
    if len(packet)==0:
        return np.prod(misses,axis=-1)
    density = np.prod(np.exp(-.5*((packet[None,:,None,:]-means[:,None,:,:])/sig)**2)/(np.sqrt(2*np.pi)*sig),axis=-1)
    detected = density[:,None,:,:]*visibility[None,:,None,:]*geometry_ok[:,None,None,:]
    a = detected[:,:,0,0]
    v = detected[:,:,0,1]
    if len(packet)==1:
        return a*misses[:,:,1]+v*misses[:,:,0]
    return a*detected[:,:,1,1]+v*detected[:,:,1,0]

def evaluate(cases, cohort, arm, model):
    support,mass,kernel = model
    all_offsets = offsets[:,None,:]+np.column_stack([support,np.zeros_like(support)])
    shifts, inds = np.unique(all_offsets.reshape(-1,2),axis=0,return_inverse=True)
    inds = inds.reshape(all_offsets.shape[:-1])
    kernel_h = {h: np.linalg.matrix_power(kernel,h) for h in horizons}
    clear_h = {h: np.linalg.matrix_power(P,h) for h in horizons}
    for i, (base, offset_axes, trajectory, packets, role_truth, truth) in enumerate(cases):
        emissions = []
        for packet, pose in zip(packets,trajectory,strict=True):
            means = means_from_geometry(base,offset_axes,pose,shifts)
            emissions.append(likelihood(packet,means)[inds])
        belief = static_prior[:,None,None]*mass[None,:,None]*clear_prior[None,None,:]
        count = np.zeros((2,len(horizons)))
        total = np.zeros((2,len(horizons),5))
        for k, l in enumerate(emissions):
            belief = np.einsum('iza,zy,ab->iyb',belief,kernel,P,optimize=True)*l
            belief /= belief.sum()
            for j,h in enumerate(horizons):
                if k+h >= len(truth):
                    continue
                pred = np.einsum('iza,zy,ab->iyb',belief,kernel_h[h],clear_h[h],optimize=True)
                distribution = np.bincount(inds.ravel(),weights=pred.sum(axis=-1).ravel(),minlength=len(shifts))
                distribution /= distribution.sum()
                lo,hi = shifts[np.searchsorted(np.cumsum(distribution),[.05,.95]),0]
                estimated = distribution @ shifts[:,0]
                has_future_packet = bool(h and k+h < len(packets))
                score = np.log(np.sum(pred*emissions[k+h])) if has_future_packet else 0.
                categories = [0] if 0 in role_truth[k] else [0,1]
                for subgroup in categories:
                    count[subgroup,j] += 1.
                    total[subgroup,j] += ((estimated-truth[k+h])**2,lo<=truth[k+h]<=hi,hi-lo,
                                          score,has_future_packet)
        yield count,total

def summarize(count, total):
    estimate = total[...,:3] / np.maximum(1,count[...,None])
    estimate[...,0] = np.sqrt(estimate[...,0])
    has_future_packet = total[...,4] > 0
    logp = np.divide(total[...,3],total[...,4],out=np.zeros_like(total[...,3]),where=has_future_packet)
    scores = np.asarray(logp, dtype=object)
    scores[~has_future_packet] = None
    return {
        'count': count.astype(int).tolist(),
        'rmse_mm': estimate[...,0].tolist(),
        'interval90_coverage': estimate[...,1].tolist(),
        'interval90_width_mm': estimate[...,2].tolist(),
        'future_packet_score_count': total[...,4].astype(int).tolist(),
        'mean_future_log_packet_density': scores.tolist(),
    }

for name in cohorts:
    folder = root / name
    if not (folder / 'SUMMARY.json').is_file() or not (folder / 'TRACES.npz').is_file():
        continue
    report = json.loads((folder / 'SUMMARY.json').read_text())
    cases = []
    with np.load(folder/'TRACES.npz',allow_pickle=False) as traces:
        v = traces['nominal_vessels_m']
        axes = traces['offset_axes']
        for case in report['cases']:
            i = case['case']
            frames = case['arms']['mpc_stream']['frames']
            path = traces[f'{i}_mpc_stream_fk_m'][::5][:len(frames)]
            truth = case['hidden_static_offset_mm'][0] + traces[f'{i}_z_mm']
            packets = [np.array(f['packet_mm']).reshape(-1,2) for f in frames]
            roles = [f['role_truth'] for f in frames]
            cases.append((v,axes,path,packets,roles,truth))
    cohort_output[name] = {}
    for arm, model in mode_state.items():
        values = list(evaluate(cases,name,arm,model))
        counts = np.sum([c for c,t in values],axis=0)
        total = np.sum([t for c,t in values],axis=0)
        cohort_output[name][arm] = summarize(counts, total)
        print(name,arm,'RMSE (all origins / missing artery at origin)',
              np.asarray(cohort_output[name][arm]['rmse_mm']).round(6).tolist(), flush=True)
        common_values[name,arm] = (counts,total)

primary = ['reference','minus-plus','plus-minus']
if all(name in cohort_output for name in primary):
    cohort_output['primary_combined'] = {}
    for arm in mode_state:
        count = sum(common_values[name,arm][0] for name in primary)
        total = sum(common_values[name,arm][1] for name in primary)
        cohort_output['primary_combined'][arm] = summarize(count, total)
if result_path := os.environ.get('ATLAS_SOL_FORECAST_JSON'):
    record = {'horizons_seconds': (horizons * .1).round(1).tolist(),
              'origin_groups': ['all_origins','artery_missing_at_origin'],
              'cohorts': cohort_output}
    Path(result_path).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
