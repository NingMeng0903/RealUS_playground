"""Marginalize persistent material offsets across jointly assigned vein/artery observations."""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np
from numba import njit
from scipy.special import ndtri
from scipy.stats import qmc

from registration import HERE, RegisteredReader, chart, key_groups, phase_stations, require_anonymous, stock, table_from_file
from tracker.tracker import noise_fraction


@lru_cache(maxsize=1)
def shift_samples():
    cube=qmc.Sobol(d=4,scramble=True,seed=137462).random_base2(8)
    return ndtri(np.clip(cube,.009,.991)).reshape(-1,2,2)


@njit(cache=True,nogil=True)
def gaussian(x,z,mx,mz,sx,sz):
    return np.exp(-.5*(((x-mx)/sx)**2+((z-mz)/sz)**2))/(2.*np.pi*sx*sz)


@njit(cache=True,nogil=True)
def sample_views(poses, prior, gates, rotations, counts, echoes, fraction, shifts, fade):
    """Per-layout likelihood and mutually exclusive candidate assignments."""
    T=len(poses)
    K=len(shifts)
    posterior=np.zeros((K,T,7,2),dtype=np.float32)
    ll=np.zeros((K,T),dtype=np.float32)
    sx=1.4+.9*fraction
    sz=1.1+.7*fraction
    hit=.91-.22*fraction
    fusion_rate=.44+.35*fraction
    ghost_rate=.31+.35*fraction
    background=np.zeros((T,7))
    for t in range(T):
        for k in range(counts[t]):
            x,z=echoes[t,k]
            # Unassigned detections include other chart tracks and coherent replicas.
            rate=.30/3850.
            for r in range(6):
                cx,cz=prior[t,r]
                if not np.isfinite(cx) or abs(cx)>28 or cz<0 or cz>84:
                    continue
                visible=hit*(1.-.96*gates[t,r])
                if r>1:
                    rate += visible*gaussian(x,z,cx,cz,sx+.7,sz+.65)
                rate += ghost_rate*.12*gaussian(x,z,cx+2.7,cz+2.,1.25,1.25)
            background[t,k]=max(.00022,rate)
    for h in range(K):
        for t in range(T):
            n=counts[t]
            if n==0 or fade[t]==0.:
                continue
            v=np.zeros(7);a=np.zeros(7);f=np.zeros(7)
            cx=prior[t,:2,0].copy()
            cz=prior[t,:2,1].copy()
            detect=np.zeros(2)
            for r in range(2):
                v1,v2=shifts[h,r]
                cx[r]+=v1*rotations[t,0,0]+v2*rotations[t,0,1]
                cz[r]+=v1*rotations[t,1,0]+v2*rotations[t,1,1]
                if np.isfinite(cx[r]):
                    visible=min(25.0-abs(cx[r]),cz[r]-3.,80.-cz[r])
                    detect[r]=hit*(1.-.96*gates[t,r])*(.5+.5*np.tanh(visible/1.6))
            gap=(cx[0]-cx[1])**2+(cz[0]-cz[1])**2
            overlap=fusion_rate*np.exp(-.5*gap/4.7**2) if detect[0]*detect[1]>1.e-8 else 0.
            for k in range(n):
                x,z=echoes[t,k]
                if detect[0]>1.e-8:
                    v[k]=detect[0]*gaussian(x,z,cx[0],cz[0],sx,sz)/background[t,k]
                if detect[1]>1.e-8:
                    a[k]=detect[1]*gaussian(x,z,cx[1],cz[1],sx,sz)/background[t,k]
                if overlap>1.e-6:
                    f[k]=overlap*detect[0]*detect[1]*gaussian(x,z,.5*(cx[0]+cx[1]),.5*(cz[0]+cz[1]),.80*sx,.80*sz)/background[t,k]
            miss_v=1.-detect[0]
            miss_a=1.-detect[1]
            vol=(v[:n].sum()*a[:n].sum()-(v[:n]*a[:n]).sum())*(1.-overlap)
            total=max(1.e-9,miss_v*miss_a+miss_v*a[:n].sum()+miss_a*v[:n].sum()+vol+f[:n].sum())
            ll[h,t]=np.log(total)
            for k in range(n):
                posterior[h,t,k,0]=v[k]*(miss_a+(a[:n].sum()-a[k])*(1.-overlap))/total
                posterior[h,t,k,1]=a[k]*(miss_v+(v[:n].sum()-v[k])*(1.-overlap))/total
    return ll,posterior


def public_hypotheses(surface,prepared,table,focus):
    fraction=noise_fraction(prepared.quality,prepared.sectors,prepared.counts)
    k=phase_stations([focus])[0]
    sd=(1.-fraction)*table[0,k,:2]+fraction*table[1,k,:2]
    shifts=shift_samples()*(sd+.12)[None]
    fade=np.exp(-.5*((prepared.stations-focus)/7.)**2)
    fade[(abs(prepared.stations-focus)>16.)]=0.
    gates=chart(surface).gates(prepared,np.zeros_like(prepared.nominal))
    ll,p=sample_views(prepared.stations,prepared.nominal,gates,prepared.transform,prepared.counts,
                      prepared.echoes,fraction,shifts,fade)
    return fade,ll,p,fraction,shifts


REGIMES = {'low':.27, 'medium':.60, 'high':1.05}


def combine(ids, ingredients, temperature):
    if not len(ids):
        return np.empty(0)
    focus,ll,p,fraction,_=ingredients
    strength=temperature*(1.-.56*fraction)
    # The realized layout posterior is tempered for repeated correlated pulses.
    logw=ll @ (strength*focus)
    w=np.exp(logw-logw.max())
    w/=w.sum()
    return np.einsum('h,hi->i',w,p[:,ids[:,0],ids[:,1],0])


class HypothesisReader(RegisteredReader):
    """Deploy the calibrated layout mixture on the acquired public prefix."""

    def __init__(self,surface,config=None):
        super().__init__(surface,config)
        self.selected=json.loads((HERE/'hypothesis_config.json').read_text())['selected']
        self.temperature=REGIMES[self.selected['regime']]

    def scores(self,frames,groups):
        require_anonymous(frames)
        groups=[list(g) for g in groups]
        if not any(groups):
            return [(np.empty((0,2),dtype=int),np.empty(0)) for _ in groups]
        begin=max(0,min(i for g in groups for i in g)-95)
        p=chart(self.surface).prepare(frames[begin:])
        candidates=key_groups(p,[[i-begin for i in g] for g in groups])
        focus=float(np.median([frames[i][0] for g in groups for i in g]))
        inputs=public_hypotheses(self.surface,p,self.prior,focus)
        fraction=inputs[3]
        mix=self.selected['ordinary']*(1.-fraction)+self.selected['stress']*fraction
        if mix<=1.e-12:
            return stock(self.surface).scores(frames,groups)
        legacy=stock(self.surface).scores(frames,groups) if mix<1.-1.e-12 else [(ix+np.array([begin,0]),None) for ix in candidates]
        out=[]
        for ix,(keys,base) in zip(candidates,legacy,strict=True):
            np.testing.assert_array_equal(ix+np.array([begin,0]),keys)
            model=combine(ix,inputs,self.temperature)
            out.append((keys,model if base is None else mix*model+(1.-mix)*base))
        return out
