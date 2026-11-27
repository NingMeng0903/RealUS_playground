"""Candidate identity ranking with registered material coordinates and cross-window evidence."""

from pathlib import Path
import json

import joblib
import numpy as np
from numba import njit

from registration import HERE, PROGRAMS, RegisteredReader, chart, key_groups, posterior, require_anonymous, stock, table_from_file
from tracker.tracker import noise_fraction


@njit(cache=True,nogil=True)
def independent_support(ids, ticks, stations, sectors, counts, relative, geom):
    """Accumulate identity-specific neighbors outside the correlated acquisition burst."""
    out = np.zeros((len(ids),16))
    for j in range(len(ids)):
        i,k = ids[j]
        for other in range(max(0,i-190),min(len(ticks),i+191)):
            dt = abs(ticks[i]-ticks[other])
            ds = abs(stations[i]-stations[other])
            if dt<6 or dt>100 or ds>15.:
                continue
            angle = abs((sectors[i]-sectors[other]+12.)%24.-12.)
            group = int(angle>=.85)
            local_sum = np.zeros(4)
            for role in range(2):
                dx,dz = relative[i,k,role]
                for mark in range(counts[other]):
                    x,z = relative[other,mark,role]
                    weight = np.exp(-.5*((x-dx)**2+(z-dz)**2)/(2.2+.10*ds)**2)
                    local_sum[2*role] += weight*geom[other,mark,role]
                    local_sum[2*role+1] += weight*geom[other,mark,1-role]
            for t in range(4):
                out[j,8*group+t] += local_sum[t]/(1.+.17*ds)
                out[j,8*group+4+t] = max(out[j,8*group+4+t],local_sum[t])
    return np.log1p(out)


class RankFeatures:
    def __init__(self, surface, config=None):
        self.surface = surface
        self.registration = chart(surface)
        self.config, self.prior = table_from_file(config)
        self.pair = stock(surface)
        choice = self.config['selected']
        self.program = PROGRAMS[choice['program']]

    def features(self, anonymous, groups):
        require_anonymous(anonymous)
        groups = [list(g) for g in groups]
        if not anonymous or not any(groups):
            empty = (np.empty((0,2),dtype=int),np.empty((0,0)),np.empty(0),np.empty(0))
            return [empty for _ in groups],0.
        if any(i < 0 or i >= len(anonymous) for g in groups for i in g):
            raise ValueError('The observed prefix does not contain the requested image group')
        p = self.registration.prepare(anonymous)
        fraction = noise_fraction(p.quality,p.sectors,p.counts)
        q, offsets, spread = posterior(self.surface,p,self.prior,self.program,anatomy=True)
        fixed = posterior(self.surface,p,self.prior,PROGRAMS['atlas'])
        material = np.einsum('tsrc,tcq->tsrq',
                np.nan_to_num(p.echoes[:,:,None,:]-p.nominal[:,None,:,:],nan=50.),p.transform)
        relative = material[:,:,:2] - offsets[:,None,:2]
        old_iter = self.pair.paired_scores(anonymous,groups)
        ids = key_groups(p,groups)
        choice = self.config['selected']
        rw = (1.-fraction)*choice['geometry_weight']+fraction*choice.get('stress_weight',choice['geometry_weight'])
        out = []
        for indices, (then,old) in zip(ids,old_iter,strict=True):
            np.testing.assert_array_equal(indices,then)
            i,k = indices.T
            rg,ap = q[i,k],fixed[i,k]
            if not len(i):
                out.append((indices,np.empty((0,0)),np.empty(0),np.empty(0)))
                continue
            support = independent_support(indices,p.ticks,p.stations,p.sectors,p.counts,relative,q)
            sep = np.linalg.norm(p.nominal[i,0]-p.nominal[i,1],axis=1)
            gap = np.linalg.norm(p.nominal[i,0,None]-p.nominal[i,2:],axis=-1)
            prev = np.maximum(0,i-1)
            misc = np.column_stack([p.stations[i]/160.,np.sin(p.sectors[i]*np.pi/12.),
                        np.cos(p.sectors[i]*np.pi/12.),p.quality[i],p.counts[i],
                        np.full(len(i),fraction),p.ticks[i]-p.ticks[prev],sep,gap])
            mat = np.column_stack((old,rg,ap, material[i,k].reshape(len(i),-1),
                relative[i,k].reshape(len(i),-1), np.linalg.norm(relative[i,k],axis=-1),
                spread[i,:2].reshape(len(i),-1), support, misc))
            combined = rw*rg[:,0]+(1.-rw)*old[:,0]
            out.append((indices,np.nan_to_num(mat,nan=50.,posinf=50.,neginf=-50.).astype('float32'),old[:,0],combined))
        return out,fraction


class EvidenceReader(RegisteredReader):
    """Apply a fixed supervised identity head to anonymous real-history candidates."""

    def __init__(self, surface, config=None, model_path=None):
        super().__init__(surface,config)
        self.encoder = RankFeatures(surface,config)
        self.model_path = Path(model_path or HERE/'rank_head.joblib')
        self.weights = json.loads(self.model_path.with_name('rank_config.json').read_text())['selected']
        self.models = joblib.load(self.model_path)

    def scores(self, frames, groups):
        blocks, fraction = self.encoder.features(frames,groups)
        weight=self.weights['ordinary']*(1.-fraction)+self.weights['stress']*fraction
        results=[]
        for ids,matrix,legacy,reg in blocks:
            if not len(ids):
                results.append((ids,np.empty(0)))
                continue
            calibrated = self.models[self.weights['head']].predict_proba(matrix)[:,0] if weight else reg
            base = self.weights['base']*reg+(1.-self.weights['base'])*legacy
            results.append((ids,weight*calibrated+(1.-weight)*base))
        return results
