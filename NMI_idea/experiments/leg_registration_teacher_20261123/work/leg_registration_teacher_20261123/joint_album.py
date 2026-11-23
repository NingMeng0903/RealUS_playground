"""Select entrance and downstream witnesses using the same material-coordinate identity."""

from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np

from registration import HERE, chart, public, stock, require_anonymous

MAX_NOMINEES=18


def nominees(probability):
    return np.argsort(-probability[:,0],kind='stable')[:MAX_NOMINEES]


def pair_features(surface,frames,left,right):
    """Form geometry features for pairs from two disjoint sets of physical echoes."""
    groups=[list(left),list(right)]
    require_anonymous(frames)
    paired=stock(surface).paired_scores(frames,groups)
    if any(not len(ids) for ids,_ in paired):
        return None
    prep=chart(surface).prepare(frames)
    ids=[]
    scores=[]
    residual=[]
    for origins,q in paired:
        ii=nominees(q)
        ij=origins[ii]
        f,e=ij.T
        view=np.nan_to_num(prep.echoes[f,e,None,:]-prep.nominal[f],nan=45.)
        world=np.einsum('krc,kcd->krd',view,prep.transform[f])
        ids.append(ij)
        scores.append(q[ii])
        residual.append(world)
    m,n=np.meshgrid(np.arange(len(ids[0])),np.arange(len(ids[1])),indexing='ij')
    m,n=m.ravel(),n.ravel()
    a,b=ids[0][m],ids[1][n]
    pa,pb=scores[0][m],scores[1][n]
    ra,rb=residual[0][m],residual[1][n]
    delta=ra-rb
    ta,tb=prep.stations[a[:,0]],prep.stations[b[:,0]]
    xa,xb=prep.sectors[a[:,0]],prep.sectors[b[:,0]]
    same=np.linalg.norm(delta,axis=-1)
    quality=np.column_stack((prep.quality[a[:,0]],prep.quality[b[:,0]]))
    small=np.column_stack((np.minimum(pa[:,0],pb[:,0]),np.maximum(pa[:,0],pb[:,0]),
                           abs(ta-tb),abs((xa-xb+12.)%24.-12.), ta/160.,
                           np.sin(xa*np.pi/12.),np.cos(xa*np.pi/12.),
                           prep.counts[a[:,0]],prep.counts[b[:,0]],m,n))
    geometric=np.column_stack((same,np.linalg.norm(ra,axis=-1),np.linalg.norm(rb,axis=-1),
                                   delta.reshape(len(a),-1),
                                   ra[:,:2].reshape(len(a),-1),rb[:,:2].reshape(len(a),-1)))
    x=np.column_stack((pa,pb,small,quality,geometric)).astype('float32')
    return a,b,pa[:,0],pb[:,0],x


def select(data,head,commit):
    a,b,pa,pb,x=data
    eligible=(pa>=.45)&(pb>=.45)
    if not eligible.any():
        return None
    qa=head.predict_proba(x)[:,1]
    pair=np.log(np.maximum(pa,.001))+np.log(np.maximum(pb,.001))
    value=pair+commit*np.log(np.maximum(qa,.01))
    j=int(np.argmax(np.where(eligible,value,-1000.)))
    return (int(a[j,0]),int(a[j,1]),float(pa[j])), (int(b[j,0]),int(b[j,1]),float(pb[j]))


@lru_cache(maxsize=1)
def get_model(path):
    return joblib.load(path)


class JointAlbum:
    """Causal band decisions from the frozen tracker and a learned joint terminal decision."""

    threshold=.45
    sanitize=staticmethod(public)

    def __init__(self,surface,model=None):
        self.surface=surface
        self.base=stock(surface)
        self.record=get_model(Path(model or HERE/'album_head.joblib'))
        self.weight=json.loads((HERE/'album_config.json').read_text())['weight']

    def scores(self,frames,groups):
        return self.base.scores(frames,groups)

    def best(self,frames,groups):
        return self.base.best(frames,groups)

    def choose(self,frames,groups):
        data=pair_features(self.surface,frames,*groups)
        choice=None if data is None else select(data,self.record,self.weight)
        return self.base.best(frames,groups) if choice is None else choice
