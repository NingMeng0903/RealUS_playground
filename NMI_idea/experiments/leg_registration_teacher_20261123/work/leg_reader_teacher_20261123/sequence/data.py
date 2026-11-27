"""Common public feature extraction, independent simulator splits and scoring windows."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
FOLLOW = HERE.parents[1] / 'leg_follow_condition'
sys.path[:0] = [str(FOLLOW / 'learned_reader'), str(FOLLOW)]
from capture_teacher import scan_request
from core import get_world, req, scan
from study import exercise
from identify import FeatureReader
from tracker import Reader

SEEDS = dict(train=690890647, develop=914865029, archive=815601941, fresh=532807421, public=215198977)
COUNT = dict(train=1280, develop=160, archive=128, fresh=96, public=96)
PT = 'Posterior_tibial_vein.r'
CH = 7


@lru_cache(maxsize=12)
def chart(surface):
    return Reader(SimpleNamespace(scanner=SimpleNamespace(surface=surface)))


def inputs(surface, frames):
    if any(f[i] is not None for f in frames for i in (3,6,7)):
        raise ValueError('Expected anonymous frames')
    c = chart(surface)
    d = c.prepare(frames)
    n = len(frames)
    valid = np.arange(CH)[None,:] < d.counts[:,None]
    nominal = d.nominal
    delta = d.echoes[:,:,None,:] - nominal[:,None,:,:]
    world = np.einsum('tsrc,tcd->tsrd', delta, d.transform)
    gate = c.gates(d, np.zeros_like(d.nominal))
    geom = nominal-nominal[:,0:1]
    geom_world = np.einsum('trc,tcd->trd',geom,d.transform)
    neighbours = d.echoes[:,:,None,:]-d.echoes[:,None,:,:]
    far = ~valid[:,None,:] | np.eye(CH,dtype=bool)[None]
    dist = np.where(far,1.e5,np.linalg.norm(neighbours,axis=3))
    j = dist.argmin(axis=2)
    neighbour = neighbours[np.arange(n)[:,None],np.arange(CH)[None,:],j]
    neighbour[dist.min(axis=2)>1.e4]=0.
    def cap(v,s):
        return np.nan_to_num(np.clip(v/s,-4.,4.),nan=4.,posinf=4.,neginf=-4.)
    angle = d.sectors*np.pi/12.
    origin = np.nan_to_num(nominal,nan=140.)
    visible = np.isfinite(nominal).all(axis=-1)
    metadata = np.column_stack((d.quality, d.counts/5., d.stations/160.,np.sin(angle),np.cos(angle),
        np.clip(np.diff(d.ticks,prepend=d.ticks[0])/8.,0.,2.),cap(origin,[25.,60.]).reshape(n,-1),
        cap(geom_world,7.).reshape(n,-1),gate,visible.astype(float)))
    local = np.concatenate((cap(d.echoes,[25.,60.]),cap(delta,[3.,2.5]).reshape(n,CH,-1),
        cap(world,3.).reshape(n,CH,-1),cap(neighbour,[3.,2.5]),cap(np.linalg.norm(delta,axis=-1),4.)),axis=2)
    local[~valid]=0.
    return local.astype('float32'),metadata.astype('float32'),valid


def slice_for_entry(frames, candidates, station):
    # Include the last material interval of history before the earliest eligible old image.
    start=min(candidates,default=len(frames)-1)
    return max(0, min(start-48, next((i for i,f in enumerate(frames) if f[0]>=station-27.),start)))


def gather(case,split):
    env,spec,t,lim,_,items=get_world(case)
    item=next(i for i in items if i[0]==PT)
    rng=np.random.default_rng(scan.rng_seed(case.action_seed,64121))
    use_teacher=split in ('archive','fresh') or (split in ('train','develop') and rng.random()<.6)
    if use_teacher:
        trace,frames,cut,spans=scan_request(env,spec,t,lim,item,True,return_frames=True)
        if split=='archive':
            old=json.loads((FOLLOW/'capture_validation_balanced'/f'{case.world_seed}.json').read_text())['requests'][PT]
            if json.loads(json.dumps(trace)) != old:
                raise RuntimeError('Archived acquisition failed reproduction')
    else:
        mode='equal' if split=='public' or rng.random()<.3 else 'random'
        frames,cut,spans,_=exercise(env,spec,t,lim,item,mode,rng,None)
    return env,t,item,frames,cut,spans


def feed_inputs(surface, public, entry, cut, spans):
    old=[i for i in range(cut) if entry-3.-1.e-8<=public[i][0]<=entry+2.+1.e-8]
    offset=slice_for_entry(public,old,entry)
    x,meta,valid=inputs(surface,public[offset:])
    group=np.zeros((len(x),5),bool)
    for i in old:
        group[i-offset,0]=True
    group[cut-offset:,1]=True
    for j,(lo,hi) in enumerate(spans):
        group[lo-offset:hi-offset,2+j]=True
    return dict(x=x,meta=meta,valid=valid,group=group,ends=np.asarray([s[1]-offset for s in spans]),
                tick=np.array([f[8] for f in public[offset:]]),offset=offset)


def generate(job):
    case,split,folder=job
    path=folder/f'{case.world_seed}.npz'
    if path.exists():
        return
    env,t,item,frames,cut,spans=gather(case,split)
    data=feed_inputs(env.scanner.surface,FeatureReader.public(frames),item[1],cut,spans)
    visible=frames[data['offset']:]
    y=np.full(data['valid'].shape,-100,np.int8)
    for i,f in enumerate(visible):
        y[i,:len(f[3])]=np.where(f[3]==-1,7,np.where(f[3]==-2,6,f[3]))
    # Auxiliary targets are accessible only in the training loss (per-individual anatomical offsets).
    station=np.asarray([f[0] for f in visible])
    mi=np.floor(station).astype(int)
    ma=np.minimum(mi+1,len(t.layout.centers)-1)
    frac=(station-mi)[:,None,None]
    diff=t.layout.centers[:,:2]-chart(env.scanner.surface).atlas[:,:2]
    offsets=diff[mi]*(1.-frac)+diff[ma]*frac
    layout=sha256(np.nan_to_num(t.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
    np.savez_compressed(path,**{**data,'x':data['x'].astype('float16'),'meta':data['meta'].astype('float16')},
                        labels=y,anatomical_offset=offsets.astype('float16'),layout=layout,world=case.world_seed,shape=case.shape)


def main():
    p=ArgumentParser()
    p.add_argument('split',choices=SEEDS);p.add_argument('folder',type=Path)
    p.add_argument('--size',type=int);p.add_argument('--workers',type=int,default=7)
    a=p.parse_args();folder=a.folder/a.split;folder.mkdir(parents=True,exist_ok=True)
    size=a.size or COUNT[a.split]
    partition='train' if a.split=='train' else 'validate' if a.split=='develop' else 'test'
    cases=scan.cases(partition,size,SEEDS[a.split])
    with ProcessPoolExecutor(a.workers) as pool:
        for i,_ in enumerate(pool.map(generate,[(c,a.split,folder) for c in cases])):
            if (i+1)%80==0:print(a.split,i+1,flush=True)


if __name__=='__main__':main()
