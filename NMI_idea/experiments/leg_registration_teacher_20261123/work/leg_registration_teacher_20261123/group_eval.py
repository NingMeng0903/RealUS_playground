"""Read held-out echoes with the submission-aware model and a dev-selected frozen blend."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import sys,json
from pathlib import Path

import numpy as np
import torch
from registration import HERE,OLDER,public,stock
from fit_reader import COUNT,SEEDS,gather,scan,NAMES,counts,score,objective,N_METRICS
sys.path.insert(0,str(OLDER/'sequence'))
from data import feed_inputs
from model import IdentityModel,single_probs


@lru_cache(maxsize=1)
def model():
    torch.set_num_threads(1)
    m=IdentityModel(temporal=True)
    m.load_state_dict(torch.load(HERE/'group_reader.pt',weights_only=True,map_location='cpu')['weights'])
    m.eval()
    return m


def collect(data):
    from hashlib import sha256
    case,phase=data
    env,t,item,raw,cut,spans=gather(case,phase)
    safe=public(raw)
    y=feed_inputs(env.scanner.surface,safe,item[1],cut,spans)
    results={}
    pairs=stock(env.scanner.surface)
    for j,(lo,hi) in enumerate(spans):
        names=[NAMES[j]]
        group=[list(range(lo,hi))]
        if j==2:
            names+=['entry','fresh']
            group += [[i for i in range(cut) if item[1]-3.-1.e-8<=raw[i][0]<=item[1]+2.+1.e-8],list(range(cut,hi))]
        end=hi-y['offset']
        p=single_probs(model(),y['x'][:end].astype('float16'),y['meta'][:end].astype('float16'),y['valid'][:end])
        if np.any([len(g) and min(g)<y['offset'] for g in group]):
            raise ValueError('A visible frame is outside the training window')
        for n,(ix,old) in zip(names,pairs.scores(safe[:hi],group),strict=True):
            q=p[ix[:,0]-y['offset'],ix[:,1]]
            results[n]=(old,q,np.asarray([raw[i][3][k] for i,k in ix],dtype=int))
    return dict(world=int(case.world_seed),layout=sha256(np.nan_to_num(t.layout.centers).tobytes()+bytes([case.shape])).hexdigest(),
                profile=case.profile,answer=results)


def decide(r,alpha):
    return {n:score((1.-alpha)*old+alpha*q,y) for n,(old,q,y) in r['answer'].items()}


def stats(r,alpha):
    val=np.sum([counts(decide(row,alpha)) for row in r],axis=0).astype(int)
    return dict(zip(N_METRICS,val.tolist()))|{'n':len(r)}


def main():
    a=ArgumentParser();a.add_argument('--workers',type=int,default=8)
    a.add_argument('--cache',type=Path,required=True)
    opts=a.parse_args()
    training=opts.cache/'train'
    train=set()
    for p in training.glob('*.npz'):
        with np.load(p) as f:train.add(str(f['layout']))
    if not train:
        raise ValueError('Training layout inventory is required')
    with ProcessPoolExecutor(opts.workers) as pool:
        for phase in ['develop','archive','fresh','public']:
            r=list(pool.map(collect,((c,phase) for c in scan.cases('validate' if phase=='develop' else 'test',COUNT[phase],SEEDS[phase]))))
            if phase=='develop':
                r=[d for d in r if d['layout'] not in train]
                options=[(m,stats(r,m)) for m in (0.,.20,.4,.60,.8,1.)]
                mix,d=max(options,key=lambda a:objective(a[1]))
                (HERE/'group_config.json').write_text(json.dumps(dict(alpha=mix,develop=options),indent=2))
                print('development',mix,d,options[0],flush=True)
            else:
                if {z['layout'] for z in r}&train:raise ValueError('Overlap')
                cases=[{k:z[k] for k in ('world','layout','profile')}|
                          {'old':decide(z,0.),'group':decide(z,mix),'group_only':decide(z,1.)} for z in r]
                info={str(m):stats(r,m) for m in (0.,mix,1.)}
                (HERE/f'group_{phase}.json').write_text(json.dumps(dict(summary=info,rows=cases),separators=(',',':')))
                print(phase,info,flush=True)


if __name__=='__main__':
    main()
