"""Freeze the influence of the material offset posterior before reading held-out tracks."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
import json

import numpy as np
from registration import HERE, chart, key_groups, public, stock, table_from_file
from hypotheses import REGIMES, combine, public_hypotheses
from fit_reader import COUNT,NAMES,SEEDS,counts,gather,objective,score,scan,N_METRICS

DEVELOP=291942307
DEVELOP_N=112
NEW_TEST=855192389
WEIGHTS=(0.,.35,.65,1.)


@lru_cache(maxsize=1)
def table():
    return table_from_file()[1]


def collect(job):
    case,split=job
    env,trial,item,raw,cut,spans=gather(case,split)
    surface=env.scanner.surface
    frames=public(raw)
    album=[i for i in range(cut) if item[1]-3.-1.e-8<=raw[i][0]<=item[1]+2.+1.e-8]
    ret={}
    for j,(lo,hi) in enumerate(spans):
        groups=[list(range(lo,hi))]
        names=[NAMES[j]]
        if j==2:
            names+=['entry','fresh']
            groups += [album, list(range(cut,hi))]
        begin=max(0,min(i for g in groups for i in g)-95)
        p=chart(surface).prepare(frames[begin:hi])
        candidates=key_groups(p,[[i-begin for i in g] for g in groups])
        focus=float(np.median([frames[i][0] for g in groups for i in g]))
        inputs=public_hypotheses(surface,p,table(),focus)
        legacy=stock(surface).scores(frames[:hi],groups)
        for ix,name,(keys,base) in zip(candidates,names,legacy,strict=True):
            np.testing.assert_array_equal(ix+np.array([begin,0]),keys)
            target=np.array([raw[i][3][k] for i,k in keys],dtype=int)
            ret[name]={'legacy':base,'fraction':inputs[3],'y':target,
                **{mode:combine(ix,inputs,value) for mode,value in REGIMES.items()}}
    layout=sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
    return dict(layout=layout,world=int(case.world_seed),profile=case.profile,groups=ret)


def pick(row,config):
    out={}
    for name,g in row['groups'].items():
        mix=config['ordinary']+(config['stress']-config['ordinary'])*g['fraction']
        p=(1.-mix)*g['legacy']+mix*g[config['regime']]
        out[name]=score(p,g['y'])
    return out


def totals(rows,config):
    data=[counts(pick(r,config)) for r in rows]
    return dict(zip(N_METRICS,np.sum(data,axis=0).astype(int).tolist()))|{'n':len(rows)}


def main():
    a=ArgumentParser();a.add_argument('--workers',type=int,default=11);a.add_argument('--only',nargs='+',default=['develop','archive','fresh','novel'])
    opts=a.parse_args()
    file=HERE/'hypothesis_config.json'
    frozen=json.loads(file.read_text()) if 'develop' not in opts.only else None
    options=[dict(ordinary=a,stress=b,regime=m) for a in WEIGHTS for b in WEIGHTS for m in REGIMES]
    base=options[0]
    with ProcessPoolExecutor(opts.workers) as pool:
        for s in opts.only:
            partitions={'develop':('validate','develop',DEVELOP,DEVELOP_N),
                        'archive':('test','archive',SEEDS['archive'],COUNT['archive']),
                        'fresh':('test','fresh',SEEDS['fresh'],COUNT['fresh']),
                        'novel':('test','public',NEW_TEST,96)}
            split,route,seed,count=partitions[s]
            rows=list(pool.map(collect,((c,route) for c in scan.cases(split,count,seed)),chunksize=1))
            if s=='develop':
                primary=json.loads((HERE/'reader_config.json').read_text())
                train=set(primary['prior_layout_ids'])
                rows=[r for r in rows if r['layout'] not in train]
                all_=[(x,totals(rows,x)) for x in options]
                goal,stats=max(all_,key=lambda x:objective(x[1]))
                frozen=dict(selected=goal,baseline=totals(rows,base),development=stats,
                            layouts=sorted(train|{r['layout'] for r in rows}),prior_seed=primary['prior_seed'],
                            design='public frame likelihood of alternate persistent material offsets')
                file.write_text(json.dumps(frozen,indent=2))
                print('development',frozen['selected'],frozen['baseline'],stats,flush=True)
            else:
                if set(frozen['layouts'])&{r['layout'] for r in rows}:
                    raise ValueError('Held-out layouts overlap')
                configs={'pair':base,'hypotheses':frozen['selected']} | {
                    f'only_{m}':dict(ordinary=1.,stress=1.,regime=m) for m in REGIMES}
                table={n:totals(rows,p) for n,p in configs.items()}
                print(s, table,flush=True)
                details=[{k:r[k] for k in ('world','layout','profile')} | {'answers':{n:pick(r,p) for n,p in configs.items()}} for r in rows]
                (HERE/f'hypothesis_{s}.json').write_text(json.dumps(dict(totals=table,rows=details),separators=(',',':')))


if __name__=='__main__':
    main()
