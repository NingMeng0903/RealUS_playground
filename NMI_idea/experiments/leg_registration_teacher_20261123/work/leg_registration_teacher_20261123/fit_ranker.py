"""Compare a frozen geometry reader and supervised readout trained on hard identity failures."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from registration import HERE, public
from ranking import RankFeatures
from fit_reader import COUNT, DEV_NAME, EXTRA_SEED, NAMES, gather, scan, SEEDS, score, counts, objective, N_METRICS

TRAIN_SEED = 710632515
TRAIN_N = 448
DEVELOP_SEED = 855014993
DEVELOP_N = 160


@lru_cache(maxsize=9)
def features(surface):
    return RankFeatures(surface)


def collect(job):
    case,split=job
    env,trial,item,raw,cut,spans=gather(case,split)
    encoder=features(env.scanner.surface)
    safe=public(raw)
    album=[i for i in range(cut) if item[1]-3.-1.e-8<=raw[i][0]<=item[1]+2.+1.e-8]
    info={}
    for j,(lo,hi) in enumerate(spans):
        names=[NAMES[j]]
        groups=[list(range(lo,hi))]
        if j==2:
            names += ['entry','fresh']
            groups += [album,list(range(cut,hi))]
        blocks,fraction=encoder.features(safe[:hi],groups)
        for name,(ix,x,old,reg) in zip(names,blocks,strict=True):
            truth=np.asarray([raw[i][3][k] for i,k in ix],dtype=np.int8)
            info[name]=dict(x=x,old=old,reg=reg,y=truth,fraction=fraction,indices=ix)
    return dict(world=int(case.world_seed),profile=case.profile,
                layout=sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest(), data=info)


def fit(rows):
    per_group=[d for r in rows for k,d in r['data'].items() if len(d['y']) and k!='fresh']
    x=np.concatenate([d['x'] for d in per_group])
    y=np.concatenate([d['y'] for d in per_group])
    labels=np.where(y==0,0,np.where(y==1,1,np.where(y==-2,2,3)))
    old=np.concatenate([d['old'] for d in per_group])
    # Retain all eligible training candidates. Give rank errors above .55 extra influence.
    wrong=(y!=0)*(old>=.55)
    weights={'plain':np.ones(len(x)), 'hard':1.+wrong*(.7+.55*(y==1))}
    out={}
    for label,w in weights.items():
        model=HistGradientBoostingClassifier(max_iter=95,max_leaf_nodes=23,min_samples_leaf=28,
                   l2_regularization=2.2, learning_rate=.085, random_state=240177,early_stopping=False)
        out[label]=model.fit(x,labels,sample_weight=w)
    return out,int(len(x)),int(sum(wrong))


def predict(rows,models):
    for row in rows:
        for d in row['data'].values():
            d['head']={name:m.predict_proba(d['x'])[:,0] if len(d['x']) else np.empty(0) for name,m in models.items()}


def choices(row,spec):
    out={}
    for k,d in row['data'].items():
        fraction=d['fraction']
        a=spec['ordinary']*(1.-fraction)+spec['stress']*fraction
        reg=spec['base']*d['reg']+(1.-spec['base'])*d['old']
        p=a*d['head'][spec['head']]+(1.-a)*reg
        out[k]=score(p,d['y'])
    return out


def evaluate(rows,spec):
    raw=np.asarray([counts(choices(r,spec)) for r in rows])
    return dict(zip(N_METRICS,raw.sum(axis=0).astype(int)))|{'n':len(rows)}


BASELINES = dict(pair=dict(head='plain',ordinary=0.,stress=0.,base=0.),
                 registration=dict(head='plain',ordinary=0.,stress=0.,base=1.))


def choose(rows):
    configs=list(BASELINES.values())+[
        dict(head=n,ordinary=u,stress=v,base=b)
        for n in ('plain','hard') for u in (0.,.33,.66,1.) for v in (0.,.33,.66,1.)
        for b in (.5,1.) if u or v
    ]
    evaluated=[(a,evaluate(rows,a)) for a in configs]
    winner=max(evaluated,key=lambda x:objective(x[1]))
    print('frozen',winner,'controls',[(n,evaluate(rows,d)) for n,d in BASELINES.items()],flush=True)
    return winner[0], [a|{'utility':objective(q),'metrics':{k:int(v) for k,v in q.items()}} for a,q in evaluated]


def get(pool,split,partition,seed,num):
    return list(pool.map(collect,((c,split) for c in scan.cases(partition,num,seed)),chunksize=1))


def main():
    a=ArgumentParser();a.add_argument('--workers',type=int,default=13)
    a.add_argument('--eval-only', action='store_true')
    a.add_argument('--eval', nargs='+', default=('archive','fresh','public','extra'))
    opts=a.parse_args()
    with ProcessPoolExecutor(opts.workers) as pool:
        if not opts.eval_only:
            train=get(pool,'train','train',TRAIN_SEED,TRAIN_N)
            layouts={r['layout'] for r in train}
            print('collected',len(train),len(layouts),flush=True)
            models,n,h=fit(train)
            print('fitted',n,h,flush=True)
            del train
            dev=[r for r in get(pool,'develop','validate',DEVELOP_SEED,DEVELOP_N) if r['layout'] not in layouts]
            predict(dev,models)
            selected,scores=choose(dev)
            config=dict(selected=selected,development=scores,train=TRAIN_N,train_candidates=n,hard_candidates=h,
                        train_seed=TRAIN_SEED,development_seed=DEVELOP_SEED,
                        layouts=sorted(layouts|{r['layout'] for r in dev}))
            joblib.dump(models,HERE/'rank_head.joblib',compress=3)
            (HERE/'rank_config.json').write_text(json.dumps(config,indent=2))
        else:
            models=joblib.load(HERE/'rank_head.joblib')
            config=json.loads((HERE/'rank_config.json').read_text())
            selected=config['selected']
        arms=BASELINES|{'trained_reader':selected}
        for s in opts.eval:
            seed,num=(EXTRA_SEED,96) if s=='extra' else (SEEDS[s],COUNT[s])
            rows=get(pool,'public' if s=='extra' else s,'test',seed,num)
            if {r['layout'] for r in rows}&set(config['layouts']):
                raise RuntimeError('Test layouts entered the supervised data')
            predict(rows,models)
            totals={n:{k:int(v) for k,v in evaluate(rows,m).items()} for n,m in arms.items()}
            print(s,totals,flush=True)
            details=[{k:r[k] for k in ('world','layout','profile')} |
                     {'out':{n:choices(r,m) for n,m in arms.items()}} for r in rows]
            (HERE/f'rank_{s}.json').write_text(json.dumps(dict(totals=totals,rows=details),separators=(',',':')))


if __name__=='__main__':
    main()
