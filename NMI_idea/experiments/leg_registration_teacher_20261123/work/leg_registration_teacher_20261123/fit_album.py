"""Supervise track-consistent image pairs and grade exactly the same acquired echoes."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from registration import HERE,public,stock
from joint_album import pair_features
from fit_reader import COUNT,SEEDS,gather,scan

TRAIN_N=480
TRAIN_SEED=662125711
DEVELOP_N=144
DEVELOP_SEED=295679877
HOLDOUT_SEED=601745721
WEIGHTS=(0.,.18,.35,.65,1.,1.6,2.5,4.)


def collect(args):
    c,split=args
    env,t,item,raw,cut,spans=gather(c,split)
    safe=public(raw)
    album=[i for i in range(cut) if item[1]-3.-1.e-8<=raw[i][0]<=item[1]+2.+1.e-8]
    groups=[album,list(range(cut,len(raw)))]
    data=pair_features(env.scanner.surface,safe,*groups)
    r=dict(world=int(c.world_seed),layout=sha256(np.nan_to_num(t.layout.centers).tobytes()+bytes([c.shape])).hexdigest(),profile=c.profile)
    if data is None:
        p=stock(env.scanner.surface).best(safe,groups)
        return r|dict(data=None,old=[(None,0.) if q is None else (int(raw[q[0]][3][q[1]]),float(q[2])) for q in p])
    a,b,x,y,features=data
    u=np.array([raw[i][3][j] for i,j in a],dtype=np.int8)
    v=np.array([raw[i][3][j] for i,j in b],dtype=np.int8)
    return r|dict(data=(x,y,features,u,v))


def fit(train):
    obs=[r['data'] for r in train if r['data'] is not None]
    x=np.concatenate([z[2] for z in obs])
    labels=np.concatenate([(z[3]==0)&(z[4]==0) for z in obs])
    for d in obs:
        assert len(d[0])<=18**2
    head=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=25,learning_rate=.09,
                min_samples_leaf=38,l2_regularization=2.5,early_stopping=False,random_state=1171703)
    head.fit(x,labels)
    return head,len(x)


def prepare(rows,head):
    for r in rows:
        if r['data'] is not None:
            x,y,f,u,v=r['data']
            r['p']=head.predict_proba(f)[:,1]


def choice(row,w):
    if row['data'] is None:
        obs=row['old']
    else:
        x,y,f,u,v=row['data']
        log0=np.log(np.maximum(x,.001))+np.log(np.maximum(y,.001))
        allowed=(x>=.45)&(y>=.45)
        score0=log0+w*np.log(np.maximum(row['p'],.01))
        j=int((np.where(allowed,score0,-1000.) if allowed.any() else log0).argmax())
        obs=[(int(u[j]),float(x[j])),(int(v[j]),float(y[j]))]
    correct=sum(k==0 and p>=.45 for k,p in obs)
    wrong=sum(k not in (0,None) and p>=.45 for k,p in obs)
    artery=sum(k==1 and p>=.45 for k,p in obs)
    return dict(both=int(correct==2),correct=correct,wrong=wrong,artery=artery,selected=obs)


def totals(rows,w):
    values=[choice(r,w) for r in rows]
    return {k:sum(d[k] for d in values) for k in ('both','correct','wrong','artery')}


def criterion(s):
    return 2.*s['both'] + .40*s['correct'] - .58*s['wrong'] - .40*s['artery']


def stream(pool,split,partition,seed,size):
    return list(pool.map(collect,((c,split) for c in scan.cases(partition,size,seed)),chunksize=1))


def main():
    a=ArgumentParser();a.add_argument('--workers',type=int,default=12);a.add_argument('--eval-only',action='store_true');v=a.parse_args()
    with ProcessPoolExecutor(v.workers) as pool:
        if v.eval_only:
            head=joblib.load(HERE/'album_head.joblib')
            record=json.loads((HERE/'album_config.json').read_text())
            weight=record['weight']
        else:
            train=stream(pool,'train','train',TRAIN_SEED,TRAIN_N)
            seen={r['layout'] for r in train}
            head,samples=fit(train)
            print('trained',TRAIN_N,samples,flush=True)
            del train
            dev=[r for r in stream(pool,'develop','validate',DEVELOP_SEED,DEVELOP_N) if r['layout'] not in seen]
            prepare(dev,head)
            measured=[(w,totals(dev,w)) for w in WEIGHTS]
            weight,win=max(measured,key=lambda x:(criterion(x[1]),-x[0]))
            record=dict(weight=weight,samples=samples,n=len(dev),training_layouts=len(seen),
                        layouts=sorted(seen|{r['layout'] for r in dev}),selection=measured)
            print('frozen',weight,win,'baseline',measured[0],flush=True)
            joblib.dump(head,HERE/'album_head.joblib',compress=3)
            (HERE/'album_config.json').write_text(json.dumps(record,indent=2))
        for s in ['archive','fresh','public','novel']:
            rows=stream(pool,'public' if s=='novel' else s,'test',HOLDOUT_SEED if s=='novel' else SEEDS[s],96 if s=='novel' else COUNT[s])
            if {d['layout'] for d in rows}&set(record['layouts']):
                raise ValueError('Training and test geometry overlap')
            prepare(rows,head)
            summary={str(m):totals(rows,m) for m in (0.,weight)}
            print(s,summary,flush=True)
            notes=[{k:r[k] for k in ('layout','world','profile')}| {'pair':choice(r,0.),'joint':choice(r,weight)} for r in rows]
            (HERE/f'album_{s}.json').write_text(json.dumps(dict(totals=summary,rows=notes),separators=(',',':')))


if __name__=='__main__':
    main()
