"""Freeze the public identity model by anatomical-layout-held-out development performance."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from model import IdentityModel, load_data, predict_case, summarise

HERE=Path(__file__).resolve().parent


def batch(rows,rng,width=104):
    B=len(rows)
    x=np.zeros((B,width,7,34),np.float32);m=np.zeros((B,width,42),np.float32)
    v=np.zeros(x.shape[:3],bool);t=np.zeros(x.shape[:2],bool)
    y=np.full(x.shape[:3],-100,np.int64);d=np.full((B,width,4),np.nan,np.float32)
    for k,row in enumerate(rows):
        length=len(row['x'])
        end=int(rng.choice(row['ends'])) if rng.random()<.7 else int(rng.integers(min(length,35),length+1))
        start=max(0,end-width);n=end-start
        x[k,:n]=row['x'][start:end];m[k,:n]=row['meta'][start:end]
        v[k,:n]=row['valid'][start:end];t[k,:n]=True;y[k,:n]=row['labels'][start:end]
        d[k,:n]=row['anatomical_offset'][start:end].reshape(n,4)/3.
    return [torch.from_numpy(a) for a in (x,m,v,t,y,d)]


def association_loss(z,y,m,rng):
    # Compare candidate identity across two distinct acquisition times, weighting a changed window.
    lag=int(rng.integers(2,30))
    left=y[:,:-lag];right=y[:,lag:]
    real=(left.unsqueeze(-1)>=0)&(left.unsqueeze(-1)<2)&(right.unsqueeze(-2)>=0)&(right.unsqueeze(-2)<2)
    indices=real.nonzero()
    if not len(indices):return z.sum()*0.
    if len(indices)>1400: indices=indices[torch.randperm(len(indices))[:1400]]
    b,i,k,j=indices.T
    pos=(y[b,i,k]==y[b,i+lag,j]).float()
    same=(z[b,i,k]*z[b,i+lag,j]).sum(-1)
    diff=(m[b,i,3:5]-m[b,i+lag,3:5]).square().sum(-1).clamp(max=1.)
    return F.binary_cross_entropy_with_logits(5.*same,pos,weight=1.+diff)


def train(args):
    torch.set_num_threads(args.threads);torch.manual_seed(76237)
    rng=np.random.default_rng(76237)
    train=load_data(args.data/'train')
    ids={r['layout'] for r in train}
    dev=[r for r in load_data(args.data/'develop') if r['layout'] not in ids]
    if len(train)!=1280 or len(dev)<105:raise RuntimeError('Training or development split incomplete')
    model=IdentityModel(temporal=not args.frame_only)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0017,weight_decay=.001)
    out=HERE/('frame' if args.frame_only else 'temporal')
    out.mkdir(exist_ok=True)
    best=-np.inf;log=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        for group in np.array_split(rng.permutation(len(train)),max(1,len(train)//24)):
            X,M,v,t,y,d=batch([train[i] for i in group],rng)
            logits,offset,z=model(X,M,v,t)
            ce=F.cross_entropy(logits.reshape(-1,8),y.reshape(-1),ignore_index=-100)
            valid_d=d.isfinite()&t.unsqueeze(-1)
            aux=F.smooth_l1_loss(offset[valid_d],d[valid_d])
            link=association_loss(z,y,M,rng) if not args.frame_only else 0.
            loss=ce+.18*aux+.14*link
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            optimizer.step()
        if epoch in (8,16):
            for g in optimizer.param_groups:g['lr']*=.5
        if epoch in (4,8,12,16,20,24,args.epochs):
            model.eval()
            pred=[predict_case(model,r) for r in dev]
            opts=[]
            for q in (.4,.45,.5,.55,.6,.65):
                s,_=summarise(pred,q)
                utility=(s['both']+s['chain']+.20*s['correct_bands']-.55*s['wrong']-.12*s['wrong_bands'])/s['n']-.5*s['brier']
                opts.append((utility,-abs(q-.45),q,s))
            val=max(opts,key=lambda o:o[:2])
            old=summarise(pred,.45)[0]
            print(epoch,round(val[0],4),val[2],old['both'],old['chain'],round(old['brier'],4),flush=True)
            log.append(dict(epoch=epoch,best_q=val[2],utility=val[0],fixed=old))
            if val[0]>best:
                best=val[0]
                torch.save(dict(weights=model.state_dict(),threshold=val[2],temporal=not args.frame_only,
                    epoch=epoch),out/'model.pt')
                (out/'development.json').write_text(json.dumps(pred,separators=(',',':')))
        (out/'training.json').write_text(json.dumps(dict(training_worlds=len(train),training_layouts=len(ids),
            development_worlds=len(dev),development_layouts=len({r['layout'] for r in dev}),history=log),indent=2))
    return out


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('data',type=Path);p.add_argument('--frame-only',action='store_true')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--epochs',type=int,default=24)
    train(p.parse_args())
