"""Fine-tune the existing public sequence model for the ranking within an actual submission group."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.nn import functional as F

from registration import HERE,OLDER
sys.path.insert(0,str(OLDER/'sequence'))
from model import IdentityModel, load_data,predict_case,summarise
from train import association_loss


def training_window(rows,rng,width=146):
    n=len(rows)
    features=np.zeros((n,width,7,34),np.float32)
    metadata=np.zeros((n,width,42),np.float32)
    present=np.zeros(features.shape[:3],bool)
    time=np.zeros((n,width),bool)
    y=np.full(features.shape[:3],-100,np.int64)
    offsets=np.full((n,width,4),np.nan,np.float32)
    targets=np.zeros((n,width,5),bool)
    for i,row in enumerate(rows):
        k=int(rng.integers(5))
        end=int(row['ends'][min(2,max(0,k-2))] if k>=2 else row['ends'][-1])
        start=max(0,end-width)
        length=end-start
        features[i,:length]=row['x'][start:end]
        metadata[i,:length]=row['meta'][start:end]
        present[i,:length]=row['valid'][start:end]
        time[i,:length]=True
        y[i,:length]=row['labels'][start:end]
        offsets[i,:length]=row['anatomical_offset'][start:end].reshape(length,4)/3.
        if k>=2:
            targets[i,:length,k]=row['group'][start:end,k]
        else:
            targets[i,:length,:2]=row['group'][start:end,:2]
    return [torch.from_numpy(z) for z in (features,metadata,present,time,y,offsets,targets)]


def group_loss(logits,y,sets):
    # Emphasize the false echo competing against the best actual vein in the same album.
    log_odds=logits[...,0]-torch.logsumexp(logits[...,1:],dim=-1)
    votes=log_odds.unsqueeze(1).expand(-1,5,-1,-1)
    chosen=sets.transpose(1,2).unsqueeze(-1)
    positives=chosen & (y.unsqueeze(1)==0)
    negatives=chosen & (y.unsqueeze(1)>=1)
    usable=positives.flatten(2).any(-1)&negatives.flatten(2).any(-1)
    if not usable.any():
        return log_odds.sum()*0.
    true_score=votes.masked_fill(~positives,-30.).flatten(2).amax(-1)
    other=votes+(.18*(y==1)).unsqueeze(1)
    false_score=other.masked_fill(~negatives,-30.).flatten(2).amax(-1)
    return F.softplus(.45+false_score[usable]-true_score[usable]).mean()


def run(cache,threads,epochs):
    torch.set_num_threads(threads)
    torch.manual_seed(515095)
    rng=np.random.default_rng(515095)
    train=load_data(cache/'train')
    used={r['layout'] for r in train}
    dev=[r for r in load_data(cache/'develop') if r['layout'] not in used]
    model=IdentityModel(True)
    model.load_state_dict(torch.load(OLDER/'sequence/temporal/model.pt',map_location='cpu',weights_only=True)['weights'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00055,weight_decay=.0015)
    best=None; history=[]
    for e in range(1,epochs+1):
        model.train()
        for ids in np.array_split(rng.permutation(len(train)),len(train)//24):
            x,m,valid,time,y,d,g=training_window([train[i] for i in ids],rng)
            logits,geom,z=model(x,m,valid,time)
            cross=F.cross_entropy(logits.reshape(-1,8),y.reshape(-1),ignore_index=-100)
            finite=torch.isfinite(d)&time.unsqueeze(-1)
            residual=F.smooth_l1_loss(geom[finite],d[finite])
            ranking=group_loss(logits,y,g)
            sequence=association_loss(z,y,m,rng)
            loss=cross+.8*residual+.32*ranking+.07*sequence
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.5)
            optimizer.step()
        if e not in (2,4,7,10,epochs):
            continue
        model.eval()
        pred=[predict_case(model,r) for r in dev]
        stats=summarise(pred,.45)[0]
        v=(2.*stats['both']+1.35*stats['chain']+.30*stats['correct_bands']-.63*stats['wrong']
               -.23*stats['artery']-.11*stats['wrong_bands'])/stats['n']
        history.append(dict(epoch=e,utility=v,statistics=stats))
        print(e,round(v,4),stats['both'],stats['chain'],stats['wrong'],stats['artery'],flush=True)
        if best is None or v>best:
            best=v
            torch.save(dict(weights=model.state_dict(),epoch=e,threshold=.45),HERE/'group_reader.pt')
            (HERE/'group_dev.json').write_text(json.dumps(pred,separators=(',',':')))
    (HERE/'group_training.json').write_text(json.dumps(dict(n=len(train),selected_by='development',history=history),indent=2))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('--threads',type=int,default=6)
    p.add_argument('--epochs',type=int,default=10)
    a=p.parse_args();run(a.cache,a.threads,a.epochs)
