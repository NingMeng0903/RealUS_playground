"""Anonymous set encoder, temporal context, and supervised anatomical association."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class IdentityModel(nn.Module):
    def __init__(self, temporal=True):
        super().__init__()
        self.temporal=temporal
        self.echo=nn.Sequential(nn.Linear(34,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,32))
        self.frame=nn.Sequential(nn.Linear(42+64,48),nn.LayerNorm(48),nn.SiLU())
        self.context=nn.ModuleList(nn.Sequential(nn.Conv1d(48,48,3,padding=d,dilation=d),
            nn.SiLU(),nn.Conv1d(48,48,1))
            for d in ((1,3,9,18) if temporal else ()))
        self.normalize=nn.ModuleList(nn.LayerNorm(48) for _ in self.context)
        self.offset=nn.Linear(48,4)
        self.identify=nn.Sequential(nn.Linear(32+48+10,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.output=nn.Linear(24,8)
        self.embedding=nn.Linear(24,8)

    def forward(self,x,meta,present,time_valid):
        valid=present.unsqueeze(-1)
        e=self.echo(x)
        mean=(e*valid).sum(-2)/valid.sum(-2).clamp_min(1.)
        peak=e.masked_fill(~valid,-10.).amax(-2)
        peak=torch.where(valid.any(-2),peak,0.)
        ctx=self.frame(torch.cat((mean,peak,meta),-1))*time_valid.unsqueeze(-1)
        for block,norm in zip(self.context,self.normalize):
            ctx=norm(ctx+block(ctx.transpose(1,2)).transpose(1,2))*time_valid.unsqueeze(-1)
        d=self.offset(ctx)
        relative=x[:,:,:,14:18]-d.unsqueeze(2)
        norm=relative.reshape(*relative.shape[:-1],2,2).square().sum(-1).clamp(max=30.)
        c=ctx.unsqueeze(2).expand(-1,-1,x.shape[2],-1)
        extra=torch.cat((relative,norm,d.unsqueeze(2).expand_as(relative)),-1)
        token=self.identify(torch.cat((e,c,extra),-1))
        return self.output(token),d,F.normalize(self.embedding(token),dim=-1)


def select_group(probs, labels, eligible):
    rows=np.flatnonzero(eligible)
    indices,slots=np.where(labels[rows]!=-100)
    if not len(indices):
        return dict(role=None,p=0.,i=-1,k=-1,capture=0)
    frame=rows[indices]
    q=probs[frame,slots]
    best=q.argmax()
    i=int(frame[best]); k=int(slots[best])
    return dict(role=int(labels[i,k]),p=float(q[best]),i=i,k=k,capture=int(np.any(labels[frame,slots]==0)))


def single_probs(model,x,meta,valid):
    X=torch.from_numpy(x.astype('float32',copy=False)).unsqueeze(0)
    M=torch.from_numpy(meta.astype('float32',copy=False)).unsqueeze(0)
    ok=torch.from_numpy(valid).unsqueeze(0)
    with torch.inference_mode():
        logits,_,_=model(X,M,ok,torch.ones(X.shape[:2],dtype=torch.bool))
        return logits[0].softmax(-1)[:,:,0].numpy()


def predict_case(model,row):
    scores={}
    cal=[]
    B=row['labels']; group=row['group']
    for j,end in enumerate(row['ends']):
        pp=single_probs(model,row['x'][:end],row['meta'][:end],row['valid'][:end])
        for k in ((j+2,) if j<2 else (j+2,0,1)):
            name=('entry','fresh','band1','band2','band3')[k]
            scores[name]=select_group(pp,B[:end],group[:end,k])
            if k>=2:
                y=B[:end][group[:end,k]];prob=pp[group[:end,k]]
                obs=y!=-100
                if obs.any(): cal.extend(np.square(prob[obs]-(y[obs]==0)).tolist())
    return dict(world=row['world'],layout=row['layout'],answers=scores,brier=float(np.mean(cal)) if cal else None)


def summarise(rows,cut):
    out={}
    results=[]
    for r in rows:
        answers=r['answers']
        def values(key):
            v=answers[key];take=v['role'] is not None and v['p']>=cut
            return int(take and v['role']==0), int(take and v['role']!=0), int(take and v['role']==1)
        e,f=values('entry'),values('fresh')
        bands=[values('band'+str(i)) for i in range(1,4)]
        results.append(dict(entry=e[0],fresh=f[0],both=e[0]*f[0],wrong=e[1]+f[1],artery=e[2]+f[2],
            chain=int(all(v[0] for v in bands)),prefix=5*next((i for i,v in enumerate(bands) if not v[0]),3),
            correct_bands=sum(v[0] for v in bands),wrong_bands=sum(v[1] for v in bands),artery_bands=sum(v[2] for v in bands),
            high_wrong=sum(v['role']!=0 and v['p']>=.90 for k,v in answers.items() if k in ('entry','fresh')),
            high_total=sum(v['p']>=.90 for k,v in answers.items() if k in ('entry','fresh'))))
    for k in results[0]:
        out[k]=float(np.mean([p[k] for p in results])) if k=='prefix' else int(sum(p[k] for p in results))
    out.update(n=len(rows),threshold=cut,brier=float(np.mean([r['brier'] for r in rows if r['brier'] is not None])))
    return out,results


def load_data(folder):
    rows=[]
    for name in sorted(Path(folder).glob('*.npz')):
        with np.load(name,allow_pickle=False) as d:
            row={k:d[k] for k in d.files}
        row['world']=int(row['world']);row['layout']=str(row['layout'])
        rows.append(row)
    return rows
