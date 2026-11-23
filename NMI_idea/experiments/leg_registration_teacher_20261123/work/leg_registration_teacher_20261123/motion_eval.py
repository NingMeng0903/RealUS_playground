"""Compare equal-time hold, public belief control and a privileged evidence-gathering teacher."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np

from registration import HERE, RegisteredReader, chart, key_groups, posterior, public, stock
from ranking import EvidenceReader
from core import (Landmark, advance, edge_poses, fixed_slot,get_world,legal_slots,named_nodes,
                  observe_hold,path,req,reserve,scan,trace_plan,visit)
from experiments import conditional
from recovery import entry_trip

GOAL = 'Posterior_tibial_vein.r'
TEST_SEED = 650388679
DEVELOP_SEED = 920218131
ARMS = ('fixed','public','capture_teacher','evidence_teacher','verified_teacher')


@dataclass
class State:
    node: object
    cut: int | None = None
    spans: tuple = ()


@lru_cache(maxsize=8)
def reading(surface,kind):
    if kind=='pair':
        return stock(surface)
    if kind=='registered':
        return RegisteredReader(surface)
    if kind=='evidence':
        return EvidenceReader(surface)
    if kind=='geometry':
        return GeometryOnly(surface)
    if kind=='joint':
        from joint_album import JointAlbum
        return JointAlbum(surface)
    raise ValueError(kind)


class GeometryOnly(RegisteredReader):
    """Use the selected geometric association directly during prospective search."""
    def scores(self, frames, groups):
        if not frames or not any(groups):
            return [(np.empty((0,2),dtype=int),np.empty(0)) for _ in groups]
        prepared, probabilities, _, _ = self.field(frames)
        return [(ix, probabilities[ix[:,0],ix[:,1],0]) for ix in key_groups(prepared,groups)]


def angles_and_path(env,spec,start,sector,anchor,turn):
    n = abs(start-anchor)
    sectors = (sector + np.rint(np.linspace(0.,turn,n+1)).astype(int)) % 24
    forward = sectors if anchor>start else sectors[::-1]
    lo = min(start,anchor)
    if path(env,lo,forward) is None:
        return None
    positions = edge_poses(env,spec,lo,forward)
    return positions if anchor>start else positions[::-1]


def episode_menu(env,spec,node,h,limits,clock,entry):
    """Produce feasible return excursions to distinct role-separation landmarks."""
    s = int(node.station)
    a = int(node.sector)
    time_budget, distance_budget = reserve(node,spec,limits)
    max_ticks = max(0, min(58, time_budget-3*clock-4))
    max_mm = max(0., distance_budget-25.)
    arms = [None]
    for q in [*entry_trip(env,spec,node,h,limits,entry,clock),trace_plan(env,spec,node,h,limits,clock,prefer_named=True)]:
        if q is not None and len(q.lengths)<=max_ticks and q.lengths.sum()<=max_mm:
            arms.append(q)
    surf = env.scanner.surface
    prior = surf.priors[2]
    links = surf.reference_links[2]
    candidates = []
    if max_ticks>=8 and max_mm>=18.:
        witnessed = public(node.reader.frames)
        rd = reading(surf,'geometry')
        prepared,p,offset,_ = rd.field(witnessed)
        keep = np.asarray(prepared.stations >= s-7.)
        p_recent = p[keep]
        rivals = p_recent[:,:,1:6].sum(axis=(0,1)) if len(p_recent) else np.ones(5)
        gap = np.linalg.norm(prior[s,1:]-prior[s,0],axis=1)
        proximity = np.exp(-.5*np.nan_to_num(gap,nan=100.)**2/5.**2)
        rival = int(np.argmax((rivals+.25)*proximity))+1
        shift = np.mean(offset[-12:],axis=0)
        base_gap = prior[s,rival]-prior[s,0]
        bridges = named_nodes(env)
        for st in range(max(spec[1],s-8),min(spec[2],s+8)+1):
            d = abs(st-s)
            if d<2 or d*10>max_mm:
                continue
            lo,hi = sorted((s,st))
            if not np.all(links[lo:hi,0]):
                continue
            connected = np.all(links[lo:hi,rival])
            if not (st in bridges or connected):
                continue
            pair_delta = prior[st,rival] - prior[st,0]
            for turn in (0,-2,2,-1,1):
                sector = (a+turn)%24
                first = angles_and_path(env,spec,s,a,st,turn)
                if first is None or 2*(len(first)-1)+3>max_ticks:
                    continue
                projected = surf.project(float(st),float(sector),prior[st]+shift)
                shadow = surf.shadow(float(st),float(sector),np.nan_to_num(projected,nan=150.))
                t,r = projected[0],projected[rival]
                visible = (np.abs(projected[:,0])<25.) & (projected[:,1]>3.) & (projected[:,1]<80.)
                detect = visible*(1.-.96*shadow)
                rg = np.linalg.norm(t-r) if connected else 0.
                bend = np.linalg.norm(pair_delta-base_gap) if connected else 0.
                distinct=detect[0]*detect[rival]*(1.-np.exp(-.5*(rg/4.5)**2))
                different=detect[0]*(1.-detect[rival]) if connected else 0.
                merit=.70*distinct + .20*different + .13*min(3.,bend) + .25*int(st in bridges)
                if merit <= 0.:
                    continue
                poses = np.vstack((first,np.repeat(first[-1:],3,axis=0),first[-2::-1]))
                segments=env.scanner._path_metrics(poses,spec)
                if segments is None or segments.sum()>max_mm:
                    continue
                name='separation_distal' if st>s else 'separation_proximal'
                candidates.append(Landmark(poses,segments,name,rival,float(merit),st))
        for name in ('separation_proximal','separation_distal'):
            subsets=[p for p in candidates if p.kind==name]
            if subsets:
                arms.append(max(subsets,key=lambda p:p.score))
    unique=[None]
    stamps=set()
    for q in arms[1:]:
        tag=(int(round(q.station)),tuple(np.round(q.poses[:,1]%24,3)))
        if tag not in stamps:
            stamps.add(tag)
            unique.append(q)
    return unique, max((len(q.lengths) for q in unique if q is not None),default=0)


class Episode:
    def __init__(self, case, compact_search=False):
        self.case = case
        self.env,self.spec,self.trial,self.limits,_,items = get_world(case)
        self.item = next(r for r in items if r[0]==GOAL)
        _,self.entry,self.origin,h,_,_,self.clock,_ = self.item
        self.plans,self.duration = episode_menu(self.env,self.spec,self.origin,h,self.limits,self.clock,self.entry)
        self.start=State(self.origin)
        self.surface=self.env.scanner.surface
        self.layout=sha256(np.nan_to_num(self.trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
        self.old=tuple(i for i,f in enumerate(self.origin.reader.frames) if self.entry-3.-1.e-8<=f[0]<=self.entry+2.+1.e-8)
        self.compact_search=compact_search

    def legal(self,state,stage):
        if stage==0:
            return range(len(self.plans))
        return legal_slots(self.env,self.spec,state.node,self.limits,self.clock,4-stage)

    def go(self,state,stage,action,trial):
        n=state.node.fork()
        start=len(n.reader.frames)
        if stage==0:
            q=self.plans[action]
            if q is not None:
                visit(self.env,self.spec,n,trial,q,True)
            observe_hold(self.env,n,self.spec,trial,self.duration-(len(q.lengths) if q is not None else 0))
            assert (n.station,n.sector,n.tick)==(state.node.station,state.node.sector,state.node.tick+self.duration)
            out=State(n,len(n.reader.frames),())
        else:
            advance(self.env,self.spec,n,trial,self.limits,action,self.clock)
            out=State(n,state.cut,state.spans+((start,len(n.reader.frames)),))
        if n.tick>self.limits.ticks or n.distance>self.limits.mm+1.e-6:
            raise ValueError('A trial action exceeded the existing physical envelope')
        return out

    def groups(self,end):
        raw=end.node.reader.frames
        old=list(self.old)
        now=[i for i,f in enumerate(raw[:end.cut]) if self.entry-3.-1.e-8<=f[0]<=self.entry+2.+1.e-8]
        downstream=list(range(end.cut,len(raw)))
        return dict(entry=now,fresh=downstream,locked=old,**{
            'band'+str(i+1):list(range(lo,hi)) for i,(lo,hi) in enumerate(end.spans)
        })

    def anatomy_choice(self,state,legal,mode):
        n=state.node
        s,a = int(n.station),int(n.sector)
        st = s+1.
        surface=self.surface
        if mode=='public':
            history = public(n.reader.frames)
            prepared,p,offset,_ = reading(surface,'geometry').field(history)
            relevance=np.exp(-.35*np.square(prepared.stations[-18:]-s))
            displacement=(offset[-18:]*relevance[:,None,None]).sum(axis=0)/max(.1,relevance.sum())
            centers=surface.priors[2][s+1]+displacement
        vals=[]
        for action in legal:
            angle=float(self.env.route.sectors[a,action])
            if mode=='public':
                xy=surface.project(st,angle,centers)
                shadow=surface.shadow(st,angle,np.nan_to_num(xy,nan=150.))
            else:
                xy,shadow=self.env.scanner._anatomy.training_truth(surface,self.trial,st,angle)
            safe=np.nan_to_num(xy,nan=140.)
            outer=np.minimum.reduce((25.-abs(safe[:,0]),safe[:,1]-3.,80.-safe[:,1]))
            view=np.clip((outer+2.)/5.,0.,1.)*(1.-.96*shadow)
            distances=np.linalg.norm((safe[1:]-safe[0])/[4.6,3.5],axis=1)
            interference=np.max(view[1:]*np.exp(-.5*distances**2))
            score=view[0]*(1.-.67*interference) + .025*np.clip(outer[0]/9.,0.,1.)
            vals.append(score)
        return int(legal[int(np.argmax(vals))])

    def public_action(self,state,stage,legal):
        if stage>0:
            return self.anatomy_choice(state,legal,'public')
        if len(self.plans)==1:
            return 0
        frame=state.node.reader.frames
        reader=reading(self.surface,'geometry')
        block=reader.scores(public(frame),[list(self.old)])[0][1]
        margin=float(block.max()) if len(block) else 0.
        if margin>=.76:
            return 0
        return int(max(range(1,len(self.plans)),key=lambda j:self.plans[j].score))

    def prospective(self,state,stage,action,branch,kind,readout=None,causal=False):
        s=self.go(state,stage,action,branch)
        for j in range(stage+1,4):
            opts=self.legal(s,j)
            nxt=self.anatomy_choice(s,opts,'true')
            s=self.go(s,j,nxt,branch)
        groups=self.groups(s)
        raw=s.node.reader.frames
        if kind=='capture_teacher':
            outcomes=[bool(any(np.any(raw[i][3]==0) for i in idx)) for idx in groups.values()]
            return 2.*(outcomes[0] and outcomes[1])+.35*sum(outcomes)+.55*all(outcomes[3:])
        names=tuple(groups)
        graded=self.score(s,readout or 'geometry',groups,causal=causal)
        vals=[graded[k] for k in names]
        both=vals[0]['right'] and vals[1]['right']
        chain=all(v['right'] for v in vals[3:])
        signed=lambda r: float(r['right'])-float(r['wrong'])
        progress=sum(signed(r) for r in vals[3:])/3.
        margins=np.mean([np.tanh(4.*(q['target_score']-q['wrong_score'])) for q in vals])
        return 2.2*both+.75*chain+.52*progress+.34*(signed(vals[0])+signed(vals[1])) + .35*signed(vals[2])+.47*margins

    def teacher_action(self,state,stage,legal,kind,draws):
        legal=list(legal)
        if len(legal)==1:
            return int(legal[0]),None
        old=state.node
        # Draw the same independent future innovations for every alternative.
        branches=[conditional(self.trial,old.tick,scan.rng_seed(self.case.action_seed,stage,414277,k))
                  for k in range(draws)]
        means=[]
        for a in legal:
            values=[self.prospective(state,stage,int(a),branch,kind) for branch in branches]
            means.append(float(np.mean(values)))
        if len(legal)>2 and kind=='evidence_teacher' and draws>=2 and not self.compact_search:
            # Compare the provisional top two against the equal-time default using fresh simulated futures.
            if stage==0:
                default=0
            else:
                default=fixed_slot(self.env,self.spec,old,self.limits,legal)
            promising=sorted(set([*np.argsort(means)[-2:].tolist(),legal.index(default)]))
            hold=[conditional(self.trial,old.tick,scan.rng_seed(self.case.action_seed,stage,918583,k))
                  for k in range(draws)]
            for i in promising:
                m=np.mean([self.prospective(state,stage,int(legal[i]),b,kind) for b in hold])
                means[i] = float((means[i]+m)*.5)
            ranked=max(promising,key=lambda k:(means[k],-k))
        else:
            ranked=int(np.argmax(means))
        return int(legal[ranked]),means

    def verified_action(self,state,stage,legal,readout,draws):
        """Select after a same-reader comparison with a protected continuation."""
        legal = [int(a) for a in legal]
        if len(legal) == 1:
            return legal[0],None
        if stage == 0:
            reference=0
            frames=state.node.reader.frames
            indices,p=reading(self.surface,readout).scores(public(frames),[list(self.old)])[0]
            if len(p):
                i,k=indices[int(np.argmax(p))]
                # A verified, clearly submitted old image already fulfills the purpose of the review.
                if p.max() >= .55 and frames[i][3][k]==0:
                    return 0,None
        else:
            reference=fixed_slot(self.env,self.spec,state.node,self.limits,legal)
        cheap=[conditional(self.trial,state.node.tick,scan.rng_seed(self.case.action_seed,stage,573133,j))
               for j in range(min(draws,3))]
        rough=[float(np.mean([self.prospective(state,stage,a,b,'evidence_teacher') for b in cheap])) for a in legal]
        pool=[reference]+[a for a in sorted(legal,key=lambda k:rough[legal.index(k)],reverse=True)
                          if a != reference][:2]
        fresh=[conditional(self.trial,state.node.tick,scan.rng_seed(self.case.action_seed,stage,915113,j))
               for j in range(draws if stage==0 else min(4,draws))]
        result={}
        for a in pool:
            result[a]=np.asarray([self.prospective(state,stage,a,b,'evidence_teacher',readout,
                                                   causal=stage>=2) for b in fresh])
        anchor=result[reference]
        champion=max(pool,key=lambda a:float(np.mean(result[a]-anchor)))
        values=[None if a not in result else float(result[a].mean()) for a in legal]
        return champion, values

    def score(self,end,kind,groups=None,*,causal=True):
        raw = end.node.reader.frames
        public_frames = public(raw)
        reader=reading(self.surface,kind)
        groupmap=self.groups(end) if groups is None else groups
        joint = {}
        if kind=='joint':
            joint = dict(zip(('entry','fresh'),reader.choose(public_frames,[groupmap['entry'],groupmap['fresh']]),strict=True))
        def grade(prefix,keys):
            for k,(ix,p) in zip(keys,reader.scores(public_frames[:prefix],[groupmap[key] for key in keys]),strict=True):
                if not len(ix):
                    out[k]=dict(right=False,wrong=False,top1=False,role=None,p=0.,capture=False,target_score=0.,wrong_score=0.)
                    continue
                label=np.array([raw[i][3][j] for i,j in ix],dtype=int)
                target=label==0
                winner = joint.get(k)
                if winner is None:
                    w = int(np.argmax(p))
                else:
                    i, j = winner[:2]
                    w = int(np.flatnonzero((ix[:,0]==i) & (ix[:,1]==j))[0])
                accept=bool(p[w]>=.45)
                out[k]=dict(right=bool(accept and target[w]),wrong=bool(accept and not target[w]),top1=bool(target[w]),
                            role=int(label[w]),p=float(p[w]),capture=bool(target.any()),
                            target_score=float(p[target].max(initial=0.)),wrong_score=float(p[~target].max(initial=0.)))
        out={}
        if causal:
            for i,(_,b) in enumerate(end.spans[:2],1):
                grade(b,['band'+str(i)])
        grade(len(raw),[k for k in groupmap if k not in out])
        return out

    def play(self,mode,draws,reader_kind):
        state=self.start
        decisions=[]
        for stage in range(4):
            legal=self.legal(state,stage)
            values=None
            if mode=='fixed':
                pick=0 if stage==0 else fixed_slot(self.env,self.spec,state.node,self.limits,legal)
            elif mode=='public':
                pick=self.public_action(state,stage,legal)
            elif mode=='verified_teacher':
                pick,values=self.verified_action(state,stage,legal,reader_kind,draws)
            else:
                pick,values=self.teacher_action(state,stage,legal,mode,draws)
            node=state.node
            decisions.append(dict(action=int(pick),legal=[int(x) for x in legal],values=values,
                                  position=[int(node.station),int(node.sector),int(node.tick)]))
            state=self.go(state,stage,pick,self.trial)
        if state.node.station!=self.origin.station+3 or state.node.tick!=self.origin.tick+self.duration+3*self.clock:
            raise RuntimeError('The scan missed the common end pose/time')
        return state,decisions,self.score(state,reader_kind)


def summarize(parts):
    order=('band1','band2','band3')
    return dict(both=int(parts['entry']['right'] and parts['fresh']['right']),
                chain=int(all(parts[g]['right'] for g in order)),
                locked=int(parts['locked']['right']),capture_both=int(parts['entry']['capture'] and parts['fresh']['capture']),
                artery=sum(parts[g]['role']==1 and parts[g]['wrong'] for g in ('entry','fresh')),
                wrong=sum(parts[g]['wrong'] for g in ('entry','fresh')),
                prefix_mm=next((5*i for i,k in enumerate(order) if not parts[k]['right']),15))


def shared_job(job):
    case,draws,primary,modes,compact=job
    ep=Episode(case,compact)
    initial=reading(ep.surface,primary)
    ids,p=initial.scores(public(ep.origin.reader.frames),[list(ep.old)])[0]
    before=None
    if len(p):
        j=int(np.argmax(p))
        before=int(ep.origin.reader.frames[ids[j,0]][3][ids[j,1]])==0 and p[j]>=.45
    by={}
    for mode in modes:
        end,decisions,score=ep.play(mode,draws,primary)
        alternate=ep.score(end,'pair') if primary!='pair' else score
        result=dict(primary=summarize(score),pair=summarize(alternate),old_before=bool(before),
                    details=score,decisions=decisions,ticks=int(end.node.tick-ep.origin.tick),
                    distance=float(end.node.distance-ep.origin.distance),
                    selected_probe='hold' if decisions[0]['action']==0 else ep.plans[decisions[0]['action']].kind)
        by[mode]=result
    return dict(world=int(case.world_seed),layout=ep.layout,profile=case.profile,shape=case.shape,
                windows=[None if p is None else dict(kind=p.kind,rival=p.rival,station=p.station,ticks=len(p.lengths)) for p in ep.plans],
                score=by)


def main():
    cli=ArgumentParser()
    cli.add_argument('--split',choices=('develop','test'),default='test')
    cli.add_argument('--count',type=int,default=96)
    cli.add_argument('--workers',type=int,default=12)
    cli.add_argument('--draws',type=int,default=2)
    cli.add_argument('--reader',choices=('pair','registered','evidence','geometry','joint'),default='evidence')
    cli.add_argument('--modes',nargs='+',choices=ARMS,default=ARMS)
    cli.add_argument('--compact',action='store_true')
    cli.add_argument('--name',default='teacher')
    cli.add_argument('--seed',type=int)
    a=cli.parse_args()
    seed=a.seed or (TEST_SEED if a.split=='test' else DEVELOP_SEED)
    cases=scan.cases('validate' if a.split=='develop' else 'test',a.count,seed)
    jobs=((c,a.draws,a.reader,a.modes,a.compact) for c in cases)
    file=HERE/(a.name+'_raw.jsonl')
    rows=[]
    with ProcessPoolExecutor(a.workers) as pool, file.open('w') as handle:
        for case in pool.map(shared_job,jobs,chunksize=1):
            handle.write(json.dumps(case,separators=(',',':'))+'\n')
            handle.flush()
            rows.append(case)
            if len(rows)%8==0:
                print(len(rows),{m:sum(r['score'][m]['primary']['both'] for r in rows) for m in a.modes},flush=True)
    data={m:{key:sum(r['score'][m]['primary'][key] for r in rows)
             for key in rows[0]['score'][m]['primary']} for m in a.modes}
    print('completed',len(rows),data,flush=True)
    (HERE/(a.name+'_summary.json')).write_text(json.dumps(dict(split=a.split,n=len(rows),seed=seed,
                reader=a.reader,draws=a.draws,compact=a.compact,results=data),indent=2))


if __name__=='__main__':
    main()
