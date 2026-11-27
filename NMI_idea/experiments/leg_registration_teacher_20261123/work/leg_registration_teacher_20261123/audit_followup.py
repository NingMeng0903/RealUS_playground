"""Check recorded actions, paired effects and evidence ablation without fitting on evaluation data."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np

from motion_eval import HERE, Episode, reading, public, scan, summarize


def interval(rows, values):
    grouped=defaultdict(list)
    for r,v in zip(rows,values,strict=True):
        grouped[r['layout']].append(v)
    count=np.asarray([len(z) for z in grouped.values()])
    sums=np.asarray([sum(z) for z in grouped.values()])
    ids=np.random.default_rng(8699).integers(len(count),size=(5000,len(count)))
    draw=sums[ids].sum(axis=1)/count[ids].sum(axis=1)
    return np.quantile(draw,[.025,.975]).tolist()


def effects(rows):
    paired={}
    for method in rows[0]['score']:
        if method=='fixed':
            continue
        x={}
        for endpoint in ('both','chain','locked','wrong','artery','prefix_mm'):
            d=[r['score'][method]['primary'][endpoint]-r['score']['fixed']['primary'][endpoint] for r in rows]
            x[endpoint]=dict(net=int(sum(d)),rescued=sum(v>0 for v in d),lost=sum(v<0 for v in d),
                             delta=float(np.mean(d)),interval95=interval(rows,d))
        paired[method]=x
    return paired


def recorded(job):
    case,row,reader,policy=job
    episode=Episode(case)
    state=episode.start
    assert episode.layout==row['layout'] and int(case.world_seed)==row['world']
    orig=row['score'][policy]
    actions=orig['decisions']
    for stage,d in enumerate(actions):
        assert d['legal']==[int(k) for k in episode.legal(state,stage)]
        state=episode.go(state,stage,d['action'],episode.trial)
    grades=episode.score(state,reader)
    assert orig['primary']==summarize(grades)
    for k,a in orig['details'].items():
        np.testing.assert_allclose(a['p'],grades[k]['p'],rtol=0.,atol=1.e-9)
        assert a['role']==grades[k]['role']
    rd=reading(episode.surface,reader)
    raw=state.node.reader.frames
    result=dict(world=row['world'],layout=row['layout'],motion=False)
    if actions[0]['action']>0:
        initial_count=len(episode.origin.reader.frames)
        n=len(episode.plans[actions[0]['action']].lengths)
        drop=set(range(initial_count,initial_count+n))
        assert not set(episode.old)&drop and all(x<state.cut for x in drop)
        kept=[i for i in range(len(raw)) if i not in drop]
        j={full:pos for pos,full in enumerate(kept)}
        reduced=public([raw[i] for i in kept])
        hidden=rd.scores(reduced,[[j[k] for k in episode.old]])[0]
        full=rd.scores(public(raw),[episode.old])[0]

        def choice(output,back):
            indices,p=output
            if not len(p):
                return (False,False,None,0.)
            k=int(p.argmax())
            i,tag=indices[k]
            role=int(raw[back[i]][3][tag])
            take=bool(p[k]>=.45)
            return bool(take and role==0),bool(take and role!=0),role,float(p[k])

        after=choice(full,list(range(len(raw))))
        before=choice(hidden,kept)
        assert after[0]==grades['locked']['right']
        result|=dict(motion=True,removed=n,hidden=before,full=after)
    return result


def main():
    p=ArgumentParser();p.add_argument('--workers',type=int,default=12);p.add_argument('--name',default='teacher_verified')
    a=p.parse_args()
    manifest=json.loads((HERE/(a.name+'_summary.json')).read_text())
    rows=[json.loads(x) for x in (HERE/(a.name+'_raw.jsonl')).read_text().splitlines()]
    assert len(rows)==manifest['n'] and all(len({x['ticks'] for x in r['score'].values()})==1 for r in rows)
    cases=scan.cases('validate' if manifest['split']=='develop' else 'test',manifest['n'],manifest['seed'])
    method='verified_teacher' if 'verified_teacher' in rows[0]['score'] else 'evidence_teacher'
    with ProcessPoolExecutor(a.workers) as pool:
        history=list(pool.map(recorded,((c,r,manifest['reader'],method) for c,r in zip(cases,rows,strict=True))))
    cases_with=[r for r in history if r['motion']]
    net=[int(r['full'][0])-int(r['hidden'][0]) for r in cases_with]
    result=dict(n=len(rows),layouts=len({r['layout'] for r in rows}),paired=effects(rows),
                mask=dict(n=len(net),locked_full=sum(r['full'][0] for r in cases_with),
                           locked_hidden=sum(r['hidden'][0] for r in cases_with),
                           correction=sum(d==1 for d in net),spoiled=sum(d==-1 for d in net),
                           ci95=interval(cases_with,net) if net else None),
                replayed_all_action_paths=True,rows=history)
    (HERE/(a.name+'_audit.json')).write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
