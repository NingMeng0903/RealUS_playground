"""Cross-score fixed, publicly planned and privileged real trajectories with the new public readers."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from motion_eval import HERE,Episode,scan,summarize

READERS=('pair','registered','evidence','joint')


def one(task):
    c,r=task
    ep=Episode(c)
    assert ep.layout==r['layout']
    by={}
    for k,record in r['score'].items():
        state=ep.start
        for stage,d in enumerate(record['decisions']):
            state=ep.go(state,stage,d['action'],ep.trial)
        ret={n:summarize(ep.score(state,n)) for n in READERS}
        assert ret['pair']==record['primary']
        by[k]=ret
    return dict(world=r['world'],layout=r['layout'],profile=r['profile'],arms=by)


def main():
    a=ArgumentParser()
    a.add_argument('--workers',type=int,default=13)
    opts=a.parse_args()
    design=json.loads((HERE/'teacher_verified_summary.json').read_text())
    with (HERE/'teacher_verified_raw.jsonl').open() as f:
        previous=[json.loads(line) for line in f]
    cases=scan.cases('test',design['n'],design['seed'])
    with ProcessPoolExecutor(opts.workers) as pool:
        rows=list(pool.map(one,zip(cases,previous,strict=True)))
    metrics=('both','chain','locked','wrong','artery')
    table={n:{name:{m:sum(r['arms'][name][n][m] for r in rows) for m in metrics}
                    for name in rows[0]['arms']} for n in READERS}
    (HERE/'cross_replay.json').write_text(json.dumps(dict(totals=table,rows=rows),separators=(',',':')))
    print(json.dumps(table),flush=True)


if __name__=='__main__':
    main()
