"""Fit material-coordinate uncertainty on training anatomy and freeze the reader on development tracks."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np

from registration import HERE, OLDER, PROGRAMS, chart, key_groups, phase_stations, posterior, public, stock
from tracker.tracker import noise_fraction
sys.path.insert(0, str(OLDER/'sequence'))
from data import COUNT, SEEDS, gather, scan

PRIOR_SEED = 674113872
PRIORS_N = 320
DEV_NAME = 'develop'
EXTRA_SEED = 350067921
WEIGHTS = (.25, .50, .75, 1.)
NAMES = ('band1','band2','band3','entry','fresh')


def training_geometry(case):
    env = scan.ref.reference.load_environment(str(scan.ATLAS),case.shape)
    from environment import spec_for
    _, trial, _ = env.start(case,spec_for(env.scanner,case))
    base = env.scanner.surface.priors[case.source]
    target = trial.layout.centers
    layout = sha256(np.nan_to_num(target).tobytes() + bytes([case.shape])).hexdigest()
    return layout,case.profile,target-base


def fit_prior(pool):
    rows = list(pool.map(training_geometry,scan.cases('train',PRIORS_N,PRIOR_SEED),chunksize=1))
    table = []
    for profile in ('ordinary','stress'):
        field = np.asarray([x for _,y,x in rows if y==profile])
        cuts = phase_stations(np.arange(field.shape[1]))
        global_sd = np.sqrt(np.nanmean(np.minimum(field**2,25.),axis=(0,1)))
        need = []
        for k in range(14):
            here = field[:,cuts==k]
            n = np.isfinite(here).sum(axis=(0,1))
            square = np.nansum(np.minimum(here**2,36.),axis=(0,1))
            value = np.sqrt((square + 30.*global_sd**2) / (n+30.))
            need.append(np.nan_to_num(value,nan=1.).clip(.31,3.8))
        table.append(need)
    return np.asarray(table), {r[0] for r in rows}


def setup(case, split, table):
    env,t,item,frames,cut,spans = gather(case,split)
    surface = env.scanner.surface
    anonymous = public(frames)
    prior = [i for i in range(cut) if item[1]-3.-1.e-8 <= frames[i][0] <= item[1]+2.+1.e-8]
    data = {}
    fraction = None
    for j,(begin,end) in enumerate(spans):
        groups = [list(range(begin,end))]
        names = [NAMES[j]]
        if j==2:
            groups.extend((prior,list(range(cut,end))))
            names.extend(NAMES[3:])
        history = anonymous[:end]
        prepared = chart(surface).prepare(history)
        fraction = noise_fraction(prepared.quality,prepared.sectors,prepared.counts)
        keys = key_groups(prepared,groups)
        current = stock(surface).scores(history,groups)
        alternative = {name:posterior(surface, prepared, table, plan)[:,:,0] for name,plan in PROGRAMS.items()}
        for name,ix,(old_ix,pr) in zip(names,keys,current,strict=True):
            np.testing.assert_array_equal(ix,old_ix)
            lab = np.asarray([frames[i][3][k] for i,k in ix],dtype=int)
            data[name] = dict(y=lab,old=pr,fraction=fraction,reg={k:p[ix[:,0],ix[:,1]] for k,p in alternative.items()})
    layout = sha256(np.nan_to_num(t.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
    return dict(world=int(case.world_seed),layout=layout,shape=case.shape,profile=case.profile,
                fraction=fraction,data=data)


def score(p,y):
    if not len(y):
        return (None,0.,None)
    k = int(np.argmax(p))
    return (int(y[k]),float(p[k]),float(np.mean((p-(y==0))**2)))


def decisions(row, name, weight, stress_weight=None):
    out = {}
    for k,d in row['data'].items():
        fraction = d['fraction']
        mix = weight if stress_weight is None else weight*(1.-fraction) + stress_weight*fraction
        old = d['old']
        p = old if not mix else (1.-mix)*old + mix*d['reg'][name]
        out[k] = score(p,d['y'])
    return out


def counts(choices,threshold=.45):
    good = [choices[k][0]==0 and choices[k][1]>=threshold for k in NAMES]
    wrong = [choices[k][0] not in (0,None) and choices[k][1]>=threshold for k in NAMES]
    artery = [choices[k][0]==1 and choices[k][1]>=threshold for k in NAMES]
    return np.array([int(all(good[3:])),int(all(good[:3])),sum(good[:3]),sum(wrong[3:]),sum(artery[3:]),
                     sum(wrong[:3]),sum(good[3:]),next((i*5 for i,g in enumerate(good[:3]) if not g),15)],dtype=float)


N_METRICS = ('both','chain','bands','wrong','artery','band_wrong','images','prefix_mm')


def summary(rows, spec, *, threshold=.45):
    n = len(rows)
    ret = [counts(decisions(r,spec['program'],spec['geometry_weight'],spec.get('stress_weight')),threshold) for r in rows]
    tot = dict(zip(N_METRICS,np.sum(ret,axis=0).astype(int).tolist()))
    return tot | {'n':n,'mean_prefix_mm':tot['prefix_mm']/n}


def objective(result):
    return (2.*result['both'] + 1.35*result['chain'] + .30*result['bands'] - .63*result['wrong']
            - .23*result['artery'] - .11*result['band_wrong']) / result['n']


def choose(rows):
    tests=[dict(program='atlas',geometry_weight=0.)]
    tests += [dict(program=k,geometry_weight=x,stress_weight=y) for k in PROGRAMS for x in WEIGHTS for y in (0.,*WEIGHTS)]
    evaluated=[(t,summary(rows,t)) for t in tests]
    best=max(evaluated,key=lambda item: objective(item[1]))
    print('development',best, 'baseline', evaluated[0][1],flush=True)
    return best[0], [t | {'score':s,'utility':objective(s)} for t,s in evaluated]


def evaluate(rows,selected):
    base = dict(program='atlas',geometry_weight=0.)
    arms = {'pair':base,'registered':selected} | {name:dict(program=name,geometry_weight=1.) for name in PROGRAMS}
    each = {name:summary(rows,spec) for name,spec in arms.items()}
    print('measured', {key:v for key,v in each.items() if key in ('pair','registered')},flush=True)
    out_rows=[]
    for r in rows:
        out_rows.append({k:r[k] for k in ('world','layout','shape','profile')} |
                        {'scores':{k:decisions(r,s['program'],s['geometry_weight'],s.get('stress_weight'))
                                   for k,s in arms.items()}})
    return dict(summary=each, rows=out_rows)


def collection(pool, split, table, count=None, extra_seed=None):
    cases = scan.cases('validate' if split=='develop' else 'test',count or COUNT[split],
                      extra_seed or SEEDS[split])
    return list(pool.map(setup_wrap,((case,split,table) for case in cases),chunksize=1))


def setup_wrap(args):
    return setup(*args)


def main():
    a = ArgumentParser()
    a.add_argument('--workers',type=int,default=12)
    a.add_argument('--evaluate',nargs='+',choices=['archive','fresh','public','extra'])
    a.add_argument('--only-evaluate',action='store_true')
    args = a.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        if args.only_evaluate:
            config=json.loads((HERE/'reader_config.json').read_text())
            table=np.asarray(config['prior_sd_mm'])
            used=set(config['prior_layout_ids']+config['develop_layout_ids'])
            chosen=config['selected']
        else:
            table, layouts = fit_prior(pool)
            print('fitted prior',len(layouts), np.round(table[:,11,:2],2).tolist(),flush=True)
            dev = [r for r in collection(pool,DEV_NAME,table) if r['layout'] not in layouts]
            chosen, records = choose(dev)
            dev_ids = {r['layout'] for r in dev}
            used = layouts | dev_ids
            config = dict(prior_seed=PRIOR_SEED,prior_worlds=PRIORS_N,prior_layout_ids=sorted(layouts),
                          prior_sd_mm=table.tolist(),development_worlds=len(dev),develop_layout_ids=sorted(dev_ids),
                          objective='correct album and visible continuation with a false-identity penalty',
                          selected=chosen,development=records)
            (HERE/'reader_config.json').write_text(json.dumps(config,indent=2))
        for s in args.evaluate or []:
            print('evaluation',s,flush=True)
            batch = collection(pool,'public',table,96,EXTRA_SEED) if s=='extra' else collection(pool,s,table)
            if any(r['layout'] in used for r in batch):
                raise ValueError('Geometry from the test set overlaps training or tuning')
            result = evaluate(batch,chosen)
            (HERE/f'reader_{s}.json').write_text(json.dumps(result,separators=(',',':')))


if __name__ == '__main__':
    main()
