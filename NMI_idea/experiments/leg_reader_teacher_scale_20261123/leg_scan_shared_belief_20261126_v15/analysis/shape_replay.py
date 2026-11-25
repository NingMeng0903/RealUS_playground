"""Re-render development scans after checked distal vessel warps on fixed poses."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import run_v15 as run

data = run.data
V14 = None


def init(atlas, out):
    global V14
    run.initialize(atlas, out, data.digest(out/'reader_freeze.json'), 'teach')
    sys.path.insert(1, str(data.V14))
    import experiment_v14 as prior
    prior.init_cpu(atlas,data.OLD)
    run._V14 = prior,data.read_json(data.V14/'run/graph.json')
    V14 = prior


def compare(case):
    original = data.old_roll(case)
    reports = {}
    for kind in ('unwarped','translation','deformation'):
        if kind == 'unwarped':
            layout, warp = original.live.scan.tape.layout, {}
        else:
            layout, warp = V14._transformed_layout(original,kind,data.ordinal(case))
            if layout is None:
                reports[kind] = {'invalid':warp}
                continue
        fixed = V14.fixed_pose_replay(original,layout)
        if run._signature(fixed)!=run._signature(original):
            raise RuntimeError('A shape replay changed the sampled poses or clock')
        old = data.deployment(fixed.read,data.policy_v13.READER,data.policy_v13.MODE)
        base = run._summary(fixed,old)
        if kind == 'unwarped' and base['selected_ids'] != data.policy_v13.grade(original)['selected_ids']:
            raise RuntimeError('The zero-warp replay changed the old same-pose delivery')
        p = run.identity.apply(fixed.read,run._CONTROLLER.reader,data.policy_v13.READER,data.policy_v13.MODE)
        reports[kind] = {'warp':warp,'v13':base, 'v14_terminal':run._v14_terminal(fixed,case,kind,base),
                         'v15_terminal':run._summary(replace(fixed,read=p))}
    return {'case':case.identifier,'profile':case.profile,'layout_id':original.live.scan.tape.layout.identifier,'geometry':reports}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path,default=data.DEFAULT_PACK)
    parser.add_argument('--out',type=Path,default=data.ROOT/'run')
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    cases = list(data.roster('reader_dev'))[:24]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn'),
                             initializer=init,initargs=(args.atlas,args.out)) as pool:
        rows = list(pool.map(compare,cases))
    summary = {}
    for k in ('unwarped','translation','deformation'):
        sp = [r['geometry'][k] for r in rows if 'invalid' not in r['geometry'][k]]
        summary[k] = {a:run.measures([r[a] for r in sp]) for a in ('v13','v14_terminal','v15_terminal')}
        summary[k]['invalid_layouts'] = len(rows)-len(sp)
        if k!='unwarped':
            summary[k]['median_realized'] = {a:float(np.median([r['warp'][a] for r in sp]))
                                             for a in ('r6_median_mm','r6_max_mm','max_rotation_degrees')}
    data.atomic_json(args.out/'shape_replay.json',{'study':'development only','summary':summary,'layouts':rows})
    print({k:{a:s['strict'] for a,s in g.items() if a in run.ARMS} for k,g in summary.items()},flush=True)


if __name__ == '__main__':
    main()
