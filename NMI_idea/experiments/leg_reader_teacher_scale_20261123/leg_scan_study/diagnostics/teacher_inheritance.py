"""Can the old CT teacher's counterfactual improvements be chosen using public observations?"""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


def load(folder):
    states=[]
    for phase, file in [('teacher','ct_only_teacher.jsonl.gz'),('student','ct_only_student_failures.jsonl.gz')]:
        with gzip.open(folder/file, 'rt') as stream:
            for raw in stream:
                episode=json.loads(raw)
                groups=defaultdict(list)
                # On these four six-station training routes, sampled decision
                # points have distinct measured stations in each episode.
                for x,y in episode['counterfactuals']:
                    groups[x[24]].append((x,y))
                assert len(groups)==episode['counts']['reached_nodes']
                for rows in groups.values():
                    x,y=map(np.asarray,zip(*rows))
                    states.append((phase,episode['shape'],episode['seed'],x,y))
    return states


def summarize(rows):
    y=np.asarray([r[1] for r in rows]); pred=np.asarray([r[2] for r in rows]); best=np.asarray([r[3] for r in rows])
    gate=pred>.025
    return {'prefixes':len(rows),'any_branch_better_than_mpc':int((best>.025).sum()),
            'selected':int(gate.sum()),'selected_realized_gain_gt_margin':int(((y>.025)&gate).sum()),
            'selected_realized_gain_negative':int(((y<0.)&gate).sum()),
            'selected_realized_gain_zero_to_margin':int(((y>=0.)&(y<=.025)&gate).sum()),
            'gain_per_visited_prefix':float(np.mean(y*gate)),
            'hindsight_max_gain_per_prefix':float(np.mean(np.maximum(0.,best))),
            'positive_candidate_fraction':sum(r[4] for r in rows)/sum(r[5] for r in rows)}


def run(folder,out):
    all_states=load(folder)
    rows=[]; repeated=defaultdict(list)
    for shape in sorted(set(row[1] for row in all_states)):
        train=[r for r in all_states if r[1]!=shape]
        x=np.vstack([r[3] for r in train]); y=np.concatenate([r[4] for r in train])
        model=HistGradientBoostingRegressor(max_iter=115,max_leaf_nodes=15,min_samples_leaf=20,
                      l2_regularization=2.5, early_stopping=False,random_state=30301).fit(x,y)
        for phase,s,seed,features,gain in all_states:
            if phase!='student' or s!=shape:continue
            prediction=model.predict(features); j=int(prediction.argmax())
            rows.append((shape,float(gain[j]),float(prediction[j]),float(gain.max()),int((gain>.025).sum()),len(gain)))
            for xx,yy in zip(features,gain):
                key=(s,hashlib.sha256(np.ascontiguousarray(xx).tobytes()).digest())
                repeated[key].append((float(yy),seed))
    duplicates=[vv for vv in repeated.values() if len({s for _,s in vv})>=2]
    conflict=[vs for vs in duplicates if max(y for y,_ in vs)>.025 and min(y for y,_ in vs)<0]
    output={'cohort':'historical scaled CT student trajectories, any task; one CT transported to 7 beta shapes',
            'target':'recorded terminal utility advantage versus same-prefix complete MPC, one realized noise tape per candidate',
            'validation':'train on teacher + student prefixes of other six shapes, test on reached student prefixes of held shape; seven folds, no Blender',
            'null':'MPC continuation, advantage 0','all':summarize(rows),
            'per_held_shape':{str(s):summarize([r for r in rows if r[0]==s]) for s in range(7)},
            'exact_duplicates':{'definition':'bitwise same public feature-and-action vector in same held shape, from at least two different recorded seeds',
              'groups':len(duplicates),'groups_with_both_gain_and_loss':len(conflict),'evaluations_in_conflict_groups':sum(map(len,conflict))},
            'interpretation':'Hindsight ceiling uses the realized future. These prefixes were chosen with failure-biased sampling. They are diagnostic, not a deployment success estimate.'}
    out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:output[k] for k in ('all','exact_duplicates')}))

if __name__=='__main__':
    parser=ArgumentParser(description=__doc__); parser.add_argument('--data',type=Path,default=Path('/media/camp/PEI_T7/1_team meeting/atlas_ppt_build/assets/blender_topology_holdout/ct_student_scaling'));parser.add_argument('--out',type=Path,default=Path(__file__).with_name('teacher_inheritance.json'));args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.data,args.out)
