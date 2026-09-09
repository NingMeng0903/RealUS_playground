"""Rebuild interpolation coefficients from saved rule-pose fits, without fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from ..generic_lower_compile_v17 import load_lower_subject, fixed_lower_fit_poses
from ..coupled_leg_articulation_v17 import (CoupledLegArticulationV17,
    interpolate_coupled_leg_values_v17, SCALE_DEG)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    subject=load_lower_subject(args.compiled)
    old=subject.leg_articulation
    if not isinstance(old,CoupledLegArticulationV17):
        raise ValueError('saved coupled rule-pose response required')
    values=np.zeros((len(old.centers),12));seen=set()
    for record in old.report['fits']:
        index=int(record['sample'])
        if index in seen or not 0<index<len(values):raise ValueError('invalid saved sample index')
        fits=record['fits']
        if [row['side'] for row in fits]!=['L','R']:raise ValueError('invalid saved side order')
        values[index]=np.deg2rad(np.asarray([row['angles_deg'] for row in fits])).reshape(12)
        seen.add(index)
    if seen!=set(range(1,len(values))) or np.any(old.centers[0]):
        raise ValueError('incomplete rule-pose fit archive')
    field=interpolate_coupled_leg_values_v17(old.centers*SCALE_DEG,values,
        old.head_centers_local,old.report)
    subject.leg_articulation=field
    subject.report=dict(subject.report,baked_leg_articulation=field.report,
        interpolation_repacked_without_refitting=True,
        interpolation_source_manifest_sha256=hashlib.sha256((args.compiled/'manifest.json').read_bytes()).hexdigest())
    subject.save(args.output/'compiled')
    loaded=load_lower_subject(args.output/'compiled');checks=[]
    for name,pose in fixed_lower_fit_poses().items():
        exact=np.array_equal(subject.apply_pose(pose),loaded.apply_pose(pose))
        if not exact:raise ValueError('saved replay differs: '+name)
        checks.append(dict(pose=name,bitexact=True))
    (args.output/'report.json').write_text(json.dumps(dict(subject.report,saved_replay=checks),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),training_reconstruction_max_deg=
        field.report['training_reconstruction_max_deg'])),flush=True)


if __name__=='__main__':main()
