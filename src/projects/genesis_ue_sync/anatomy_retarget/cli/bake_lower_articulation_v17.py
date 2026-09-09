"""Bake a generic rule-grid leg response into a saved V17 subject."""
from pathlib import Path
import argparse,json,time
import numpy as np
from ..generic_lower_compile_v17 import load_lower_subject
from ..coupled_leg_articulation_v17 import bake_coupled_leg_articulation_v17
from ..v8_artifacts import load_source_operator
from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
from ..smplx_body_surface_v7 import load_smplx_model_v7,require_frozen_smplx_male_v7

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiled',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replace-articulation',action='store_true',help='compile a new response from the same immutable rest into a new output')
    p.add_argument('--extend-protocol',action='store_true',help='reuse saved bone-skin fits and fit the added universal rule poses')
    p.add_argument('--operator',type=Path,default=Path('outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8'))
    p.add_argument('--calibration',type=Path,default=Path('outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1'))
    p.add_argument('--model',type=Path,default=Path('ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.extend_protocol and a.replace_articulation:raise ValueError('choose replace or extend, not both')
    subject=load_lower_subject(a.compiled);operator=load_source_operator(a.operator)
    if a.replace_articulation:subject.leg_articulation=None
    cal=load_anatomical_calibration_v1(a.calibration,operator=operator)
    if cal.source_operator_digest!=subject.report['source_operator_digest']:raise ValueError('source identity mismatch')
    path,digest=require_frozen_smplx_male_v7(a.model);model=load_smplx_model_v7(path)
    start=time.monotonic()
    try:
        subject.leg_articulation=bake_coupled_leg_articulation_v17(subject,cal,model,
            progress=lambda row:print(json.dumps(row),flush=True),
            checkpoint_path=a.output/'fit_checkpoint.npz',reuse_saved_fits=a.extend_protocol)
    except Exception as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(status='bake_failed',
            error=str(exc),runtime_package_saved=False),indent=2)+'\n')
        raise
    subject.report=dict(subject.report,baked_leg_articulation=subject.leg_articulation.report,
                        articulation_bake_seconds=time.monotonic()-start,model_sha256=digest)
    subject.save(a.output/'compiled')
    replay=load_lower_subject(a.output/'compiled')
    from ..generic_lower_compile_v17 import fixed_lower_fit_poses
    checks=[]
    for name,pose in fixed_lower_fit_poses().items():
        identical=np.array_equal(subject.apply_pose(pose),replay.apply_pose(pose))
        if not identical:raise ValueError('saved articulation replay differs: '+name)
        checks.append(dict(pose=name,bitexact=True))
    (a.output/'report.json').write_text(json.dumps(dict(subject.report,saved_replay=checks),indent=2)+'\n')

if __name__=='__main__':main()
