"""Export saved V17 anatomy and target-beta skin for supplied pose NPZ files.

Only theta/transl are read from the supplied geometry. Its subject shape and
skin are never reused. This command does not compile or optimize the subject.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from ..generic_lower_compile_v17 import load_lower_subject
from ..consistent_runtime_v14 import load_compiled_subject
from ..smplx_body_surface_v7 import load_smplx_model_v7, require_frozen_smplx_male_v7
from .run_material_matrix_v13 import _pose_joints_and_skin
from .export_capture_joint_review_v13 import _tissue_codes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-rest-only',action='store_true',
                        help='evaluate a V17 reference without its optional articulation field')
    parser.add_argument('--poses',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl'))
    args=parser.parse_args()
    if any(not path.is_file() for path in args.poses):
        raise FileNotFoundError('missing pose inputs: '+str([str(p) for p in args.poses if not p.is_file()]))
    if len({path.stem for path in args.poses}) != len(args.poses):
        raise ValueError('duplicate pose stems')
    if args.output.exists():
        raise FileExistsError(args.output)
    subject=load_lower_subject(args.compiled)
    reference_manifest=json.loads((args.reference/'manifest.json').read_text())
    if reference_manifest.get('schema')=='CompiledLowerSubjectV17':
        reference=load_lower_subject(args.reference)
        if args.reference_rest_only:reference.leg_articulation=None
        if not np.array_equal(reference.target_betas,subject.target_betas):
            raise ValueError('comparison subject has a different target beta')
        reference_runtime=reference.runtime;reference_shift=np.zeros(3)
    else:
        reference=load_compiled_subject(args.reference)
        reference_runtime=reference;reference_shift=np.asarray(subject.report['target_root_shift_m'])
    if reference_runtime.source_pack.operator_runtime_digest!=subject.report['source_operator_digest']:
        raise ValueError('comparison reference has a different source identity')
    model_path,model_sha=require_frozen_smplx_male_v7(args.model)
    model=load_smplx_model_v7(model_path)
    asset=subject.runtime.source_asset
    args.output.mkdir(parents=True)
    records=[]
    for path in args.poses:
        destination=args.output/(path.stem+'.npz')
        if destination.exists():
            raise ValueError('duplicate pose stem: '+path.stem)
        with np.load(path,allow_pickle=False) as data:
            pose=np.asarray(data['pose'],dtype=np.float64)
            transl=np.asarray(data['transl'],dtype=np.float64)
        try:
            candidate=subject.apply_pose(pose,transl)
        except ValueError as exc:
            records.append(dict(pose_source=str(path.resolve()),status='unsupported_or_evaluation_error',
                error=str(exc),geometry_exported=False))
            continue
        before=reference.apply_pose(pose,transl)+reference_shift
        skin,faces,joints=_pose_joints_and_skin(model,betas=subject.target_betas,pose=pose)
        np.savez_compressed(destination,before_vertices=before,source_vertices=before,
            candidate_vertices=candidate,faces=asset.faces,skin_vertices=skin+transl,
            skin_faces=faces,smplx_joints=joints+transl,pose=pose,transl=transl,
            vertex_tissue=_tissue_codes(asset),mesh_names=np.asarray(asset.source_mesh_names),
            mesh_ranges=asset.source_vertex_ranges,mesh_tissues=np.asarray(asset.source_tissues),
            target_betas=subject.target_betas)
        records.append(dict(pose_source=str(path.resolve()),
            pose_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),output=destination.name))
    (args.output/'manifest.json').write_text(json.dumps(dict(
        compiled=str(args.compiled.resolve()),target_betas=subject.target_betas.tolist(),
        reference=str(args.reference.resolve()),reference_rest_only=args.reference_rest_only,
        model_sha256=model_sha,source_skin_reused=False,runtime_fit=False,
        runtime_rebind=False,runtime_blender=False,anatomical_passed=False,poses=records),indent=2)+'\n')


if __name__=='__main__':
    main()
