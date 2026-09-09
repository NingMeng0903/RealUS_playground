"""Compile one target beta with the same rule-generated lower-chain fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from ..generic_lower_compile_v17 import (
    compile_lower_subject, load_lower_subject, make_canonical_motion_reference,
    fixed_lower_fit_poses,
)
from ..consistent_runtime_v14 import load_compiled_subject
from ..v8_artifacts import load_source_operator
from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
from ..smplx_body_surface_v7 import load_smplx_model_v7, require_frozen_smplx_male_v7
from .run_material_matrix_v13 import _pose_joints_and_skin
from .export_capture_joint_review_v13 import _tissue_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    beta = parser.add_mutually_exclusive_group(required=True)
    beta.add_argument('--beta-values', type=float, nargs=10)
    beta.add_argument('--betas-npz', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--save-reference', type=Path)
    parser.add_argument('--operator', type=Path, default=Path('outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8'))
    parser.add_argument('--calibration', type=Path, default=Path('outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1'))
    parser.add_argument('--model', type=Path, default=Path('ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-evaluations', type=int, default=1200)
    parser.add_argument('--rest-fit-poses',nargs='+',help='fixed protocol for rest fit; articulation is baked separately')
    parser.add_argument('--bake-articulation',action='store_true',help='compile fixed leg pose response in the same subject package')
    parser.add_argument('--export-fit-poses', nargs='*', default=['tpose','knees_90','hips_60_knees_90'])
    parser.add_argument('--validation-geometry', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.betas_npz:
        with np.load(args.betas_npz, allow_pickle=False) as data:
            key = 'betas' if 'betas' in data.files else 'shapes'
            betas = np.asarray(data[key], dtype=np.float64).reshape(-1)
        beta_source = dict(path=str(args.betas_npz.resolve()),
                           sha256=hashlib.sha256(args.betas_npz.read_bytes()).hexdigest())
    else:
        betas = np.asarray(args.beta_values, dtype=np.float64)
        beta_source = dict(values=betas.tolist())
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise ValueError('exactly ten finite target beta values required; extra dimensions are not discarded')
    operator = load_source_operator(args.operator)
    calibration = load_anatomical_calibration_v1(args.calibration, operator=operator)
    reference = load_compiled_subject(args.reference) if args.reference else make_canonical_motion_reference(operator)
    origin = np.asarray(operator.mechanism_coefficients['unified_fit.beta_origin'],dtype=np.float32).reshape(10)
    if (reference.source_pack.operator_runtime_digest != operator.runtime_digest(validate=False) or
            not np.array_equal(reference.betas.astype(np.float32), origin)):
        raise ValueError('reference identity differs from the canonical operator origin')
    if args.save_reference:
        if args.save_reference.exists():
            raise FileExistsError(args.save_reference)
        reference.save(args.save_reference)
    model_path, model_hash = require_frozen_smplx_male_v7(args.model)
    model = load_smplx_model_v7(model_path)
    args.output.mkdir(parents=True)
    progress_path = args.output/'progress.jsonl'

    def progress(row):
        with progress_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if 'evaluation' in row:
            print('fit', row['evaluation'], 'objective', round(row['objective'], 4), flush=True)
        else:
            print(json.dumps(row,allow_nan=False),flush=True)

    compiled = compile_lower_subject(betas, reference, calibration, model,
                                     max_evaluations=args.max_evaluations, progress=progress,
                                     rest_fit_poses=args.rest_fit_poses,bake_articulation=args.bake_articulation)
    compiled.report.update(beta_source=beta_source,
        model_sha256=model_hash,
        source_operator_digest=operator.runtime_digest(validate=False))
    compiled.save(args.output/'compiled')
    loaded = load_lower_subject(args.output/'compiled')
    geometry = args.output/'geometry'; geometry.mkdir()
    probes = fixed_lower_fit_poses()
    validations = {name: (pose, np.zeros(3)) for name, pose in probes.items()}
    for path in args.validation_geometry:
        with np.load(path, allow_pickle=False) as data:
            name = 'validation_'+path.stem
            validations[name] = (np.asarray(data['pose'], dtype=np.float64), np.asarray(data['transl'], dtype=np.float64))
    asset = reference.source_asset
    rows = []
    for name, (pose, transl) in validations.items():
        try:
            vertices = compiled.apply_pose(pose, transl)
        except ValueError as exc:
            rows.append(dict(name=name,status='unsupported_or_evaluation_error',error=str(exc),
                             saved_replay_bitexact=None,geometry_exported=False))
            print('replay unsupported',name,str(exc),flush=True)
            continue
        replay = loaded.apply_pose(pose, transl)
        if not np.array_equal(vertices, replay):
            raise ValueError('saved runtime pose differs: '+name)
        before = reference.apply_pose(pose, transl)+np.asarray(compiled.report['target_root_shift_m'])
        exported = name not in probes or name in args.export_fit_poses
        if exported:
            skin, faces, joints = _pose_joints_and_skin(model, betas=betas, pose=pose)
            np.savez_compressed(geometry/(name+'.npz'), before_vertices=before,
                candidate_vertices=vertices, source_vertices=before, faces=asset.faces,
                skin_vertices=skin+transl, skin_faces=faces, smplx_joints=joints+transl,
                pose=pose, transl=transl, vertex_tissue=_tissue_codes(asset),
                mesh_names=np.asarray(asset.source_mesh_names), mesh_ranges=asset.source_vertex_ranges,
                mesh_tissues=np.asarray(asset.source_tissues), target_betas=betas)
        rows.append(dict(name=name, saved_replay_bitexact=True,
                         geometry_exported=exported,
                         maximum_vertex_change_mm=float(np.linalg.norm(vertices-before, axis=1).max()*1000)))
        print('replay checked', name, 'exported', exported, flush=True)
    report = dict(compiled.report, saved_replay=rows, completed=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
