"""Compile an explicit collar response and pivot change without rest refitting.

The neutral mesh stays bitexact. Both the source motion reference and the
target anatomical pivot are updated, then all local binds and inverse binds
are derived from those authorities. This is an ablation, not an acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import collar_response_v14
from ..consistent_runtime_v14 import load_compiled_subject, driver_rotation_maps_v14
from ..motion_response_v14 import BakedMotionResponseV14
from ..v8_artifacts import load_source_operator
from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
from .fit_consistent_arm_v14 import ArmMap, OP, CAL, signed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--geometry-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subject', choices=['213328', '213712'], required=True)
    parser.add_argument('--rotation-transport', choices=['preserve', 'driver_axes'], default='preserve',
                        help='Optionally combine the independently checked angular frame conversion')
    args = parser.parse_args()
    old = load_compiled_subject(args.compiled)
    if old.corrector is not None or old.motion_response is not None:
        raise ValueError('collar ablation requires a rest-fitted package without an already fitted pose response')
    op = load_source_operator(OP)
    if op.runtime_digest(validate=False) != old.source_pack.operator_runtime_digest:
        raise ValueError('source operator differs from original compile')
    effective = collar_response_v14.make_collar_pivot_response_asset_v14(old.source_asset)
    response = BakedMotionResponseV14.from_assets(old.source_asset, effective, provenance=dict(
        method='collar_local_rotation_and_joint13_pivot', controller_ids=[129],
        smplx_joint_ids=[13], offline_coupling_baked=True,
        anatomical_passed=False, fit_data_used=False))
    target_bind = old.target_bind.copy()
    if old.provenance.get('shape_reference_kind') == 'frozen_operator_template':
        # The first template-subject artifact preceded explicit root-alignment
        # provenance and used the template coordinates directly.
        alignment = np.asarray(old.provenance.get('shape_root_alignment', np.eye(4)), dtype=float)
        if 'shape_root_alignment' not in old.provenance and not np.allclose(
                old.target_bind[0], old.reference_bind[0], atol=2e-6, rtol=0):
            raise ValueError('legacy unregistered shape has a different root; recompile it first')
        target_bind[129, :3, 3] = (alignment @ np.r_[op.template_asset.rest_joints[13], 1])[:3]
    elif old.provenance.get('shape_reference_kind') == 'materialized_subject':
        target_bind[129, :3, 3] = old.source_asset.rest_joints[13]
    else:
        raise ValueError('unknown geometric reference for the collar anatomical pivot')
    provenance = dict(old.provenance)
    provenance['collar_pivot_ablation'] = dict(
        base_compiled_manifest_sha256=hashlib.sha256((args.compiled/'manifest.json').read_bytes()).hexdigest(),
        original_target_pivot_m=old.target_bind[129, :3, 3].tolist(),
        corrected_target_pivot_m=target_bind[129, :3, 3].tolist(),
        static_geometry_unchanged=True, rest_refit_performed=False,
        anatomical_passed=False)
    rotation_maps=old.rotation_maps
    if args.rotation_transport=='driver_axes':
        rotation_maps=driver_rotation_maps_v14(effective.target_bind_global,target_bind)
        provenance['rotation_transport']='driver_axes'
    compiled = replace(old, reference_bind=effective.target_bind_global,
                       target_bind=target_bind, motion_response=response, provenance=provenance,
                       rotation_maps=rotation_maps)
    args.output.mkdir(parents=True, exist_ok=False)
    compiled.save(args.output/'compiled')
    replay = load_compiled_subject(args.output/'compiled')
    cal = load_anatomical_calibration_v1(CAL, operator=op)
    arm_ids = ArmMap(old.source_asset, cal).all_ids
    source = old.source_asset
    rows = []
    for path in sorted(args.geometry_dir.glob(f'subject_{args.subject}_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        p = arrays['pose']
        before = old.apply_pose(p)
        after, globals_ = compiled.apply_pose(p, return_globals=True)
        after_reload, globals_reload = replay.apply_pose(p, return_globals=True)
        if not np.array_equal(after, after_reload) or not np.array_equal(globals_, globals_reload):
            raise ValueError(f'{path.stem}: package replay mismatch')
        if not np.any(p) and not np.array_equal(before, after):
            raise ValueError('a pivot-only response ablation altered neutral geometry')
        before_s = signed(before[arm_ids], arrays['skin_vertices'], arrays['skin_faces'])
        after_s = signed(after[arm_ids], arrays['skin_vertices'], arrays['skin_faces'])
        arrays.update(before_vertices=before, candidate_vertices=after,
                      source_global=compiled.source_globals(p),
                      candidate_controller_globals=globals_)
        np.savez_compressed(args.output/path.name, **arrays)
        rows.append(dict(pose=path.stem, before_arm_max_outside_mm=float(max(0, before_s.max())*1000),
                         after_arm_max_outside_mm=float(max(0, after_s.max())*1000),
                         arm_count_above_1mm=int(np.count_nonzero(after_s > .001)),
                         full_anatomy_max_change_mm=float(np.linalg.norm(after-before,axis=1).max()*1000),
                         save_load_vertices_and_globals_bitexact=True))
        print(rows[-1], flush=True)
    if not rows:
        raise ValueError('no explicit pose/skin geometry inputs found')
    report = dict(anatomical_passed=False, publishable=False, operational_evaluation_completed=True,
                  subject=args.subject, rest_refit_performed=False,
                  rotation_transport=args.rotation_transport,
                  source_weights_unchanged=bool(np.array_equal(source.driver_weights, compiled.source_asset.driver_weights)),
                  source_faces_unchanged=bool(np.array_equal(source.faces, compiled.source_asset.faces)),
                  target_rest_bitexact=bool(np.array_equal(old.target_rest, compiled.target_rest)),
                  source_manifest=str(args.compiled/'manifest.json'), metrics=rows,
                  response=response.provenance, provenance=provenance)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
