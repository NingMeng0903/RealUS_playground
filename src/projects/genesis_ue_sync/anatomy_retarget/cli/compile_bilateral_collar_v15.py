"""Compile both collar responses while preserving the frozen rest geometry.

This isolated correction is independently reviewable. It does not claim to
finish beta fitting, repair bone collisions, or compile a soft-material field.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

from ..consistent_runtime_v14 import load_compiled_subject, driver_rotation_maps_v14
from ..motion_response_v14 import BakedMotionResponseV14
from ..v8_artifacts import load_source_operator


def compile_bilateral_response(old, operator, calibration=None):
    from ..collar_response_v15 import make_bilateral_collar_pivot_response_asset_v15
    from ..translation_rebuild_v15 import rebuild_rest_world_jacobians_v15
    if old.corrector is not None:
        raise ValueError('old pose corrector was fitted under different response; refit it offline')
    if operator.runtime_digest(validate=False) != old.source_pack.operator_runtime_digest:
        raise ValueError('original shape and motion operator identities differ')
    if calibration is None and old.provenance.get('translation_transport') != 'motion_reference_axes':
        from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
        calibration = load_anatomical_calibration_v1(
            Path('outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1'),
            operator=operator)
    world_jacobians, translation_provenance = rebuild_rest_world_jacobians_v15(
        old, operator, calibration)
    effective = make_bilateral_collar_pivot_response_asset_v15(old.source_asset)
    # Look up anatomical sides explicitly; controller ordering differs between
    # the left and right shoulder chains in the authored rig.
    ids = [list(old.source_asset.source_bone_names).index(name)
           for name in ('Clavicle_Rot_L', 'Clavicle_Rot_R')]
    response = BakedMotionResponseV14.from_assets(old.source_asset, effective,
        provenance=dict(method='bilateral_collar_local_rotation_and_anatomical_pivot_v15',
                        controller_ids=ids, smplx_joint_ids=[13, 14],
                        offline_coupling_baked=True, anatomical_passed=False,
                        fit_data_used=False))
    bind = old.target_bind.copy()
    if old.provenance.get('shape_reference_kind') == 'frozen_operator_template':
        alignment = np.asarray(old.provenance.get('shape_root_alignment', np.eye(4)), dtype=float)
        if 'shape_root_alignment' not in old.provenance and not np.allclose(
                old.target_bind[0], old.reference_bind[0], atol=2e-6, rtol=0):
            raise ValueError('missing authored-shape root registration')
        for controller, joint in zip(ids, (13, 14)):
            bind[controller, :3, 3] = (alignment @ np.r_[operator.template_asset.rest_joints[joint], 1])[:3]
    elif old.provenance.get('shape_reference_kind') == 'materialized_subject':
        bind[ids, :3, 3] = old.source_asset.rest_joints[[13, 14]]
    else:
        raise ValueError('unknown rest shape authority')
    new_reference = np.asarray(effective.target_bind_global, dtype=np.float64)
    # Rebuild the spatial rest-field Jacobian first. Legacy V14 packages may
    # express their saved maps in shape-reference axes, so treating those maps
    # as motion-reference coordinates would silently preserve the old error.
    translation = bind[:, :3, :3].swapaxes(1, 2) @ world_jacobians @ new_reference[:, :3, :3]
    provenance = dict(old.provenance)
    provenance.update(rotation_transport='driver_axes',
        translation_transport='motion_reference_axes',
        translation_rebuild_v15=translation_provenance,
        bilateral_collar_correction_v15=dict(controller_ids=ids, smplx_joint_ids=[13,14],
            rest_refit_performed=False, target_rest_unchanged=True,
            calibrated_response_replaces_previous_response=True,
            anatomical_passed=False))
    return replace(old, reference_bind=new_reference, target_bind=bind,
                   translation_maps=translation,
                   rotation_maps=driver_rotation_maps_v14(new_reference, bind),
                   motion_response=response, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--operator', type=Path, default=Path('outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8'))
    parser.add_argument('--calibration', type=Path,
        default=Path('outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    old = load_compiled_subject(args.compiled)
    operator = load_source_operator(args.operator)
    from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
    calibration = load_anatomical_calibration_v1(args.calibration, operator=operator)
    new = compile_bilateral_response(old, operator, calibration)
    args.output.mkdir(parents=True)
    new.save(args.output/'compiled')
    reloaded = load_compiled_subject(args.output/'compiled')
    probes = [np.zeros((55,3))]
    for joint in (13,14):
        probe=np.zeros((55,3)); probe[joint]=[.1,.2,-.3]; probes.append(probe)
    exact = []
    for probe in probes:
        a, ga=new.apply_pose(probe, return_globals=True)
        b, gb=reloaded.apply_pose(probe, return_globals=True)
        exact.append(bool(np.array_equal(a,b) and np.array_equal(ga,gb)))
    if not all(exact): raise ValueError('compiled response is not reproducible after loading')
    if not np.array_equal(old.target_rest,new.target_rest): raise ValueError('neutral geometry changed')
    report=dict(kind='bilateral_collar_response_diagnostic_v15',
        input_compiled=str(args.compiled.resolve()),
        input_manifest_sha256=hashlib.sha256((args.compiled/'manifest.json').read_bytes()).hexdigest(),
        betas=new.betas.tolist(), target_rest_bitexact=True,
        source_weights_bitexact=bool(np.array_equal(old.source_asset.driver_weights,new.source_asset.driver_weights)),
        source_faces_bitexact=bool(np.array_equal(old.source_asset.faces,new.source_asset.faces)),
        translation_rebuild=new.provenance['translation_rebuild_v15'],
        saved_pose_replay_bitexact=all(exact), tested_contract_pose_count=len(probes),
        runtime_recompile=False, runtime_blender=False, runtime_optimization=False,
        beta_fit_completed=False, shared_soft_field_compiled=False,
        anatomical_passed=False, publishable=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
