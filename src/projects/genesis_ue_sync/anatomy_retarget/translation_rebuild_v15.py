"""Reconstruct rest-field translation bases before changing motion references.

Legacy numeric maps are never interpreted as motion-reference maps by default.
The supported legacy fit is reconstructed from saved parameters, without fitting,
and its complete neutral geometry and bind must reproduce the saved package.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .consistent_runtime_v14 import (
    _digest, _rigid, compile_subject, original_shape_reference_v14,
)


def _world_jacobians(compiled):
    reference = _rigid(compiled.reference_bind, len(compiled.parents), 'motion reference')
    target = _rigid(compiled.target_bind, len(compiled.parents), 'target bind')
    maps = np.asarray(compiled.translation_maps, dtype=np.float64)
    if maps.shape != (len(reference), 3, 3) or not np.isfinite(maps).all():
        raise ValueError('invalid translation maps')
    # Bind rotations are stored with finite precision. Invert rather than
    # transpose when recovering J so the saved map is reconstructed exactly.
    jac = np.linalg.solve(target[:, :3, :3].swapaxes(1, 2), maps)
    jac = jac @ np.linalg.inv(reference[:, :3, :3])
    if np.any(np.linalg.det(jac) <= 0) or not np.isfinite(jac).all():
        raise ValueError('rest Jacobian is singular, reflected or non-finite')
    return jac


def rebuild_rest_world_jacobians_v15(old, operator, calibration):
    """Return authenticated spatial rest Jacobians and reconstruction evidence."""
    identity = operator.runtime_digest(validate=False)
    if identity != old.source_pack.operator_runtime_digest:
        raise ValueError('source operator identity mismatch')
    provenance = old.provenance
    basis = provenance.get('translation_transport')
    if basis == 'motion_reference_axes':
        return _world_jacobians(old), dict(
            method='recover_spatial_jacobian_from_declared_motion_reference_axes',
            source_operator_digest=identity, previous_translation_transport=basis,
            runtime_fit=False, world_jacobian_reconstructed=True)
    if basis not in (None, 'legacy_shape_reference_axes'):
        raise ValueError('unknown translation coordinate basis')
    if provenance.get('method') != 'connected_left_arm_caps_v14':
        raise ValueError('legacy translation basis cannot be certified for this fit method')
    if old.corrector is not None:
        raise ValueError('legacy reconstruction requires a package without a pose corrector')
    if calibration is None:
        raise ValueError('legacy translation reconstruction requires frozen calibration')
    calibration.validate()
    if calibration.source_operator_digest != identity:
        raise ValueError('calibration and source operator identities differ')
    if provenance.get('rest_field_application_count') != 1:
        raise ValueError('unknown legacy rest-field composition')

    from .cli.fit_consistent_arm_v14 import ArmMap

    kind = provenance.get('shape_reference_kind')
    if kind == 'frozen_operator_template':
        if provenance.get('shape_operator_runtime_digest') != identity:
            raise ValueError('legacy geometric operator identity mismatch')
        original = operator.template_asset
        vertices, bind, alignment = original_shape_reference_v14(operator, old.source_asset)
        recorded_alignment = np.asarray(provenance.get('shape_root_alignment', np.eye(4)), dtype=float)
        # Earliest template packages predate the alignment field and used the
        # template unchanged. Accept that history only when registration is
        # actually identity; full rest/bind reconstruction below still applies.
        if 'shape_root_alignment' not in provenance and not np.allclose(
                alignment, np.eye(4), atol=1e-10, rtol=0):
            raise ValueError('unrecorded nonidentity legacy shape root alignment')
        if recorded_alignment.shape != (4, 4) or not np.allclose(
                alignment, recorded_alignment, atol=1e-10, rtol=0):
            raise ValueError('legacy shape root alignment cannot be reproduced')
        asset = replace(original, vertices_rest=vertices, target_rest_global=bind)
        shape_operator = operator
    elif kind == 'materialized_subject':
        original = old.source_asset
        asset = original
        bind = np.asarray(asset.target_bind_global, dtype=float).copy()
        alignment = np.eye(4)
        shape_operator = None
    else:
        raise ValueError('unknown legacy shape reference')
    for key, value in (
        ('shape_reference_vertices_digest', original.vertices_rest),
        ('shape_reference_bind_digest', original.target_bind_global),
    ):
        if provenance.get(key) != _digest(value):
            raise ValueError(f'legacy {key} cannot be certified')

    # Only explicitly declared collar pivot revisions may differ from the
    # authored geometric reference. Their anatomical positions are recomputed.
    revised = set()
    if 'collar_pivot_ablation' in provenance:
        revised.add('Clavicle_Rot_L')
    bilateral = provenance.get('bilateral_collar_correction_v15')
    if bilateral is not None:
        declared = bilateral.get('controller_ids', [])
        expected = [list(asset.source_bone_names).index(n)
                    for n in ('Clavicle_Rot_L', 'Clavicle_Rot_R')]
        if list(declared) != expected:
            raise ValueError('unknown bilateral pivot declaration')
        revised.update(('Clavicle_Rot_L', 'Clavicle_Rot_R'))
    for name, joint in (('Clavicle_Rot_L', 13), ('Clavicle_Rot_R', 14)):
        if name not in revised:
            continue
        controller = list(asset.source_bone_names).index(name)
        point = (alignment @ np.r_[original.rest_joints[joint], 1.0])[:3]
        if not np.allclose(point, old.target_bind[controller, :3, 3], atol=1e-7, rtol=0):
            raise ValueError('saved collar pivot differs from its declared anatomical position')
        bind[controller, :3, 3] = point
    asset = replace(asset, target_rest_global=bind)
    roll = bool(provenance.get('rest_roll_enabled', False))
    elbow_wrist = np.asarray(provenance.get('elbow_wrist_delta_mm'), dtype=float)
    cap_offset = np.asarray(provenance.get('forearm_cap_offset_mm'), dtype=float)
    if elbow_wrist.shape != (2, 3) or cap_offset.shape != (3,):
        raise ValueError('saved rest-fit parameters are incomplete')
    parameters = np.r_[elbow_wrist.ravel(), cap_offset]
    if roll:
        angles = np.asarray(provenance.get('rest_roll_deg'), dtype=float)
        if angles.shape != (2,):
            raise ValueError('saved rest-roll parameters are incomplete')
        parameters = np.r_[parameters, angles]
    builder = ArmMap(asset, calibration, motion_reference_bind=old.reference_bind,
                     rest_roll=roll)
    config = replace(builder.config(parameters), shape_reference_operator=shape_operator,
                     shape_reference_bind=bind, motion_response=old.motion_response)
    rebuilt = compile_subject(old.betas, old.source_pack, config)
    rest_error = float(np.max(np.abs(rebuilt.target_rest - old.target_rest)))
    bind_error = float(np.max(np.abs(rebuilt.target_bind - old.target_bind)))
    reference_error = float(np.max(np.abs(rebuilt.reference_bind - old.reference_bind)))
    if rest_error > 1e-7 or bind_error > 1e-7 or reference_error > 1e-7:
        raise ValueError('saved rest fit does not reproduce geometry/bind/reference: '
                         f'{rest_error:g}, {bind_error:g}, {reference_error:g}')
    jac = _world_jacobians(rebuilt)
    return jac, dict(
        method='reconstruct_saved_arm_rest_field_without_refitting',
        source_operator_digest=identity, calibration_domain_digest=calibration.fixed_domain_digest,
        previous_translation_transport=basis or 'legacy_unlabelled',
        saved_parameter_sha256=_digest(parameters), world_jacobian_sha256=_digest(jac),
        rest_max_abs_error_m=rest_error, bind_max_abs_error=bind_error,
        reference_max_abs_error=reference_error, world_jacobian_reconstructed=True,
        runtime_fit=False, optimization_performed=False)
