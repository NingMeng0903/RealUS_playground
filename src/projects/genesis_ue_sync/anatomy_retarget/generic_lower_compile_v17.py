"""Offline lower-chain station fitting using one rule for every target beta.

The V14 child is an immutable motion reference. Its beta labels that reference,
not the target person. The V17 envelope stores the actual target beta separately.
This first stage adapts the lower chains; it is not a whole-body acceptance gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from .chain_rest_fit_v1 import _global_to_local
from .consistent_runtime_v14 import load_compiled_subject
from .pose_map_v1 import _fk
from .sparse_lbs_v14 import SparseLBSV14


def _beta(values):
    beta = np.asarray(values, dtype=np.float64)
    if beta.shape != (10,) or not np.isfinite(beta).all():
        raise ValueError('target betas must be exactly ten finite values')
    return beta.copy()


def fixed_lower_fit_poses():
    """Subject-independent moderate flexion protocol, no recorded test frames."""
    poses = {'tpose': np.zeros((55, 3), dtype=np.float64)}
    for degrees in (30, 60, 90):
        pose = np.zeros((55, 3), dtype=np.float64)
        pose[[4, 5], 0] = np.deg2rad(degrees)
        poses[f'knees_{degrees}'] = pose
    for hip, knee in ((30, 45), (60, 90)):
        pose = np.zeros((55, 3), dtype=np.float64)
        pose[[1, 2], 0] = -np.deg2rad(hip)
        pose[[4, 5], 0] = np.deg2rad(knee)
        poses[f'hips_{hip}_knees_{knee}'] = pose
    return poses


def make_canonical_motion_reference(operator):
    """Build one authored reference, independent of the requested target beta."""
    from .v8_artifacts import materialize_subject
    from .consistent_runtime_v14 import compile_subject, CompileConfigV14, original_shape_reference_v14
    from .collar_response_v15 import make_bilateral_collar_pivot_response_asset_v15
    from .motion_response_v14 import BakedMotionResponseV14

    if 'unified_fit.beta_origin' not in operator.mechanism_coefficients:
        raise ValueError('canonical source has no explicit beta origin')
    origin = _beta(np.asarray(operator.mechanism_coefficients['unified_fit.beta_origin']).reshape(-1))
    pack = materialize_subject(operator, betas=origin, gender='male')
    effective = make_bilateral_collar_pivot_response_asset_v15(pack.rigged_asset)
    response = BakedMotionResponseV14.from_assets(pack.rigged_asset, effective,
        provenance=dict(method='canonical_bilateral_reference_v17', fit_data_used=False,
                        anatomical_passed=False))
    _vertices, bind, alignment = original_shape_reference_v14(operator, pack.rigged_asset)
    for side, joint in (('L', 13), ('R', 14)):
        index = list(pack.rigged_asset.source_bone_names).index(f'Clavicle_Rot_{side}')
        bind[index, :3, 3] = (alignment @ np.r_[operator.template_asset.rest_joints[joint], 1])[:3]
    return compile_subject(origin, pack, CompileConfigV14(
        shape_reference_operator=operator, shape_reference_bind=bind, motion_response=response,
        provenance=dict(method='canonical_reference_v17', target_beta_used=False)))


class SkinFitProxy:
    """Triangle distance with winding sign; open skin is not a solid certificate.

    A closest triangle's normal is insufficient for concave surfaces or points
    whose closest point lies on a triangle edge. In the authored SMPL-X skin it
    misclassified deep internal nerves as grossly outside. Winding evaluates
    the whole surface instead of treating that one face as a half-space.
    """
    def __init__(self, vertices, faces):
        import igl
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.tree = igl.AABB()
        self.tree.init(self.vertices, self.faces)

    def signed(self, points):
        import igl
        points = np.ascontiguousarray(points, dtype=np.float64)
        squared, _faces, _closest = self.tree.squared_distance(self.vertices, self.faces, points)
        winding = igl.fast_winding_number(self.vertices, self.faces, points)
        sign = np.where(np.abs(winding) > .5, -1., 1.)
        return sign * np.sqrt(np.maximum(squared, 0))


def _fit_vertex_ids(reference):
    asset = reference.source_asset
    names = list(asset.source_bone_names)
    descendants = []
    roots = [names.index(f'Femur_Rot_{side}') for side in ('L', 'R')]
    for i in range(len(names)):
        parent = i
        while parent >= 0 and parent not in roots:
            parent = int(reference.parents[parent])
        if parent in roots:
            descendants.append(i)
    mass = np.sum(reference.weights * np.isin(reference.indices, descendants), axis=1)
    ids, bones = [], []
    for name, tissue, (start, stop) in zip(asset.source_mesh_names, asset.source_tissues,
                                         asset.source_vertex_ranges):
        selected = np.arange(int(start), int(stop))[mass[int(start):int(stop)] > .05]
        if not len(selected):
            continue
        is_bone = str(tissue).lower() == 'bone'
        count = 600 if str(name).startswith('Femur_') else 350 if str(name).startswith('Tibia_') else 100 if is_bone else 220
        selected = selected[np.linspace(0, len(selected)-1, min(count, len(selected))).astype(int)]
        ids.extend(selected.tolist()); bones.extend([is_bone] * len(selected))
    return np.asarray(ids, dtype=np.int64), np.asarray(bones, dtype=bool)


@dataclass
class CompiledLowerSubjectV17:
    target_betas: np.ndarray
    runtime: object
    report: dict
    leg_articulation: object = None

    def apply_pose(self, pose55, transl=None, *, return_globals=False):
        if self.leg_articulation is not None:
            from .lower_chain_pose_fit_v17 import articulated_leg_globals
            twists=self.leg_articulation.evaluate(pose55)
            if not np.any(twists):
                return self.runtime.apply_pose(pose55,transl,return_globals=return_globals)
            global_=articulated_leg_globals(self.runtime,pose55,twists,self.leg_articulation.head_centers_local)
            vertices=self.runtime._lbs(global_@self.runtime.target_inverse)
            if transl is not None:
                t=np.asarray(transl,dtype=np.float64)
                if t.shape!=(3,) or not np.isfinite(t).all():raise ValueError('three finite translation values required')
                vertices=vertices+t;global_=global_.copy();global_[:,:3,3]+=t
            vertices=np.asarray(vertices,dtype=np.float32)
            if not np.isfinite(vertices).all():raise ValueError('nonfinite baked articulation output')
            return (vertices,global_) if return_globals else vertices
        return self.runtime.apply_pose(pose55, transl, return_globals=return_globals)

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.runtime.save(path/'runtime')
        np.savez_compressed(path/'target.npz', betas=_beta(self.target_betas))
        manifest = dict(schema='CompiledLowerSubjectV17', target_beta_file='target.npz',
                        target_sha256=hashlib.sha256((path/'target.npz').read_bytes()).hexdigest(),
                        runtime_manifest_sha256=hashlib.sha256((path/'runtime/manifest.json').read_bytes()).hexdigest(),
                        motion_reference_beta=self.runtime.betas.tolist(),
                        actual_target_beta=_beta(self.target_betas).tolist(), report=self.report,
                        runtime_blender=False, runtime_fit=False, runtime_rebind=False,
                        anatomical_passed=False, scope='lower_chain_station_fit')
        if self.leg_articulation is not None:
            self.leg_articulation.save(path/'leg_articulation.npz')
            manifest['leg_articulation_sha256']=hashlib.sha256((path/'leg_articulation.npz').read_bytes()).hexdigest()
            manifest['scope']='lower_chain_station_and_baked_articulation'
        (path/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def load_lower_subject(directory):
    path = Path(directory)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest.get('schema') != 'CompiledLowerSubjectV17':
        raise ValueError('not a V17 lower subject')
    if hashlib.sha256((path/'target.npz').read_bytes()).hexdigest() != manifest['target_sha256']:
        raise ValueError('target beta archive hash mismatch')
    if hashlib.sha256((path/'runtime/manifest.json').read_bytes()).hexdigest() != manifest.get('runtime_manifest_sha256'):
        raise ValueError('runtime manifest hash mismatch or unauthenticated early prototype')
    with np.load(path/'target.npz', allow_pickle=False) as data:
        beta = _beta(data['betas'])
    if not np.array_equal(beta, manifest['actual_target_beta']):
        raise ValueError('target beta identity mismatch')
    runtime = load_compiled_subject(path/'runtime')
    report = manifest['report']
    if not (np.array_equal(runtime.betas, manifest['motion_reference_beta']) and
            np.array_equal(runtime.betas, report['source_reference_beta']) and
            np.array_equal(beta, report['target_beta'])):
        raise ValueError('target/reference beta identities disagree')
    if report.get('source_operator_digest') != runtime.source_pack.operator_runtime_digest:
        raise ValueError('source operator identity disagrees with the V17 envelope')
    articulation=None
    if 'leg_articulation_sha256' in manifest:
        from .baked_leg_articulation_v17 import BakedLegArticulationV17
        if hashlib.sha256((path/'leg_articulation.npz').read_bytes()).hexdigest()!=manifest['leg_articulation_sha256']:
            raise ValueError('baked leg articulation hash mismatch')
        with np.load(path/'leg_articulation.npz',allow_pickle=False) as data:
            articulation_schema=json.loads(str(data['metadata_json']))['schema']
        if articulation_schema=='CoupledLegArticulationV17':
            from .coupled_leg_articulation_v17 import CoupledLegArticulationV17
            articulation=CoupledLegArticulationV17.load(path/'leg_articulation.npz')
        else:
            articulation=BakedLegArticulationV17.load(path/'leg_articulation.npz')
    return CompiledLowerSubjectV17(beta, runtime, report,articulation)


def compile_lower_subject(betas, reference, calibration, model, *, max_evaluations=1200,
                          progress=None, rest_fit_poses=None, bake_articulation=False):
    """Solve the same twelve station parameters for any finite target beta.

    No subject IDs, capture poses, AMASS test frames, or per-person constants are
    supplied. Optimization uses a fixed rule-generated motion set. True surface
    validation and Genesis review remain separate from this fit proxy.
    """
    from .lower_chain_rest_v17 import LowerChainRestMapV17
    from .cli.run_material_matrix_v13 import _pose_joints_and_skin

    started = time.monotonic()
    beta = _beta(betas)
    if (reference.provenance.get('method') != 'canonical_reference_v17' or
            reference.provenance.get('target_beta_used') is not False or
            reference.provenance.get('shape_reference_kind') != 'frozen_operator_template' or
            'lower_chain_rest_v17' in reference.provenance):
        raise ValueError('fit reference must be an unfitted canonical authored reference')
    neutral = np.zeros((55, 3), dtype=np.float64)
    skin0, faces, joints0 = _pose_joints_and_skin(model, betas=beta, pose=neutral)
    # Register the entire reference to the target SMPL-X root before applying
    # local limb maps. The same translation moves all anatomy and bind origins.
    reference_j0 = np.asarray(reference.source_asset.rest_joints[0], dtype=np.float64)
    shift = joints0[0] - reference_j0
    bind = reference.target_bind.copy(); bind[:, :3, 3] += shift
    aligned = replace(reference, target_rest=reference.target_rest+shift, target_bind=bind)
    mapper = LowerChainRestMapV17(aligned, calibration)
    ids, is_bone = _fit_vertex_ids(aligned)
    index, weights = aligned.indices[ids], aligned.weights[ids]
    poses = fixed_lower_fit_poses()
    if rest_fit_poses is not None:
        selected=list(rest_fit_poses)
        if not selected or len(set(selected))!=len(selected) or set(selected)-poses.keys():
            raise ValueError('invalid fixed rest fit pose selection')
        poses={name:poses[name] for name in selected}
    frames = []
    for name, pose in poses.items():
        skin, skin_faces, joints = _pose_joints_and_skin(model, betas=beta, pose=pose)
        local = _global_to_local(aligned.source_globals(pose), aligned.parents)
        source_delta = aligned.reference_local_inverse @ local
        frames.append((name, source_delta, SkinFitProxy(skin, skin_faces)))
    names = list(aligned.source_asset.source_bone_names)
    stations = np.array([aligned.target_bind[names.index(n), :3, 3] for n in
                         ('Knee_Rotate_L','Ankle_Rot_L','Knee_Rotate_R','Ankle_Rot_R')])
    desired = joints0[[4, 7, 5, 8]]-stations
    # Initial station offsets are an initialization only, never a pose-time
    # attraction to raw SMPL-X joint centres. Bone-end constraints are in mapper.
    desired = desired.reshape(12)
    # Bounds follow the target geometry, not a beta whitelist or a fixed number
    # of centimetres. The separate cap map enforces the shaft length limits.
    # All continuation candidates and the optimizer use these same bounds.
    reference_length = np.median([np.linalg.norm(s['q']-s['p']) for s in mapper.segments])
    bound_radius = np.abs(desired) + .15*reference_length
    margin = np.where(is_bone, .003, .0015)
    best = [np.inf, np.zeros(12), None]
    evaluations = 0

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        try:
            sampled = mapper.sample(np.asarray(parameters), vertex_ids=ids)
        except ValueError:
            return 1e8 + 1e8*float(np.dot(parameters, parameters))
        rest = sampled['vertices_rest']
        target = sampled['target_bind']
        target_local = _global_to_local(target, aligned.parents)
        target_inverse = np.linalg.inv(target)
        rotation = sampled['rotation_maps']; translations = sampled['translation_maps']
        lbs = SparseLBSV14(rest, index, weights)
        loss = 0.0
        pose_losses = {}
        for name, source_delta, proxy in frames:
            delta = source_delta.copy()
            delta[:, :3, :3] = rotation.swapaxes(1, 2) @ delta[:, :3, :3] @ rotation
            delta[:, :3, 3] = np.einsum('bij,bj->bi', translations, delta[:, :3, 3])
            global_ = _fk(target_local @ delta, aligned.parents)
            posed = lbs(global_ @ target_inverse)
            excess = np.maximum(proxy.signed(posed)-margin, 0)*1000
            # Balance bones and soft material so small meshes and large vessels
            # both affect the solution. The tail term emphasizes gross protrusion.
            value = sum(float(np.mean(excess[part]**2)) for part in (is_bone, ~is_bone))
            value += .1*float(np.max(excess, initial=0)**2)
            pose_losses[name] = value; loss += value/len(frames)
        loss += .025*float(np.mean((parameters*1000)**2))
        loss += .01*float(np.mean(((parameters-desired)*1000)**2))
        if loss < best[0]:
            best[:] = [loss, np.asarray(parameters).copy(), pose_losses]
            if progress is not None:
                progress(dict(evaluation=evaluations, objective=loss, parameters_mm=(best[1]*1000).tolist()))
        return loss

    before = objective(np.zeros(12))
    # A common deterministic continuation avoids starting outside cap limits.
    for alpha in (.25, .5, .75, 1.0):
        objective(desired*alpha)
    initial = best[1].copy()
    result = minimize(objective, initial, method='Powell', bounds=list(zip(-bound_radius,bound_radius)),
                      options=dict(maxfev=int(max_evaluations), xtol=1e-4, ftol=1e-4))
    runtime = mapper.compile(best[1])
    report = dict(method='generic_lower_station_fit_v17', anatomical_passed=False,
                  whole_body_beta_fit_completed=False, subject_specific_branches=False,
                  target_beta=beta.tolist(), source_reference_beta=reference.betas.tolist(),
                  source_operator_digest=reference.source_pack.operator_runtime_digest,
                  target_root_shift_m=shift.tolist(), fit_pose_names=list(poses),
                  capture_or_amass_frames_used_for_fit=False, fit_vertex_count=len(ids),
                  fit_distance='triangle_distance_with_abs_fast_winding_above_0.5_inside; open_skin_not_solid_certificate',
                  station_bound_radius_m=bound_radius.tolist(),
                  initial_objective=before, final_objective=best[0],
                  parameters_m=best[1].tolist(), pose_losses=best[2],
                  optimizer_success=bool(result.success), optimizer_message=str(result.message),
                  evaluations=evaluations, elapsed_s=time.monotonic()-started)
    subject=CompiledLowerSubjectV17(beta, runtime, report)
    if bake_articulation:
        from .coupled_leg_articulation_v17 import bake_coupled_leg_articulation_v17
        subject.leg_articulation=bake_coupled_leg_articulation_v17(subject,calibration,model,progress=progress)
        subject.report['baked_leg_articulation']=subject.leg_articulation.report
        subject.report['elapsed_s']=time.monotonic()-started
    return subject
