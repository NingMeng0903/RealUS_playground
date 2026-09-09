"""Experimental closed-form closure for the baked V17 lower chain.

The coupled V17 response gives a useful hip/knee orientation, but its ankle
origin can drift between the finite fit samples.  This module is an offline
diagnostic prototype which closes one two-link chain after that response.  It
does not alter a runtime package, weights, bind matrices, or source pose.

For each side, the hip head ``H`` is fixed, the current baked knee and ankle
origins define the two radii, and the SMPL-X ankle target is reached whenever
the two-link triangle is feasible.  The knee is selected on the intersection
circle by minimizing its distance to the current knee plane.  The final ankle
counterrotation is applied to the complete ankle subtree, so the foot remains
hierarchically coherent while its authored global orientation is restored.

This is deliberately experimental: it has no skin query, optimizer, nearest
surface operation, or pose-time matrix solve.  Use ``close_subject_pose_v17``
or the command-line diagnostic to inspect the resulting geometry before any
integration decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .chain_rest_fit_v1 import _global_to_local
from .pose_map_v1 import _fk


_SIDE_SPECS = (('L', 4, 7), ('R', 5, 8))
_EPS = 1.0e-10


def _finite_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite with shape {shape}, got {array.shape}')
    return array.copy()


def _unit(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length > _EPS:
        return vector / length
    if fallback is None:
        raise ValueError('cannot normalize a zero vector without a fallback')
    fallback = np.asarray(fallback, dtype=np.float64)
    fallback_length = float(np.linalg.norm(fallback))
    if fallback_length <= _EPS:
        raise ValueError('zero vector fallback')
    return fallback / fallback_length


def _orthogonal(axis: np.ndarray) -> np.ndarray:
    """Return a deterministic unit vector perpendicular to ``axis``."""
    axis = _unit(axis, np.array([1.0, 0.0, 0.0]))
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    return _unit(np.cross(axis, basis))


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.eye(3) * c + (1.0 - c) * np.outer(axis, axis) + s * skew


def _minimal_arc_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest proper rotation taking one nonzero direction to another."""
    source = _unit(source)
    target = _unit(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1.0e-9:
        if cosine >= 0.0:
            return np.eye(3)
        # The antiparallel case has no unique shortest axis.  The least
        # aligned coordinate axis makes the result deterministic and adds no
        # authored axial twist.
        return _rodrigues(_orthogonal(source), np.pi)
    return _rodrigues(cross / sine, float(np.arctan2(sine, cosine)))


def _world_rotation_about(pivot: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(pivot, dtype=np.float64) - matrix[:3, :3] @ pivot
    return matrix


def _descendants(parents: np.ndarray, root: int) -> np.ndarray:
    parents = np.asarray(parents, dtype=np.int64)
    result = [index for index in range(len(parents))
              if index == root or root in _ancestors(parents, index)]
    return np.asarray(result, dtype=np.int64)


def _ancestors(parents: np.ndarray, index: int) -> set[int]:
    result: set[int] = set()
    parent = int(parents[index])
    while parent >= 0:
        result.add(parent)
        parent = int(parents[parent])
    return result


def _apply_subtree(global_: np.ndarray, indices: np.ndarray, matrix: np.ndarray) -> None:
    """Left-apply one world rigid motion to every controller in a subtree."""
    global_[indices] = np.einsum('ij,njk->nik', matrix, global_[indices])


def _knee_plane_pole(head: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    pole = np.cross(knee - head, ankle - knee)
    if np.linalg.norm(pole) <= _EPS:
        pole = _orthogonal(_unit(knee - head))
    return _unit(pole)


def _circle_knee_point(
    head: np.ndarray,
    desired_ankle: np.ndarray,
    current_knee: np.ndarray,
    current_ankle: np.ndarray,
    femur_radius: float,
    shank_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose a knee on the two-sphere intersection, without iteration.

    The returned ankle is the closest point to the requested ankle on the
    reachable annulus when the request is outside the triangle inequality.
    Within the intersection circle, the point whose signed distance to the
    current knee plane is smallest is selected; a residual tie is resolved by
    proximity to the current knee.
    """
    head = np.asarray(head, dtype=np.float64)
    desired_ankle = np.asarray(desired_ankle, dtype=np.float64)
    current_knee = np.asarray(current_knee, dtype=np.float64)
    current_ankle = np.asarray(current_ankle, dtype=np.float64)
    r1, r2 = float(femur_radius), float(shank_radius)
    if not (np.isfinite(r1) and np.isfinite(r2) and r1 > _EPS and r2 > _EPS):
        raise ValueError('two positive finite leg radii are required')

    target_vector = desired_ankle - head
    target_distance = float(np.linalg.norm(target_vector))
    minimum_distance = abs(r1 - r2)
    maximum_distance = r1 + r2
    effective_distance = float(np.clip(target_distance, minimum_distance, maximum_distance))
    radius_residual = abs(effective_distance - target_distance)

    # A zero-length target direction is underdetermined.  Preserve the
    # current knee direction while choosing the nearest reachable radius.
    if target_distance <= _EPS:
        direction = _unit(current_ankle - head, _unit(current_knee - head))
    else:
        direction = target_vector / target_distance
    reachable_ankle = head + direction * effective_distance

    pole = _knee_plane_pole(head, current_knee, current_ankle)
    if effective_distance <= _EPS:
        # This only occurs for two equal radii and coincident centres.  Keep
        # the current bend direction because the target ankle has no axis.
        knee_direction = _unit(current_knee - head)
        knee = head + r1 * knee_direction
        return knee, reachable_ankle, dict(
            selection='coincident_centres_current_knee_direction',
            target_distance_m=target_distance,
            reachable_distance_m=effective_distance,
            reachable_residual_m=radius_residual,
            circle_radius_m=0.0,
            plane_distance_m=0.0,
            current_knee_plane_pole=pole.tolist(),
        )

    axis = direction
    x = (r1 * r1 - r2 * r2 + effective_distance * effective_distance) / (2.0 * effective_distance)
    circle_radius_squared = max(0.0, r1 * r1 - x * x)
    circle_radius = float(np.sqrt(circle_radius_squared))
    center = head + x * axis
    e1 = _orthogonal(axis)
    e2 = _unit(np.cross(axis, e1))

    # For K(theta)=center+rho*(e1*cos(theta)+e2*sin(theta)), its signed
    # distance to the old bend plane is c0+rho*(a*cos(theta)+b*sin(theta)).
    c0 = float(np.dot(pole, center - head))
    a = float(np.dot(pole, e1))
    b = float(np.dot(pole, e2))
    amplitude = circle_radius * float(np.hypot(a, b))
    candidates: list[float] = []
    if circle_radius <= 1.0e-9:
        candidates = [0.0]
    elif amplitude > 1.0e-12 and abs(c0) <= amplitude + 1.0e-12:
        phase = float(np.arctan2(b, a))
        offset = float(np.arccos(np.clip(-c0 / amplitude, -1.0, 1.0)))
        candidates = [phase + offset, phase - offset]
    elif amplitude > 1.0e-12:
        phase = float(np.arctan2(b, a))
        candidates = [phase if abs(c0 + amplitude) <= abs(c0 - amplitude)
                      else phase + np.pi]
    else:
        # The current plane is parallel to the circle; use the closest point
        # to the current knee as a deterministic secondary rule.
        current_offset = current_knee - center
        candidates = [float(np.arctan2(np.dot(current_offset, e2),
                                       np.dot(current_offset, e1)))]

    def point(theta: float) -> np.ndarray:
        return center + circle_radius * (e1 * np.cos(theta) + e2 * np.sin(theta))

    points = [point(theta) for theta in candidates]
    distances_to_plane = [abs(float(np.dot(pole, p - head))) for p in points]
    best_plane = min(distances_to_plane)
    tied = [index for index, value in enumerate(distances_to_plane)
            if value <= best_plane + 1.0e-10]
    selected = min(tied, key=lambda index: float(np.linalg.norm(points[index] - current_knee)))
    knee = points[selected]
    return knee, reachable_ankle, dict(
        selection='two_sphere_circle_nearest_current_knee_plane',
        target_distance_m=target_distance,
        reachable_distance_m=effective_distance,
        reachable_residual_m=radius_residual,
        circle_radius_m=circle_radius,
        plane_distance_m=distances_to_plane[selected],
        current_knee_plane_pole=pole.tolist(),
        circle_center=center.tolist(),
        circle_axis=axis.tolist(),
        candidate_count=len(points),
    )


def _matrix_orientation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def close_leg_globals_v17(
    current_globals: np.ndarray,
    authored_globals: np.ndarray,
    target_bind: np.ndarray,
    smplx_rest_to_pose: np.ndarray,
    parents: np.ndarray,
    bone_names: Iterable[str],
    head_centers_local: np.ndarray,
    transl: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Close both lower chains after the baked hip/knee response.

    ``current_globals`` must be the coupled/baked result.  ``authored_globals``
    is the same runtime pose before the coupled correction; its ankle rotation
    is restored after the geometric closure.  All transforms remain 4x4
    controller globals and are subsequently evaluated by the caller's normal
    235-controller LBS.
    """
    current = _finite_matrix(current_globals, (235, 4, 4), 'current_globals')
    authored = _finite_matrix(authored_globals, (235, 4, 4), 'authored_globals')
    bind = _finite_matrix(target_bind, (235, 4, 4), 'target_bind')
    smplx = _finite_matrix(smplx_rest_to_pose, (55, 4, 4), 'smplx_rest_to_pose')
    parent_ids = np.asarray(parents, dtype=np.int64)
    if parent_ids.shape != (235,) or np.any(parent_ids >= np.arange(235)) or np.any(parent_ids < -1):
        raise ValueError('invalid 235-controller parent hierarchy')
    names = [str(value) for value in bone_names]
    if len(names) != 235 or len(set(names)) != 235:
        raise ValueError('expected 235 unique controller names')
    centers = np.asarray(head_centers_local, dtype=np.float64)
    if centers.shape != (2, 3) or not np.isfinite(centers).all():
        raise ValueError('head_centers_local must be [2,3] and finite')
    if transl is None:
        translation = np.zeros(3, dtype=np.float64)
    else:
        translation = np.asarray(transl, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError('transl must be three finite values')

    # The authored global transforms are evaluated without frame translation;
    # add the same world translation used for current_globals before using
    # their pivot/orientation data.
    authored = authored.copy()
    authored[:, :3, 3] += translation
    result = current.copy()
    reports: list[dict[str, Any]] = []

    for side_index, (suffix, ankle_joint, _unused) in enumerate((('L', 7, 0), ('R', 8, 0))):
        hip = names.index(f'Femur_Rot_{suffix}')
        knee_id = names.index(f'Knee_Rotate_{suffix}')
        ankle = names.index(f'Ankle_Rot_{suffix}')
        hip_subtree = _descendants(parent_ids, hip)
        knee_subtree = _descendants(parent_ids, knee_id)
        ankle_subtree = _descendants(parent_ids, ankle)

        # The head centre is expressed in the hip bind-local coordinates by
        # head_centers_local_v17.  Applying authored hip global gives its
        # fixed world position, independent of the current hip twist.
        center_homogeneous = np.r_[centers[side_index], 1.0]
        head = (authored[hip] @ center_homogeneous)[:3]
        current_head = (current[hip] @ center_homogeneous)[:3]
        current_knee = current[knee_id, :3, 3].copy()
        current_ankle = current[ankle, :3, 3].copy()
        femur_radius = float(np.linalg.norm(current_knee - head))
        shank_radius = float(np.linalg.norm(current_ankle - current_knee))

        desired_ankle = (smplx[ankle_joint] @ bind[ankle])[:3, 3] + translation
        knee_target, reachable_ankle, circle_report = _circle_knee_point(
            head, desired_ankle, current_knee, current_ankle,
            femur_radius, shank_radius)

        hip_rotation = _minimal_arc_rotation(current_knee - head, knee_target - head)
        hip_world = _world_rotation_about(head, hip_rotation)
        _apply_subtree(result, hip_subtree, hip_world)

        knee_after_hip = result[knee_id, :3, 3].copy()
        ankle_after_hip = result[ankle, :3, 3].copy()
        knee_rotation = _minimal_arc_rotation(ankle_after_hip - knee_target,
                                              reachable_ankle - knee_target)
        knee_world = _world_rotation_about(knee_target, knee_rotation)
        _apply_subtree(result, knee_subtree, knee_world)

        ankle_after_knee = result[ankle, :3, 3].copy()
        authored_ankle_rotation = authored[ankle, :3, :3]
        counter_rotation = authored_ankle_rotation @ result[ankle, :3, :3].T
        ankle_world = _world_rotation_about(ankle_after_knee, counter_rotation)
        _apply_subtree(result, ankle_subtree, ankle_world)

        final_knee = result[knee_id, :3, 3].copy()
        final_ankle = result[ankle, :3, 3].copy()
        final_hip = (result[hip] @ center_homogeneous)[:3]
        final_femur_radius = float(np.linalg.norm(final_knee - final_hip))
        final_shank_radius = float(np.linalg.norm(final_ankle - final_knee))
        reports.append(dict(
            side=suffix,
            hip_controller=f'Femur_Rot_{suffix}',
            knee_controller=f'Knee_Rotate_{suffix}',
            ankle_controller=f'Ankle_Rot_{suffix}',
            hip_subtree_count=int(len(hip_subtree)),
            knee_subtree_count=int(len(knee_subtree)),
            ankle_subtree_count=int(len(ankle_subtree)),
            authored_head_world_m=head.tolist(),
            current_head_world_m=current_head.tolist(),
            head_drift_before_m=float(np.linalg.norm(current_head - head)),
            knee_before_m=current_knee.tolist(),
            ankle_before_m=current_ankle.tolist(),
            desired_ankle_m=desired_ankle.tolist(),
            reachable_ankle_m=reachable_ankle.tolist(),
            knee_after_m=final_knee.tolist(),
            ankle_after_m=final_ankle.tolist(),
            femur_radius_before_m=femur_radius,
            femur_radius_after_m=final_femur_radius,
            shank_radius_before_m=shank_radius,
            shank_radius_after_m=final_shank_radius,
            femur_radius_error_m=abs(final_femur_radius - femur_radius),
            shank_radius_error_m=abs(final_shank_radius - shank_radius),
            ankle_error_before_mm=float(np.linalg.norm(current_ankle - desired_ankle) * 1000.0),
            ankle_error_after_mm=float(np.linalg.norm(final_ankle - desired_ankle) * 1000.0),
            reachable_residual_mm=float(circle_report['reachable_residual_m'] * 1000.0),
            hip_direction_arc_deg=float(np.degrees(np.arccos(np.clip(
                np.dot(_unit(current_knee - head), _unit(knee_target - head)), -1.0, 1.0)))),
            knee_direction_arc_deg=float(np.degrees(np.arccos(np.clip(
                np.dot(_unit(ankle_after_hip - knee_target), _unit(reachable_ankle - knee_target)),
                -1.0, 1.0)))),
            ankle_orientation_error_deg=_matrix_orientation_error_deg(
                authored_ankle_rotation, result[ankle, :3, :3]),
            restored_authored_ankle_orientation=True,
            original_axial_twist_preserved_by='minimal_arc_direction_rotation',
            **circle_report,
        ))

    # World subtree motions preserve the hierarchy by construction.  Keep an
    # explicit all-235 FK residual in the artifact so a future wrapper cannot
    # silently consume a malformed transform array.
    local = _global_to_local(result, parent_ids)
    fk_rebuilt = _fk(local, parent_ids)
    fk_error = float(np.max(np.abs(fk_rebuilt - result)))
    report = dict(
        schema='AnalyticLegClosureV17',
        experimental=True,
        method='closed_form_two_sphere_two_link_after_baked_hip_knee',
        optimizer_used=False,
        skin_query_used=False,
        nearest_surface_query_used=False,
        runtime_matrix_solve=False,
        source_theta_unchanged=True,
        all_235_fk=True,
        original_weights_unchanged=True,
        hip_head_fixed=True,
        foot_subtrees_coherent=True,
        bone_lengths_changed=False,
        transl_m=translation.tolist(),
        fk_reconstruction_max_abs_m=fk_error,
        sides=reports,
    )
    if not np.isfinite(result).all() or fk_error > 2.0e-8:
        raise ValueError(f'closed transforms failed the all-235 FK check: {fk_error}')
    return result, report


def close_subject_pose_v17(
    subject: Any,
    pose55: np.ndarray,
    smplx_rest_to_pose: np.ndarray,
    *,
    transl: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the experimental closure to one already-baked subject pose."""
    if getattr(subject, 'leg_articulation', None) is None:
        raise ValueError('subject must contain the saved coupled/baked leg response')
    runtime = subject.runtime
    pose = np.asarray(pose55, dtype=np.float64)
    if pose.shape != (55, 3) or not np.isfinite(pose).all():
        raise ValueError('pose55 must be finite [55,3]')
    translation = np.zeros(3, dtype=np.float64) if transl is None else np.asarray(transl, dtype=np.float64)
    current_vertices, current_globals = subject.apply_pose(pose, translation, return_globals=True)
    authored_globals = runtime.globals_from_source(runtime.source_globals(pose))
    closed_globals, report = close_leg_globals_v17(
        current_globals=current_globals,
        authored_globals=authored_globals,
        target_bind=runtime.target_bind,
        smplx_rest_to_pose=smplx_rest_to_pose,
        parents=runtime.parents,
        bone_names=runtime.source_asset.source_bone_names,
        head_centers_local=subject.leg_articulation.head_centers_local,
        transl=translation,
    )
    closed_vertices = runtime._lbs(closed_globals @ runtime.target_inverse)
    closed_vertices = np.asarray(closed_vertices, dtype=np.float32)
    if not np.isfinite(closed_vertices).all():
        raise ValueError('nonfinite analytic closure vertices')
    report['coupled_vertex_count'] = int(len(current_vertices))
    report['coupled_to_closed_vertex_max_abs_mm'] = float(
        np.max(np.abs(np.asarray(closed_vertices, dtype=np.float64) -
                      np.asarray(current_vertices, dtype=np.float64))) * 1000.0)
    return closed_vertices, closed_globals, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _default_scene_names(geometry: Path) -> list[Path]:
    names = {
        'knees_90', 'hips_60_knees_90', 'sitstand_frame422', 'sitstand_frame792',
    }
    paths = []
    for path in sorted(geometry.glob('*.npz')):
        if path.stem in names:
            paths.append(path)
            continue
        pieces = path.stem.split('_')
        if len(pieces) == 3 and pieces[0].isdigit() and pieces[1] == 'capture' and pieces[0] == pieces[2]:
            paths.append(path)
    return paths


def export_closure_diagnostic_v17(
    compiled: Path | str,
    geometry: Path | str,
    output: Path | str,
    model_path: Path | str,
    scene_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Export a small closure diagnostic package for later surface evaluation."""
    from .generic_lower_compile_v17 import load_lower_subject
    from .smplx_body_surface_v7 import (
        _smplx_joint_kinematics_v7,
        load_smplx_model_v7,
        require_frozen_smplx_male_v7,
    )

    compiled_path = Path(compiled).resolve()
    geometry_path = Path(geometry).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    subject = load_lower_subject(compiled_path)
    model_file, model_sha = require_frozen_smplx_male_v7(model_path)
    model = load_smplx_model_v7(model_file)
    paths = (_default_scene_names(geometry_path) if scene_names is None else
             [geometry_path / f'{name}.npz' for name in scene_names])
    if not paths:
        raise ValueError('no diagnostic scenes selected')
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    output_path.mkdir(parents=True)
    records = []
    asset = subject.runtime.source_asset
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            pose = np.asarray(data['pose'], dtype=np.float64)
            translation = np.asarray(data['transl'], dtype=np.float64)
            original_candidate = np.asarray(data['candidate_vertices'], dtype=np.float64)
            original_before = np.asarray(data['before_vertices'], dtype=np.float64)
            smplx_skin = np.asarray(data['skin_vertices'], dtype=np.float64)
            smplx_faces = np.asarray(data['skin_faces'], dtype=np.int32)
            smplx_joints = np.asarray(data['smplx_joints'], dtype=np.float64)
            _, _, rest_to_pose = _smplx_joint_kinematics_v7(
                model, betas=subject.target_betas, pose_axis_angle=pose)
            closed, closed_globals, report = close_subject_pose_v17(
                subject, pose, rest_to_pose, transl=translation)
            coupled, _coupled_globals = subject.apply_pose(pose, translation, return_globals=True)
            coupled = np.asarray(coupled, dtype=np.float64)
            replay_error_mm = float(np.max(np.abs(coupled - original_candidate)) * 1000.0)
            destination = output_path / path.name
            np.savez_compressed(
                destination,
                # The evaluator's before/candidate pair intentionally measures
                # the incremental closure against the current coupled result.
                before_vertices=coupled,
                source_vertices=coupled,
                candidate_vertices=closed,
                original_reference_before_vertices=original_before,
                original_coupled_candidate_vertices=original_candidate,
                faces=np.asarray(data['faces'], dtype=np.int32),
                skin_vertices=smplx_skin,
                skin_faces=smplx_faces,
                smplx_joints=smplx_joints,
                pose=pose,
                transl=translation,
                vertex_tissue=np.asarray(data['vertex_tissue'], dtype=np.int8),
                mesh_names=np.asarray(data['mesh_names']),
                mesh_ranges=np.asarray(data['mesh_ranges'], dtype=np.int32),
                mesh_tissues=np.asarray(data['mesh_tissues']),
                target_betas=subject.target_betas,
            )
        report['input_scene'] = str(path)
        report['input_scene_sha256'] = _sha256(path)
        report['output_scene'] = destination.name
        report['coupled_replay_max_abs_mm'] = replay_error_mm
        records.append(report)

    manifest = dict(
        schema='AnalyticLegClosureV17Diagnostic',
        compiled=str(compiled_path),
        geometry_input=str(geometry_path),
        target_betas=subject.target_betas.tolist(),
        model_path=str(model_file),
        model_sha256=model_sha,
        method='closed_form_two_sphere_two_link_after_baked_hip_knee',
        experimental=True,
        optimizer_used=False,
        skin_query_used=False,
        runtime_integration=False,
        before_pair='saved coupled candidate; original reference baseline retained as original_reference_before_vertices',
        scenes=records,
    )
    (output_path / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', type=Path, required=True,
                        help='saved V17 coupled package compiled directory')
    parser.add_argument('--geometry', type=Path, required=True,
                        help='saved coupled geometry directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='new diagnostic geometry output directory')
    parser.add_argument('--model', type=Path, required=True,
                        help='authenticated SMPLX_MALE.pkl')
    parser.add_argument('--scene', action='append', default=None,
                        help='scene stem to include; default is four rules plus own capture')
    args = parser.parse_args()
    manifest = export_closure_diagnostic_v17(
        args.compiled, args.geometry, args.output, args.model, args.scene)
    print(json.dumps(dict(output=str(args.output.resolve()), scene_count=len(manifest['scenes']),
                          ankle_error_after_mm={
                              r['input_scene'].split('/')[-1]: {
                                  side['side']: side['ankle_error_after_mm']
                                  for side in r['sides']
                              } for r in manifest['scenes']
                          }), indent=2))


if __name__ == '__main__':
    main()
