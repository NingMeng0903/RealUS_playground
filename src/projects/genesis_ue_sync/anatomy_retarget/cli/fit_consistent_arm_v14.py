"""Fit a connected left arm with one cap-preserving rest/bind map.

Elbow and wrist stations are fitted together, shoulder fixed. This is a
bounded first-stage experiment, never an anatomical acceptance declaration.
The source is materialized 142; neither V11 nor V12e geometry is inherited.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path
import numpy as np
import igl
from scipy.optimize import minimize, OptimizeResult

from ..anatomy_lbs import source_bone_posed_global
from ..anatomical_calibration_v1 import load_anatomical_calibration_v1
from ..axial_caps_v14 import AxialCapsFieldV14, extract_proper_rotation_v14
from ..chain_rest_fit_v1 import _global_to_local, _weighted_rest_correction
from ..consistent_runtime_v14 import (compile_subject, CompileConfigV14, original_shape_reference_v14,
                                      driver_rotation_maps_v14)
from ..pose_map_v1 import _fk
from ..segment_similarity_rest_v10 import _rotation_align
from ..v8_artifacts import load_source_operator, materialize_subject
from ..validation_poses_v13 import load_held_out_poses_v13
from ..deep_flex_poses_v12 import _donor_axis, solve_hinge_magnitude
from ..smplx_body_surface_v7 import require_frozen_smplx_male_v7, load_smplx_model_v7
from .run_material_matrix_v13 import _load_capture, _pose_joints_and_skin
from .export_capture_joint_review_v13 import _tissue_codes


ROOT = Path(__file__).resolve().parents[5]
OP = ROOT / 'outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8'
CAL = ROOT / 'outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1'
MODEL = ROOT / 'ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl'

# Rest-roll is an explicit, bounded degree of freedom.  The two entries are
# stored in degrees in the fit parameter vector, while all geometry and bind
# matrices remain metres / homogeneous matrices as before.
REST_ROLL_LIMIT_DEG = 60.0
REST_ROLL_PARAMETER_COUNT = 2


def _rest_roll_rotation(axis, angle_deg):
    """Return a proper column-vector rotation for one target segment axis."""

    direction = np.asarray(axis, dtype=float).reshape(-1)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError('rest-roll axis must be finite [3]')
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError('rest-roll axis is degenerate')
    angle = float(angle_deg)
    if not np.isfinite(angle) or abs(angle) > REST_ROLL_LIMIT_DEG:
        raise ValueError(
            f'rest-roll angle must be finite and within +/-{REST_ROLL_LIMIT_DEG:g} degrees'
        )
    direction = direction / norm
    x, y, z = direction
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    identity = np.eye(3)
    radians = np.deg2rad(angle)
    rotation = (
        np.cos(radians) * identity
        + (1.0 - np.cos(radians)) * np.outer(direction, direction)
        + np.sin(radians) * skew
    )
    if not np.allclose(rotation.T @ rotation, identity, atol=1e-12, rtol=0.0):
        raise ValueError('rest-roll rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError('rest-roll rotation is not proper')
    return rotation


def _axis_rotation_map(origin, axis, angle_deg):
    """Homogeneous map rotating around a target segment line."""

    rotation = _rest_roll_rotation(axis, angle_deg)
    point = np.asarray(origin, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)):
        raise ValueError('rest-roll axis origin is non-finite')
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = point - rotation @ point
    return result


def _initial_parameters_from_report(
    report_path,
    *,
    parameter_count,
    legacy_parameter_count,
    rest_roll_enabled,
):
    """Load an evaluate-only vector, migrating a matching legacy report.

    Reports written before the opt-in rest-roll DOF contain exactly the old
    six or nine values.  When the caller explicitly enables rest-roll, append
    two zero-degree values so the old neutral result remains reproducible.
    Reports with a different layout fail closed rather than silently shifting
    contact or angle parameters.
    """

    payload = json.loads(Path(report_path).read_text())
    previous = np.asarray(payload['best_parameters_mm'], dtype=float).reshape(-1)
    if rest_roll_enabled and previous.size == legacy_parameter_count:
        previous = np.concatenate((previous, np.zeros(REST_ROLL_PARAMETER_COUNT)))
    if previous.size != parameter_count:
        raise ValueError(
            f'initial report has {previous.size} parameters; this invocation requires '
            f'{parameter_count} ({"with" if rest_roll_enabled else "without"} rest-roll)'
        )
    if not np.all(np.isfinite(previous)):
        raise ValueError('initial report parameters contain non-finite values')
    return previous


class ArmMap:
    """Same map sampler for optimization and the final whole-body compiler.

    ``rest_roll`` is opt-in so historical six- and nine-parameter callers keep
    their exact parameter layout.  When enabled, two extra parameters are
    appended after the legacy station/contact values: humerus ``S->E`` and
    forearm ``E->W`` axial rolls, both in degrees and bounded to +/-60.
    """
    def __init__(self, asset, calibration, *, motion_reference_bind=None,
                 rest_roll=False):
        self.a = asset
        self.b0 = np.asarray(asset.target_bind_global, dtype=float)
        self.parents = np.asarray(asset.source_bone_parents, dtype=int)
        self.names = list(asset.source_bone_names)
        self.s, self.e, self.w = [self.b0[self.names.index(n), :3, 3] for n in
                                ['Shoulder_Rotate_L', 'Elbow_Rot_L', 'Wrist_Rotate_L']]
        self.humerus = [self.names.index(n) for n in ['Shoulder_Rotate_L', 'Elbow_Rot_L']]
        self.forearm = [self.names.index(n) for n in ['Forearm_Bone_L', 'Forearm_Twist_L']]
        self.hand = []
        root = self.names.index('Wrist_Rotate_L')
        for i in range(len(self.names)):
            p = i
            while p >= 0 and p != root:
                p = self.parents[p]
            if p == root:
                self.hand.append(i)
        self.groups = [self.humerus, self.forearm, self.hand]
        self.rest_roll_enabled = bool(rest_roll)
        self.rest_roll_bounds_deg = (-REST_ROLL_LIMIT_DEG, REST_ROLL_LIMIT_DEG)
        self.domains = calibration.domains
        self.motion_reference_bind = self.b0 if motion_reference_bind is None else np.asarray(motion_reference_bind,dtype=float)
        self.reference_local_inv = np.linalg.inv(_global_to_local(self.motion_reference_bind, self.parents))
        v = np.asarray(asset.vertices_rest, dtype=float)
        bounds = []
        keys = [(['calibration/left/shoulder/humerus.fit'], ['elbow/left/humerus.fit']),
                (['elbow/left/radius.fit', 'elbow/left/ulna.fit'],
                 ['calibration/left/wrist/radius.fit', 'calibration/left/wrist/ulna.fit'])]
        for (p, q), (prox, dist) in zip([(self.s, self.e), (self.e, self.w)], keys):
            axis = (q-p)/np.linalg.norm(q-p)
            lo = (v[np.concatenate([self.domains[k] for k in prox])] - p) @ axis
            hi = (v[np.concatenate([self.domains[k] for k in dist])] - p) @ axis
            bounds.append(((float(lo.min()), float(lo.max())), (float(hi.min()), float(hi.max()))))
        self.bounds = bounds
        self.mesh_ids = {}
        self.mesh_faces = {}
        labels = _tissue_codes(asset)
        hand_owners = set(self.hand)
        for n, tissue, owner, (start, stop) in zip(asset.source_mesh_names, asset.source_tissues,
                                                  asset.source_mesh_controller_bones, asset.source_vertex_ranges):
            if n in ['Humerus_L', 'Radius_L', 'Ulna_L'] or (tissue == 'bone' and int(owner) in hand_owners):
                ids = np.arange(start, stop)
                self.mesh_ids[str(n)] = ids
                mask = np.all((asset.faces >= start) & (asset.faces < stop), axis=1)
                self.mesh_faces[str(n)] = np.asarray(asset.faces[mask] - start, dtype=np.int64)
        self.all_ids = np.unique(np.concatenate(list(self.mesh_ids.values())))
        self.fit_ids = np.concatenate([ids if n in ['Humerus_L','Radius_L','Ulna_L'] else
                                       ids[np.linspace(0, len(ids)-1, min(30, len(ids))).astype(int)]
                                       for n, ids in self.mesh_ids.items()])
        self.fit_ids = np.unique(self.fit_ids)
        self.label = labels

    def config(self, parameters_mm):
        parameters_mm = np.asarray(parameters_mm, dtype=float).reshape(-1)
        expected_sizes = (8, 11) if self.rest_roll_enabled else (6, 9)
        if parameters_mm.size not in expected_sizes:
            if self.rest_roll_enabled:
                raise ValueError(
                    'Expected legacy 6/9 values plus two rest-roll angles '
                    '(8/11 parameters)'
                )
            raise ValueError('Expected elbow/wrist deltas and optional forearm cap offset')
        if not np.all(np.isfinite(parameters_mm)):
            raise ValueError('fit parameters contain non-finite values')
        delta = parameters_mm[:6].reshape(2, 3) * .001
        contact_delta = parameters_mm[6:9] * .001 if parameters_mm.size in (9, 11) else np.zeros(3)
        if np.linalg.norm(contact_delta) > .012:
            raise ValueError('Forearm cap rest offset exceeds 12 mm')
        if self.rest_roll_enabled:
            roll_offset = 9 if parameters_mm.size == 11 else 6
            rest_roll_deg = np.asarray(
                parameters_mm[roll_offset : roll_offset + REST_ROLL_PARAMETER_COUNT],
                dtype=float,
            )
            if np.any(np.abs(rest_roll_deg) > REST_ROLL_LIMIT_DEG):
                raise ValueError(
                    f'Rest-roll angles must lie within +/-{REST_ROLL_LIMIT_DEG:g} degrees'
                )
        else:
            rest_roll_deg = np.zeros(REST_ROLL_PARAMETER_COUNT, dtype=float)
        e, w = self.e + delta[0], self.w + delta[1]
        maps = np.tile(np.eye(4), (235, 1, 1)); fields = [None] * 235
        rigid_ends = []
        scales = []
        for segment_index, (group, (p, q), (p1, q1), bounds) in enumerate(
            zip(
                self.groups[:2],
                [(self.s, self.e), (self.e, self.w)],
                [(self.s, e), (e + contact_delta, w)],
                self.bounds,
            )
        ):
            length = float(np.linalg.norm(q-p)); length1 = float(np.linalg.norm(q1-p1))
            scale = length1 / length; axis = (q-p)/length
            field = AxialCapsFieldV14(p, axis, length, scale, bounds[0], bounds[1])
            r = _rotation_align(q-p, q1-p1)
            t = p1-r@p
            segment_map = np.eye(4); segment_map[:3,:3] = r; segment_map[:3,3] = t
            # The cap-preserving field runs first in source segment space;
            # this rigid roll runs second around the already mapped target
            # segment axis.  Store their composition in the same rigid map
            # consumed by ArmMap.sample and compile_subject.
            roll_origin = np.asarray(p1, dtype=float)
            target_axis = (q1 - p1) / length1
            roll_map = _axis_rotation_map(
                roll_origin, target_axis, rest_roll_deg[segment_index]
            )
            m = segment_map if rest_roll_deg[segment_index] == 0.0 else roll_map @ segment_map
            for i in group:
                maps[i]=m;fields[i]=field
            base_end = segment_map.copy()
            base_end[:3,3] += r @ (axis * field.delta_m)
            end = base_end if rest_roll_deg[segment_index] == 0.0 else roll_map @ base_end
            rigid_ends.append(end);scales.append(scale)
        maps[self.hand] = rigid_ends[-1]
        provenance = {
            'method':'connected_left_arm_caps_v14', 'elbow_wrist_delta_mm':(delta * 1000).tolist(),
            'forearm_cap_offset_mm':(contact_delta * 1000).tolist(),
            'segment_length_scales':scales, 'fit_domain_partition':'fit',
            'shoulder_station_fixed':True, 'anatomical_passed':False,
            'rest_roll_enabled': self.rest_roll_enabled,
            'rest_roll_deg': rest_roll_deg.tolist(),
            'rest_roll_unit': 'degrees',
            'rest_roll_bounds_deg': [-REST_ROLL_LIMIT_DEG, REST_ROLL_LIMIT_DEG],
            'rest_roll_axes': ['S->E', 'E->W'],
            'hand_rest_map_follows_forearm': True,
        }
        return CompileConfigV14(
            rigid_maps=maps,
            axial_fields=tuple(fields),
            provenance=provenance,
        )

    def sample(self, config, vertex_ids):
        a = self.a; maps = config.rigid_maps; fields=config.axial_fields
        x = np.asarray(a.vertices_rest[vertex_ids], dtype=float)
        index = a.driver_indices[vertex_ids]; weight=a.driver_weights[vertex_ids]
        y=x.copy(); bind=self.b0.copy()
        # A controller with no selected geometry field still needs the
        # authored motion-reference frame change.  Initialising these rows to
        # identity makes the scorer disagree with the runtime whenever the
        # geometry and motion reference binds use different axes.  The rows
        # below are overwritten for controllers with a mapped field; the
        # untouched rows must start at B0.T @ motion_reference_bind.
        base_rotation = np.asarray(self.b0[:, :3, :3], dtype=float)
        motion_rotation = np.asarray(
            self.motion_reference_bind[:, :3, :3], dtype=float
        )
        residual = base_rotation.swapaxes(1, 2) @ motion_rotation
        for group in self.groups:
            i=group[0]; field=fields[i]; m=maps[i]
            mass=np.sum(np.where(np.isin(index,group),weight,0),axis=1)
            moved=x if field is None else field.map_points(x)
            moved=moved@m[:3,:3].T+m[:3,3]
            y += mass[:,None]*(moved-x)
        # Reproduce compile_subject's per-controller frame and translation
        # response for every controller, including controllers with no
        # selected vertices.  The latter still receive the authored identity
        # map in the compiler, and their slightly non-orthogonal authored
        # frame must therefore produce the same stored response matrix.
        for j in range(len(self.names)):
            field = fields[j]
            m = maps[j]
            origin = self.b0[j,:3,3][None]
            mapped = origin if field is None else field.map_points(origin)
            jac = np.eye(3) if field is None else field.jacobian(origin)[0]
            jacobian = m[:3,:3] @ jac
            frame_rotation = extract_proper_rotation_v14(jacobian)
            target_rotation = frame_rotation @ self.b0[j,:3,:3]
            bind[j,:3,:3] = target_rotation
            bind[j,:3,3] = (mapped @ m[:3,:3].T + m[:3,3])[0]
            # Keep the residual in the source motion-reference frame used by
            # runtime delta translation.  In particular, axial stretch stays
            # in this linear response and never leaks into bind rotation.
            residual[j] = (
                target_rotation.T
                @ jacobian
                @ self.motion_reference_bind[j,:3,:3]
            )
        return y,bind,residual


def signed(points, v, f):
    return igl.signed_distance(np.ascontiguousarray(points,dtype=float),
                               np.ascontiguousarray(v,dtype=float),np.ascontiguousarray(f,dtype=np.int64))[0]


_FULL_ARM_CONTACT_KEYS = (
    'shoulder_humerus_to_scapula',
    'shoulder_scapula_to_humerus',
    'elbow_humerus_to_radius',
    'elbow_humerus_to_ulna',
    'elbow_radius_to_humerus',
    'elbow_ulna_to_humerus',
    'elbow_radius_to_ulna',
    'elbow_ulna_to_radius',
    'wrist_radius_to_hand',
    'wrist_ulna_to_hand',
    'wrist_hand_to_radius',
    'wrist_hand_to_ulna',
)
_REPULSIVE_ONLY_CONTACT_KEYS = frozenset(
    {'wrist_ulna_to_hand', 'wrist_hand_to_ulna'}
)


def _frozen_full_arm_contact_specs():
    """Return the probe's frozen FIT contact graph with penalty policies.

    Keeping the source tuple in ``probe_arm_rest_roll_v14`` makes the FIT
    objective and the earlier bounded rest-roll ablation use exactly the same
    domains.  The lazy import avoids a module cycle because that probe imports
    :class:`ArmMap` for its own diagnostic run.
    """

    from .probe_arm_rest_roll_v14 import CONTACT_SPECS

    specs = tuple(CONTACT_SPECS)
    keys = tuple(str(row[0]) for row in specs)
    if keys != _FULL_ARM_CONTACT_KEYS:
        raise RuntimeError(
            'rest-roll probe CONTACT_SPECS changed; refusing a mixed FIT graph'
        )
    result = []
    for row in specs:
        if len(row) != 3:
            raise RuntimeError('CONTACT_SPECS rows must be (key, domain, target)')
        key, domain, target = (str(value) for value in row)
        if not domain.endswith('.fit'):
            raise ValueError(f'contact graph requires a frozen FIT domain: {domain!r}')
        result.append((key, domain, target, key not in _REPULSIVE_ONLY_CONTACT_KEYS))
    return tuple(result)


def _contact_mesh_layout(asset, mesh_names):
    """Build strict global-ID/local-face layouts for contact target meshes."""

    names = [str(value) for value in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    faces = np.asarray(asset.faces, dtype=np.int64)
    if ranges.shape != (len(names), 2):
        raise ValueError('source mesh ranges do not match mesh names')
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError('source mesh ranges are invalid')
    result = {}
    for mesh_name in tuple(mesh_names):
        mesh_name = str(mesh_name)
        if mesh_name not in names:
            raise ValueError(f'source asset is missing contact mesh {mesh_name!r}')
        mesh_index = names.index(mesh_name)
        start, stop = (int(value) for value in ranges[mesh_index])
        if stop > len(asset.vertices_rest):
            raise ValueError(f'contact mesh {mesh_name!r} exceeds source vertices')
        complete = np.all((faces >= start) & (faces < stop), axis=1)
        partial = np.any((faces >= start) & (faces < stop), axis=1) & ~complete
        if np.any(partial) or not np.any(complete):
            raise ValueError(
                f'contact mesh {mesh_name!r} does not have complete local triangles'
            )
        face_ids = np.flatnonzero(complete).astype(np.int64)
        result[mesh_name] = {
            'vertex_start': start,
            'vertex_stop': stop,
            'vertex_ids': np.arange(start, stop, dtype=np.int64),
            'faces_local': (faces[face_ids] - start).astype(np.int64),
            'face_count': int(len(face_ids)),
        }
    return result


def _strict_contact_lookup(global_ids, lookup, *, label):
    """Translate global source vertex IDs, failing closed on every omission."""

    ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    table = np.asarray(lookup, dtype=np.int64).reshape(-1)
    if np.any(ids < 0) or np.any(ids >= len(table)):
        raise ValueError(f'{label} contains a source vertex outside lookup')
    local = table[ids]
    if np.any(local < 0):
        missing = ids[local < 0]
        preview = ', '.join(str(int(value)) for value in missing[:8])
        raise ValueError(f'{label} has vertices missing from used set: {preview}')
    return local


def _full_arm_contact_metrics(
    vertices,
    calibration,
    mesh_layout,
    lookup,
    specs=None,
):
    """Score the frozen shoulder/elbow/wrist graph and return per-query data.

    All values are computed from FIT domains.  ``gap_penalty_enabled`` is
    false for both Ulna--Scaphoid directions, so those edges only repel
    penetration and never pull the meshes together across a natural gap.
    """

    if specs is None:
        specs = _frozen_full_arm_contact_specs()
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError('contact graph vertices must be finite [N,3]')
    total = 0.0
    metrics = {}
    for key, domain_key, target_name, gap_enabled in specs:
        if not str(domain_key).endswith('.fit'):
            raise ValueError(f'contact graph requires a frozen FIT domain: {domain_key!r}')
        if domain_key not in calibration.domains:
            raise KeyError(f'missing frozen FIT domain {domain_key!r}')
        query_global = np.asarray(calibration.domains[domain_key], dtype=np.int64).reshape(-1)
        if len(query_global) < 3:
            raise ValueError(f'contact query {domain_key!r} has fewer than three vertices')
        query_local = _strict_contact_lookup(
            query_global, lookup, label=f'contact query {domain_key!r}'
        )
        if target_name not in mesh_layout:
            raise KeyError(f'missing contact mesh layout {target_name!r}')
        target = mesh_layout[target_name]
        target_global = np.asarray(target['vertex_ids'], dtype=np.int64)
        target_local = _strict_contact_lookup(
            target_global, lookup, label=f'contact target {target_name!r}'
        )
        target_vertices = values[target_local]
        target_faces = np.asarray(target['faces_local'], dtype=np.int64)
        signed_values = np.asarray(
            signed(values[query_local], target_vertices, target_faces),
            dtype=np.float64,
        ).reshape(-1)
        if len(signed_values) != len(query_global) or not np.all(np.isfinite(signed_values)):
            raise ValueError(f'contact query {key!r} returned invalid signed distances')
        gap = float(np.min(np.abs(signed_values)))
        maximum_penetration = float(np.maximum(-signed_values, 0.0).max())
        gap_excess = max(gap - 0.003, 0.0) if gap_enabled else 0.0
        penetration_excess = max(maximum_penetration - 0.0005, 0.0)
        penalty = 5.0 * (gap_excess * gap_excess + penetration_excess * penetration_excess)
        total += penalty
        metrics[key] = {
            'query_domain': domain_key,
            'query_count': int(len(query_global)),
            'query_vertices_in_used': True,
            'target_mesh': target_name,
            'target_mesh_vertex_range': [
                int(target['vertex_start']), int(target['vertex_stop'])
            ],
            'target_vertex_count': int(len(target_global)),
            'target_face_count': int(len(target_faces)),
            'target_vertices_in_used': True,
            'minimum_absolute_gap_mm': gap * 1000.0,
            'minimum_signed_mm': float(np.min(signed_values) * 1000.0),
            'maximum_penetration_lower_bound_mm': maximum_penetration * 1000.0,
            'penetration_gt_0.5mm_count': int(
                np.count_nonzero(signed_values < -0.0005)
            ),
            'gap_penalty_enabled': bool(gap_enabled),
            'penetration_penalty_enabled': True,
            'gap_attraction_used': bool(gap_enabled),
            'gap_excess_mm': gap_excess * 1000.0,
            'penetration_excess_mm': penetration_excess * 1000.0,
            'constraint_penalty_m2': float(penalty),
            'signed_depth_is_sampled_lower_bound': True,
            'calibration_partition': 'fit',
            'validation_domains_used': False,
        }
    return float(total), metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',default='213328',choices=['213328','213712'])
    parser.add_argument('--max-evaluations',type=int,default=240)
    parser.add_argument('--optimizer',choices=['Powell','SLSQP'],default='Powell')
    parser.add_argument('--joint-rest-offset',action='store_true',
                        help='Fit the forearm pair rest contact position with the same geometry/bind map')
    parser.add_argument('--initial-report',type=Path)
    parser.add_argument('--evaluate-only',action='store_true',
                        help='Recompile the exact parameters from --initial-report without optimizing')
    parser.add_argument('--fit-domain',choices=['capture_and_hinge','rest'],default='capture_and_hinge')
    parser.add_argument('--skin-fit-inset-mm',type=float,default=-1.,
                        help='Signed fitting target: +1 fits 1 mm inside; -1 permits 1 mm outside')
    parser.add_argument('--full-hand-fit',action='store_true',help='Use every hand-bone vertex during fitting')
    parser.add_argument('--shape-source',choices=['materialized','original142'],default='original142',
                        help='Use authenticated original whole-anatomy shape instead of inherited beta vertex deformation')
    parser.add_argument('--rotation-transport',choices=['driver_axes','material_axes'],default='driver_axes')
    parser.add_argument(
        '--contact-graph',
        choices=['legacy_elbow', 'full_arm'],
        default='legacy_elbow',
        help=(
            'Contact objective: preserve the historical one-way humerus->radius/ulna '
            'snapshot, or use all frozen FIT shoulder/elbow/wrist directions'
        ),
    )
    parser.add_argument(
        '--rest-roll',
        nargs='*',
        type=float,
        metavar='DEG',
        help=(
            'Enable two fitted rest-roll angles (humerus S->E, forearm E->W) '
            'in degrees, bounded to +/-60; optionally provide their initial values'
        ),
    )
    parser.add_argument('--collar-pivot-response',action='store_true',
                        help='Compile the calibrated collar13 driver and anatomical pivot before fitting')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.evaluate_only and args.initial_report is None:
        parser.error('--evaluate-only requires --initial-report')
    rest_roll_values = [] if args.rest_roll is None else list(args.rest_roll)
    if rest_roll_values and len(rest_roll_values) != REST_ROLL_PARAMETER_COUNT:
        parser.error('--rest-roll accepts zero values or exactly two angle values in degrees')
    rest_roll_enabled = args.rest_roll is not None
    if rest_roll_values and np.any(np.abs(rest_roll_values) > REST_ROLL_LIMIT_DEG):
        parser.error(
            f'--rest-roll angles must lie within +/-{REST_ROLL_LIMIT_DEG:g} degrees'
        )
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    op=load_source_operator(OP);cal=load_anatomical_calibration_v1(CAL,operator=op)
    model_path,model_sha=require_frozen_smplx_male_v7(MODEL);model=load_smplx_model_v7(model_path)
    captures={};betas={}
    for name in ['213328','213712']:
        b,p,_=_load_capture(ROOT/f'smplx_outputs/20260713_{name}/moment_0000/smplx_result.npz',model_path=model_path)
        betas[name]=b;captures[name]=p
    beta=betas[args.subject];pack=materialize_subject(op,betas=beta,gender='male');a=pack.rigged_asset
    geometry_asset = a; motion_asset = a; response = None
    alignment = np.eye(4)
    if args.shape_source == 'original142':
        shape_vertices,shape_bind,alignment = original_shape_reference_v14(op,a)
        geometry_asset=replace(op.template_asset,vertices_rest=shape_vertices,target_rest_global=shape_bind)
    if args.collar_pivot_response:
        from ..collar_response_v14 import make_collar_pivot_response_asset_v14
        from ..motion_response_v14 import BakedMotionResponseV14
        motion_asset=make_collar_pivot_response_asset_v14(a)
        response=BakedMotionResponseV14.from_assets(a,motion_asset,provenance=dict(
            method='collar_local_rotation_and_joint13_pivot',controller_ids=[129],smplx_joint_ids=[13],
            offline_coupling_baked=True,anatomical_passed=False))
        geometry_bind=np.asarray(geometry_asset.target_bind_global,dtype=float).copy()
        point=op.template_asset.rest_joints[13] if args.shape_source=='original142' else a.rest_joints[13]
        geometry_bind[129,:3,3]=(alignment@np.r_[point,1])[:3]
        geometry_asset=replace(geometry_asset,target_rest_global=geometry_bind)
    builder=ArmMap(geometry_asset,cal,motion_reference_bind=motion_asset.target_bind_global,
                   rest_roll=rest_roll_enabled)
    if args.full_hand_fit:
        builder.fit_ids=builder.all_ids.copy()
    poses={'tpose':np.zeros((55,3),dtype=np.float32),**{'pose_'+k:v for k,v in captures.items()}}
    def joints_of(p):return _pose_joints_and_skin(model,betas=beta,pose=p)[2]
    axis=_donor_axis('elbow_L',captures)
    for angle in [30,60,90,120]:
        magnitude,_=solve_hinge_magnitude('elbow_L',target_deg=angle,axis=axis,joints_of=joints_of)
        p=np.zeros((55,3),dtype=np.float32);p[18]=axis*magnitude;poses[f'elbow_L_{angle}']=p
    fit_names=['tpose'] if args.fit_domain == 'rest' else list(poses)
    # These motion files were used diagnostically in V13; they are regression
    # cases, not newly unseen evidence. A separate continuous holdout follows.
    heldout,heldout_prov=load_held_out_poses_v13()
    poses.update({k:v for k,v in heldout.items() if k in ['heldout_sitting','heldout_kicking']})
    frames={}
    for name,p in poses.items():
        skin,faces,joints=_pose_joints_and_skin(model,betas=beta,pose=p)
        g=source_bone_posed_global(motion_asset,p)
        d=builder.reference_local_inv@_global_to_local(g,builder.parents)
        raw_g=g if response is None else source_bone_posed_global(a,p)
        frames[name]=dict(skin=skin,faces=faces,joints=joints,source_global=g,raw_source_global=raw_g,delta=d)
    contact_specs = ()
    contact_layout = {}
    if args.contact_graph == 'full_arm':
        contact_specs = _frozen_full_arm_contact_specs()
        contact_layout = _contact_mesh_layout(
            geometry_asset,
            tuple(dict.fromkeys(row[2] for row in contact_specs)),
        )
    used_parts = [
        np.asarray(builder.all_ids, dtype=np.int64),
        *[
            np.asarray(cal.domains[f'elbow/left/{b}.fit'], dtype=np.int64)
            for b in ['humerus', 'radius', 'ulna']
        ],
    ]
    if args.contact_graph == 'full_arm':
        # The graph includes the shoulder query and complete Scapula_L target,
        # plus every frozen FIT query/target used by the wrist edges.  Keeping
        # these vertices in one sample is required for signed targets to share
        # precisely the same left-arm field as the skin and runtime sample.
        used_parts.extend(
            np.asarray(cal.domains[domain_key], dtype=np.int64)
            for _key, domain_key, _target, _gap_enabled in contact_specs
        )
        used_parts.extend(
            np.asarray(contact_layout[target_name]['vertex_ids'], dtype=np.int64)
            for target_name in contact_layout
        )
    used=np.unique(np.concatenate(used_parts))
    lookup=np.full(len(a.vertices_rest),-1,dtype=int);lookup[used]=np.arange(len(used))
    if args.contact_graph == 'full_arm':
        # Validate the complete graph once before Powell starts.  Every
        # objective evaluation then uses the same checked global-to-local
        # correspondence and can never silently drop a -1 lookup.
        for _key, domain_key, target_name, _gap_enabled in contact_specs:
            _strict_contact_lookup(
                cal.domains[domain_key], lookup, label=f'contact query {domain_key!r}'
            )
            _strict_contact_lookup(
                contact_layout[target_name]['vertex_ids'],
                lookup,
                label=f'contact target {target_name!r}',
            )
    legacy_parameter_count = 9 if args.joint_rest_offset else 6
    parameter_count = legacy_parameter_count + (
        REST_ROLL_PARAMETER_COUNT if rest_roll_enabled else 0
    )
    best={'score':float('inf'),'parameters':np.zeros(parameter_count)}; evaluations=0
    def evaluated(parameters, report=False):
        nonlocal evaluations,best
        try:
            config=builder.config(parameters)
            config=replace(config,rotation_transport=args.rotation_transport,motion_response=response,
                           shape_reference_bind=geometry_asset.target_bind_global if response is not None else None)
            if args.shape_source == 'original142':
                config=replace(config,shape_reference_operator=op)
        except ValueError:return (1e6+float(np.dot(parameters,parameters))) if not report else None
        rest,bind,residual=builder.sample(config,used)
        local=_global_to_local(bind,builder.parents); inv=np.linalg.inv(bind)
        angular_maps=(driver_rotation_maps_v14(builder.motion_reference_bind,bind) if args.rotation_transport=='driver_axes'
           else np.tile(np.eye(3),(235,1,1)))
        indices=a.driver_indices[used];weights=a.driver_weights[used]
        cost=0.;metrics={}
        for name in (poses if report else fit_names):
            f=frames[name];d=f['delta'].copy();d[:,:3,3]=np.einsum('bij,bj->bi',residual,d[:,:3,3])
            d[:,:3,:3]=angular_maps.swapaxes(1,2)@d[:,:3,:3]@angular_maps
            globals_=_fk(local@d,builder.parents)
            vertices=rest if name=='tpose' else _weighted_rest_correction(rest,indices,weights,globals_@inv)
            ids=builder.all_ids if report else builder.fit_ids
            skin_s=signed(vertices[lookup[ids]],f['skin'],f['faces'])
            outside=np.maximum(skin_s + args.skin_fit_inset_mm * .001,0)
            cost += float(np.mean(outside**2)+np.max(outside)**2)
            joints={}
            contact_graph_metrics = None
            if args.contact_graph == 'full_arm':
                contact_score, contact_graph_metrics = _full_arm_contact_metrics(
                    vertices,
                    cal,
                    contact_layout,
                    lookup,
                    contact_specs,
                )
                cost += contact_score
            else:
                # Keep the historical objective and its metric snapshot
                # byte-for-byte in the default mode.  Full-arm mode below is
                # an explicit opt-in graph and must not alter old fits.
                for part,mesh in [('radius','Radius_L'),('ulna','Ulna_L')]:
                    q=vertices[lookup[cal.domains['elbow/left/humerus.fit']]]
                    bone=vertices[lookup[builder.mesh_ids[mesh]]]
                    s=signed(q,bone,builder.mesh_faces[mesh])
                    gap=float(np.min(np.abs(s)));depth=float(np.maximum(-s-.0005,0).max())
                    cost += 5*(max(gap-.003,0)**2+depth**2)
                    joints[part]=dict(closest_m=gap,min_signed_m=float(s.min()))
            metrics[name]=dict(max_outside_m=float(np.maximum(skin_s,0).max()),
                               outside_count=int(np.count_nonzero(skin_s>.001)),
                               fit_joint_queries=joints)
            if args.contact_graph == 'full_arm':
                metrics[name]['fit_contact_graph'] = contact_graph_metrics
        cost+=1e-9*float(np.dot(parameters,parameters))
        if not report:
            evaluations+=1
            if cost<best['score']:
                best=dict(score=cost,parameters=np.asarray(parameters).copy())
            if evaluations%20==0:
                payload=dict(evaluations=evaluations,best_score=best['score'],best_parameters_mm=best['parameters'].tolist(),elapsed_s=time.perf_counter()-started)
                (args.output/'progress.json').write_text(json.dumps(payload,indent=2)+'\n')
                print(payload,flush=True)
        return (metrics,config) if report else cost
    zero=np.zeros(parameter_count);evaluated(zero)
    initial=zero.copy()
    if rest_roll_values:
        initial[legacy_parameter_count:] = np.asarray(rest_roll_values, dtype=float)
    if args.initial_report:
        previous = _initial_parameters_from_report(
            args.initial_report,
            parameter_count=parameter_count,
            legacy_parameter_count=legacy_parameter_count,
            rest_roll_enabled=rest_roll_enabled,
        )
        initial[:] = previous
        evaluated(initial)
    bounds=[(-25.,25.)]*6 + ([(-8.,8.)]*3 if args.joint_rest_offset else [])
    if rest_roll_enabled:
        bounds += [(-REST_ROLL_LIMIT_DEG, REST_ROLL_LIMIT_DEG)] * REST_ROLL_PARAMETER_COUNT
    if args.evaluate_only:
        result=OptimizeResult(success=False,message='not run: evaluating supplied fixed parameters')
    elif args.optimizer == 'SLSQP':
        result=minimize(lambda x: 1e6 * evaluated(x),initial,method='SLSQP',bounds=bounds,
                        options=dict(maxiter=args.max_evaluations,eps=.05,ftol=1e-5))
    else:
        result=minimize(evaluated,initial,method='Powell',bounds=bounds,
                        options=dict(maxfev=args.max_evaluations,xtol=.2,ftol=.001))
    chosen=initial if args.evaluate_only else best['parameters'];metrics,config=evaluated(chosen,report=True)
    baseline,_=evaluated(zero,report=True)
    compiled=compile_subject(beta,pack,config)
    compiled.save(args.output/'compiled')
    for name in poses:
        f=frames[name]; candidate=compiled.apply_pose(poses[name])
        raw=_weighted_rest_correction(a.vertices_rest,a.driver_indices,a.driver_weights,
                                      f['raw_source_global']@np.linalg.inv(a.target_bind_global)).astype(np.float32)
        np.savez_compressed(args.output/f'subject_{args.subject}_{name}.npz',
                            source_vertices=raw,candidate_vertices=candidate,faces=a.faces,
                            skin_vertices=f['skin'],skin_faces=f['faces'],smplx_joints=f['joints'],
                            pose=poses[name],vertex_tissue=builder.label)
        # Score and runtime must have exactly the same transformation semantics.
        rest,bind,res=builder.sample(config,used);d=f['delta'].copy()
        d[:,:3,3]=np.einsum('bij,bj->bi',res,d[:,:3,3])
        angular_maps=(driver_rotation_maps_v14(builder.motion_reference_bind,bind) if args.rotation_transport=='driver_axes'
           else np.tile(np.eye(3),(235,1,1)))
        d[:,:3,:3]=angular_maps.swapaxes(1,2)@d[:,:3,:3]@angular_maps
        g=_fk(_global_to_local(bind,builder.parents)@d,builder.parents)
        scored=rest if name=='tpose' else _weighted_rest_correction(rest,a.driver_indices[used],a.driver_weights[used],g@np.linalg.inv(bind))
        error=float(np.linalg.norm(candidate[used]-scored,axis=1).max())
        if error>2e-6:raise ValueError(f'{name}: score/runtime discrepancy {error}m')
        metrics[name]['score_runtime_max_m']=error
    parameter_layout = [
        'elbow_dx_mm', 'elbow_dy_mm', 'elbow_dz_mm',
        'wrist_dx_mm', 'wrist_dy_mm', 'wrist_dz_mm',
    ]
    if args.joint_rest_offset:
        parameter_layout.extend(['forearm_cap_dx_mm', 'forearm_cap_dy_mm', 'forearm_cap_dz_mm'])
    if rest_roll_enabled:
        parameter_layout.extend(['humerus_rest_roll_deg', 'forearm_rest_roll_deg'])
    report=dict(status='complete',operational_evaluation_completed=True,anatomical_passed=False,
                publishable=False,subject=args.subject,fit_poses=fit_names,
                regression_poses=['heldout_sitting','heldout_kicking'],
                independent_validation_complete=False,best_parameters_mm=chosen.tolist(),
                parameter_count=len(parameter_layout), parameter_layout=parameter_layout,
                optimizer=args.optimizer, joint_rest_offset=args.joint_rest_offset,
                optimizer_was_run=not args.evaluate_only,
                fit_domain=args.fit_domain,
                skin_fit_inset_mm=args.skin_fit_inset_mm,
                full_hand_fit=args.full_hand_fit,
                contact_graph=args.contact_graph,
                contact_graph_specs=(
                    [
                        {
                            'key': key,
                            'query_domain': domain_key,
                            'target_mesh': target_name,
                            'gap_penalty_enabled': bool(gap_enabled),
                            'penetration_penalty_enabled': True,
                        }
                        for key, domain_key, target_name, gap_enabled in contact_specs
                    ]
                    if args.contact_graph == 'full_arm' else []
                ),
                contact_graph_thresholds_mm={
                    'gap_limit': 3.0,
                    'penetration_limit': 0.5,
                },
                contact_graph_fit_partition='fit',
                contact_graph_validation_domains_used=False,
                shape_source=args.shape_source,
                rotation_transport=args.rotation_transport,collar_pivot_response=args.collar_pivot_response,
                rest_roll_enabled=rest_roll_enabled,
                rest_roll_initial_deg=rest_roll_values if rest_roll_values else [0.0, 0.0],
                rest_roll_parameter_indices=(
                    [legacy_parameter_count, legacy_parameter_count + 1]
                    if rest_roll_enabled else []
                ),
                rest_roll_unit='degrees',
                rest_roll_bounds_deg=[-REST_ROLL_LIMIT_DEG, REST_ROLL_LIMIT_DEG],
                optimizer_success=bool(result.success),optimizer_message=str(result.message),
                evaluations=evaluations,metrics=metrics,baseline=baseline,
                config=config.provenance,elapsed_s=time.perf_counter()-started,
                source_operator_digest=op.runtime_digest(validate=False),smplx_model_sha256=model_sha,
                regression_provenance=heldout_prov)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('complete',chosen.tolist(),flush=True)


if __name__=='__main__':main()
