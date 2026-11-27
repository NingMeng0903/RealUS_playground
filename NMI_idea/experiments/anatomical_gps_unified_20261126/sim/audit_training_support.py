"""Run the training-only coverage, registration, and proposal-yield audit.

With --require-reader-inputs, a failed engineering preflight exits 2 after
writing the report. Dataset limitations are reported separately.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np

from .anatomy_assets import (
    FAMILY_HELD_OUT_ATLASES, HOLDOUT_SOURCE_ID, LOAO_BODYPARTS, LOAO_OPEN3D, MEDSEG_RIGHT,
    NEUTRAL_SMPLX_SURFACE_PATH, NYSORA_STATIONS, THIGH_NYSORA_Y_BOUNDS_MM, TrainingAnatomyBank, VesselRole,
)
from .material_chart import RightLegMaterialChart
from .metric_graph_gp import _CONTACT_ATLAS, MetricGraphAnatomyGenerator
from .smplx_dynamic_atlas import NEUTRAL_BETA, SMPLXLegKinematics


_SPLITS = {
    'LOAO_OPEN3D': LOAO_OPEN3D,
    'LOAO_BODYPARTS': LOAO_BODYPARTS,
    'FAMILY_HELD_OUT_ATLASES': FAMILY_HELD_OUT_ATLASES,
}
_JUNCTION_STATION = {'S2_SFJ_MICKEY': 'SFJ', 'S3_ART_BIF': 'CFA_bif', 'S4_VEN_BIF': 'CFV_bif'}


def _training_only_io(event, args):
    if event not in ('open', 'os.listdir', 'os.scandir') or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0]).lower()
    if any(part in path for part in ('blender_holdout', 'blender_topology_holdout', 'sim/holdout_eval_assets',
                                     'smplx_vascular_retarget_audit', 'arterial_paths.json')):
        raise PermissionError(f'training preflight attempted to access a holdout path: {path}')


def station_support_masks(h_mm, supported, graph):
    """Raw same-source role co-presence and Gate 2 station windows, in row order."""
    lo, hi = THIGH_NYSORA_Y_BOUNDS_MM
    in_domain = (lo <= h_mm) & (h_mm <= hi)
    copresence, windows = {}, {}
    for name, roles in NYSORA_STATIONS.items():
        present = supported[:, sorted(roles)].all(axis=1) & in_domain
        if name == 'S5_POPLITEAL':
            present &= h_mm <= -825.
        copresence[name] = present
        junction_name = _JUNCTION_STATION.get(name)
        if junction_name is not None:
            junction = graph.junctions.get(junction_name)
            if junction is None:
                present = np.zeros_like(present)
            else:
                present = present & (abs(h_mm - graph.vertices[junction.vertex_id].h_mm) <= 15.)
        elif name == 'S1_CFV_CFA':
            present = present & (h_mm >= -560.)
        windows[name] = present
    return copresence, windows


def _coverage(bank, generator):
    sources = {}
    for source_id in bank.source_ids:
        graph = generator.graph(source_id)
        source = bank.source(source_id)
        copresence, windows = station_support_masks(source.h_mm, source.supported, graph)
        height = source.h_mm[source.supported.any(axis=1)]
        sources[source_id] = {
            'source_family': source.provenance['source_family'],
            'source_supported_y_range_mm': [float(height.min()), float(height.max())],
            'supported_role_rows': dict(zip((r.name for r in VesselRole), source.supported.sum(axis=0).tolist())),
            'nysora_role_copresence_rows': {name: int(mask.sum()) for name, mask in copresence.items()},
            'junction_window_role_copresence_rows': {name: int(mask.sum()) for name, mask in windows.items()},
            'modeled_junctions': {name: asdict(evidence) for name, evidence in graph.junctions.items()},
            'unlocated_junctions': graph.unlocated_junctions,
        }
    splits = {}
    for name, split in _SPLITS.items():
        train = {station for source in split.train_source_ids
                 for station, n in sources[source]['nysora_role_copresence_rows'].items() if n}
        supported = {source: sorted(station for station in train if sources[source]['nysora_role_copresence_rows'][station])
                     for source in split.test_source_ids}
        splits[name] = {**asdict(split), 'supported_stations': sorted({s for values in supported.values() for s in values}),
                        'per_test_source_supported_stations': supported}
    return {'clinical_station_labels_available': False,
            'scope': 'Role co-presence is necessary, not a clinical label; junction-window rows are a topological surrogate.',
            'sources': sources, 'splits': splits}


def _numerical_registration(chart):
    kin = SMPLXLegKinematics()
    q = np.deg2rad([18., -8., 6., 23.])
    heights = np.array([-880., -850.7, -824.8, -780.4, -672.1, -580.8, -549.6, -507.1, -480.4, -475.])
    xi = np.column_stack((np.linspace(-2.85, 3.05, len(heights)), heights, np.linspace(.22, .82, len(heights))))
    xyz = chart.xi_to_world(xi)
    recon = chart.world_to_xi(xyz) - xi
    recon[:, 0] = (recon[:, 0] + np.pi) % (2. * np.pi) - np.pi
    tilde = xi.copy()
    tilde[:, 2] = chart.eta_from_d(xi[:, 2])
    # Height differences use open-domain samples, unlike the pose check.
    analytic_checks = []
    for points, forward, accum in ((xi[:-1], chart.xi_to_world, chart.dworld_dxi),
                                    (tilde[:-1], chart.tilde_xi_to_world, chart.dworld_dtilde_xi)):
        eye = np.eye(3) * 1.e-5
        numeric = np.stack([(forward(points + d) - forward(points - d)) / 2.e-5 for d in eye], axis=-1)
        analytic_checks.append(float(np.max(abs(numeric - accum(points)))))
    _, _, jq, _ = kin.pose_points_and_material_frame(NEUTRAL_BETA, q, chart, tilde)
    numeric = np.stack([(kin.forward_pose_canonical(q + d, xyz)[0] - kin.forward_pose_canonical(q - d, xyz)[0]) / 2.e-5
                        for d in np.eye(4) * 1.e-5], axis=-1)
    with np.load(NEUTRAL_SMPLX_SURFACE_PATH, allow_pickle=False) as pack:
        skin = pack['vertices']
    indices = np.arange(len(skin))
    canonical = kin.forward_vertices(NEUTRAL_BETA, np.zeros(4), indices)[0]
    legacy = kin.forward_vertices(None, np.zeros(4), indices)[0]
    return {
        'canonical_mesh_beta0_max_component_difference_mm': float(np.max(abs(canonical - skin))),
        'legacy_shape_max_component_difference_mm': float(np.max(abs(legacy - skin))),
        'legacy_beta': kin.reference_beta.tolist(),
        'chart_round_trip_max_error': float(np.max(abs(recon))),
        'chart_round_trip_world_mm_max_error': float(np.max(abs(chart.xi_to_world(chart.world_to_xi(xyz)) - xyz))),
        'analytic_chart_jacobian_max_error': max(analytic_checks),
        'dworld_dxi_max_error': analytic_checks[0],
        'dworld_dtilde_xi_max_error': analytic_checks[1],
        'beta0_pose_jacobian_mm_per_rad_max_error': float(np.max(abs(numeric - jq))),
        'beta0_pose_jacobian_y_range_mm': [float(heights.min()), float(heights.max())],
        'legacy_harmonic_shaft_minmax_world_y_mm': (kin.atlas.harmonic_vertices[:, 1][[np.argmin(kin.atlas.harmonic_vertices[:, 1]),
                                                                                       np.argmax(kin.atlas.harmonic_vertices[:, 1])]] * 1000.).tolist(),
    }


def _registration(bank, material, legacy):
    result = _numerical_registration(RightLegMaterialChart())
    sources = {}
    for source_id in bank.source_ids:
        graph = material.graph(source_id)
        prior = material.prior(source_id)
        inside, skin_known, bone_hit, _ = material.classify_points(graph.knot_xyz_mm)
        raw = bank.source(source_id)
        station_role = {}
        for i, j in zip(*np.where(graph.supported), strict=True):
            station_role.setdefault(graph.sample_knots[i, j], []).append(VesselRole(j).name)
        issues = [{'knot_index': int(i), 'roles': station_role[i], 'xyz_mm': graph.knot_xyz_mm[i].tolist(),
                   'outside_canonical_skin': bool(skin_known[i] and not inside[i]), 'in_excluded_bone': bool(bone_hit[i])}
                  for i in np.flatnonzero((skin_known & ~inside) | bone_hit)]
        centers = prior.decode_knots_to_world(prior.mean)
        shaft_lo, shaft_hi = result['legacy_harmonic_shaft_minmax_world_y_mm']
        sources[source_id] = {
            'legacy_contact_nominal_geometry': asdict(legacy.audit_centers(graph, graph.knot_xyz_mm)),
            'canonical_raw_nominal_geometry': asdict(material.audit_centers(graph, graph.knot_xyz_mm)),
            'raw_knots_inside_legacy_shaft_y_range': int(((shaft_lo <= graph.knot_xyz_mm[:, 1]) & (graph.knot_xyz_mm[:, 1] <= shaft_hi)).sum()),
            'canonical_nominal_exclusion_knots': issues,
            'nominal_projection_audit': prior.nominal_projection_audit,
            'projected_material_nominal_geometry': asdict(material.audit_centers(graph, centers, prior=prior, latent=prior.mean)),
            'source_station_rows': int(raw.supported.any(axis=1).sum()),
        }
    return {**result, 'legacy_contact_skin_file': str(_CONTACT_ATLAS), 'canonical_skin_file': str(NEUTRAL_SMPLX_SURFACE_PATH),
            'canonical_skin_sha256': bank.provenance['neutral_smplx_surface_sha256'],
            'legacy_contact_lower_body_boundary_edges': legacy._validity.lower_body_boundary_edges,
            'canonical_lower_body_boundary_edges': material._validity.lower_body_boundary_edges, 'sources': sources}


def _yield(generator, source, difficulty, n_proposals, seed):
    draws = generator.sample_batch(source, range(n_proposals), np.random.default_rng(seed), difficulty, geometry_policy='raw')
    audits = [draw.audit for draw in draws]
    prior = draws[0].prior
    nominal = prior.decode_knots_to_world(prior.mean)
    distances = np.concatenate([np.linalg.norm(draw.knot_centers_mm - nominal, axis=1) for draw in draws])
    accepted = sum(a.all_locations_known and a.passes_known_checks for a in audits)
    return {
        'n_proposals': len(draws), 'fully_known_and_passing': accepted, 'effective_valid_yield': accepted / len(draws),
        **{reason: sum(getattr(a, reason) for a in audits) for reason in
           ('checked_points', 'skin_unknown', 'bone_unknown', 'outside_skin', 'in_excluded_bone',
            'unsupported_material_points', 'axial_reversals')},
        'achieved_euclidean_shift_mm_median_p90': np.percentile(distances, (50, 90)).tolist(),
        'achieved_euclidean_rms_shift_mm_median_p90': np.percentile([d.achieved_euclidean_rms_shift_mm for d in draws], (50, 90)).tolist(),
    }


def build_report(n_proposals: int, seed: int) -> dict:
    if n_proposals < 1:
        raise ValueError('n_proposals must be positive')
    bank = TrainingAnatomyBank.load()
    material = MetricGraphAnatomyGenerator(bank)
    cartesian = MetricGraphAnatomyGenerator(bank, coordinate_mode='cartesian_xyz')
    legacy = MetricGraphAnatomyGenerator(bank, skin_path=_CONTACT_ATLAS, coordinate_mode='cartesian_xyz')
    coverage = _coverage(bank, material)
    registration = _registration(bank, material, legacy)
    profiles = {gen.coordinate_mode: {source: {difficulty: _yield(gen, source, difficulty, n_proposals, seed)
                                               for difficulty in ('ordinary', 'stress')} for source in bank.source_ids}
                for gen in (cartesian, material)}
    required = int(np.ceil(.90625 * n_proposals))
    thresholds = {'ordinary': 14., 'stress': 25.}
    blockers = []
    zero_holdout_leakage = (HOLDOUT_SOURCE_ID not in bank.source_ids and 'sim.holdout_eval_assets' not in sys.modules
                            and os.getenv('AGPS_EVAL_HOLDOUT_ALLOWED') != '1')
    if not zero_holdout_leakage:
        blockers.append({'code': 'training_holdout_isolation'})
    for key, limit in (('chart_round_trip_max_error', 1.e-6), ('analytic_chart_jacobian_max_error', 1.e-4),
                       ('beta0_pose_jacobian_mm_per_rad_max_error', 1.e-5)):
        if not registration[key] < limit:
            blockers.append({'code': key, 'value': registration[key], 'strict_upper_bound': limit})
    for source, by_profile in profiles['material'].items():
        raw = registration['sources'][source]['canonical_raw_nominal_geometry']
        if raw['skin_unknown'] or raw['bone_unknown']:
            blockers.append({'code': 'unknown_nominal_geometry', 'source': source})
        for difficulty, profile in by_profile.items():
            reasons = []
            if profile['skin_unknown'] or profile['bone_unknown']:
                reasons.append('unknown_geometry')
            if profile['fully_known_and_passing'] < required:
                reasons.append('insufficient_valid_yield')
            if profile['achieved_euclidean_shift_mm_median_p90'][0] < thresholds[difficulty]:
                reasons.append('insufficient_achieved_shift')
            blockers.extend({'code': reason, 'source': source, 'difficulty': difficulty} for reason in reasons)
    return {
        'axis_1_original_ct_coverage': coverage,
        'axis_2_atlas_registration': registration,
        'axis_3_perturbed_sample_yield': {'domain': 'thigh_nysora', 'seed': seed, 'n_proposals': n_proposals,
                                         'scope': 'Raw Gaussian draws. Same beta=0 checker in both modes; shifts are relative to each mode\'s zero-latent world centers.',
                                         **profiles},
        'evidence_gaps': ['no_independent_clinical_station_labels', 'ct_fov_proximal_only', 'bodyparts3d_lacks_dfa'],
        'engineering_preflight': {'passed': not blockers, 'zero_holdout_leakage': zero_holdout_leakage,
                                  'minimum_fully_known_and_passing': required,
                                  'minimum_median_achieved_shift_mm': thresholds, 'blockers': blockers},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--proposals', type=int, default=64)
    parser.add_argument('--seed', type=int, default=58)
    parser.add_argument('--require-reader-inputs', action='store_true')
    args = parser.parse_args()
    sys.addaudithook(_training_only_io)
    report = build_report(args.proposals, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(args.out)
    return 2 if args.require_reader_inputs and not report['engineering_preflight']['passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
