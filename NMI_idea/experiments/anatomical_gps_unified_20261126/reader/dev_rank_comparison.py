"""Compare v6 and one frozen evidence-ranking change on the two Step 3R confirmation cohorts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from sim.anatomy_assets import LOAO_OPEN3D, TrainingAnatomyBank, VesselRole

from . import dev_tracking_interventions as tracking
from . import dev_visible_continuation as visible
from .baselines import Full_AGPS_Reader
from .eval_reader import HOLDOUT_ENV, _load, _sha256, _source_sha256, _tape_truth
from .evidence_ranked_reader import EvidenceRankedReader
from .split_prior import DynamicScanSimulator, build_split_prior
from .visible_stations import DECISION_THRESHOLD, STATIONS, pose_features, visible_features


_ARMS = ('v6', 'evidence_ranked')
_SOURCES = ('bodyparts3d_right', 'bodyparts3d_left_mirrored', 'medseg_right')
_STATION_ROOT, _STATION_REPEATS, _STATION_FRAMES = 731903, 8, 4
_DYNAMIC_ROOT, _DYNAMIC_REPEATS, _DYNAMIC_FRAMES = 931902, 12, 8
_CHECKPOINT = visible._CHECKPOINT
_FIT_REPORT = _CHECKPOINT.parents[2] / 'step3r_dev/visible.json'
_OLD_VERIFIER_STAGE = 'visible_identity_tracking_and_old_verifier_accepts_without_violation'
_DYNAMIC_JOINT = 'joint_representable_masks_match_and_all_role_centers_below_6_mm'


class _RankedDiagnostic(tracking._DiagnosticReader, EvidenceRankedReader):
    """Trace the ordinary candidate search through the diagnostic's cooperative hooks."""


def _paired_flags(before, after):
    # Index first, then aggregate: a gain and a loss are both reported even when net is zero.
    bins = [0, 0, 0, 0]
    for old, new in zip(before, after, strict=True):
        bins[2 * bool(old) + bool(new)] += 1
    neither, gains, losses, both = bins
    return {'fixed_denominator': sum(bins), 'v6': both + losses, 'evidence_ranked': both + gains,
            'both': both, 'neither': neither, 'gains': gains, 'losses': losses, 'net': gains - losses}


def _paired_endpoints(before, after):
    return {key: _paired_flags((a[key] for a in before), (b[key] for b in after)) for key in before[0]}


def _dynamic_schedule(root):
    return [(source, index, tracking._episode_seed(root, source_index, index))
            for source_index, source in enumerate(_SOURCES) for index in range(_DYNAMIC_REPEATS)]


def _schedules(population, checkpoint):
    fit = visible._tapes(population, visible._FIT_SEED, visible._FIT_REPEATS)
    station_dev = visible._tapes(population, visible._DEV_SEED, visible._DEV_REPEATS)
    station = visible._tapes(population, _STATION_ROOT, _STATION_REPEATS)
    dynamic = _dynamic_schedule(_DYNAMIC_ROOT)
    if len(station) != 80 or len(dynamic) != 36 or tuple(source for source, _ in population) != _SOURCES:
        raise ValueError('confirmation population differs from the frozen Step 3R design')
    station_seeds = [t.seed for t in (*fit, *station_dev, *station)]
    dynamic_seeds = [seed for _, _, seed in (*_dynamic_schedule(tracking._COHORT_SEEDS['dev']), *dynamic)]
    seeds = station_seeds + dynamic_seeds
    frozen_seeds = {t['seed'] for name in ('fit_tapes', 'temperature_tapes', 'calibration_tapes')
                    for t in checkpoint[name]}
    if len(set(seeds)) != len(seeds) or frozen_seeds.intersection(seeds):
        raise ValueError('Step 3R cohorts overlap one another or the frozen checkpoint cohorts')
    return fit, station_dev, station, dynamic


def _check_episode(episode, source, station, frames):
    if (episode.source_id, episode.station_id, len(episode.frames)) != (source, station, frames):
        raise ValueError('rendered tape differs from its scheduled source, request or frame count')


def _run_station(sim, schedule, windows, prior, scorer, params, models, certifier):
    from .station_classifier import make_segment_record

    readers = dict(zip(_ARMS, (Full_AGPS_Reader(prior, sim.kinematics, scorer=scorer, **params),
                              EvidenceRankedReader(prior, sim.kinematics, scorer=scorer, **params)), strict=True))
    tapes = []
    for i, tape in enumerate(schedule):
        episode = sim.episode(tape.source_id, tape.seed, station_id=tape.requested_station,
                              benchmark='station', frames=_STATION_FRAMES)
        _check_episode(episode, tape.source_id, tape.requested_station, _STATION_FRAMES)
        tape_hash = tracking._tape_hash(episode)
        for reader in readers.values():
            reader.reset()
        steps = {arm: [] for arm in _ARMS}
        elapsed_ns = {arm: [] for arm in _ARMS}
        first = []
        for j, frame in enumerate(episode.frames):
            public = (frame.obs, frame.probe_pose, frame.f_contact, frame.q_hat, frame.Sigma_q, frame.dt)
            order = _ARMS[::-1] if (i + j) % 2 else _ARMS
            first.append(order[0])
            for arm in order:
                step = readers[arm].step
                start = perf_counter_ns()
                estimate = step(*public)
                end = perf_counter_ns()
                elapsed_ns[arm].append(end - start)
                steps[arm].append(estimate)
        if tracking._tape_hash(episode) != tape_hash:
            raise RuntimeError('station tape changed during paired replay')

        heights = visible._pose_heights(readers[_ARMS[0]]._height, episode.frames)
        pose_only = pose_features(heights)
        source_windows = windows[tape.source_id]
        labels = [visible.label_frame(f.supervision, source_windows) for f in episode.frames]
        coverage = np.any([x.coverage for x in labels], axis=0)
        seen = np.any([x.visible for x in labels], axis=0)
        acquisition = _tape_truth(episode).acquisition()
        no_singleton = not any(x.clean_singletons for x in labels)
        shared = {**asdict(tape), 'tape_sha256': tape_hash, 'acquisition': acquisition,
                  'h_start_mm_scoring_only': episode.h_start_mm, 'no_clean_singleton_segment': no_singleton,
                  'first_online_step_by_frame': first,
                  'frames': [{'acquisition_h_mm': f.supervision.acquisition_h_mm, 'public_h_mm': float(h),
                              'nearest_native_row_h_mm': None if x.row is None else float(source_windows.h_mm[x.row]),
                              'coverage_C': visible._names(x.coverage), 'visible_V': visible._names(x.visible), **x.counts}
                             for f, h, x in zip(episode.frames, heights, labels, strict=True)]}
        samples = {}
        for arm, estimates in steps.items():
            features = {'pose_only': pose_only,
                        'visible': visible_features(heights, (x.identity_probabilities for x in estimates))}
            log = {'requested_station': tape.requested_station, 'empty_segment': acquisition['empty_echo_tape'],
                   'no_clean_singleton_segment': no_singleton, 'replay_tape_sha256': tape_hash,
                   'online_step_ns': elapsed_ns[arm],
                   'request_diagnostic': visible._request_diagnostic(tape.requested_station, episode.frames,
                                                                    labels, estimates),
                   'probabilities': {mode: dict(zip(STATIONS, model.predict_proba(features[mode]).tolist(), strict=True))
                                     for mode, model in models.items()}}
            record = make_segment_record(tape.source_id, tape.episode_id, episode.frames, estimates)
            samples[arm] = visible._Sample(log, seen, coverage, source_windows.opportunities, features, record)
        tapes.append((shared, samples))
    for arm in _ARMS:
        visible._audit([samples[arm] for _, samples in tapes], certifier)
    return tapes


def _station_flags(flags):
    return {(_OLD_VERIFIER_STAGE if k == visible._FUNNEL_STAGES[-1] else k): v for k, v in flags.items()}


def _station_summary(tapes):
    by_arm = {arm: [samples[arm] for _, samples in tapes] for arm in _ARMS}
    outcomes = []
    measured = []
    result = {}
    for arm, samples in by_arm.items():
        summary = visible._continuation(samples)
        summary['joint_counts'] = _station_flags(summary['joint_counts'])
        summary['old_verifier_decisions'] = summary.pop('raw_whole_segment_certificate')
        ns = [value for s in samples for value in s.log['online_step_ns']]
        outcomes.append([_station_flags(s.log['funnel_joint']) for s in samples])
        measured.append([err is not None for s in samples
                         for err in s.log['request_diagnostic']['world_errors_mm'].values()])
        result[arm] = {**summary, 'frozen_readouts': visible._classification(samples),
                       'required_geometry_missing_on_first_complete_frame': measured[-1].count(False),
                       'online_step': {'samples': len(ns), 'median_ms': float(np.median(ns) / 1.e6)}}
    return {'readers': result, 'paired_joint_counts': _paired_endpoints(*outcomes),
            'paired_required_geometry_measured_on_first_complete_frame': _paired_flags(*measured),
            'no_echo_tapes': sum(t['acquisition']['empty_echo_tape'] for t, _ in tapes),
            'no_lumen_tapes': sum(t['acquisition']['empty_lumen_tape'] for t, _ in tapes),
            'no_lumen_frames': sum(t['acquisition']['frames'] - t['acquisition']['frames_with_lumen_echo']
                                   for t, _ in tapes)}


def _station_tape_log(shared, samples):
    logs = {arm: dict(s.log) for arm, s in samples.items()}
    for log in logs.values():
        log['old_verifier_decision'] = log.pop('certificate')
        log['funnel_joint'] = _station_flags(log['funnel_joint'])
    return {**shared, 'readers': logs}


def _run_dynamic(sim, schedule, prior, scorer, params):
    readers = dict(zip(_ARMS, (tracking._DiagnosticReader(prior, sim.kinematics, scorer=scorer, **params),
                              _RankedDiagnostic(prior, sim.kinematics, scorer=scorer, **params)), strict=True))
    tapes = []
    for source, index, seed in schedule:
        episode = sim.episode(source, seed, benchmark='dynamic', frames=_DYNAMIC_FRAMES)
        _check_episode(episode, source, None, _DYNAMIC_FRAMES)
        tape_hash, truth = tracking._tape_hash(episode), _tape_truth(episode)
        cells = {arm: tracking._replay(reader, episode, truth, 'real_tracker', tape_hash)
                 for arm, reader in readers.items()}
        for cell in cells.values():
            if any(p.shape != m.shape for p, m in zip(cell.identity, truth.echo_masks, strict=True)):
                raise ValueError('paired dynamic replay changed the fixed echo-slot denominator')
        tapes.append(dict(episode_id=f'{_DYNAMIC_ROOT}:{source}:{index}', source=source, seed=seed,
                          tape_sha256=tape_hash, h_start_mm=episode.h_start_mm, truth=truth, cells=cells,
                          frames=[{'exact_masks': masks.tolist(), 'ghosts': list(f.supervision.echo_ghosts),
                                   'acquisition_h_mm': f.supervision.acquisition_h_mm}
                                  for f, masks in zip(episode.frames, truth.echo_masks, strict=True)]))
    return tapes


def _dynamic_criteria(tape, arm):
    result = tracking._tape_criteria(tape['truth'], tape['cells'][arm])
    # Exact NULL masks remain identity observations, but a no-lumen tape never earns task success.
    return {**result, 'lumen_and_all_masks_representable_and_correct':
            result['has_rendered_lumen'] and result['all_rendered_masks_representable_and_match_reference']}


def _nonempty_representable_prefix(truth, cell):
    representable = True
    for masks, row in zip(truth.echo_masks, cell.trace, strict=True):
        representable = representable and all(row['supported_exact_slots'])
        if masks.size:
            yield representable and row['real_search_funnel']['retained']['true_prefix']


def _geometry_pair(residuals, design, observed, finite_truth):
    measured = [np.isfinite(r).all(axis=-1) for r in residuals]
    comparable = observed & finite_truth & design[0] & design[1] & measured[0] & measured[1]
    return {'design_support': _paired_flags(*(d[observed] for d in design)),
            'design_support_with_finite_truth': _paired_flags(*((d & finite_truth)[observed] for d in design)),
            'geometry_measured': _paired_flags(*(m[observed] for m in measured)),
            'tracking_on_common_measured_roles': {arm: tracking._tracking(r[comparable], comparable.sum())
                                                  for arm, r in zip(_ARMS, residuals, strict=True)}}


def _dynamic_summary(tapes):
    observed = np.concatenate([t['truth'].observed for t in tapes])
    finite_truth = np.concatenate([t['truth'].supported for t in tapes])
    exact = np.concatenate([m for t in tapes for m in t['truth'].echo_masks])
    design, support, residuals, matches, prefix, criteria = [], [], [], [], [], []
    for arm in _ARMS:
        cells = [t['cells'][arm] for t in tapes]
        traces = [r for c in cells for r in c.trace]
        design.append(np.array([np.isin(np.arange(len(VesselRole)), r['design_roles']) for r in traces]))
        support.append([ok for r in traces for ok in r['supported_exact_slots']])
        residuals.append(np.concatenate([c.residuals for c in cells]))
        matches.append(np.concatenate([m for c in cells for m in c.identity]) == exact)
        prefix.append([survived for t, cell in zip(tapes, cells, strict=True)
                       for survived in _nonempty_representable_prefix(t['truth'], cell)])
        criteria.append([_dynamic_criteria(t, arm) for t in tapes])
    return {'readers': {arm: tracking._summary(tapes, arm) for arm in _ARMS},
            'paired_whole_tape_criteria': _paired_endpoints(*criteria),
            'paired_exact_echo_masks': _paired_flags(*matches),
            'paired_structurally_representable_exact_slots': _paired_flags(*support),
            'paired_representable_retained_true_prefix_on_nonempty_frames': _paired_flags(*prefix),
            'paired_geometry': _geometry_pair(residuals, design, observed, finite_truth),
            'paired_geometry_by_role': {role.name: _geometry_pair([r[:, role] for r in residuals],
                                                                [d[:, role] for d in design],
                                                                observed[:, role], finite_truth[:, role])
                                        for role in VesselRole},
            'no_echo_tapes': sum(t['truth'].acquisition()['empty_echo_tape'] for t in tapes),
            'no_lumen_tapes': sum(not t['truth'].observed.any() for t in tapes),
            'no_lumen_frames': int((~observed.any(axis=1)).sum())}


def _dynamic_tape_log(tape):
    truth = tape['truth']
    return {**{key: tape[key] for key in ('episode_id', 'source', 'seed', 'h_start_mm', 'tape_sha256', 'frames')},
            'acquisition': truth.acquisition(),
            'readers': {arm: {'replay_tape_sha256': cell.tape_sha256, 'frames': cell.trace,
                              'same_tape_criteria': _dynamic_criteria(tape, arm),
                              'tracking': tracking._tracking(cell.residuals, truth.observed.sum()),
                              'identity': tracking._identity(np.concatenate(truth.echo_masks), np.concatenate(cell.identity))}
                        for arm, cell in tape['cells'].items()}}


def _decision(station, dynamic):
    station_counts = station['paired_joint_counts']
    dynamic_counts = dynamic['paired_whole_tape_criteria']
    joint = {**{f'station.{name}': station_counts[name]['net']
                for name in (*visible._FUNNEL_STAGES[2:-1], _OLD_VERIFIER_STAGE)},
             f'dynamic.{_DYNAMIC_JOINT}': dynamic_counts[_DYNAMIC_JOINT]['net']}
    guards = {**joint,
              **{f'dynamic.{name}': dynamic_counts[name]['net']
                 for name in ('lumen_and_all_masks_representable_and_correct',
                              'all_rendered_role_centers_strictly_below_6_mm')},
              'station.geometry_measured': station['paired_required_geometry_measured_on_first_complete_frame']['net'],
              'dynamic.geometry_measured': dynamic['paired_geometry']['geometry_measured']['net']}
    improved = [name for name, change in joint.items() if change > 0]
    regressed = [name for name, change in guards.items() if change < 0]
    eligible = bool(improved and not regressed)
    return {'rule': 'At least one pooled downstream joint count must improve; none of the pooled downstream '
                    'counts or geometry measured counts on fixed opportunities may decrease. '
                    'Search survival, conditional RMSE and readout F1 are descriptive only.',
            'downstream_joint_count_deltas': joint, 'nonregression_count_deltas': guards,
            'improved_joint_counts': improved, 'regressed_guard_counts': regressed,
            'eligible_for_further_validation': eligible,
            'recommendation': 'continue_validation' if eligible else 'retain_v6_only', 'default_reader': 'v6',
            'candidate_independently_calibrated': False, 'formal_gate_3_opened': False, 'step_4_authorized': False}


def _verifier_metadata(certifier):
    state = certifier.state_dict()

    def finite_cutoff(cutoff):
        return cutoff if math.isfinite(cutoff) else 'unbounded'

    return {'alpha_used_by_v6': certifier.alpha, 'v6_calibration_segments': certifier.calibration_segments,
            'image_tolerance_mm': certifier.tolerance_mm, 'rank_used_by_v6': certifier.rank,
            'cutoff_used_by_v6': finite_cutoff(certifier.cutoff),
            'strata_used_by_v6': [{'rank': rank, 'cutoff': finite_cutoff(cutoff), 'segments': count}
                                  for rank, cutoff, count in state['_strata']],
            'candidate_use': 'same old verifier decision rules, not independent calibration or coverage evidence'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if HOLDOUT_ENV in os.environ:
        parser.error(f'{HOLDOUT_ENV} must be unset for train-source confirmation')
    if LOAO_OPEN3D.train_source_ids != _SOURCES or LOAO_OPEN3D.test_source_ids != ('open3dmodel_right',):
        parser.error('LOAO_OPEN3D source authority has changed')
    out = args.out.resolve()
    if args.out.is_symlink() or out.exists() or out.is_relative_to(_CHECKPOINT.parents[1]):
        parser.error('output must be new and outside the historical v6 results directory')
    for frozen_input in (_CHECKPOINT, _FIT_REPORT):
        if not frozen_input.is_file():
            parser.error(f'frozen input not found: {frozen_input}')

    bank = TrainingAnatomyBank.load()
    # The shared loader materializes the registered inventory; neither the
    # prior nor the renderer may access the split's held source from this bank.
    bank = replace(bank, _sources={source: bank.source(source) for source in _SOURCES})
    sim = DynamicScanSimulator(bank, LOAO_OPEN3D)
    windows = {s: visible.RawStationWindows.from_source(bank.source(s), sim._graph(s)) for s in _SOURCES}
    population = [(s, sim.station_ids(s)) for s in _SOURCES]
    if any(list(stations) != visible._names(windows[s].opportunities) for s, stations in population):
        raise ValueError('unmodified station sampler disagrees with raw Gate 2 windows')
    scorer, certifier, checkpoint = _load(argparse.Namespace(checkpoint=_CHECKPOINT, split='LOAO_OPEN3D'),
                                           LOAO_OPEN3D, population)
    if checkpoint['segment_frames'] != _STATION_FRAMES:
        raise ValueError('fixed station length must match the old verifier segment length')
    fit, station_dev, station_schedule, dynamic_schedule = _schedules(population, checkpoint)
    checkpoint_hash = _sha256(_CHECKPOINT)
    fit_bytes = _FIT_REPORT.read_bytes()
    _, models = visible._reuse_fit(json.loads(fit_bytes), checkpoint_hash, fit, station_dev)
    params = checkpoint['filter_parameters']
    prior = build_split_prior(bank, LOAO_OPEN3D)
    station_tapes = _run_station(sim, station_schedule, windows, prior, scorer, params, models, certifier)
    dynamic_tapes = _run_dynamic(sim, dynamic_schedule, prior, scorer, params)
    station, dynamic = _station_summary(station_tapes), _dynamic_summary(dynamic_tapes)
    report = {
        'schema': 'step3r-evidence-rank-confirm-v1',
        'scope': 'Independent training-source development seeds only: same prior, real associations and tracker pose. '
                 'No held-source rendering, Blender, oracle candidate, policy training, new calibration or formal gate.',
        'clinical_station_labels_available': False,
        'provenance': {'split': 'LOAO_OPEN3D', 'train_sources': _SOURCES, 'held_sources_rendered': [],
                       'training_source_provenance': {s: dict(bank.source(s).provenance) for s in _SOURCES},
                       'shared_geometry_provenance': checkpoint['shared_geometry_provenance'],
                       'checkpoint': {'path': str(_CHECKPOINT), 'sha256': checkpoint_hash,
                                      'schema_version': checkpoint['schema_version'], 'filter_parameters': params,
                                      'code_sha256': checkpoint['source_sha256']},
                       'frozen_readout_fit': {'path': str(_FIT_REPORT), 'sha256': hashlib.sha256(fit_bytes).hexdigest()},
                       'code_sha256': {**_source_sha256(), **{f'reader/{name}': _sha256(Path(__file__).with_name(name))
                                       for name in ('dev_rank_comparison.py', 'dev_visible_continuation.py',
                                                    'visible_stations.py', 'dev_tracking_interventions.py',
                                                    'evidence_ranked_reader.py')}},
                       'tape_sha256': 'agps-step3r-tracking-tape-v1: public controls and anonymous echoes plus '
                                      'offline pose, role centers, acquisition height, contributor masks and ghosts'},
        'thresholds': {'visible_readout_probability_at_least': DECISION_THRESHOLD, 'raw_row_half_bin_mm': 2.5,
                       'world_error_strictly_below_mm': 6., 'old_whole_segment_verifier': _verifier_metadata(certifier)},
        'station': {
            'schedule': {'root_seed': _STATION_ROOT, 'segments_per_supported_pair': _STATION_REPEATS,
                         'frames': _STATION_FRAMES, 'supported_pairs': dict(population),
                         'seed_derivation': 'first 8 SHA-256 bytes (big endian) of recorded episode_id'},
            'metric_scope': 'Raw nearest native rows (ties lower, no gap fill); first frame with independent clean '
                            'lumen echoes for the complete request. Whole-segment verifier uses all four online estimates. '
                            'Frozen development readouts are descriptive; neither was refitted.',
            'timing_scope': 'Local wall-clock per actual uninstrumented online step; alternating first arm by tape and '
                            'frame. Includes online scoring, excludes rendering, reset, I/O and audit. Not formal latency.',
            'all_sources': station,
            'by_source': {s: _station_summary([t for t in station_tapes if t[0]['source_id'] == s]) for s in _SOURCES},
            'tapes': [_station_tape_log(shared, samples) for shared, samples in station_tapes],
        },
        'dynamic': {
            'schedule': {'root_seed': _DYNAMIC_ROOT, 'segments_per_source': _DYNAMIC_REPEATS, 'frames': _DYNAMIC_FRAMES,
                         'seed_derivation': 'SeedSequence([root_seed, source_index, tape_index]).generate_state(1, '
                                            'dtype=uint64)[0]; both indices are zero-based'},
            'metric_scope': 'Real/tracker for both; truth only for scoring and passive stage tracing. Every rendered '
                            'role/frame counts even without geometry; complete RMSE is null if any role is missing '
                            'or no lumen rendered. Joint success requires every exact mask representable and correct, '
                            'and every rendered center measured with world error <6 mm on a tape with lumen.',
            'all_sources': dynamic,
            'by_source': {s: _dynamic_summary([t for t in dynamic_tapes if t['source'] == s]) for s in _SOURCES},
            'tapes': [_dynamic_tape_log(t) for t in dynamic_tapes],
        },
        'decision': _decision(station, dynamic),
    }
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(output)
    print(f'Wrote paired Step 3R training-source confirmation: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
