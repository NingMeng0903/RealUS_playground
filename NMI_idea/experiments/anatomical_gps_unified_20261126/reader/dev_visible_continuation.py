"""Training-source-only Step 3R visible-combination and continuation opportunity audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sim.anatomy_assets import LOAO_OPEN3D, NYSORA_STATIONS, TrainingAnatomyBank
from sim.audit_training_support import station_support_masks

from .visible_stations import DECISION_THRESHOLD, STATIONS, VisibleStationReadout, pose_features, visible_features


_SCOPE = 'LOAO_OPEN3D'
_ALLOWED_SOURCES = ('bodyparts3d_right', 'bodyparts3d_left_mirrored', 'medseg_right')
_CHECKPOINT = Path(__file__).resolve().parents[1] / 'results/step3_blind_v6/checkpoints/LOAO_OPEN3D.pt'
_FIT_SEED, _FIT_REPEATS = 731901, 12
_DEV_SEED, _DEV_REPEATS = 731902, 8
_CONFIRM_SEED, _CONFIRM_REPEATS = 731903, 8
_EVALUATION_COHORTS = {'dev': (_DEV_SEED, _DEV_REPEATS), 'confirm': (_CONFIRM_SEED, _CONFIRM_REPEATS)}
_READOUT_MODES = ('visible', 'pose_only')
_FRAMES = 4
_HALF_BIN_MM = 2.5
_TRACKING_TOLERANCE_MM = 6.
_ROLE_SETS = tuple(NYSORA_STATIONS[name] for name in STATIONS)
_FUNNEL_STAGES = ('scheduled', 'requested_combo_visible', 'visible_and_exact_identity',
                  'visible_identity_and_world_tracking', 'visible_identity_tracking_and_valid_certificate')
_CERT_REASONS = ('accepted', 'unbounded', 'singleton_ambiguity_or_empty_factor',
                 'geometric_ellipse_too_large', 'geometric_ellipse_unavailable', 'no_lumen')


@dataclass(frozen=True)
class RawStationWindows:
    """Original Gate 2 masks on the full native grid, including unsupported rows."""

    h_mm: np.ndarray
    masks: np.ndarray  # (native rows, five combinations), in strictly ascending row order

    def __post_init__(self):
        heights = np.asarray(self.h_mm, dtype=float)
        masks = np.asarray(self.masks, dtype=bool)
        if (heights.ndim != 1 or not heights.size or not np.isfinite(heights).all()
                or not np.all(np.diff(heights) > 0.) or masks.shape != (len(heights), len(STATIONS))):
            raise ValueError('raw Gate 2 masks require a finite ascending native grid and all five stations')
        object.__setattr__(self, 'h_mm', heights)
        object.__setattr__(self, 'masks', masks)

    @classmethod
    def from_source(cls, source, graph) -> RawStationWindows:
        _, masks = station_support_masks(source.h_mm, source.supported, graph)
        order = np.argsort(source.h_mm)
        return cls(source.h_mm[order], np.column_stack([masks[name][order] for name in STATIONS]))

    def nearest_row(self, height_mm: float) -> int | None:
        # Match against *all* native rows before testing station support. Going
        # directly to the closest supported row would fill missing windows.
        grid = self.h_mm
        if not grid[0] <= height_mm <= grid[-1]:
            return None
        upper = int(grid.searchsorted(height_mm, side='left'))
        lower = max(0, upper - 1)
        row = lower if height_mm - grid[lower] <= grid[upper] - height_mm else upper
        return row if abs(height_mm - grid[row]) <= _HALF_BIN_MM else None

    @property
    def opportunities(self) -> np.ndarray:
        return self.masks.any(axis=0)


@dataclass(frozen=True)
class Tape:
    episode_id: str
    source_id: str
    requested_station: str
    pair_index: int
    seed: int


def _tapes(population, root_seed, repeats) -> tuple[Tape, ...]:
    tapes = []
    for source_id, station_ids in population:
        for requested in station_ids:
            for idx in range(repeats):
                key = f'step3r-visible-v1:{root_seed}:{source_id}:{requested}:{idx}'
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
                tapes.append(Tape(key, source_id, requested, idx, seed))
    return tuple(tapes)


def _pose_heights(height_from_pose, frames) -> np.ndarray:
    # Use the reader's same height inversion, but before reading observations.
    # No echo-conditioned pose means, branch weights or station posteriors.
    return np.fromiter((height_from_pose(frame.probe_pose, frame.q_hat) for frame in frames), dtype=float)


@dataclass(frozen=True)
class FrameLabels:
    row: int | None
    coverage: np.ndarray
    visible: np.ndarray
    clean_singletons: tuple[tuple[int, int], ...]  # (echo slot, role)
    counts: dict


def label_frame(supervision, windows: RawStationWindows) -> FrameLabels:
    row = windows.nearest_row(supervision.acquisition_h_mm)
    coverage = windows.masks[row] if row is not None else np.zeros(len(STATIONS), dtype=bool)
    clean = []
    counts = dict(echoes=len(supervision.echo_targets), clean_singletons=0, true_null=0,
                  pure_ghost=0, mixed_ghost_lumen=0, unresolved_merge=0)
    for slot, (targets, ghost) in enumerate(zip(supervision.echo_targets, supervision.echo_ghosts, strict=True)):
        if ghost:
            counts['mixed_ghost_lumen' if targets else 'pure_ghost'] += 1
        elif len(targets) == 1:
            clean.append((slot, targets[0]))
        else:
            counts['unresolved_merge' if targets else 'true_null'] += 1
    counts['clean_singletons'] = len(clean)
    observed_roles = {role for _, role in clean}
    visible = coverage & np.array([roles <= observed_roles for roles in _ROLE_SETS])
    return FrameLabels(row, coverage, visible, tuple(clean), counts)


def _request_diagnostic(requested, frames, labels, estimates) -> dict:
    station = STATIONS.index(requested)
    earliest = next((idx for idx, frame_labels in enumerate(labels) if frame_labels.visible[station]), None)
    result = {'earliest_true_complete_frame_index': earliest, 'required_singleton_slots': [],
              'required_singleton_hits': 0, 'exact_identity_right': False, 'world_errors_mm': {},
              'world_tracking_under_6_mm': False}
    if earliest is None:
        return result
    estimate, truth = estimates[earliest], frames[earliest].supervision
    roles = sorted(_ROLE_SETS[station])
    witnesses = [(slot, role) for slot, role in labels[earliest].clean_singletons if role in roles]
    decisions = np.argmax(estimate.identity_probabilities, axis=1)
    hits = sum(int(decisions[slot]) == 1 << role for slot, role in witnesses)
    errors = np.linalg.norm(estimate.centers_mm[roles] - truth.centers_mm[roles], axis=1)
    result.update(required_singleton_slots=[{'echo_slot': slot, 'required_mask': 1 << role} for slot, role in witnesses],
                  required_singleton_hits=int(hits), exact_identity_right=bool(hits == len(witnesses)),
                  world_errors_mm={role.name: float(error) if np.isfinite(error) else None
                                   for role, error in zip(roles, errors, strict=True)},
                  world_tracking_under_6_mm=bool(np.all(np.isfinite(errors) & (errors < _TRACKING_TOLERANCE_MM))))
    return result


def _funnel_flags(request, certification) -> dict:
    """Instantaneous checks may pass individually on different tapes; intersect here first."""
    visible = request['earliest_true_complete_frame_index'] is not None
    identified = visible and request['exact_identity_right']
    tracked = identified and request['world_tracking_under_6_mm']
    completed = tracked and certification['accepted'] and not (certification['miscovered']
                                                                 or certification['false_certified'])
    return dict(zip(_FUNNEL_STAGES, (True, visible, identified, tracked, completed), strict=True))


def _certificate_reason(predicted_set, tolerance_mm) -> str:
    """First blocker in the frozen verifier's check order, on its actual predictive set."""
    if predicted_set.unbounded:
        return 'unbounded'
    has_lumen = False
    for frame in predicted_set.frames:
        candidates = frame.echo_candidates
        if (candidates.sum(axis=1) != 1).any():
            return 'singleton_ambiguity_or_empty_factor'
        mask = int(np.bitwise_or.reduce(candidates.argmax(axis=1), initial=0))
        roles = [role for role in range(len(frame.valid_geometry)) if mask & (1 << role)]
        if roles:
            has_lumen = True
            if not frame.valid_geometry[roles].all():
                return 'geometric_ellipse_unavailable'
            if (frame.ellipse_radii_mm[roles] > tolerance_mm).any():
                return 'geometric_ellipse_too_large'
    return 'accepted' if has_lumen else 'no_lumen'


@dataclass
class _Sample:
    log: dict
    visible: np.ndarray
    coverage: np.ndarray
    eligible: np.ndarray
    features: dict
    certificate_record: object | None


def _collect(sim, reader, tapes, windows_by_source, *, audit=False) -> list[_Sample]:
    from .station_classifier import make_segment_record

    samples = []
    for tape in tapes:
        if tape.source_id not in _ALLOWED_SOURCES:
            raise ValueError(f'not a primary training source: {tape.source_id!r}')
        episode = sim.episode(tape.source_id, tape.seed, station_id=tape.requested_station,
                              benchmark='station', frames=_FRAMES)
        if len(episode.frames) != _FRAMES:
            raise ValueError(f'unexpected station segment length on {tape.episode_id}')
        public_heights = _pose_heights(reader._height, episode.frames)
        reader.reset()
        estimates = tuple(reader.step(frame.obs, frame.probe_pose, frame.f_contact, frame.q_hat, frame.Sigma_q, frame.dt)
                          for frame in episode.frames)
        features = {'pose_only': pose_features(public_heights),
                    'visible': visible_features(public_heights, (est.identity_probabilities for est in estimates))}

        # Only these offline labels, metrics and diagnostics touch simulator supervision.
        windows = windows_by_source[tape.source_id]
        labels = tuple(label_frame(frame.supervision, windows) for frame in episode.frames)
        coverage = np.any([lab.coverage for lab in labels], axis=0)
        visible = np.any([lab.visible for lab in labels], axis=0)
        frame_logs = [
            {'acquisition_h_mm': float(frame.supervision.acquisition_h_mm), 'pose_estimated_h_mm': float(h),
             'nearest_native_row_h_mm': None if lab.row is None else float(windows.h_mm[lab.row]),
             'coverage_C': _names(lab.coverage), 'visible_V': _names(lab.visible), **lab.counts}
            for frame, h, lab in zip(episode.frames, public_heights, labels, strict=True)
        ]
        log = {**asdict(tape), 'h_start_mm_scoring_only': float(episode.h_start_mm),
               'supported_opportunities': _names(windows.opportunities), 'coverage_C': _names(coverage),
               'visible_V': _names(visible), 'frames_C_by_station': _counts([lab.coverage for lab in labels]),
               'frames_V_by_station': _counts([lab.visible for lab in labels]),
               'empty_segment': not any(lab.counts['echoes'] for lab in labels),
               'no_clean_singleton_segment': not any(lab.clean_singletons for lab in labels), 'frames': frame_logs}
        record = None
        if audit:
            log['request_diagnostic'] = _request_diagnostic(tape.requested_station, episode.frames, labels, estimates)
            record = make_segment_record(tape.source_id, tape.episode_id, episode.frames, estimates)
        samples.append(_Sample(log, visible, coverage, windows.opportunities, features, record))
    return samples


def _names(mask) -> list[str]:
    return [name for name, present in zip(STATIONS, mask, strict=True) if present]


def _counts(masks) -> dict:
    return dict(zip(STATIONS, np.asarray(masks, dtype=bool).reshape(-1, len(STATIONS)).sum(axis=0).tolist(), strict=True))


def _matrix(samples, attr) -> np.ndarray:
    return np.asarray([getattr(sample, attr) for sample in samples], dtype=bool).reshape(-1, len(STATIONS))


def _class_counts(samples) -> dict:
    eligible = _matrix(samples, 'eligible')
    coverage, visible = _matrix(samples, 'coverage'), _matrix(samples, 'visible')
    requested = Counter(s.log['requested_station'] for s in samples)
    return {name: {'scheduled_requests': requested[name], 'scored_supported': int(eligible[:, idx].sum()),
                   'unscored_unsupported': int((~eligible[:, idx]).sum()),
                   'coverage_C': int((eligible[:, idx] & coverage[:, idx]).sum()),
                   'visible_V': int((eligible[:, idx] & visible[:, idx]).sum()),
                   'C_without_V': int((eligible[:, idx] & coverage[:, idx] & ~visible[:, idx]).sum())}
            for idx, name in enumerate(STATIONS)}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _method_metrics(samples, mode) -> dict:
    eligible = _matrix(samples, 'eligible')
    truth = _matrix(samples, 'visible')
    pred = np.array([[s.log['probabilities'][mode][name] >= DECISION_THRESHOLD for name in STATIONS]
                     for s in samples], dtype=bool).reshape(-1, len(STATIONS))
    result = {}
    for i, name in enumerate(STATIONS):
        k, y, p = eligible[:, i], truth[:, i], pred[:, i]
        positives, predictions, tp = int((k & y).sum()), int((k & p).sum()), int((k & y & p).sum())
        result[name] = {'true_positives': tp, 'false_positives': predictions - tp, 'false_negatives': positives - tp,
                        'true_negatives': int((k & ~y & ~p).sum()), 'predicted_positives': predictions,
                        'positive_truth': positives, 'unscored_unsupported_predictions': int((~k & p).sum()),
                        'precision': _ratio(tp, predictions), 'recall': _ratio(tp, positives),
                        'f1': 2 * tp / (positives + predictions) if positives + predictions else 0.}
    zero_tp = [name for name, m in result.items() if not m['true_positives']]
    zero_truth = [name for name, m in result.items() if not m['positive_truth']]
    return {'by_station': result, 'macro_f1_including_all_five_classes': float(np.mean([m['f1'] for m in result.values()])),
            'classes_with_zero_tp': zero_tp, 'classes_with_no_positive_truth': zero_truth,
            'evidence_status': 'insufficient_evidence' if zero_tp else 'all_classes_have_true_positive_predictions'}


def _classification(samples) -> dict:
    visible, pose = (_method_metrics(samples, mode) for mode in ('visible', 'pose_only'))
    key = 'macro_f1_including_all_five_classes'
    return {'tapes': len(samples), 'class_counts': _class_counts(samples),
            'empty_segments': sum(s.log['empty_segment'] for s in samples),
            'no_clean_singleton_segments': sum(s.log['no_clean_singleton_segment'] for s in samples),
            'visible': visible, 'pose_only': pose, 'visible_minus_pose_macro_f1': visible[key] - pose[key]}


def _shared_height_control(samples, windows_by_source) -> dict:
    shared = defaultdict(list)
    possible = {(source, float(height)): _names(mask) for source, windows in windows_by_source.items()
                for height, mask in zip(windows.h_mm, windows.masks, strict=True) if mask.sum() > 1}
    for sample in samples:
        key = sample.log['source_id'], sample.log['h_start_mm_scoring_only']
        if key in possible:
            shared[key].append(sample)
    observed, duplicate = [], []
    for (source, height), group in sorted(shared.items()):
        requests = Counter(s.log['requested_station'] for s in group)
        observed.append({'source_id': source, 'h_start_mm': height, 'raw_gate2_stations': possible[source, height],
                         'counts_by_request': dict(requests), 'multiple_realized_requests': len(requests) > 1})
        if len(requests) > 1:
            duplicate.extend(group)
    return {'definition': 'same raw source and start height with different scheduled requests; independent tapes, not paired intervention',
            'possible_shared_height_groups': len(possible), 'observed_shared_height_groups': observed,
            'tapes_at_any_shared_height': sum(map(len, shared.values())),
            'tapes_at_realized_duplicate_request_heights': len(duplicate),
            'classification_at_realized_duplicate_request_heights': _classification(duplicate)}


def _audit(samples, certifier) -> None:
    records = [s.certificate_record for s in samples]
    for sample, record, outcome in zip(samples, records, certifier.outcomes(records), strict=True):
        miscovered, false_certified, accepted = map(bool, outcome)
        reason = _certificate_reason(certifier.predict_set(record.prediction), certifier.tolerance_mm)
        if accepted != (reason == 'accepted'):
            raise RuntimeError('reported reason disagrees with frozen certificate verifier')
        certification = dict(miscovered=miscovered, false_certified=false_certified, accepted=accepted,
                             first_verifier_outcome=reason)
        sample.log['certificate'] = certification
        sample.log['funnel_joint'] = _funnel_flags(sample.log['request_diagnostic'], certification)


def _continuation(samples) -> dict:
    count = len(samples)
    audit = [s.log['certificate'] for s in samples]
    reasons = Counter(a['first_verifier_outcome'] for a in audit)
    accepted = sum(a['accepted'] for a in audit)
    false = sum(a['false_certified'] for a in audit)
    violations_on_accepted = sum(a['accepted'] and (a['false_certified'] or a['miscovered']) for a in audit)
    return {'denominator_scheduled_tapes': count,
            'joint_counts': {key: sum(s.log['funnel_joint'][key] for s in samples) for key in _FUNNEL_STAGES},
            'raw_whole_segment_certificate': {
                'tapes': count, 'accepted': accepted, 'acceptance_rate': _ratio(accepted, count),
                'miscovered': sum(a['miscovered'] for a in audit), 'false_certified': false,
                'miscovered_on_accepted': sum(a['accepted'] and a['miscovered'] for a in audit),
                'violations_on_accepted': violations_on_accepted,
                'conditional_false_certification_rate': _ratio(false, accepted),
                'conditional_violation_rate': _ratio(violations_on_accepted, accepted),
                'first_verifier_outcome_counts': {key: reasons[key] for key in _CERT_REASONS}}}


def _reuse_fit(report, checkpoint_sha256, fit_tapes, dev_tapes) -> tuple[dict, dict]:
    """Restore exactly the development artifact's weights; never refit on confirmation."""
    provenance = report.get('provenance', {})
    if (report.get('schema') != 'step3r-visible-continuation-dev-v1'
            or report.get('cohort', 'dev') != 'dev'
            or provenance.get('split') != _SCOPE
            or provenance.get('train_source_ids') != list(_ALLOWED_SOURCES)
            or provenance.get('checkpoint_sha256') != checkpoint_sha256):
        raise ValueError('confirmation requires a development fit report from the same frozen split/checkpoint')
    fit = report['fit']
    for logged, expected in ((fit['tapes'], fit_tapes), (report['dev']['tapes'], dev_tapes)):
        tapes = [{key: row[key] for key in Tape.__dataclass_fields__} for row in logged]
        if tapes != [asdict(tape) for tape in expected]:
            raise ValueError('confirmation requires the fixed development fit and evaluation tape schedules')
    models = {mode: VisibleStationReadout.from_state_dict(report['fitted_readouts'][mode])
              for mode in _READOUT_MODES}
    if any(model.mode != mode for mode, model in models.items()):
        raise ValueError('development fit report has mismatched comparator states')
    return fit, models


def _validate_args(parser, args):
    if 'AGPS_EVAL_HOLDOUT_ALLOWED' in os.environ:
        parser.error('Step 3R training-source development requires the holdout authorization variable to be absent')
    if (LOAO_OPEN3D.train_source_ids != _ALLOWED_SOURCES or LOAO_OPEN3D.test_source_ids != ('open3dmodel_right',)
            or set(_ALLOWED_SOURCES) & set(LOAO_OPEN3D.test_source_ids)):
        parser.error('LOAO_OPEN3D source authority has changed')
    out = args.out.resolve()
    if args.out.is_symlink() or out.exists() or out.is_relative_to(_CHECKPOINT.parents[1]):
        parser.error('output must be new and outside the historical Step 3 checkpoint/results directory')
    if (args.cohort == 'confirm') != (args.fit_report is not None):
        parser.error('--fit-report is required for confirmation and not allowed for development')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=_EVALUATION_COHORTS, default='dev',
                        help='run development (default) or the disjoint confirmation cohort')
    parser.add_argument('--fit-report', type=Path, help='existing development JSON supplying the unchanged weights')
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    from .baselines import Full_AGPS_Reader
    from .eval_reader import _load, _sha256, _source_sha256
    from .split_prior import DynamicScanSimulator, build_split_prior

    bank = TrainingAnatomyBank.load()
    bank = replace(bank, _sources={source: bank.source(source) for source in _ALLOWED_SOURCES})
    split = LOAO_OPEN3D
    sim = DynamicScanSimulator(bank, split)
    windows = {source: RawStationWindows.from_source(bank.source(source), sim._graph(source))
               for source in _ALLOWED_SOURCES}
    population = [(source, sim.station_ids(source)) for source in _ALLOWED_SOURCES]
    for source, stations in population:
        if list(stations) != _names(windows[source].opportunities):
            raise ValueError(f'unmodified sampler and raw Gate 2 scoring opportunities disagree for {source}')
    scorer, certifier, checkpoint = _load(argparse.Namespace(checkpoint=_CHECKPOINT, split=_SCOPE), split, population)
    if checkpoint['segment_frames'] != _FRAMES:
        raise ValueError('development tape length must match frozen certificate segment length')
    params = checkpoint['filter_parameters']
    reader = Full_AGPS_Reader(build_split_prior(bank, split), sim.kinematics, scorer=scorer, **params)
    fit_tapes = _tapes(population, _FIT_SEED, _FIT_REPEATS)
    eval_root, eval_repeats = _EVALUATION_COHORTS[args.cohort]
    eval_tapes = _tapes(population, eval_root, eval_repeats)
    old_dev_tapes = _tapes(population, _DEV_SEED, _DEV_REPEATS) if args.cohort == 'confirm' else ()
    schedules = (*fit_tapes, *eval_tapes, *old_dev_tapes)
    all_seeds = {tape.seed for tape in schedules}
    checkpoint_seeds = {t['seed'] for field in ('fit_tapes', 'temperature_tapes', 'calibration_tapes')
                        for t in checkpoint[field]}
    if len(all_seeds) != len(schedules) or all_seeds & checkpoint_seeds:
        raise RuntimeError('development/confirmation tapes overlap one another or frozen-reader tapes')
    checkpoint_sha256 = _sha256(_CHECKPOINT)
    if args.cohort == 'dev':
        fit = _collect(sim, reader, fit_tapes, windows)
        y = _matrix(fit, 'visible')
        trained = {mode: VisibleStationReadout.fit(mode, [s.features[mode] for s in fit], y) for mode in _READOUT_MODES}
        fit_summary = {'class_counts': _class_counts(fit), 'tapes': [s.log for s in fit]}
        reused_fit = None
    else:
        fit_bytes = args.fit_report.read_bytes()
        fit_summary, trained = _reuse_fit(json.loads(fit_bytes), checkpoint_sha256, fit_tapes, old_dev_tapes)
        reused_fit = {'path': str(args.fit_report.resolve()), 'sha256': hashlib.sha256(fit_bytes).hexdigest()}
    evaluated = _collect(sim, reader, eval_tapes, windows, audit=True)
    for sample in evaluated:
        sample.log['probabilities'] = {mode: dict(zip(STATIONS, model.predict_proba(sample.features[mode]).tolist(),
                                                     strict=True)) for mode, model in trained.items()}
    _audit(evaluated, certifier)
    by_source = {source: [s for s in evaluated if s.log['source_id'] == source] for source in _ALLOWED_SOURCES}
    report = {
        'schema': f'step3r-visible-continuation-{args.cohort}-v1', 'cohort': args.cohort,
        'scope': 'training-source development protocol only; no held evaluation, RL success or clinical station claims',
        'clinical_station_labels_available': False,
        'label_policy': {'C': 'any actual acquisition in the original same-source Gate 2 window, nearest native '
                              'row within 2.5 mm, ties lower, no gap fill or extrapolation',
                         'V': 'exists one C frame with distinct clean exact singleton echoes for all required roles',
                         'opportunities': 'raw same-source window nonempty; mask used for scoring, not features or fit selection',
                         'macro_f1': 'all five classes, zero F1 when both truth and predicted positive counts are zero'},
        'funnel_scope': 'joint counts on scheduled four-frame station tapes, not continuous tracking or RL success; '
                        'world error uses first truly complete frame and frozen certificate uses all four online estimates',
        'certification_scope': 'frozen whole-segment verifier applied descriptively on the chosen balanced cohort; '
                               'no rank coverage or conditional risk guarantee claimed',
        'provenance': {
            'split': _SCOPE, 'train_source_ids': list(split.train_source_ids), 'held_sources_rendered': [],
            'checkpoint': str(_CHECKPOINT), 'checkpoint_sha256': checkpoint_sha256, 'reused_fit_report': reused_fit,
            'checkpoint_schema_version': checkpoint['schema_version'], 'filter_parameters': params,
            'checkpoint_code_sha256': checkpoint['source_sha256'],
            'current_code_sha256': {**_source_sha256(), **{f'reader/{filename}': _sha256(Path(__file__).with_name(filename))
                                                        for filename in ('visible_stations.py', 'dev_visible_continuation.py')}},
            'shared_geometry_provenance': checkpoint['shared_geometry_provenance'],
            'training_source_provenance': {source: dict(bank.source(source).provenance) for source in _ALLOWED_SOURCES},
            'segment_frames': _FRAMES, 'benchmark': 'unmodified station', 'decision_threshold': DECISION_THRESHOLD,
            'development_seeds': {'fit_root': _FIT_SEED, 'fit_per_pair': _FIT_REPEATS,
                                  'dev_root': _DEV_SEED, 'dev_per_pair': _DEV_REPEATS,
                                  'confirmation_root': _CONFIRM_SEED, 'confirmation_per_pair': _CONFIRM_REPEATS,
                                  'evaluation_root': eval_root, 'evaluation_per_pair': eval_repeats},
            'episode_seed_derivation': 'first 8 SHA-256 bytes (big endian) of episode_id; no retries or culling',
            'role_sets': {name: sorted(role.name for role in NYSORA_STATIONS[name]) for name in STATIONS},
            'raw_gate2_rows_mm_by_source': {source: {name: w.h_mm[w.masks[:, i]].tolist() for i, name in enumerate(STATIONS)}
                                          for source, w in windows.items()},
            'frozen_certificate': {'tolerance_mm_image_space': certifier.tolerance_mm, 'alpha': certifier.alpha,
                                   'calibration_segments': certifier.calibration_segments}},
        'fitted_readouts': {mode: model.state_dict() for mode, model in trained.items()},
        'fit': fit_summary,
        args.cohort: {'all_sources': _classification(evaluated), 'continuation': _continuation(evaluated),
                      'per_source': {source: {'classification': _classification(items), 'continuation': _continuation(items)}
                                     for source, items in by_source.items()},
                      'shared_height_control': _shared_height_control(evaluated, windows),
                      'tapes': [s.log for s in evaluated]},
    }
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(output)
    print(f'Wrote training-source Step 3R diagnostic: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
