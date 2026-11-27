"""Step 3R development-only 2×2 decomposition of dynamic reader errors.

Truth is used offline for scoring and, in designated cells, to choose contributor
sets or replace the public pose measurement. No truth center or acquisition
height enters the reader. Nothing here is a deployment mode or a held-source gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import log_expit

from sim.anatomy_assets import LOAO_OPEN3D, TrainingAnatomyBank, VesselRole

from .baselines import Full_AGPS_Reader
from .dynamic_factor_graph import (
    ARTERY, ROLE_COUNT, _Association, _COMPRESSIVE, _LOG_2PI, _contributor_masks, _physical_scores,
)
from .eval_reader import (
    HOLDOUT_ENV, _bootstrap_weights, _load, _sha256, _source_sha256, _station_population, _tape_truth,
)
from .split_prior import DynamicScanSimulator, build_split_prior
from .station_classifier import STATION_IDS


_SPLIT = 'LOAO_OPEN3D'
_SOURCES = ('bodyparts3d_right', 'bodyparts3d_left_mirrored', 'medseg_right')
_COHORT_SEEDS = {'dev': 931901, 'confirm': 931902}
_EPISODES_PER_SOURCE, _FRAMES, _BOOTSTRAP = 12, 8, 1000
_CHECKPOINT = Path(__file__).resolve().parents[1] / 'results/step3_blind_v6/checkpoints/LOAO_OPEN3D.pt'
_CELLS = ('real_tracker', 'forced_tracker', 'real_true_pose', 'forced_true_pose')
_CONTRASTS = (
    ('force_association_given_tracker', 'real_tracker', 'forced_tracker'),
    ('force_association_given_true_pose', 'real_true_pose', 'forced_true_pose'),
    ('replace_pose_given_real_association', 'real_tracker', 'real_true_pose'),
    ('replace_pose_given_forced_association', 'forced_tracker', 'forced_true_pose'),
)
_ROLE_NAMES = (*[r.name for r in VesselRole], 'NULL')
_UNAVAILABLE = -1


def _episode_seed(root, source_index, episode_index):
    return int(np.random.SeedSequence([root, source_index, episode_index]).generate_state(1, dtype=np.uint64)[0])


def _tape_hash(episode):
    """Length-delimited metadata and shape-tagged little-endian numeric bytes."""
    digest = hashlib.sha256(b'agps-step3r-tracking-tape-v1\0')

    def add(data):
        digest.update(struct.pack('<Q', len(data)))
        digest.update(data)

    def numbers(value):
        x = np.asarray(value, dtype='<f8')
        add(bytes([x.ndim]) + struct.pack(f'<{x.ndim}Q', *x.shape) + x.tobytes(order='C'))

    add(episode.source_id.encode())
    numbers(episode.h_start_mm)
    for frame in episode.frames:
        truth, probe = frame.supervision, frame.probe_pose
        for value in (frame.obs.echoes, probe.origin_mm, probe.rotation, frame.q_hat, frame.Sigma_q,
                      truth.true_q_rad, truth.centers_mm,
                      [probe.speed_mm_s, probe.dwell_s, frame.f_contact, frame.dt, truth.acquisition_h_mm]):
            numbers(value)
        add(bytes(_contributor_masks(truth.echo_targets)))
        add(bytes(truth.echo_ghosts))
    return digest.hexdigest()


def _unsupported_slots(groups, roles):
    counts = Counter(role for group in groups for role in group)
    role_set = set(roles)
    reasons = []
    for group in groups:
        missing = [int(r) for r in group if r not in role_set]
        reused = [int(r) for r in group if counts[r] > 1]
        reasons.append({'missing_design_roles': missing, 'reused_roles': reused} if missing or reused else None)
    return reasons


def _merge_log_probability(members, image, covariance, distance):
    """Same connected-component contact model as the bounded reader search.

    Track the search cutoff separately from its likelihood: forcing a supported
    true group bypasses truncation, not the distance-dependent contact score.
    """
    edges = []
    for n, i in enumerate(members):
        for j in members[n + 1:]:
            a, b = slice(2 * i, 2 * i + 2), slice(2 * j, 2 * j + 2)
            delta = image[i] - image[j]
            separation = np.linalg.norm(delta)
            axis = delta / max(separation, 1.e-9)
            diff = covariance[a, a] + covariance[b, b] - covariance[a, b] - covariance[b, a]
            spread = np.sqrt(max(axis @ diff @ axis, .5))
            score = float(log_expit(1.65 * (distance - separation) / spread))
            edges.append((score, i, j))
    components = {i: i for i in members}
    score, least = 0., 0.
    for weight, i, j in sorted(edges, reverse=True):
        a, b = components[i], components[j]
        if a == b:
            continue
        components = {k: a if v == b else v for k, v in components.items()}
        score += weight
        least = weight
        if len(set(components.values())) == 1:
            break
    return score, least >= np.log(.035) and (len(members) == 2 or score >= -4.5)


def _known_association(groups, supported, frame, roles, pred, covariance, caliber_mean, caliber_std,
                       *, construct):
    """Score just the fixed groups, without ranking or synthesizing centroids."""
    excess = max(0., frame.force - 2.5)
    local = {int(role): i for i, role in enumerate(roles)}
    members = [tuple(local[r] for r in group) if ok else () for group, ok in zip(groups, supported, strict=True)]
    merge = {i: _merge_log_probability(group, pred.image, covariance, 4.7 + frame.probe.speed_mm_s * .06)
             for i, group in enumerate(members) if len(group) > 1}
    admissible = tuple(ok and (i not in merge or merge[i][1]) for i, ok in enumerate(supported))
    if not construct:
        return None, admissible

    slots = np.array([i for i, group in enumerate(members) if group], dtype=np.intp)
    operator = np.zeros((2 * len(slots), 2 * len(roles)))
    for row, slot in enumerate(slots):
        group = groups[slot]
        # The production transform uses shared baseline areas and morphology-
        # class compression, never the measured diameter or simulator weights.
        compliance = _COMPRESSIVE[[int(role in ARTERY) for role in group]]
        area = (1. - np.minimum(.95, excess * compliance))**2
        columns = 2 * np.asarray(members[slot], dtype=np.intp)
        operator[2 * row, columns] = operator[2 * row + 1, columns + 1] = area / area.sum()
    valid_slots = [i for i, ok in enumerate(supported) if ok]
    valid_groups = [groups[i] for i in valid_slots]
    cues = _physical_scores(frame.echoes[valid_slots], valid_groups, excess, frame.probe.dwell_s,
                            caliber_mean, caliber_std)
    acoustic = (float(np.trace(cues)) + sum(s for s, _ in merge.values())
                + sum(not g for g in valid_groups) * np.log(.42 / (50. * 77.)))
    score = np.log1p(-pred.visible).sum() + acoustic
    residual = frame.echoes[slots, :2].ravel() - operator @ pred.image.ravel()
    if len(slots):
        matched = np.any(operator[:, ::2], axis=0)
        score += np.log(pred.visible[matched] / (1. - pred.visible[matched])).sum()
        root = cholesky(operator @ covariance @ operator.T, lower=True, check_finite=False)
        whitened = solve_triangular(root, residual, lower=True, check_finite=False)
        score -= .5 * (whitened @ whitened + len(residual) * _LOG_2PI) + np.log(root.diagonal()).sum()
    # Unsupported slots preserve indexing but have neither a geometry factor
    # nor NULL evidence. The offline recorder reports them as abstentions.
    return _Association(float(score), tuple(g if ok else () for g, ok in zip(groups, supported, strict=True)),
                        slots, operator, residual, float(acoustic)), admissible


def _matches_path(path, masks):
    for expected in reversed(masks):
        if path is None or path.masks != expected:
            return False
        path = path.previous
    return path is None


class _DiagnosticReader(Full_AGPS_Reader):
    """Instrument one offline run; real cells use the original search unchanged."""

    def _propose(self, frame, design, parents, seeds, rho):
        self.unrepresentable = _unsupported_slots(self.groups, design.roles)
        self.available = tuple(reason is None for reason in self.unrepresentable)
        self.stages = [dict(any_parent=False, true_prefix=False) for _ in range(3)]
        self.per_slot_admissible = np.zeros(len(self.groups), dtype=bool)
        self.per_slot_local_beam = self.per_slot_admissible.copy()
        self.parents_on_prefix = tuple(_matches_path(p.path, self.previous_truth) for p in parents)
        proposals = super()._propose(frame, design, parents, seeds, rho)
        if not self.forced:
            for p in proposals:
                if p.masks == self.masks:
                    self._mark(2, self.parents_on_prefix[p.parent])
        return proposals

    def _mark(self, stage, on_prefix):
        self.stages[stage]['any_parent'] = True
        self.stages[stage]['true_prefix'] |= on_prefix

    def _associate(self, frame, design, branch, pred, *, solved):
        projected = pred.H @ solved[design.cols]
        covariance = pred.variance + .5 * (projected + projected.T)
        known, admissible = _known_association(
            self.groups, self.available, frame, design.roles, pred, covariance,
            self.caliber_mean_mm, self.caliber_std_mm, construct=self.forced,
        )
        self.per_slot_admissible |= np.asarray(admissible, dtype=bool)
        if self.forced:
            return (known,)
        on_prefix = _matches_path(branch.path, self.previous_truth)
        if all(admissible):
            self._mark(0, on_prefix)
        candidates = super()._associate(frame, design, branch, pred, solved=solved)
        for a in candidates:
            mask = _contributor_masks(a.groups)
            self.per_slot_local_beam |= np.fromiter((x == y for x, y in zip(mask, self.masks, strict=True)), dtype=bool)
            if mask == self.masks:
                self._mark(1, on_prefix)
        return candidates

    def _accumulate_presence(self, masks, parent_path, rate):
        if self.forced:
            masks = tuple(m for m, ok in zip(masks, self.available, strict=True) if ok)
        return super()._accumulate_presence(masks, parent_path, rate)


@dataclass
class _Cell:
    trace: list
    residuals: np.ndarray  # (frame, rendered role, lateral/elevation/depth), NaN if unmeasured
    identity: list  # forced decisions are interventions, not reader achievements
    scorer_identity: list
    tape_sha256: str


def _replay(reader, episode, truth, cell, tape_sha256):
    reader.reset()
    reader.forced = cell.startswith('forced_')
    exact_pose = cell.endswith('_true_pose')
    zero = np.zeros((4, 4))
    reader.previous_truth = ()
    trace, labels, scorer_labels = [], [], []
    residuals = np.full_like(truth.centers, np.nan)
    for i, frame in enumerate(episode.frames):
        supervision = frame.supervision
        reader.groups = supervision.echo_targets
        reader.masks = tuple(truth.echo_masks[i].tolist())
        q = supervision.true_q_rad if exact_pose else frame.q_hat
        cov = zero if exact_pose else frame.Sigma_q
        estimate = reader.step(frame.obs, frame.probe_pose, frame.f_contact, q, cov, frame.dt)
        design_roles = reader._history[-1][1]
        branches = reader._branches
        support = np.isin(np.arange(ROLE_COUNT), design_roles)
        eligible_roles = support.copy()
        abstain = ~np.array(reader.available, dtype=bool)
        if reader.forced:
            eligible_roles[:] = False
            for group, ok in zip(reader.groups, reader.available, strict=True):
                if group and ok:
                    eligible_roles[list(group)] = True
        scorer_masks = np.argmax(estimate.identity_probabilities, axis=1)
        if reader.forced:
            scorer_masks[abstain] = _UNAVAILABLE
            decisions = np.array(branches[0].path.masks, dtype=np.intp)
            decisions[abstain] = _UNAVAILABLE
        else:
            decisions = scorer_masks
        labels.append(decisions)
        scorer_labels.append(scorer_masks)
        observed, finite_truth = truth.observed[i], truth.supported[i]
        measured = observed & finite_truth & eligible_roles & np.isfinite(estimate.centers_mm).all(axis=1)
        unmeasured = {
            role.name: ('missing_slice_truth' if not finite_truth[role] else
                        'missing_reader_design_role' if not support[role] else
                        'unrepresentable_forced_mask' if not eligible_roles[role] else 'nonfinite_reader_center')
            for role in VesselRole if observed[role] and not measured[role]
        }
        world_residuals = estimate.centers_mm[measured] - truth.centers[i, measured]
        residuals[i, measured] = world_residuals @ frame.probe_pose.rotation
        reader.previous_truth += (reader.masks,)
        funnel = None
        if not reader.forced:
            funnel = dict(zip(('gate_admissible', 'local_topk', 'refinement_proposal'), reader.stages, strict=True))
            funnel['retained'] = {
                'any_parent': any(b.path.masks == reader.masks for b in branches),
                'true_prefix': any(_matches_path(b.path, reader.previous_truth) for b in branches),
            }
        trace.append({
            'measured_world_xyz_residuals_mm': dict(zip(
                (role.name for role in VesselRole if measured[role]), world_residuals.tolist(), strict=True)),
            'frame': i, 'design_roles': design_roles.tolist(),
            'supported_exact_slots': list(reader.available),
            'unrepresentable_exact_slots': reader.unrepresentable,
            'admissible_exact_slots_any_seed': reader.per_slot_admissible.tolist(),
            'local_topk_exact_slots_any_seed': None if reader.forced else reader.per_slot_local_beam.tolist(),
            'forced_abstain_slots': np.flatnonzero(abstain).tolist() if reader.forced else [],
            'mask_decisions': [None if mask < 0 else int(mask) for mask in decisions],
            'scorer_mask_decisions': [None if mask < 0 else int(mask) for mask in scorer_masks],
            'branches_retained': len(branches), 'real_search_funnel': funnel,
            'h_error_mm': float(estimate.h_mm - supervision.acquisition_h_mm),
            'input_pose_error_rad': (q - supervision.true_q_rad).tolist(),
            'filtered_pose_error_rad': (estimate.pose_mean_rad - supervision.true_q_rad).tolist(),
            'unmeasured_rendered_role_reasons': unmeasured,
            'rendered_role_residual_probe_xyz_mm': {role.name: residuals[i, role].tolist() if measured[role] else None
                                                   for role in VesselRole if truth.observed[i, role]},
        })
    replay_hash = _tape_hash(episode)
    if replay_hash != tape_sha256:
        raise RuntimeError('public inputs or offline supervision changed during replay')
    return _Cell(trace, residuals, labels, scorer_labels, replay_hash)


def _ratio(a, b):
    return float(a / b) if b else None


def _squared(residuals):
    parts = residuals**2
    return np.stack((parts.sum(axis=-1), parts[..., 0] + parts[..., 2], parts[..., 1]), axis=-1)


_RESIDUALS = ('world', 'image_plane', 'elevation')


def _tracking(residuals, opportunities):
    sq = _squared(residuals)
    measured = int(np.isfinite(sq[..., 0]).sum())
    sse = np.nansum(sq.reshape(-1, len(_RESIDUALS)), axis=0)
    rmse = {axis: float(np.sqrt(total / measured)) if measured else None
            for axis, total in zip(_RESIDUALS, sse, strict=True)}
    return {'rendered': int(opportunities), 'measured': measured, 'unmeasured': int(opportunities) - measured,
            'world_error_strictly_below_6_mm': int((sq[..., 0] < 36.).sum()),
            'sum_squared_mm2': dict(zip(_RESIDUALS, sse.tolist(), strict=True)), 'conditional_rmse_mm': rmse,
            'complete_rmse_mm': rmse if measured == opportunities else dict.fromkeys(_RESIDUALS)}


def _identity(expected, predicted):
    valid = predicted >= 0
    true_size = np.array([int(m).bit_count() for m in expected], dtype=int)
    exact = valid & (expected == predicted)
    # Multi-label membership F1, each merge contributor an independent role
    # opportunity. NULL is a separate output; a missing decision is never NULL.
    bits = 1 << np.arange(ROLE_COUNT)
    actual_sets = np.column_stack(((expected[:, None] & bits) != 0, expected == 0))
    pred_sets = np.column_stack(((predicted[:, None] & bits) != 0, predicted == 0)) & valid[:, None]
    counts = np.stack((actual_sets.sum(axis=0), (actual_sets & valid[:, None]).sum(axis=0),
                       pred_sets.sum(axis=0), (actual_sets & pred_sets).sum(axis=0))).T
    roles = {
        role: dict(truth_memberships=truth, truth_memberships_on_decisions=decided,
                   predicted_memberships=pred, true_positives=tp,
                   f1=_ratio(2 * tp, truth + pred), f1_on_decisions=_ratio(2 * tp, decided + pred))
        for role, (truth, decided, pred, tp) in zip(_ROLE_NAMES, counts.tolist(), strict=True)
    }
    macro = {}
    for suffix, support_key, f1_key in (
        ('all_rendered', 'truth_memberships', 'f1'),
        ('decisions', 'truth_memberships_on_decisions', 'f1_on_decisions'),
    ):
        values = [r[f1_key] for r in roles.values() if r[support_key]]
        macro[f'macro_f1_on_{suffix}_truth_present'] = _ratio(sum(values), len(values))
    return {'rendered_echoes': len(expected), 'decisions': int(valid.sum()), 'abstentions': int((~valid).sum()),
            'exact_mask_hits': int(exact.sum()), 'exact_mask_accuracy_on_all_rendered': _ratio(exact.sum(), len(expected)),
            'exact_mask_accuracy_on_decisions': _ratio(exact.sum(), valid.sum()),
            'exact_by_true_contributor_count': {
                str(k): {'rendered': int((true_size == k).sum()), 'decisions': int((valid & (true_size == k)).sum()),
                         'exact_hits': int((exact & (true_size == k)).sum())} for k in range(ROLE_COUNT + 1)},
            'by_role': roles, **macro, 'micro_f1': _ratio(2 * counts[:, 3].sum(), counts[:, [0, 2]].sum()),
            'micro_f1_on_decisions': _ratio(2 * counts[:, 3].sum(), counts[:, 1:3].sum())}


def _tape_criteria(truth, result):
    """Fixed-denominator episode outcomes, not a certification or continuation test."""
    track = _tracking(result.residuals, truth.observed.sum())
    exact = np.concatenate(truth.echo_masks) == np.concatenate(result.identity)
    complete_world = track['complete_rmse_mm']['world']
    observable = bool(track['rendered'])
    all_roles_close = observable and track['world_error_strictly_below_6_mm'] == track['rendered']
    all_exact = bool(exact.size and exact.all())
    all_exact_representable = all_exact and all(all(row['supported_exact_slots']) for row in result.trace)
    return {
        'has_rendered_lumen': observable,
        'all_rendered_role_centers_measured': complete_world is not None,
        'complete_world_rmse_at_most_4_mm': complete_world is not None and complete_world <= 4.,
        'all_rendered_role_centers_strictly_below_6_mm': all_roles_close,
        'all_rendered_masks_match_reference': all_exact,
        'all_rendered_masks_representable_and_match_reference': all_exact_representable,
        'joint_representable_masks_match_and_all_role_centers_below_6_mm': all_exact_representable and all_roles_close,
    }


def _summary(tapes, cell):
    known = np.concatenate([np.concatenate(t['truth'].echo_masks) for t in tapes])
    traces = [row for t in tapes for row in t['cells'][cell].trace]
    measured = np.concatenate([t['cells'][cell].residuals for t in tapes], axis=0)
    observed = np.concatenate([t['truth'].observed for t in tapes], axis=0)
    protected = np.array([np.isin(np.arange(ROLE_COUNT), row['design_roles']) for row in traces])
    labels = np.concatenate([m for t in tapes for m in t['cells'][cell].identity])
    scorer_labels = np.concatenate([m for t in tapes for m in t['cells'][cell].scorer_identity])
    nonempty = [f for f in traces if f['mask_decisions']]
    supported = np.fromiter((ok for row in traces for ok in row['supported_exact_slots']), dtype=bool)
    unavailable = [r for row in traces for r in row['unrepresentable_exact_slots'] if r is not None]
    criteria = [_tape_criteria(t['truth'], t['cells'][cell]) for t in tapes]
    funnel = None
    if not cell.startswith('forced_'):
        funnel = {'nonempty_frames': len(nonempty),
                  'structurally_representable_frames': sum(all(f['supported_exact_slots']) for f in nonempty),
                  'exact_slots_returned_by_any_local_topk': sum(sum(f['local_topk_exact_slots_any_seed']) for f in nonempty),
                  **{stage: {scope: sum(f['real_search_funnel'][stage][scope] for f in nonempty)
                             for scope in ('any_parent', 'true_prefix')}
                     for stage in ('gate_admissible', 'local_topk', 'refinement_proposal', 'retained')}}
    return {'episodes': len(tapes), 'frames': len(traces), 'empty_frames': len(traces) - len(nonempty),
            'same_tape_criteria': {key: {'numerator': sum(r[key] for r in criteria), 'fixed_denominator': len(tapes)}
                                   for key in criteria[0]},
            'identity_basis': 'known masks injected; not identity achievement' if cell.startswith('forced_') else 'Full scorer MAP exact mask',
            'identity': _identity(known, labels),
            'scorer_on_forced_paths_diagnostic_only': _identity(known, scorer_labels) if cell.startswith('forced_') else None,
            'reader_design_supported_rendered_frame_roles': int((observed & protected).sum()),
            'structurally_representable_exact_slots': sum(sum(f['supported_exact_slots']) for f in traces),
            'unrepresentable_exact_slots_by_reason': {reason: sum(bool(r[reason]) for r in unavailable)
                                                     for reason in ('missing_design_roles', 'reused_roles')},
            'mask_hits_without_structural_support': int(((known == labels) & ~supported).sum()),
            'unmeasured_rendered_frame_roles_by_reason': dict(Counter(
                reason for row in traces for reason in row['unmeasured_rendered_role_reasons'].values())),
            'merge_gate_admissible_exact_slots_any_seed': sum(sum(f['admissible_exact_slots_any_seed']) for f in traces),
            'real_search_funnel': funnel,
            'tracking': _tracking(measured, observed.sum()),
            'tracking_by_role': {role.name: {**_tracking(measured[:, role], observed[:, role].sum()),
                                             'design_supported': int((observed[:, role] & protected[:, role]).sum())}
                                 for role in VesselRole}}


def _paired(tapes, before, after, weights):
    records = []
    for tape in tapes:
        x, y = (tape['cells'][key] for key in (before, after))
        a, b = map(_squared, (x.residuals, y.residuals))
        ma, mb = map(lambda s: np.isfinite(s[..., 0]), (a, b))
        both = ma & mb
        p, q = (np.concatenate(c.identity) for c in (x, y))
        truth = np.concatenate(tape['truth'].echo_masks)
        joint = (p >= 0) & (q >= 0)
        pc, qc = p == truth, q == truth
        common = int(both.sum())
        left, right = a[both].sum(axis=0), b[both].sum(axis=0)
        episode_delta = (np.sqrt(right / common) - np.sqrt(left / common)).tolist() if common else [None] * len(_RESIDUALS)
        records.append({'episode_id': tape['episode_id'], 'rendered': int(tape['truth'].observed.sum()),
                        'common': common, 'before_only': int((ma & ~mb).sum()), 'after_only': int((mb & ~ma).sum()),
                        'before_sse_common': left.tolist(), 'after_sse_common': right.tolist(),
                        'common_rmse_delta_mm': dict(zip(_RESIDUALS, episode_delta, strict=True)),
                        'common_mask_decisions': int(joint.sum()), 'mask_wrong_to_right': int((joint & ~pc & qc).sum()),
                        'mask_right_to_wrong': int((joint & pc & ~qc).sum())})
    count = weights @ np.array([r['common'] for r in records])
    left, right = (weights @ np.array([r[f'{side}_sse_common'] for r in records]) for side in ('before', 'after'))
    denom = np.where(count > 0, count, np.nan)[:, None]
    delta = np.sqrt(right / denom) - np.sqrt(left / denom)
    return {'before': before, 'after': after, 'identity_contrast_interventional_not_achievement':
            before.startswith('forced_') or after.startswith('forced_'),
            'counts': {key: sum(r[key] for r in records) for key in (
                'rendered', 'common', 'before_only', 'after_only', 'common_mask_decisions',
                'mask_wrong_to_right', 'mask_right_to_wrong')},
            'common_rmse_delta_mm_after_minus_before': {
                axis: {'value': float(d[0]) if count[0] else None,
                       'episode_bootstrap_ci95': np.quantile(d[1:][count[1:] > 0], [.025, .975]).tolist()
                       if np.any(count[1:] > 0) else None,
                       'defined_bootstrap_draws': int((count[1:] > 0).sum())}
                for axis, d in zip(_RESIDUALS, delta.T, strict=True)}, 'episodes': records}


def _findings(cells, paired):
    """Describe observed effects without ranking interventions or claiming a gate."""
    contrasts = {}
    for name, item in paired.items():
        world = item['common_rmse_delta_mm_after_minus_before']['world']
        delta = world['value']
        direction = ('no common readout' if delta is None else
                     'lower RMSE' if delta < 0. else 'higher RMSE' if delta > 0. else 'unchanged RMSE')
        contrasts[name] = {'world_rmse_direction': direction, 'delta_mm': world, **item['counts']}
    search = {}
    for name in ('real_tracker', 'real_true_pose'):
        f = cells[name]['real_search_funnel']
        eligible = f['structurally_representable_frames']
        admissible, topk, refined, retained = (
            f[stage]['any_parent'] for stage in ('gate_admissible', 'local_topk', 'refinement_proposal', 'retained'))
        search[name] = {'nonempty_frames': f['nonempty_frames'], 'unrepresentable_frames': f['nonempty_frames'] - eligible,
                        'representable_frames_blocked_before_local_beam': eligible - admissible,
                        'admissible_but_not_in_local_topk': admissible - topk,
                        'local_topk_but_not_refined': topk - refined, 'refined_but_not_retained': refined - retained,
                        'full_true_prefix_retained': f['retained']['true_prefix']}
    return {'common_world_tracking_contrasts': contrasts, 'real_search_losses_by_stage': search,
            'error_remaining_with_both_interventions': cells['forced_true_pose']['tracking']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=tuple(_COHORT_SEEDS), default='dev',
                        help='predeclared dev root 931901 (default) or independent confirmation root 931902')
    args = parser.parse_args(argv)
    if HOLDOUT_ENV in os.environ:
        parser.error(f'{HOLDOUT_ENV} must be unset for train-source development')
    if LOAO_OPEN3D.train_source_ids != _SOURCES or set(_SOURCES).intersection(LOAO_OPEN3D.test_source_ids):
        parser.error('unexpected primary training split')
    if args.out.exists() or args.out.is_symlink():
        parser.error(f'would overwrite an existing artifact: {args.out}')
    if not _CHECKPOINT.is_file():
        parser.error(f'base frozen reader checkpoint is missing: {_CHECKPOINT}')
    bank = TrainingAnatomyBank.load()
    # Keep held geometry out of the prior/renderer even though the common asset
    # loader constructs the complete registered bank.
    bank = replace(bank, _sources={source: bank.source(source) for source in _SOURCES})
    sim = DynamicScanSimulator(bank, LOAO_OPEN3D)
    population = _station_population(sim, _SOURCES, STATION_IDS)
    scorer, _, provenance = _load(argparse.Namespace(checkpoint=_CHECKPOINT, split=_SPLIT), LOAO_OPEN3D, population)
    filter_params = provenance['filter_parameters']
    checkpoint_hash = _sha256(_CHECKPOINT)
    prior = build_split_prior(bank, LOAO_OPEN3D)
    # Keep separate filter instances per condition but reuse prior/weights and
    # cached model basis across episodes. Reset beliefs before every replay.
    readers = {c: _DiagnosticReader(prior, sim.kinematics, scorer=scorer, **filter_params) for c in _CELLS}
    fitted = {(r['source_id'], r['seed']) for name in ('fit_tapes', 'temperature_tapes', 'calibration_tapes')
              for r in provenance[name]}
    schedules = {cohort: [(s, j, _episode_seed(root, i, j)) for i, s in enumerate(_SOURCES)
                          for j in range(_EPISODES_PER_SOURCE)] for cohort, root in _COHORT_SEEDS.items()}
    all_pairs = {(s, seed) for schedule in schedules.values() for s, _, seed in schedule}
    if len(all_pairs) != sum(map(len, schedules.values())) or all_pairs & fitted:
        raise ValueError('tracking tape seeds collide with each other or frozen checkpoint cohorts')
    root = _COHORT_SEEDS[args.cohort]
    tapes = []
    for source, j, seed in schedules[args.cohort]:
        episode = sim.episode(source, seed, benchmark='dynamic', frames=_FRAMES)
        tape_sha256 = _tape_hash(episode)
        truth = _tape_truth(episode)
        cells = {key: _replay(reader, episode, truth, key, tape_sha256) for key, reader in readers.items()}
        tapes.append(dict(episode_id=f'{root}:{source}:{j}', source=source, seed=seed, tape_sha256=tape_sha256,
                          truth=truth, cells=cells, h_start_mm=episode.h_start_mm,
                          acquisition=[{'exact_masks': masks.tolist(), 'ghosts': list(f.supervision.echo_ghosts),
                                        'h_mm': f.supervision.acquisition_h_mm}
                                       for f, masks in zip(episode.frames, truth.echo_masks, strict=True)]))
    weights = _bootstrap_weights([t['source'] for t in tapes], _BOOTSTRAP, root)
    summaries = {c: _summary(tapes, c) for c in _CELLS}
    paired = {name: _paired(tapes, left, right, weights) for name, left, right in _CONTRASTS}
    output = {
        'schema_version': 1, 'scope': 'training-source development only; no held evaluation, certification or RL outcome',
        'split': _SPLIT, 'train_sources': _SOURCES, 'reader': 'Full_AGPS_Reader',
        'checkpoint': {'path': str(_CHECKPOINT), 'sha256': checkpoint_hash, 'filter_parameters': filter_params,
                       'source_authority': provenance['training_source_ids']},
        'training_source_provenance': {source: dict(bank.source(source).provenance) for source in _SOURCES},
        'source_sha256': {**_source_sha256(), 'reader/dev_tracking_interventions.py': _sha256(Path(__file__))},
        'episode_design': {'cohort': args.cohort, 'root_seed': root, 'predeclared_cohort_seeds': _COHORT_SEEDS,
                           'tapes_per_source_per_cohort': _EPISODES_PER_SOURCE, 'frames': _FRAMES,
                           'episode_seed': 'SeedSequence([root_seed, zero_based_train_source_index, zero_based_tape_index]).generate_state(1, dtype=uint64)[0]',
                           'tape_hash': 'sha256 agps-step3r-tracking-tape-v1 over public controls, anonymous echoes and offline supervision',
                           'bootstrap': '1000 paired resamples of whole tapes, stratified by training source, seed=root_seed'},
        'interpretation': {
            'association': 'Known role masks bypass search only. Present design roles use public echo coordinates and the Full sensor model; missing role/reused-role exact groups abstain as a whole. A mixed ghost/lumen echo is scored against its lumen centroid; no ghost component is injected.',
            'pose': 'Replace q_hat and its covariance, not probe, echoes or height selection. This jointly changes tracker mean and uncertainty; it is not a pullback ablation.',
            'geometry': 'Filtered (not smoothed) role-center errors for every rendered role/frame including every merge contributor. World SSE equals lateral/depth image SSE plus elevation SSE. Conditional RMSE is on finite readouts; complete RMSE is null if any fixed opportunity is missing or the tape is empty.',
            'identity': 'Exact-set membership F1 (not v6 single-label F1); all rendered echo contributors or NULL enter fixed truth denominators. Decision-conditional F1 excludes abstained echoes. Forced F1 measures intervention coverage, not identity correctness achieved by the reader.',
            'same_tape_criteria': 'Integer numerators over all tapes, fixed across arms. No rendered lumen means geometric criteria do not pass. Exact matches on forced masks are injected references; the joint diagnostic is not a continuation or certification measure.',
            'search': 'Known groups with all design roles and no reuse are structurally representable. Gate-admissible applies the production merge contact cutoffs before bounded association search; local top-k precedes proposal cap/dedup and branch pruning. A full true prefix must match all previous frames, including empty frames.',
            'limitations': 'Contrasts are descriptive train-source diagnostics, not guaranteed isolated effects: changes in associations and pose alter the subsequent filter state. Compare paired errors with coverage and search counts; conditional RMSE alone can improve by abstaining.',
        },
        'cells': summaries,
        'by_source': {src: {c: _summary([t for t in tapes if t['source'] == src], c) for c in _CELLS}
                      for src in _SOURCES},
        'paired_contrasts': paired,
        'findings': _findings(summaries, paired),
        'tapes': [dict(episode_id=t['episode_id'], source=t['source'], seed=t['seed'], tape_sha256=t['tape_sha256'],
                       checkpoint_sha256=checkpoint_hash, h_start_mm=t['h_start_mm'],
                       acquisition=t['truth'].acquisition(), frames=t['acquisition'],
                       cells={c: {'replay_tape_sha256': result.tape_sha256, 'frames': result.trace,
                                  'same_tape_criteria': _tape_criteria(t['truth'], result),
                                  'tracking': _tracking(result.residuals, t['truth'].observed.sum()),
                                  'identity': _identity(np.concatenate(t['truth'].echo_masks), np.concatenate(result.identity))}
                              for c, result in t['cells'].items()}) for t in tapes],
    }
    encoded = json.dumps(output, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')
    print(f'Wrote {len(tapes)} paired {args.cohort} training-source tapes to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
