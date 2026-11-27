"""Population upper bounds for the fixed single-label station protocol (raw geometry only).

This offline calculation neither renders tapes nor imports the reader or checkpoint
code. The source and exact start height are oracle variables, not reader inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from sim.anatomy_assets import NYSORA_STATIONS, SPLIT_PROTOCOLS, TrainingAnatomyBank
from sim.audit_training_support import station_support_masks
from sim.metric_graph_gp import MetricGraphAnatomyGenerator

_STATIONS = tuple(NYSORA_STATIONS)
# The three station strata declared by reader.eval_reader.PROTOCOL_STATIONS.
# Do not import the evaluator: this diagnostic must not import the reader.
_EVALUATION_STATIONS = {
    'LOAO_OPEN3D': _STATIONS,
    'LOAO_BODYPARTS': (_STATIONS[0], _STATIONS[1], _STATIONS[4]),
    'FAMILY_HELD_OUT_ATLASES': _STATIONS[:4],
}


def station_population_bounds(joint_mass: np.ndarray) -> dict:
    """Oracle bounds from a joint probability table indexed by (cell, tag)."""
    joint = np.asarray(joint_mass, dtype=float)
    if joint.ndim != 2 or not np.isfinite(joint).all() or np.any(joint < 0.) or not np.isclose(joint.sum(), 1.):
        raise ValueError('expected a normalized nonnegative cell-by-tag probability table')
    site_mass, prevalence = joint.sum(axis=1), joint.sum(axis=0)
    if np.any(site_mass <= 0.) or np.any(prevalence <= 0.):
        raise ValueError('each listed cell and expected class must have positive support')

    # For a fixed predicted positive mass, taking the highest P(tag | cell)
    # maximizes true-positive mass. Along a fractional cell the resulting F1
    # is linear-fractional and monotone or constant, so a prefix suffices.
    order = np.argsort(-joint / site_mass[:, None], axis=0, kind='stable')
    tp = np.take_along_axis(joint, order, axis=0).cumsum(axis=0)
    predicted = site_mass[order].cumsum(axis=0)
    f1 = 2. * tp / (prevalence + predicted)
    best = f1.argmax(axis=0)
    per_class = [
        {
            'tag_population_mass': float(prevalence[i]),
            'f1_upper_bound': float(f1[k, i]),
            'precision_at_f1_optimum': float(tp[k, i] / predicted[k, i]),
            'recall_at_f1_optimum': float(tp[k, i] / prevalence[i]),
            'predicted_positive_mass_at_optimum': float(predicted[k, i]),
            'true_positive_mass_at_optimum': float(tp[k, i]),
            'positive_prefix_cells': int(k + 1),
        }
        for i, k in enumerate(best)
    ]
    return {
        'oracle_accuracy': float(joint.max(axis=1).sum()),
        'ambiguous_cell_mass': float(site_mass[(joint > 0.).sum(axis=1) > 1].sum()),
        'macro_f1_upper_bound': float(np.mean([row['f1_upper_bound'] for row in per_class])),
        'per_class': per_class,
    }


def _balanced_mixture(raw, sources, stations):
    windows = {sid: {tag: raw[sid][tag] for tag in stations if raw[sid][tag].size} for sid in sources}
    pairs = sum(map(len, windows.values()))
    mass = defaultdict(lambda: np.zeros(len(stations)))
    for sid in sources:
        for i, tag in enumerate(stations):
            heights = windows[sid].get(tag)
            if heights is not None:
                for h in heights:
                    mass[sid, float(h)][i] = 1. / (pairs * len(heights))
    if not mass:
        raise ValueError('no supported source/station pairs in the declared population')
    cells = sorted(mass)
    joint = np.stack([mass[cell] for cell in cells])
    bounds = station_population_bounds(joint)
    bounds['per_class'] = dict(zip(stations, bounds['per_class'], strict=True))
    return {
        'station_classes_in_mass_vector_order': list(stations),
        'balanced_source_station_pairs': pairs,
        'window_rows_mm_by_source_and_tag': {
            sid: {tag: heights.tolist() for tag, heights in by_tag.items()}
            for sid, by_tag in windows.items()
        },
        'cells': [
            {'source_id': sid, 'h_start_mm': h, 'p_cell': float(row.sum()), 'p_cell_and_tag': row.tolist()}
            for (sid, h), row in zip(cells, joint, strict=True)
        ],
        **bounds,
    }


def build_report(*, include_train: bool = False) -> dict:
    bank = TrainingAnatomyBank.load()
    generator = MetricGraphAnatomyGenerator(bank)
    raw = {}
    for sid in bank.source_ids:
        source = bank.source(sid)
        _, masks = station_support_masks(source.h_mm, source.supported, generator.graph(sid))
        raw[sid] = {tag: np.sort(source.h_mm[mask]) for tag, mask in masks.items()}

    protocols = {}
    for name, declared in _EVALUATION_STATIONS.items():
        split = SPLIT_PROTOCOLS[name]
        train_supported = tuple(tag for tag in _STATIONS if any(raw[sid][tag].size for sid in split.train_source_ids))
        if set(declared).difference(train_supported):
            raise ValueError(f'{name}: not all declared evaluation tags have train-source window support')
        node = {
            'train_source_ids': list(split.train_source_ids),
            'test_source_ids': list(split.test_source_ids),
            'evaluation': _balanced_mixture(raw, split.test_source_ids, declared),
        }
        if include_train:
            node['optional_train_diagnostic_same_balanced_pair_law'] = _balanced_mixture(
                raw, split.train_source_ids, train_supported)
        protocols[name] = node
    return {
        'scope': 'Fixed single-label topological station surrogate, not clinical classification; no Blender, reader characteristics, or fitted results.',
        'opportunity_unit': 'One scheduled station tape with one immutable requested tag and a single station prediction; not an echo, frame, or successful-capture subset.',
        'window_policy': 'Raw same-source Gate 2 role co-presence within the thigh; proximal S1 >= -560 mm, graph-vertex +/-15 mm for S2–S4, S5 <= -825 mm. Overlap is retained.',
        'mixture': 'Equal probability for each supported (source, tag) pair in the declared protocol; uniform true start height over its window rows. Optional train results use the same balanced-pair law, not an IID conformal cohort.',
        'assumptions': [
            'Oracle cell Z is (source, exact sampled start height). The tag affects only sampling Z.',
            'Conditional on Z, the public tape law is independent of the tag: acquisition nuisance, anatomy draw and acoustic noise are drawn from the same law with independent seed streams, and controls do not use the tag.',
            'The oracle information is sufficient for tag inference under this conditional independence. It is not available to the deployed reader.',
        ],
        'derivation': {
            'joint_mass': 'p(z,y)=1/(number of supported source/tag pairs * number of rows in that pair) at supported heights; zero otherwise.',
            'accuracy': 'Exact oracle population accuracy is sum_z max_y p(z,y); it need not equal the accuracy of a reader or class-balanced recall.',
            'classwise_f1': 'For each y, rank cells by p(y|z). At a given prediction mass Q, the highest-posterior cells maximize TP. Since 2*TP/(p(y)+Q) is linear-fractional within any partially selected cell, a sorted prefix attains the one-vs-rest optimum.',
            'macro_f1': 'Averaging classwise optima is an upper bound, not necessarily an achievable score: their positive subsets may overlap and fail to define a single-label prediction rule.',
        },
        'finite_sample_note': 'These are population bounds under the uniform-height and conditional-independence model. Realized finite balanced tapes can have different cell counts and empirical F1 can exceed a population bound by sampling variation. No held tape or empirical reader metric was used.',
        'protocols': protocols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='New JSON output file; default is stdout')
    parser.add_argument('--include-train', action='store_true', help='Also report train-source mixtures')
    args = parser.parse_args()
    text = json.dumps(build_report(include_train=args.include_train), indent=2, allow_nan=False) + '\n'
    if args.out is None:
        print(text, end='')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as report:
            report.write(text)


if __name__ == '__main__':
    main()
