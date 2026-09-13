#!/usr/bin/env python3
"""Export N-region evidence from original recorded images and held-out errors."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, output, name):
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(output/(name+'.'+extension), dpi=160)
    plt.close(fig)


def plot_examples(records, output):
    import cv2
    import h5py
    from matplotlib.patches import Rectangle
    from peirastic.contact_qp.features import FeatureConfig, confidence_features, random_walk_confidence

    examples = (('RH_Per_C_DtP', 771), ('RH_Per_S_PtD', 394))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (name, index) in enumerate(examples):
        record = next(r for r in records if r['name'] == name)
        cfg = FeatureConfig(**record['feature_config'])
        with h5py.File(record['source'], 'r') as saved:
            raw = cv2.imdecode(np.asarray(saved['ultrasound/jpeg'][index], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            sequence = int(saved['ultrasound/frame_index'][index])
        if raw is None:
            raise ValueError(f'{name}: invalid recorded frame {index}')
        confidence = random_walk_confidence(raw, cfg)
        feature = confidence_features(raw, confidence, cfg)
        y0, y1 = np.asarray(feature['region_roi_rows'])/cfg.height
        values = np.asarray(feature['region_confidence'])
        bad = np.asarray(feature['region_bad_mask'])
        axes[row, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, extent=(0, 1, 1, 0), aspect='auto')
        mapped = axes[row, 1].imshow(confidence, vmin=0, vmax=1, cmap='viridis', extent=(0, 1, 1, 0), aspect='auto')
        fig.colorbar(mapped, ax=axes[row, 1], fraction=.046, pad=.04)
        for i, (left, right) in enumerate(zip(cfg.region_edges[:-1], cfg.region_edges[1:])):
            color = '#f44336' if bad[i] else '#aaaaaa'
            for ax in axes[row, :2]:
                ax.add_patch(Rectangle((left, y0), right-left, y1-y0,
                                       edgecolor=color, facecolor='none', lw=1.6 if bad[i] else .7))
                ax.text((left+right)/2, y1+.04, str(i+1), color=color, ha='center', fontsize=8)
        colors = np.where(bad, '#d84438', '#207caa')
        axes[row, 2].bar(np.arange(cfg.region_count)+1, values, color=colors)
        axes[row, 2].axhline(cfg.low_confidence_threshold, color='#ba352b', ls='--', lw=1)
        axes[row, 2].set_xticks(np.arange(cfg.region_count)+1)
        axes[row, 2].set(xlabel='Equal-width image region (1-based display)', ylabel='Weighted near-field q25', ylim=(0, 1.04))
        axes[row, 0].set_title(f'{name}: H5 index {index}, seq {sequence}\nOriginal recorded B-mode', fontsize=10)
        axes[row, 1].set_title(f'Original random-walk map; {cfg.region_count} equal regions\nRed: confidence below {cfg.low_confidence_threshold:g}', fontsize=10)
        axes[row, 2].set_title('Independent quality of every region\nNo L/C/R aggregation', fontsize=10)
        axes[row, 2].grid(axis='y', alpha=.2)
        for ax in axes[row, :2]:
            ax.set(xlabel='Normalized image column', ylabel='Normalized image depth')
    fig.suptitle('Recorded pixels and N-region features; no new acquisition or image repair is shown', fontsize=12)
    save(fig, output, 'recorded_confidence_examples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.calibration/'manifest.json').read_text())['records']
    records = sorted((r for r in records if not r['failed_attempt']), key=lambda r: (not r['name'].endswith('DtP'), r['name']))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for axis, record in zip(axes.flat, records):
        with np.load(record['cache']) as data:
            t = data['capture_s']-data['capture_s'][0]
            quality = data['region_confidence'].copy()
            quality[~data['region_valid']] = np.nan
        n = quality.shape[1]
        mapped = axis.imshow(quality.T, origin='lower', aspect='auto', cmap='viridis', vmin=.4, vmax=1.,
                              extent=(float(t[0]), float(t[-1]), .5, n+.5))
        axis.set(title=record['name'], xlabel='Recorded capture time (s)', ylabel='Image region (1-based)')
        axis.set_yticks(np.arange(n)+1)
        fig.colorbar(mapped, ax=axis, fraction=.046, pad=.04, label='Region q25')
    fig.suptitle('All spatial regions retained: low confidence is a signal proxy, not contact ground truth', fontsize=11)
    save(fig, args.output, 'recorded_quality_comparison')

    report = json.loads((args.calibration/'kf_calibration.json').read_text())
    names = [item['name'] for item in report['validation']]
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(7, 4))
    for i, (key, label, color) in enumerate((('held', 'Hold delayed frame', '.4'),
                                            ('prediction_raw', 'KF raw prediction', '#1678be'),
                                            ('prediction_task', 'KF clipped task input', '#37937b'))):
        axis.bar(x+(i-1)*.25, [report['errors'][name][key]['rmse'] for name in names], width=.25, color=color, label=label)
    axis.set_xticks(x, [name.replace('RH_Per_', '') for name in names])
    axis.set(ylabel='All-region confidence RMSE', title='Held-out return paths: fixed registered delay')
    axis.legend(fontsize=8)
    axis.grid(axis='y', alpha=.2)
    save(fig, args.output, 'heldout_prediction_error')
    plot_examples(records, args.output)


if __name__ == '__main__':
    main()
