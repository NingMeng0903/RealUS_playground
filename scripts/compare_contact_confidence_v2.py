"""Small read-only real-frame comparison; no contact labels or threshold tuning."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import zoom

from peirastic.contact_qp.features import FeatureConfig, random_walk_confidence, window_quality


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=Path(
        '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/uncalibrated'))
    parser.add_argument('--audit', type=Path, default=Path('MD/contact_qp/human_audit'))
    parser.add_argument('--output', type=Path, default=Path('MD/contact_qp/confidence_v2_check'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig()
    old = replace(cfg, algorithm_version='randomwalk_thesis_v1')
    previews = []
    for path in sorted(args.audit.glob('**/frame_*_image_confidence.png')):
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[:, :cfg.width]
        previews.append((float((im < 8).mean()), str(path), path))
    previews.sort()
    # Fixed quantiles of preview dark occupancy select signal appearances only.
    # These are not ranked by mechanical contact or confidence improvement.
    chosen = [previews[int(round(f*(len(previews)-1)))][2] for f in (0, .25, .5, .75, 1)]
    frames = []
    for path in chosen:
        rel = path.relative_to(args.audit)
        source = args.source / rel.parent.with_suffix('.h5')
        index = int(path.stem.split('_')[1])
        with h5py.File(source, 'r') as handle:
            im = cv2.imdecode(np.asarray(handle['ultrasound/jpeg'][index], dtype=np.uint8),
                              cv2.IMREAD_GRAYSCALE)
        frames.append((str(rel.parent)+f' frame {index}', im,
                       {'source_h5': str(source), 'frame_index': index, 'audit_panel': str(path)}))
    blank_half = np.zeros((cfg.height, cfg.width))
    blank_half[:, :cfg.width//2] = 80
    frames.append(('Synthetic: left=80, right=0', blank_half,
                   {'synthetic': True, 'purpose': 'black-region counterexample; no acoustic truth'}))
    fig, axes = plt.subplots(len(frames), 3, figsize=(12, 3.1*len(frames)), constrained_layout=True)
    rows = []
    for axs, (name, im, provenance) in zip(axes, frames):
        maps = [random_walk_confidence(im, version) for version in (old, cfg)]
        processed = zoom(im.astype(float), (cfg.height/im.shape[0], cfg.width/im.shape[1]),
                         order=1, prefilter=False)
        axs[0].imshow(processed, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title(name+'\nB-mode (fixed 0..255)', fontsize=9)
        metrics = {}
        for ax, c, version in zip(axs[1:], maps, (old, cfg)):
            shown = ax.imshow(c, cmap='gray', vmin=0, vmax=1)
            q = window_quality(c, version)
            ax.set_title(('v1 thesis' if version is old else 'v2 CAMP B-mode')+
                         '\nq L/C/R = '+', '.join(f'{v:.3f}' for v in q), fontsize=9)
            metrics[version.algorithm_version] = {
                'quality_left_center_right': q.tolist(), 'inner_mean': float(c[1:-1].mean()),
                'inner_fraction_below_0_01': float((c[1:-1] < .01).mean())}
        for ax in axs:
            y0, y1 = (int(cfg.height*f) for f in cfg.near_depth)
            for color, (lo, hi) in zip(('cyan', 'lime', 'orange'), cfg.lateral_windows):
                x0, x1 = int(lo*cfg.width), int(hi*cfg.width)
                ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0,
                                       fill=False, edgecolor=color, linewidth=.75))
            ax.set_xticks([]); ax.set_yticks([])
        rows.append({'name': name, **provenance, **metrics})
    fig.colorbar(shown, ax=axes[:, 1:].ravel().tolist(), shrink=.5,
                 label='Confidence probability (same fixed 0..1 for v1 and v2)')
    fig.savefig(args.output/'comparison.png', dpi=140)
    report = {
        'scope': 'Five real frames selected by preview dark-fraction quantiles plus synthetic counterexample; no mechanical contact labels.',
        'selection': 'Preview gray<8 fraction quantiles 0, .25, .5, .75, 1; original H5 JPEG decoded for computation.',
        'old_config': asdict(old), 'new_config': asdict(cfg),
        'notes': ['No final-map stretching or masking; display fixed 0..1.',
                  'CAMP reference gamma default is .05; project .03 remains uncalibrated.',
                  'The MATLAB source uses 1-exp(-alpha*d); retained thesis v1 uses exp(-alpha*d).',
                  'Confidence measures propagation and does not segment non-contact black areas.'],
        'reference': 'https://github.com/TJKlein/Nakagami_Confidence_Maps', 'frames': rows}
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in rows:
        print(row['name'], *(row[v.algorithm_version]['quality_left_center_right'] for v in (old, cfg)))


if __name__ == '__main__':
    main()
