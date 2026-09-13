"""Read-only offline extraction benchmark; never opens sockets or devices.

Decode selected raw-H5 JPEGs before timing. Alternate old/new extraction order
after warming both implementations. Scheduling figures are arithmetic models,
not measured live worker timing or a total capture-to-controller latency bound.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import h5py
import numpy as np

from peirastic.apps.contact_qp_features import encode_feature_payload
from peirastic.contact_qp import features


def stats_ms(values):
    values = np.asarray(values) * 1000.
    return dict(count=len(values), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), min=float(values.min()),
                max=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--config', type=Path,
                        default=Path('peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml'))
    parser.add_argument('--baseline', default='HEAD')
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    if not 12 <= args.frames <= 30 or args.repeats < 1:
        parser.error('use 12 to 30 frames and at least one repeat')
    cv2.setNumThreads(1)
    baseline_commit = subprocess.check_output(
        ['git', 'rev-parse', args.baseline], text=True).strip()
    old_source = subprocess.check_output(
        ['git', 'show', f'{baseline_commit}:peirastic/contact_qp/features.py'], text=True)
    old = types.ModuleType('peirastic.contact_qp._offline_baseline_features')
    old.__package__ = 'peirastic.contact_qp'
    sys.modules[old.__name__] = old
    exec(compile(old_source, '<git-baseline-features>', 'exec'), old.__dict__)
    cfg = features.load_feature_config(args.config)
    extractors = dict(baseline=old.FeatureExtractor(old.FeatureConfig(**asdict(cfg))),
                      edited=features.FeatureExtractor(cfg))
    frames = []
    with h5py.File(args.source, 'r') as handle:
        # Intentionally do not access H5 attributes: some contain giant logs.
        jpeg_dataset = handle['ultrasound/jpeg']
        total_frames = len(jpeg_dataset)
        indices = np.unique(np.linspace(0, total_frames - 1, args.frames, dtype=int))
        if len(indices) != args.frames:
            raise ValueError('raw H5 has too few frames')
        for index in indices:
            jpeg = np.asarray(jpeg_dataset[index], dtype=np.uint8)
            image = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f'cannot decode frame {index}')
            meta = json.loads(handle['ultrasound/metadata_json'][index])
            capture = float(meta['capture_monotonic_ns']) * 1e-9
            kwargs = dict(frame_seq=int(meta['frame_index']),
                source_id=f"{meta['source_id']}:{meta['publisher_instance_id']}",
                registration_source_id=meta['source_id'], capture_time_s=capture,
                received_time_s=capture + .01, crop_box=meta.get('crop_box'),
                hflip=meta.get('hflip', False))
            frames.append((image, kwargs, dict(h5_index=int(index),
                frame_seq=kwargs['frame_seq'], capture_time_s=capture,
                jpeg_sha256=hashlib.sha256(jpeg.tobytes()).hexdigest(),
                image_shape=list(image.shape), mean_gray=float(image.mean()),
                crop_box=kwargs['crop_box'], hflip=kwargs['hflip'])))
    for image, kwargs, _ in frames[:3]:
        for extractor in extractors.values():
            extractor.extract(image, **kwargs)
    timings = {name: [] for name in extractors}
    equality_checks = 0
    for repetition in range(args.repeats):
        for index, (image, kwargs, record) in enumerate(frames):
            outputs = {}
            order = ('baseline', 'edited') if (repetition + index) % 2 == 0 else ('edited', 'baseline')
            for name in order:
                extractor = extractors[name]
                start = time.perf_counter()
                obs, confidence = extractor.extract(image, **kwargs)
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed)
                outputs[name] = (obs, confidence, extractor.last_features)
            before, after = outputs['baseline'], outputs['edited']
            np.testing.assert_array_equal(before[1], after[1])
            assert before[0].to_dict() == after[0].to_dict()
            assert before[2] == after[2]
            assert encode_feature_payload(before[0], before[2], 0.) == encode_feature_payload(
                after[0], after[2], 0.)
            equality_checks += 1
            record.update(quality_lcr=after[0].quality.tolist(), valid=after[0].valid.tolist(),
                registration_version=after[0].registration_version,
                unknown_column_count=int(np.count_nonzero(after[2]['unknown_columns'])),
                frame_status=after[2]['frame_status'])
    result = dict(kind='offline_extraction_benchmark', source=str(args.source.resolve()),
        baseline_commit=baseline_commit,
        baseline_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
        edited_source_sha256=hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest(),
        python=sys.executable, numpy_version=np.__version__,
        environment={key: os.environ.get(key) for key in
                     ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        config_file=str(args.config), config=asdict(cfg), window_version=cfg.window_version,
        input_frame_count=total_frames, sampled_frames=len(frames), repeats=args.repeats,
        warmup_extractions_per_implementation=3, paired_exact_equality_checks=equality_checks,
        exact_equality=['confidence_map', 'observation_dict', 'cached_features_dict',
                        'feature_payload_bytes_with_identical_processing_s'],
        timing_method='Interleaved paired extraction; alternating implementation order; JPEG decode before timing.',
        extraction_ms={name: stats_ms(values) for name, values in timings.items()},
        scheduling_model=dict(label='Arithmetic model, not measured live; excludes decode/encode/transport and arrival gaps.',
            max_hz=20., period_ms=50., old_interval_formula='baseline_compute_ms + 50',
            new_interval_formula='max(edited_compute_ms, 50)',
            old_interval_ms=stats_ms(np.asarray(timings['baseline']) + .050),
            new_interval_ms=stats_ms(np.maximum(timings['edited'], .050))),
        frames=[record for _, _, record in frames])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in
        ('sampled_frames', 'paired_exact_equality_checks', 'extraction_ms', 'scheduling_model')}, indent=2))


if __name__ == '__main__':
    main()
