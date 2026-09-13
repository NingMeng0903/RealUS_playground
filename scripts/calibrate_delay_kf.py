#!/usr/bin/env python3
"""Read-only HDF5 extraction and fixed-noise delayed-confidence calibration.

Outputs are separate artifacts. Recorded images/forces are never predictions of
what a different controller would have acquired. No hardware modules are opened.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from peirastic.contact_qp.delayed_kf import measurement_update, predict_state
from peirastic.contact_qp.features import (
    FeatureConfig, confidence_features, random_walk_confidence,
)
from peirastic.contact_qp.types import WEAK_SIDE_FEATURE_VERSION, REGION_FEATURE_VERSION


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def extract(data_root, output, region_count=10, threshold=.85):
    import cv2
    import h5py
    session = json.loads((data_root/'session.json').read_text())
    source_config = session['contact_qp']['config']
    config = FeatureConfig(**(source_config['feature']['config'] |
                             {'region_count': region_count, 'low_confidence_threshold': threshold}))
    feature_config = json.loads(json.dumps(asdict(config)))
    output.mkdir(parents=True, exist_ok=True)
    paths = [(path.stem, path, False) for path in sorted(data_root.glob('*.h5'))]
    paths += [(f'{path.parent.parent.name}_failed_{path.parent.name}', path, True)
              for path in sorted(data_root.glob('attempts/*/*/raw.h5'))
              if (path.parent/'failure.json').exists()]
    records = []
    for name, path, failed in paths:
        cache = output/(name+'.npz')
        manifest_path = output/(name+'.json')
        source_hash = sha256(path)
        if cache.exists() and manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if (old.get('source_sha256') == source_hash and
                    old.get('region_feature_version') == REGION_FEATURE_VERSION and
                    old.get('region_layout_version') == config.region_layout_version and
                    old.get('feature_config') == feature_config):
                records.append(old)
                print('cached', name, flush=True)
                continue
        with h5py.File(path, 'r') as saved:
            us = saved['ultrasound']
            n = len(us['jpeg'])
            quality, legacy, valid, capture, received, seq = [], [], [], [], [], []
            regional, regional_valid = [], []
            sources, unique = [], set()
            for index in range(n):
                metadata = json.loads(us['metadata_json'][index])
                identity = (metadata.get('publisher_instance_id'), int(us['frame_index'][index]))
                if identity in unique:
                    continue
                unique.add(identity)
                if metadata.get('timestamp_source') != 'host_frame_read_complete':
                    raise ValueError(f'{name}: uncalibrated timestamp semantics')
                for key in ('crop_box', 'hflip'):
                    if metadata.get(key) != source_config['feature']['registration'][key]:
                        raise ValueError(f'{name}: {key} does not match registered geometry')
                stamp_ns = int(metadata['capture_monotonic_ns'])
                if stamp_ns != int(us['timestamp_mono_ns'][index]):
                    raise ValueError(f'{name}: raw capture clock mismatch')
                image = cv2.imdecode(np.asarray(us['jpeg'][index], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f'{name}: invalid JPEG at {index}')
                feature = confidence_features(image, random_walk_confidence(image, config), config)
                unknown = np.asarray(feature['unknown_columns'], bool)
                side_valid = [not unknown[int(lo*config.width):int(hi*config.width)].any()
                              for lo, hi in (config.lateral_windows[0], config.lateral_windows[2])]
                quality.append(feature['confidence_lr'])
                legacy.append(feature['quality_lcr'])
                valid.append(side_valid)
                regional.append(feature['region_confidence'])
                regional_valid.append(feature['region_valid'])
                capture.append(stamp_ns*1e-9)
                received.append(float(us['received_monotonic_ns'][index])*1e-9)
                seq.append(int(us['frame_index'][index]))
                sources.append(str(identity[0]))
                if (index+1) % 200 == 0:
                    print(name, index+1, '/', n, flush=True)
            capture = np.asarray(capture)
            if len(capture) > 1 and np.any(np.diff(capture) <= 0):
                raise ValueError(f'{name}: non-monotone capture clock')
            force = np.asarray(saved['force/contact_force_n'], dtype=float)
            force_time = np.asarray(saved['force/timestamp_mono_ns'], dtype=float)*1e-9
            wrench = np.asarray(saved['force/wrench_tcp'], dtype=float)
            pose = np.asarray(saved['tcp/pose_m_rad'], dtype=float)
            pose_time = np.asarray(saved['tcp/timestamp_mono_ns'], dtype=float)*1e-9
        np.savez_compressed(cache, capture_s=capture,
                            effective_s=capture-config.effective_delay_s,
                            received_s=np.asarray(received), frame_seq=np.asarray(seq),
                            source_id=np.asarray(sources), confidence_lr=np.asarray(quality),
                            quality_lcr=np.asarray(legacy), valid_lr=np.asarray(valid),
                            region_confidence=np.asarray(regional), region_valid=np.asarray(regional_valid),
                            region_edges=config.region_edges,
                            force_s=force_time, force_n=force, wrench_tcp=wrench,
                            pose_s=pose_time, pose_m_rad=pose)
        record = dict(name=name, source=str(path), source_sha256=source_hash,
                      cache=str(cache), failed_attempt=failed, frames=len(capture),
                      weakside_feature_version=WEAK_SIDE_FEATURE_VERSION,
                      region_feature_version=REGION_FEATURE_VERSION,
                      region_layout_version=config.region_layout_version, region_count=config.region_count,
                      feature_config=feature_config,
                      effective_delay_s=config.effective_delay_s,
                      timestamp_semantics='host_frame_read_complete_minus_registered_delay',
                      received_semantics='recorder_receive_not_confidence_worker_completion',
                      force_max_n=float(np.max(force)) if force.size else None,
                      force_p95_n=float(np.quantile(force, .95)) if force.size else None)
        dump(manifest_path, record)
        records.append(record)
        print('extracted', name, len(capture), flush=True)
    dump(output/'manifest.json', dict(data_root=str(data_root), records=records))
    return records


def segments(record):
    with np.load(record['cache']) as data:
        time = data['effective_s']
        quality = data['region_confidence']
        valid = data['region_valid'].all(axis=1)
        source = data['source_id']
    split = [0]
    for i in range(1, len(time)):
        if not valid[i-1] or not valid[i] or time[i]-time[i-1] > .3 or source[i] != source[i-1]:
            split.append(i)
    split.append(len(time))
    return [(time[a:b], quality[a:b]) for a, b in zip(split[:-1], split[1:])
            if b-a >= 2 and valid[a:b].all()]


def innovation_nll(log_noise, sequences):
    q, r = np.exp(log_noise)
    result = 0.
    for time, quality in sequences:
        x = np.zeros(quality.shape[1]*2); x[::2] = quality[0]
        p = np.diag(np.tile([r, 1.], quality.shape[1]))
        for i in range(1, len(time)):
            x, p = predict_state(x, p, float(time[i]-time[i-1]), q)
            x, p, innovation, covariance = measurement_update(x, p, quality[i], r)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                return 1e100
            result += .5*(logdet + innovation @ np.linalg.solve(covariance, innovation))
    return float(result)


def prediction_error(record, q, r):
    """Compare at exact recorded effective times, with a fixed arrival delay.

    Measurement j is available only at t_j + registered_delay. Recorder receive
    does not measure worker processing latency, so this is a clearly labelled
    fixed-delay experiment, not reconstructed live worker timing.
    """
    predicted, held, actual = [], [], []
    horizon = record['effective_delay_s']
    for time, quality in segments(record):
        x = np.zeros(quality.shape[1]*2); x[::2] = quality[0]
        p = np.diag(np.tile([r, 1.], quality.shape[1]))
        posteriors = [(x.copy(), p.copy())]
        for i in range(1, len(time)):
            x, p = predict_state(x, p, float(time[i]-time[i-1]), q)
            x, p, _, _ = measurement_update(x, p, quality[i], r)
            posteriors.append((x.copy(), p.copy()))
        for i, now in enumerate(time):
            j = int(np.searchsorted(time+horizon, now, side='right'))-1
            if j < 0 or now-time[j] > .3:
                continue
            xp, _ = predict_state(*posteriors[j], float(now-time[j]), q)
            predicted.append(xp[::2])
            held.append(quality[j])
            actual.append(quality[i])
    def errors(values):
        diff = np.asarray(values)-np.asarray(actual)
        return dict(rmse=float(np.sqrt(np.mean(diff**2))), mae=float(np.mean(abs(diff))))
    if not actual:
        return dict(samples=0)
    return dict(samples=len(actual), prediction_raw=errors(predicted),
                prediction_task=errors(np.clip(predicted, 0., 1.)), held=errors(held))


def fit(output):
    records = json.loads((output/'manifest.json').read_text())['records']
    training = [r for r in records if not r['failed_attempt'] and r['name'].endswith('_DtP')]
    validation = [r for r in records if not r['failed_attempt'] and r['name'].endswith('_PtD')]
    if len(training) != 3 or len(validation) != 3:
        raise ValueError('Expected three distinct training outward and three held-out return paths')
    sequences = [segment for record in training for segment in segments(record)]
    if not sequences:
        raise ValueError('No valid training sequence')
    starts = ((.01, 1e-4), (.1, 1e-3), (1., 1e-5))
    bounds = np.log([[1e-8, 100.], [1e-10, .25]])
    results = []
    for initial in starts:
        result = minimize(innovation_nll, np.log(initial), args=(sequences,),
                          method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=80, ftol=1e-9))
        results.append(result)
        print('noise fit', result.success, result.fun, np.exp(result.x).tolist(), flush=True)
    converged = [result for result in results if result.success and np.isfinite(result.fun)]
    if not converged:
        raise RuntimeError('Noise likelihood fitting did not converge')
    best = min(converged, key=lambda result: result.fun)
    q, r = map(float, np.exp(best.x))
    counts = {item['region_count'] for item in records}
    layouts = {item['region_layout_version'] for item in records}
    if len(counts) != 1 or len(layouts) != 1:
        raise ValueError('Calibration cannot mix regional layouts')
    channels = next(iter(counts))
    report = dict(schema='confidence_region_delay_kf_calibration_v1',
                  region_feature_version=REGION_FEATURE_VERSION, region_layout_version=next(iter(layouts)),
                  kf=dict(q=q, r=r, max_age_s=.3, history_s=.3, initial_rate_variance=1.,
                          channels=channels, observation_kind='regions'),
                  method='common white-acceleration q and R=rI; training innovation maximum likelihood',
                  log_parameter_bounds=bounds.tolist(), likelihood=float(best.fun),
                  training=[dict(name=x['name'], sha256=x['source_sha256']) for x in training],
                  validation=[dict(name=x['name'], sha256=x['source_sha256']) for x in validation],
                  validation_used_for_fit=False,
                  prediction_experiment='fixed registered arrival delay; actual worker compute latency excluded',
                  errors={x['name']: prediction_error(x, q, r) for x in training+validation})
    heldout = [report['errors'][x['name']] for x in validation]
    count = sum(x['samples'] for x in heldout)
    aggregate = {}
    for model in ('held', 'prediction_raw', 'prediction_task'):
        aggregate[model] = dict(
            rmse=float(np.sqrt(sum(x['samples']*x[model]['rmse']**2 for x in heldout)/count)),
            mae=float(sum(x['samples']*x[model]['mae'] for x in heldout)/count))
    report['heldout_aggregate'] = dict(samples=count, **aggregate,
        raw_rmse_change_fraction=aggregate['prediction_raw']['rmse']/aggregate['held']['rmse']-1.,
        task_rmse_change_fraction=aggregate['prediction_task']['rmse']/aggregate['held']['rmse']-1.)
    report['prediction_improved_on_heldout'] = aggregate['prediction_task']['rmse'] < aggregate['held']['rmse']
    dump(output/'kf_calibration.json', report)
    summarize(output)
    print(json.dumps(report['kf']), flush=True)
    return report


def summarize(output, threshold=None):
    """Recount unchanged continuous features after a threshold-only change."""
    manifest = json.loads((output/'manifest.json').read_text())
    records = manifest['records']
    if threshold is not None:
        if not 0 < threshold < 1:
            raise ValueError('confidence threshold must be in (0,1)')
        for record in records:
            cfg = FeatureConfig(**record['feature_config'])
            if cfg.region_layout_version != record['region_layout_version']:
                raise ValueError('Cannot relabel a different region layout')
            record['feature_config']['low_confidence_threshold'] = threshold
            record['threshold_update_note'] = 'Threshold-only recount; map, regional continuous values and noise fitting are unchanged.'
            dump(output/(record['name']+'.json'), record)
        dump(output/'manifest.json', manifest)
    thresholds = {r['feature_config']['low_confidence_threshold'] for r in records}
    if len(thresholds) != 1:
        raise ValueError('Cannot combine different thresholds')
    c_min = next(iter(thresholds))
    detection = []
    for record in records:
        with np.load(record['cache']) as data:
            valid = data['valid_lr'].all(axis=1)
            old = data['quality_lcr'][valid][:, [0, 2]]
            new = data['confidence_lr'][valid]
            region_valid = data['region_valid']
            region = data['region_confidence']
        detection.append(dict(name=record['name'], failed_attempt=record['failed_attempt'],
            total_frames=record['frames'], valid_frames=int(valid.sum()),
            mean_weak_counts=np.sum(old < c_min, axis=0).tolist(),
            q25_weak_counts=np.sum(new < c_min, axis=0).tolist(),
            region_count=record['region_count'], all_regions_valid_frames=int(region_valid.all(axis=1).sum()),
            region_weak_counts=np.sum((region < c_min) & region_valid, axis=0).tolist(),
            any_region_weak_frames=int(np.sum(np.any((region < c_min) & region_valid, axis=1))),
            all_legacy_good_region_weak_frames=int(np.sum(np.all(data_legacy_quality(record) >= c_min, axis=1)
                                                        & np.any((region < c_min) & region_valid, axis=1))),
            mean_good_q25_weak_frames=int(np.sum(np.all(old >= c_min, axis=1) & np.any(new < c_min, axis=1)))))
    dump(output/'confidence_detection.json', dict(
        meaning='quality-threshold decisions on unchanged recorded images; no counterfactual image improvement',
        c_min=c_min, records=detection))


def data_legacy_quality(record):
    with np.load(record['cache']) as data:
        return data['quality_lcr'].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('extract', 'fit', 'all', 'summarize'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--region-count', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=.85)
    args = parser.parse_args()
    if args.output.resolve() == args.data_root.resolve() or args.data_root.resolve() in args.output.resolve().parents:
        parser.error('output must be outside the original acquisition data')
    if args.action in ('extract', 'all'):
        extract(args.data_root, args.output, args.region_count, args.threshold)
    if args.action in ('fit', 'all'):
        fit(args.output)
    if args.action == 'summarize':
        summarize(args.output, args.threshold)


if __name__ == '__main__':
    main()
