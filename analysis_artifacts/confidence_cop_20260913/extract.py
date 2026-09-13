"""Read saved scans and exact control-used JPEGs; resumable derived evidence only."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import h5py
import numpy as np
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, revision

CACHE_SCHEMA = 'exact_control_frame_centroid_cache_v1'
GROUPS = ('001', '003', '007', 'jiaqi', 'pei')


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_stamp(path):
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns)


def prepare(saved, out):
    attempt = '002' if saved.parent.name == '001' and saved.stem == 'RH_Per_S_DtP' else '001'
    folder = saved.parent / 'attempts' / saved.stem / attempt
    log = folder / 'contact_qp.jsonl'
    raw = folder / 'raw.h5'
    identity = f'{saved.parent.name}/{saved.stem}'
    dest = out / saved.parent.name / saved.stem
    dest.mkdir(parents=True, exist_ok=True)
    controls, publications, reviews, config, effective, counts = [], {}, {}, None, None, Counter()
    with log.open() as stream:
        for line in stream:
            row = json.loads(line)
            event = row['event']; counts[event] += 1
            if event == 'study_start':
                config = row['config']; effective = row.get('effective_configuration', {})
            elif event == 'control_sample':
                controls.append(row)
            elif event == 'publication':
                publications[row['control_id']] = row
            elif event in ('publication_review','publication_review_rejected'):
                reviews[row['control_id']] = row
    assert config is not None
    needed = {}
    control_temp = dest / 'control.jsonl.gz.tmp'
    with gzip.open(control_temp, 'wt') as stream:
        for row in controls:
            # Retain all control ticks: tanks start once before scan progress.
            fields = ('control_id', 'record_monotonic_s', 'reference_s', 'nominal_twist_tool',
                'candidate_twist_tool', 'previous_outer_command_tool', 'command_slew_dt_s',
                'rotation_base_tcp', 'alpha', 'alpha_preferred', 'command_hold_model_s',
                'control_actual_dt_s', 'feature', 'image_compatible', 'image_feedback_status',
                'control_wrench_tool', 'physical_wrench_candidate_tool', 'energy', 'allocation_diagnostics',
                'source', 'compute_elapsed_s')
            compact = {key: row.get(key) for key in fields}
            compact['publication'] = publications.get(row['control_id'])
            compact['publication_review'] = reviews.get(row['control_id'])
            stream.write(json.dumps(compact, separators=(',', ':')) + '\n')
            feature = row.get('feature')
            if feature is not None:
                previous = needed.setdefault(feature['frame_seq'], feature)
                assert previous['source_id'] == feature['source_id'], 'frame-sequence publisher collision'
    control_temp.replace(dest / 'control.jsonl.gz')
    feature_hash = hashlib.sha256((ROOT / 'peirastic/contact_qp/features.py').read_bytes()).hexdigest()
    sources = dict(saved=source_stamp(saved), raw=source_stamp(raw), log=source_stamp(log))
    cache_key = revision(dict(schema=CACHE_SCHEMA, config=config['feature']['config'],
                              feature_source_sha256=feature_hash, sources=sources))
    manifest = dict(identity=identity, cache_schema=CACHE_SCHEMA, cache_key=cache_key,
        sources=sources, attempt=attempt, config=config, effective_configuration=effective,
        controls=len(controls), moving_controls=sum(r['reference_s'] > 0 for r in controls),
        required_unique_frames=len(needed), event_counts=dict(counts),
        feature_source_sha256=feature_hash)
    previous_path = dest/'manifest.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous.get('cache_key')==cache_key and 'extraction' in previous:
            manifest['extraction']=previous['extraction']
    dump(dest / 'manifest.json', manifest)
    return dest, manifest, needed


def extract(dest, manifest, needed):
    cache = dest / 'frames.jsonl'
    completed = {}
    if cache.exists():
        with cache.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['cache_key'] != manifest['cache_key']:
                    raise ValueError(f'cache provenance mismatch: {cache}')
                completed[row['frame_seq']] = row
    cfg = FeatureConfig(**manifest['config']['feature']['config'])
    extractor = FeatureExtractor(cfg)
    start = time.perf_counter()
    with h5py.File(manifest['sources']['raw']['path'], 'r') as raw, cache.open('a') as stream:
        seqs = raw['ultrasound/frame_index'][:]
        if len(set(seqs)) != len(seqs):
            raise ValueError('ambiguous duplicate raw frame indexes')
        indexes = {int(seq): i for i, seq in enumerate(seqs)}
        for seq, feature in sorted(needed.items()):
            if seq in completed:
                continue
            row = dict(cache_key=manifest['cache_key'], frame_seq=seq)
            idx = indexes.get(seq)
            if idx is None:
                row.update(status='missing_exact_raw_frame')
            else:
                jpeg = np.asarray(raw['ultrasound/jpeg'][idx], dtype=np.uint8)
                metadata = json.loads(raw['ultrasound/metadata_json'][idx])
                identity = metadata.get('source_id', '') + ':' + metadata.get('publisher_instance_id', '')
                if identity != feature['source_id']:
                    row.update(status='source_identity_mismatch', raw_source_id=identity)
                else:
                    image = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise ValueError(f'invalid JPEG: {dest} frame {seq}')
                    obs, _ = extractor.extract(image, frame_seq=seq, source_id=feature['source_id'],
                        capture_time_s=feature['effective_time_s'], received_time_s=feature['received_time_s'],
                        already_aligned=True, clock_domain='offline_aligned',
                        crop_box=metadata.get('crop_box'), hflip=metadata.get('hflip', False))
                    row.update(status='exact', raw_index=idx,
                        jpeg_sha256=hashlib.sha256(jpeg.tobytes()).hexdigest(),
                        raw_timestamp_mono_ns=int(raw['ultrasound/timestamp_mono_ns'][idx]),
                        confidence_centroid_x=obs.confidence_centroid_x,
                        confidence_centroid_valid=obs.confidence_centroid_valid,
                        confidence_feature_version=obs.confidence_feature_version,
                        recomputed_quality=obs.quality.tolist(), recomputed_valid=obs.valid.tolist(),
                        quality_max_abs_error=float(np.max(abs(obs.quality-np.asarray(feature['quality'])))))
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
            completed[seq] = row
    manifest['extraction'] = dict(status_counts=dict(Counter(r['status'] for r in completed.values())),
        frames=len(completed), wall_s=time.perf_counter()-start,
        quality_max_abs_error=max((r.get('quality_max_abs_error', 0.) for r in completed.values()), default=0.))
    dump(dest / 'manifest.json', manifest)
    print(json.dumps(dict(scan=manifest['identity'], **manifest['extraction'])), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/media/camp/PEI_T7/icra 2027_contact/uncalibrated'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'cache')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--groups', nargs='+', choices=GROUPS)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    # One CPU and one BLAS thread; environment variables should be set before Python starts.
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {max(allowed) if args.cpu is None else args.cpu})
    cv2.setNumThreads(1)
    saved = [p for group in GROUPS for p in sorted((args.data/group).glob('*.h5'))]
    assert len(saved) == 49, f'expected 49 saved scans, found {len(saved)}'
    if args.groups:
        saved = [p for p in saved if p.parent.name in args.groups]
    if args.limit:
        saved = saved[:args.limit]
    manifests = []
    for path in saved:
        dest, manifest, needed = prepare(path, args.out)
        manifests.append(manifest if args.prepare_only else extract(dest, manifest, needed))
        name = 'manifest.json' if not args.groups else 'manifest-'+'-'.join(args.groups)+'.json'
        dump(args.out / name, manifests)


if __name__ == '__main__':
    main()
