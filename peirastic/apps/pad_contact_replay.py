"""Offline original-JPEG pad evidence inspection; no robot or transport I/O.

python -m peirastic.apps.pad_contact_replay SESSION... --output /tmp/pad20_replay
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from peirastic.apps.contact_qp_features import encode_feature_payload
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, region_quality
from peirastic.realman8dof.modes.pad_contact_evidence import PadContactEvidence


# Image review annotations, not confidence-derived truth or contact measurements.
ANNOTATIONS = {
    '20260921_181359': {4757: ('left_gap', 'Residual dark wedge at left aperture edge.')},
    '20260921_182016': {
        218: ('nearfield_healthy', 'Deep shadow below bright continuous near-field tissue; do not equate deep shadow with lost contact.'),
        844: ('right_gap', 'Large right-side dark wedge reaches near field.'),
        998: ('global_dark', 'Near field dark across almost the entire aperture.'),
        1307: ('nearfield_healthy', 'Near-field tissue remains visible across aperture despite deeper shadow.'),
    },
    '20260921_182309': {
        2834: ('left_gap', 'Dark left-edge wedge reaches near field.'),
        3416: ('central_ambiguous', 'Central dark column with visible tissue on both sides; direction ambiguous.'),
    },
}


class OfflineReceiver:
    observation = None

    def close(self, *, wait=False):
        pass


def nearest_control(rows, times, stamp):
    """Time correlation only: never interpret historical commands as publication."""
    if not len(times):
        return None, None
    pos = int(np.searchsorted(times, stamp))
    candidates = [i for i in (pos-1, pos) if 0 <= i < len(times)]
    index = min(candidates, key=lambda i: abs(times[i]-stamp))
    return rows[index], float(times[index]-stamp)


def temporal_split(capture_times, fraction=.6):
    """Disjoint contiguous intervals, including unsampled original frames."""
    n = len(capture_times)
    boundary = min(n-1, max(1, int(n*fraction))) if n > 1 else n
    def segment(a, b):
        return dict(frame_start=a, frame_stop_exclusive=b,
                    capture_time_range_s=None if a == b else [float(capture_times[a]), float(capture_times[b-1])])
    return dict(calibration=segment(0, boundary), validation=segment(boundary, n),
                boundary_frame=boundary, policy='contiguous_first_60_percent_last_40_percent',
                limitations='Retrospective diagnostic split; reviewed cases are already known. No parameter fitting or independent held-out performance claim.')


def extract_original(jpeg, metadata, extractor, received_s):
    import cv2
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('invalid recorded JPEG')
    capture = metadata.get('capture_monotonic_ns')
    if type(capture) is not int or capture <= 0:
        raise ValueError('recorded frame lacks original monotonic capture timestamp')
    obs, confidence = extractor.extract(
        image, frame_seq=int(metadata['frame_index']),
        source_id=metadata['source_id']+':'+metadata['publisher_instance_id'],
        registration_source_id=metadata['source_id'],
        capture_time_s=capture*1e-9, received_time_s=received_s,
        crop_box=metadata.get('crop_box'), hflip=metadata.get('hflip', False))
    return image, obs, confidence


def observation_for_layout(obs, confidence, extractor, count):
    """Reaggregate the original confidence map; never resample historical Q."""
    fc = replace(extractor.config, region_count=count)
    region = region_quality(confidence, fc, np.asarray(extractor.last_features['unknown_columns'], dtype=bool))
    fields = {k: region[k] for k in ('region_confidence', 'region_valid', 'region_edges',
                                     'region_feature_version', 'region_layout_version')}
    return replace(obs, **fields), region


def _summary(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else dict(count=len(values), median=float(np.median(values)),
                                            p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _plot_case(path, image, row, fc):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [7, 1, 1, 1]})
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    w, h = image.shape[1], image.shape[0]
    for x in fc.region_edges*w:
        axes[0].axvline(x-.5, color='cyan', alpha=.6, lw=.6)
    for i, x in enumerate((fc.region_edges[:-1]+fc.region_edges[1:])*w/2):
        axes[0].text(x, h*.025, str(i+1), color='yellow', ha='center', fontsize=9)
    # The dark mask comes from the echo band, not the (different) confidence
    # window. Use the actual feature metadata rather than its display default.
    for y in np.asarray(row['nearfield_echo_depth'])*h:
        axes[0].axhline(y, color='orange', lw=1)
    axes[0].set_title(f"{row['session']} frame row {row['frame_row']} / camera {row['frame_seq']} | {row.get('annotation', {}).get('label', 'inspection')}\n"
                      f"capture {row['capture_time_s']:.3f}s; effective {row['effective_image_time_s']:.3f}s; simulated control {row['replay_control_time_s']:.3f}s")
    axes[0].axis('off')
    for axis, key, label in zip(axes[1:3], ('q_raw', 'q_predicted'), ('Original-image region Q25', 'Replay KF predicted Q25 (trend only)')):
        values = row[key] if row[key] is not None else [np.nan]*fc.region_count
        axis.imshow([values], aspect='auto', vmin=0, vmax=1, cmap='viridis', extent=(.5, fc.region_count+.5, 0, 1))
        axis.set_xticks(range(1, fc.region_count+1)); axis.set_yticks([]); axis.set_ylabel(label, fontsize=8)
    dark = np.zeros(fc.width); dark[row['dark_columns']] = 1
    axes[3].imshow([dark], aspect='auto', vmin=0, vmax=1, cmap='Reds', extent=(0, fc.width, 0, 1))
    axes[3].set_yticks([]); axes[3].set_xlabel(f"145-column near-field raw dark mask; {row['raw_classification']}; published command unknown unless explicitly recorded")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_timeline(path, rows, controls, fc):
    import matplotlib.pyplot as plt
    origin = rows[0]['capture_time_s']
    effective = np.array([r['effective_image_time_s'] for r in rows])-origin
    now = np.array([r['replay_control_time_s'] for r in rows])-origin
    fig, axes = plt.subplots(6, 1, figsize=(15, 12), sharex=True)
    for ax, key, stamps, title in ((axes[0], 'q_raw', effective, 'Raw Q25 at effective IMAGE time'),
                                  (axes[1], 'q_predicted', now, 'Replay predicted Q25 at simulated CONTROL time')):
        matrix = np.array([r[key] if r[key] is not None else [np.nan]*fc.region_count for r in rows])
        ax.pcolormesh(stamps, np.arange(1, fc.region_count+1), matrix.T, vmin=0, vmax=1, cmap='viridis', shading='nearest')
        ax.set_yticks(range(1, fc.region_count+1, 2)); ax.set_ylabel('region'); ax.set_title(title, fontsize=10)
    mask = np.zeros((len(rows), fc.width))
    for i, row in enumerate(rows): mask[i, row['dark_columns']] = 1
    axes[2].pcolormesh(effective, np.arange(fc.width), mask.T, cmap='Reds', vmin=0, vmax=1, shading='nearest')
    axes[2].set_ylabel('echo column'); axes[2].set_title('Raw near-field dark mask (IMAGE time)', fontsize=10)
    ct = np.array([r['record_monotonic_s'] for r in controls])-origin
    def series(key): return [r.get(key) if r.get(key) is not None else np.nan for r in controls]
    axes[3].plot(ct, series('fz'), label='recorded force, control time'); axes[3].set_ylabel('N'); axes[3].legend(loc='upper right')
    for key, label in [('omega_moment', 'historical nominal moment'), ('omega_visual', 'historical visual'), ('omega_cmd', 'historical allocated/requested (NOT publication)'), ('published_omega_tool_y_rad_s', 'explicit published (if recorded)')]:
        axes[4].plot(ct, series(key), label=label, lw=.8)
    axes[4].plot(now, [r['omega_request_rad_s'] for r in rows], label='new evidence request, simulated', lw=.8)
    axes[4].set_ylabel('rad/s'); axes[4].legend(fontsize=7, loc='upper right')
    axes[5].plot(ct, series('alpha'), label='historical alpha')
    axes[5].plot(now, [r['alpha_target'] for r in rows], label='new evidence alpha target (simulated)')
    axes[5].legend(); axes[5].set_xlabel('Seconds relative to first replayed capture; historical correlation approximate, delayed; publication unknown')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def replay_session(session, output, config, *, stride=1, receive_delay_s=0.):
    import h5py
    session, output = Path(session), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    controls = [json.loads(line) for line in (session/'pad.jsonl').read_text().splitlines()]
    controls = sorted((r for r in controls if r.get('event') == 'control_sample'), key=lambda r: r['record_monotonic_s'])
    control_times = np.array([r['record_monotonic_s'] for r in controls])
    fc = FeatureConfig(**config['feature']['config']); extractor = FeatureExtractor(fc)
    receiver = OfflineReceiver(); evidence = PadContactEvidence(config, receiver=receiver)
    comparison_config = deepcopy(config)
    comparison_config['feature']['region_layout_version'] = replace(fc, region_count=10).region_layout_version
    comparison_config['feature']['config']['region_count'] = 10
    comparison_config['image_kf']['channels'] = 10
    comparison_receiver = OfflineReceiver(); comparison = PadContactEvidence(comparison_config, receiver=comparison_receiver)
    rows, cases, timings, kf_timings, payload_sizes = [], [], [], [], []
    benchmark10 = []
    with h5py.File(session/'raw.h5', 'r') as file:
        group = file['ultrasound']
        metadata = [json.loads(value) for value in group['metadata_json'][:]]
        capture = np.array([m['capture_monotonic_ns']*1e-9 for m in metadata])
        split = temporal_split(capture)
        selected = sorted(set(range(0, len(capture), stride)) | {len(capture)-1} | set(ANNOTATIONS.get(session.name, {})))
        previous_now = None
        for index in selected:
            if index >= len(capture): continue
            now = float(capture[index]+receive_delay_s)
            start = time.perf_counter()
            jpeg = group['jpeg'][index].tobytes()
            image, obs, confidence = extract_original(jpeg, metadata[index], extractor, now)
            processing_s = time.perf_counter()-start; timings.append(processing_s)
            blob = encode_feature_payload(obs, extractor.last_features, processing_s); payload_sizes.append(len(blob))
            obs10, region10 = observation_for_layout(obs, confidence, extractor, 10)
            # Stateless view before sequential update retains raw-image classification
            # even when force/contact gates prevent a control request.
            raw_kind = evidence._classify(obs)[0]; raw_facts = deepcopy(evidence._observation_facts)
            raw_kind10 = comparison._classify(obs10)[0]
            control, offset = nearest_control(controls, control_times, now)
            historical, image_offset = nearest_control(controls, control_times, obs.effective_time_s)
            dt = .02 if previous_now is None else max(1e-6, now-previous_now)
            receiver.observation = obs; comparison_receiver.observation = obs10
            kwargs = dict(now_s=now, dt_s=dt, contact_present=bool(control and control.get('contact')),
                          force_n=float(control['fz']) if control else float('nan'),
                          stick_speed_m_s=float(control.get('stick_speed_m_s', 0.)) if control else 0.)
            start = time.perf_counter(); result = evidence.update(**kwargs); kf_timings.append(time.perf_counter()-start)
            result10 = comparison.update(**kwargs)
            previous_now = now
            row = dict(session=session.name, frame_row=index, frame_seq=obs.frame_seq,
                       capture_time_s=float(capture[index]), effective_image_time_s=obs.effective_time_s,
                       replay_control_time_s=now, received_time_s=now,
                       jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),
                       recorder_received_time_s=float(group['received_monotonic_ns'][index])*1e-9 if 'received_monotonic_ns' in group else None,
                       replay_receive_time_semantics='synthetic capture + configured receive delay; original worker receive unknown',
                       historical_alignment='nearest timestamp, approximate delayed correlation; no historical frame attribution',
                       control_time_offset_s=offset, image_time_control_offset_s=image_offset,
                       historical_control_id=None if control is None else control.get('control_id'),
                       historical_image_time_control_id=None if historical is None else historical.get('control_id'),
                       published_omega_tool_y_rad_s=None if control is None else control.get('published_omega_tool_y_rad_s'),
                       publication_status='unknown' if control is None or control.get('published_omega_tool_y_rad_s') is None else 'explicit_field',
                       q_raw=obs.region_confidence.tolist(), q_predicted=result.facts.get('quality_predicted'),
                       q_raw_10=obs10.region_confidence.tolist(), q_predicted_10=result10.facts.get('quality_predicted'),
                       nearfield_echo=obs.nearfield_echo.tolist(),
                       nearfield_echo_depth=obs.nearfield_echo_depth.tolist(), dark_columns=raw_facts['dark_columns'],
                       raw_classification=raw_kind, raw_classification_10=raw_kind10,
                       evidence_reason=result.reason, alpha_target=result.alpha_target,
                       omega_request_rad_s=result.omega_request_rad_s, image_age_s=result.facts['image_age_s'],
                       feature_processing_s=processing_s, evidence_kf_update_s=kf_timings[-1], payload_bytes=len(blob),
                       partition='calibration' if index < split['boundary_frame'] else 'validation')
            annotation = ANNOTATIONS.get(session.name, {}).get(index)
            if annotation:
                row['annotation'] = dict(label=annotation[0], rationale=annotation[1], source='visual review of original recorded image; qualitative, not contact ground truth')
                ten_extractor = FeatureExtractor(replace(fc, region_count=10))
                start = time.perf_counter()
                _, timed10, _ = extract_original(group['jpeg'][index].tobytes(), metadata[index], ten_extractor, now)
                ten_s = time.perf_counter()-start
                ten_bytes = len(encode_feature_payload(timed10, ten_extractor.last_features, ten_s))
                benchmark10.append(dict(frame_row=index, feature_processing_s=ten_s, payload_bytes=ten_bytes))
                row['ten_region_benchmark'] = benchmark10[-1]
                row['q25_comparison'] = {
                    str(count): dict(minimum=float(np.min(q)), minimum_region_1based=int(np.argmin(q))+1,
                                     below_080_regions_1based=(np.flatnonzero(np.asarray(q)<.8)+1).tolist(),
                                     note='Q25 is a diagnostic, not an image-review label')
                    for count, q in ((10, row['q_raw_10']), (fc.region_count, row['q_raw']))}
                cases.append(row.copy()); _plot_case(output/f'frame_{index:06d}.png', image, row, fc)
            elif index == selected[0]: _plot_case(output/f'frame_{index:06d}.png', image, row, fc)
            rows.append(row)
    with (output/'frames.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    _plot_timeline(output/'timeline.png', rows, controls, fc)
    metrics = dict(session=session.name, original_frame_count=len(capture), replayed_frame_count=len(rows), stride=stride,
                   temporal_split=split, feature_processing_s=_summary(timings), evidence_kf_update_s=_summary(kf_timings),
                   payload_bytes=_summary(payload_sizes), image_age_s=_summary([r['image_age_s'] for r in rows]),
                   offline_feature_plus_evidence_s=_summary(np.asarray(timings)+np.asarray(kf_timings)),
                   offline_feature_plus_evidence_exceed_30hz=sum(a+b > 1/30 for a, b in zip(timings, kf_timings)),
                   offline_feature_plus_evidence_exceed_20hz=sum(a+b > .05 for a, b in zip(timings, kf_timings)),
                   estimated_age_after_offline_processing_s=_summary([r['image_age_s']+r['feature_processing_s']+r['evidence_kf_update_s'] for r in rows]),
                   receive_delay_s=receive_delay_s, frame_seq_missing_controls=sum(r.get('frame_seq') is None for r in controls),
                   ten_region_labelled_benchmark=benchmark10,
                   control_count=len(controls), closed_loop_proof=False,
                   ten_vs_twenty=dict(source='same original-JPEG confidence map independently aggregated at 10 and 20; no interpolation of old Q',
                                      region_width_processing_columns={'10': fc.width/10, str(fc.region_count): fc.width/fc.region_count},
                                      labelled_cases=len(cases), classification_disagreements=sum(r['raw_classification'] != r['raw_classification_10'] for r in rows)),
                   limitations=['Historical commands are requested/allocated, publication unknown.',
                                'Replay receive times synthetic; measured CPU latency is offline and excludes publisher/network scheduling.',
                                'Stride > 1 omits temporal evidence and cannot validate confirmation/recovery durations.',
                                'Image labels are qualitative reviewed examples, not physical contact ground truth.'])
    _json(output/'metrics.json', metrics); _json(output/'cases.json', cases)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--receive-delay-s', type=float, default=0.)
    args = parser.parse_args(argv)
    if args.stride < 1 or not np.isfinite(args.receive_delay_s) or args.receive_delay_s < 0:
        parser.error('stride must be positive and receive delay finite/nonnegative')
    from peirastic.realman8dof.modes.pad_scan_config import load_pad_scan_config
    config = load_pad_scan_config(args.config)
    for session in args.sessions:
        if args.output.resolve() == session.resolve() or session.resolve() in args.output.resolve().parents:
            parser.error('output must be outside original recording directories')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    provenance_files = [Path(__file__), root/'realman8dof/modes/pad_contact_evidence.py',
                        root/'realman8dof/modes/pad_scan_config.py', root/'contact_qp/features.py', root/'contact_qp/delayed_kf.py']
    provenance_files.append(args.config or root/'config/contact_qp/pad_scan.yaml')
    try: revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): revision = None
    _json(args.output/'provenance.json', dict(schema='pad_original_jpeg_replay_v1', config=config, git_head=revision,
          source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files},
          config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
          source_recordings=[dict(path=str(p.resolve()), raw_bytes=(p/'raw.h5').stat().st_size,
                                 pad_jsonl_sha256=hashlib.sha256((p/'pad.jsonl').read_bytes()).hexdigest()) for p in args.sessions]))
    results = []
    for session in args.sessions:
        result = replay_session(session, args.output/session.name, config, stride=args.stride, receive_delay_s=args.receive_delay_s)
        results.append(result); print(json.dumps(dict(session=session.name, frames=result['replayed_frame_count'], output=str(args.output/session.name))), flush=True)
    _json(args.output/'metrics.json', results)


if __name__ == '__main__':
    main()
