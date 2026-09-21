#!/usr/bin/env python3
"""Replay recorded images through handheld evidence; never imports robot I/O.

Output is a detector/command-logic audit, not a counterfactual closed-loop run.
Old pad logs lack frame IDs: arrival time is explicitly modelled, not recovered.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from peirastic.apps.contact_qp_features import process_parts
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor
from peirastic.realman8dof.modes.pad_contact_config import load_pad_contact_config
from peirastic.realman8dof.modes.pad_contact_evidence import PadContactEvidence


def replay(directory, output, stride=2, processing_delay=.077):
    config = load_pad_contact_config()
    extractor = FeatureExtractor(FeatureConfig(**config['feature']['config']))
    receiver = SimpleNamespace(observation=None, close=lambda **kwargs: None)
    guide = PadContactEvidence(config, receiver)
    with (directory/'pad.jsonl').open() as stream:
        controls = [r for line in stream if (r := json.loads(line)).get('event') == 'control_sample']
    times = np.array([r['record_monotonic_s'] for r in controls])
    rows = []
    selections = {1132, 1882, 1939, 2128, 2684, 3025} if directory.name.endswith('144554') else set()
    with h5py.File(directory/'raw.h5', 'r') as raw:
        group = raw['ultrasound']
        count = len(group['jpeg'])
        indices = sorted(set(range(0, count, stride)) | {i for i in selections if i < count})
        origin = json.loads(group['metadata_json'][0])['capture_monotonic_ns']*1e-9
        previous_now = None
        control_index = 0
        for i in indices:
            metadata = group['metadata_json'][i]
            meta = json.loads(metadata)
            capture = meta['capture_monotonic_ns']*1e-9
            now = capture+processing_delay
            obs = process_parts([b'ultrasound', metadata, bytes(group['jpeg'][i])], extractor, now)
            # Consume intervening force samples at their original control cadence.
            # Frame subsampling must not turn a 5 ms force difference gate
            # into a 100 ms difference gate.
            while control_index < len(controls) and times[control_index] < now:
                row = controls[control_index]
                stamp = times[control_index]
                if previous_now is None or stamp > previous_now:
                    guide.update(now_s=stamp, dt_s=.005 if previous_now is None else stamp-previous_now,
                        contact_present=bool(row['contact']), force_n=float(row['fz']),
                        stick_speed_m_s=float(row['stick_speed_m_s']))
                    previous_now = stamp
                control_index += 1
            receiver.observation = obs
            control = controls[int(np.clip(np.searchsorted(times, now, side='right')-1, 0, len(times)-1))]
            answer = guide.update(now_s=now, dt_s=.1 if previous_now is None else now-previous_now,
                contact_present=bool(control['contact']), force_n=float(control['fz']),
                stick_speed_m_s=float(control['stick_speed_m_s']))
            previous_now = now
            facts = answer.facts
            rows.append(dict(row=i, time_s=capture-origin,
                segment='calibration' if i < count//2 else 'validation',
                reason=answer.reason, alpha_target=answer.alpha_target,
                omega_request_rad_s=answer.omega_request_rad_s, contact=control['contact'],
                force_n=control['fz'], stick_m_s=control['stick_speed_m_s'],
                image_age_s=now-obs.effective_time_s,
                q_min=float(np.min(obs.region_confidence)),
                echo_min=float(np.min(obs.nearfield_echo)),
                dark_columns=json.dumps(facts.get('dark_columns', [])),
                baseline_ready=facts['baseline_ready'], selected=i in selections))
    output.mkdir(parents=True, exist_ok=True)
    with (output/(directory.name+'.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(session=str(directory), frames=len(rows), stride=stride,
        processing_delay_s=processing_delay, image_age_s=processing_delay+config['feature']['config']['effective_delay_s'],
        alignment='modelled capture+processing delay; nearest preceding original control sample',
        temporal_split='first half calibration; second half validation; no parameter fitting in this script',
        caveat='Offline evidence only. Force/stick are the original recorded trajectory, not a new-policy physical response.',
        segments={name: dict(Counter(r['reason'] for r in rows if r['segment']==name))
                  for name in ('calibration', 'validation')},
        baseline_ready_frames=sum(r['baseline_ready'] for r in rows),
        selected=[r for r in rows if r['selected']], configuration=config)
    (output/(directory.name+'.json')).write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('configuration','selected')}))
    print(json.dumps(summary['selected']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--processing-delay', type=float, default=.077)
    args = parser.parse_args()
    if args.stride < 1 or not np.isfinite(args.processing_delay) or args.processing_delay < 0:
        parser.error('stride must be positive and delay nonnegative')
    for directory in args.sessions:
        replay(directory, args.output, args.stride, args.processing_delay)


if __name__ == '__main__':
    main()
