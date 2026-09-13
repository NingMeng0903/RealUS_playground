"""Stream current attempts read-only; separate proposal ages from force ages."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def summary(path):
    counts=Counter();sources={};energies=[];charge=debit=0.;pauses=[];rejects=[]
    tracking=[];samples=[];last_energy=None
    with path.open() as stream:
        for line in stream:
            row=json.loads(line);event=row['event'];counts[event]+=1
            if event=='control_sample':
                source=row.get('source') or {}
                sources[row['control_id']]=source.get('source_t_s')
                energies.append(row['energy']['balance_j'])
                if row['reference_s']>0:
                    tracking.append(row['record_monotonic_s'])
                    samples.append((source.get('age_s'),row['image_feedback_status'],
                                    row['energy']['balance_j']))
            elif event=='logical_command_energy':
                last_energy=row
                for entry in row.get('events',[]):
                    if entry.get('event')=='logical_epoch_work':
                        change=entry['tank_work_j']
                        charge+=max(change,0.);debit+=max(-change,0.)
                        energies.append(entry['balance_j'])
            elif event=='image_feedback_recovered':
                pauses.append(row['unavailable_s'])
            elif event=='publication_review_rejected':
                source=sources.get(row['control_id'])
                rejects.append(dict(reason=row['reason'],
                    source_age_at_review_s=None if source is None else row['review_time_s']-source,
                    proposal_age_at_review_s=row['review_time_s']-row['created_time_s']))
    ages=[s[0] for s in samples if s[0] is not None]
    result=json.loads((path.parent/'result.json').read_text())
    return dict(attempt=str(path.parent),result=result,event_counts=dict(counts),
        tank_min_j=min(energies),tank_max_j=max(energies),tank_end_j=last_energy['balance_j'],
        tank_charge_j=charge,tank_debit_j=debit,
        cumulative_port_work_j=last_energy['cumulative_port_work_j'],
        cumulative_task_source_used_j=last_energy['cumulative_task_source_used_j'],
        image_pause_count=len(pauses),image_pause_total_s=sum(pauses),
        image_pause_max_s=max(pauses,default=0.),
        tracking_duration_s=tracking[-1]-tracking[0],tracking_samples=len(samples),
        tracking_image_status_counts=dict(Counter(s[1] for s in samples)),
        force_ingress_age_ms=dict(median=float(np.median(ages)*1000),
            p95=float(np.quantile(ages,.95)*1000),max=float(max(ages)*1000)),
        review_rejections=rejects)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=Path('/media/camp/yameng/icra 2027/uncalibrated/001'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('scan_counts.json'))
    args=parser.parse_args()
    summaries={p.parent.parent.name:summary(p) for p in sorted(args.session.glob('attempts/*/001/contact_qp.jsonl'))}
    args.output.write_text(json.dumps(summaries,indent=2)+'\n')
    for name,data in summaries.items():
        print(name,json.dumps({k:data[k] for k in ('tank_min_j','tank_max_j','tank_end_j',
            'tank_charge_j','tank_debit_j','image_pause_count','image_pause_total_s',
            'tracking_duration_s','tracking_image_status_counts','force_ingress_age_ms','review_rejections')}))


if __name__=='__main__':main()
