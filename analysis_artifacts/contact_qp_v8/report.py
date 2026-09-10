"""Summarize every frozen unit, including aborts, without retuning."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    records=json.loads((HERE/'metrics.json').read_text())
    lines=['# Frozen synthetic v7/v8 comparison','',
        'Twenty prescribed units: seeds 701/702, five scenarios, two policies. The unchanged shared nominal controller targets 4 N. Both policies use the identical v3 FeatureExtractor at 145×100, cmin 0.8; path 60 mm at 5 mm/s, 5 ms controls. Finite-area mechanics, actuator delay, image delay, and stop tails are retained. See protocol.json and compare.py.', '',
        '**Observed limitation:** both persistent-shadow v8 runs abort on the measured 0.35 rad angle limit, whereas v7 finishes with acoustic gaps. The fixed shadow cannot be repaired by pressing or rocking; v8 accumulates greater actual rocking and force error without reducing latent acoustic bad path. Delayed actuation carries measured angle beyond the command-side bound. The other four scenarios maintain image quality above cmin under the shared nominal controller and do not distinguish the policies. These frozen results do not demonstrate a closed-loop v8 benefit. No controller or experimental parameter was tuned after observing this result.', '',
        'This is a small exploratory simulation, not acceptance evidence, a real-image causal result, or physical certification. No hardware or controller service was connected. Closed-loop energy admission is **off** because the reused run_case has no energy input. energy_sweep.json separately checks instantaneous admission with the existing single full-port constraint and known synthetic full-six-dimensional wrenches; it does not settle or simulate a second ledger.','',
        'Force RMSE and peak include the recorded scan and stopping tail; >4.5 N is sampled measured-force exposure at 5 ms. Angular travel is integrated measured rocking, including the tail. Acoustic bad path is latent plant truth, not an image contact label. Image quality is the scan mean of the extracted left/center/right values, including held observations. Completion with gaps is distinct from clean completion. Deadline counts are the existing measured controller block >5 ms; image extraction runs outside that timer. Concurrent offline Python timing is not a real-time certification.','',
        '| Scenario / seed | Policy | Status / reason | Force RMSE N | Peak N | >4.5 N s | Angular rad | Acoustic bad mm | Quality L/C/R | p99 ms | >5 ms |',
        '|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    def number(r,k,scale=1):
        v=r.get(k)
        return '—' if v is None else f'{v*scale:.4f}'
    for r in sorted(records,key=lambda r:(r['scenario'],r['seed'],r['policy'])):
        quality='/'.join(f'{x:.3f}' for x in r.get('quality_mean_lcr') or []) or '—'
        lines.append('| '+ ' | '.join([f"{r['scenario']} / {r['seed']}",r['policy'],r['status']+(' / '+r['reason'] if r.get('reason') else ''),number(r,'rmse'),number(r,'force_peak_n'),number(r,'force_above_4p5_s'),number(r,'angular_travel_rad'),number(r,'acoustic_bad_path_m',1000),quality,number(r,'controller_p99_ms'),str(r.get('simulated_deadline_overruns','—'))])+' |')
    lines+=['','Paired changes below are v8 minus v7, averaged over the two fixed seeds. They are descriptive; two seeds do not establish generalization.','',
        '| Scenario | Δ RMSE N | Δ peak N | Δ >4.5 N s | Δ angular rad | Δ acoustic bad mm |',
        '|---|---:|---:|---:|---:|---:|']
    for scenario in sorted(set(r['scenario'] for r in records)):
        deltas=[]
        for key,scale in [('rmse',1),('force_peak_n',1),('force_above_4p5_s',1),('angular_travel_rad',1),('acoustic_bad_path_m',1000)]:
            v=[]
            for seed in (701,702):
                pair={r['policy']:r for r in records if r['scenario']==scenario and r['seed']==seed}
                if len(pair)==2 and all(key in r for r in pair.values()):
                    v.append(pair['differential_repair_v8'][key]-pair['legacy_v7'][key])
            deltas.append(f'{np.mean(v)*scale:+.5f}' if len(v)==2 else 'incomplete')
        lines.append('| '+scenario+' | '+' | '.join(deltas)+' |')
    counts={p:{s:sum(r['policy']==p and r['status']==s for r in records) for s in sorted(set(r['status'] for r in records))} for p in ('legacy_v7','differential_repair_v8')}
    lines+=['',f'Completed records: {len(records)}/20. Status counts: `{json.dumps(counts)}`.']
    start=json.loads((HERE/'source_hashes_start.json').read_text())
    end=json.loads((HERE/'source_hashes_end.json').read_text())
    changed=[k for k in start if start[k]!=end.get(k)]
    lines+=['',f'Source files changed during execution: `{json.dumps(changed)}`.']
    energy=json.loads((HERE/'energy_sweep.json').read_text())
    admitted=[r for r in energy['rows'] if r['success']]
    lines+=['',f"Separate energy sweep: {energy['admitted']}/{energy['units']} admitted; {energy['admitted_constraint_violations']} admission violations. Minimum work margin {min(r['margin_work_j'] for r in admitted):.3g} J (numerical tolerance 1e-9 J). Ample-budget maximum deviation from the energy-off optimum {max(r['difference_from_energy_off'] for r in admitted if r['budget_j']==1):.3g}. Zero-budget alpha spans {min(r['alpha'] for r in admitted if r['budget_j']==0):.4f}–{max(r['alpha'] for r in admitted if r['budget_j']==0):.4f}; net full-port cancellation can permit nonzero motion. These are static snapshots, not closed-loop energy histories."]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
