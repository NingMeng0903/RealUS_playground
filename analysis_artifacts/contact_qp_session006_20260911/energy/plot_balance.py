from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent
r=json.loads((out/'energy_audit.json').read_text())
fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
for a in r['attempts']:
 if a.get('outcome')!='completed':continue
 c=json.loads((out/(a['attempt'].replace('/','_')+'_balance.json')).read_text())
 t0=a['supervisor_scan_window']['requested_start_s']
 label=a['attempt'].split('/')[0].removeprefix('RH_Per_')
 ax.plot([x['t_s']-t0 for x in c],[x['balance_j'] for x in c],label=label,lw=1.4)
ax.axhline(.05,color='black',ls='--',lw=1,label='Reserve 0.05 J')
ax.axhline(.15,color='gray',ls=':',lw=1,label='Capacity 0.15 J')
ax.axvline(0,color='gray',lw=.7)
ax.set(xlabel='Time from observed scan start (s)',ylabel='Logical tank balance (J)',title='Session 006: all six completed scans retain spendable energy',ylim=(.045,.155))
ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8,loc='lower center')
fig.savefig(out/'completed_balance.png',dpi=180)
fig.savefig(out/'completed_balance.svg')
