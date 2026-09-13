"""Standalone figure of recorded and frozen-input command calculations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).parent
scans=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*/*/summary.json'))]
labels=[r['identity'].replace('RH_Per_','R ').replace('LH_Per_','L ') for r in scans]
x=np.arange(len(scans))
fig,axes=plt.subplots(4,1,figsize=(19,15),sharex=True,layout='constrained')
colors=('#5d6770','#0072b2','#d55e00')
names=('Recorded old','Angular: frozen inputs / ideal inner','Angular + CoP: frozen inputs / ideal inner')
for i,(name,color) in enumerate(zip(names,colors)):
    mode=None if i==0 else ('confidence_angular_v1','confidence_cop_v1')[i-1]
    data=[r['recorded_old'] if mode is None else r['modes'][mode] for r in scans]
    tanks=[d['tank_j']['min'] if mode is None else d['tank_min_j'] for d in data]
    weak=[np.nan if d['weak_side_speed_m_s']['mean'] is None else d['weak_side_speed_m_s']['mean']*1000 for d in data]
    axes[0].plot(x,tanks,'o-',ms=3,lw=1,color=color,label=name)
    axes[1].plot(x,weak,'o-',ms=3,lw=1,color=color)
    if mode:
        pair=[np.nan if d['pairing_residual_abs_m_s']['p95'] is None else d['pairing_residual_abs_m_s']['p95']*1000 for d in data]
        axes[2].plot(x,pair,'o-',ms=3,lw=1,color=color)
        axes[3].bar(x+(.17 if i==2 else -.17),[d['counts'].get('deferred_or_rejected',0) for d in data],width=.34,color=color)
axes[0].axhline(.05,color='#a22',ls='--',lw=1,label='Stopping reserve: .05 J')
axes[0].set_ylabel('Minimum tank (J)')
axes[0].legend(ncol=2,loc='lower left',fontsize=9)
axes[1].axhline(0,color='#777',lw=.7)
axes[1].set_ylabel('Mean weak-window\nnormal speed (mm/s)')
axes[2].set_ylabel('Normal pairing residual\np95 (mm/s)')
axes[2].set_ylim(0,max(.001,max((r['modes'][name]['pairing_residual_abs_m_s']['p95'] or 0.)*1000
                               for r in scans for name in ('confidence_angular_v1','confidence_cop_v1'))*1.1))
axes[3].set_ylabel('Deferred/rejected\nproposals (count)')
axes[3].set_ylim(0,max(1,max(r['modes'][name]['counts'].get('deferred_or_rejected',0)
                            for r in scans for name in ('confidence_angular_v1','confidence_cop_v1'))*1.1))
axes[3].set_xticks(x,labels,rotation=90,fontsize=8)
for ax in axes:
    ax.grid(axis='y',alpha=.2)
    ax.set_xlim(-.7,len(scans)-.3)
    for j in range(1,len(scans)):
        if scans[j]['identity'].split('/')[0]!=scans[j-1]['identity'].split('/')[0]:
            ax.axvline(j-.5,color='#999',lw=.8,alpha=.5)
fig.suptitle('49 saved scans: command calculations on frozen recorded inputs\nNo counterfactual ultrasound, force, anatomy, or physical stability prediction',fontsize=15)
fig.savefig(ROOT/'comparison.png',dpi=170)
plt.close(fig)
