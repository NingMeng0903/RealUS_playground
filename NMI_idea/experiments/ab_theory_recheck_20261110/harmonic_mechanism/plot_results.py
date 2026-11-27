from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
result=json.loads((ROOT/'RESULTS.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
for side,color in [('right','#2869b2'),('left','#d37b20')]:
 rows=result['results'][side];degrees=[r['knee_deg'] for r in rows]
 axes[0].plot(degrees,[r['frozen_world_point_error_mm']['p95'] for r in rows],'-o',color=color,label=f'{side}: frozen world point')
 axes[1].plot(degrees,[r['wrong_vector_normal_error_deg']['p95'] for r in rows],'-o',color=color,label=f'{side}: wrong $Fn$')
 axes[2].plot(degrees,[r['raw_gradient_reparameterization_angle_deg']['median'] for r in rows],'-o',color=color,label=f'{side}: raw chart gradient')
axes[0].plot(degrees,np.zeros(len(degrees)),'k--',label='Material / ordinary tet (coincide)')
axes[1].plot(degrees,np.zeros(len(degrees)),'k--',label=r'Correct $F^{-T}n$ reference')
axes[2].plot(degrees,np.zeros(len(degrees)),'k--',label='Physical metric / Cartesian (coincide)')
for ax,title,ylabel in zip(axes,['Address transport under actual leg poses','Local plane normal transport','Same energy, different chart scales'],
 ['p95 position error vs posed reference (mm)','p95 normal error (degrees)','Direction change (degrees)']):
 ax.set(title=title,xlabel='Knee flexion (degrees)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper left')
fig.suptitle('Production harmonic shaft assets: 256 cells per leg, seven saved poses',fontsize=12)
fig.savefig(ROOT/'mechanism_summary.png',dpi=170)
