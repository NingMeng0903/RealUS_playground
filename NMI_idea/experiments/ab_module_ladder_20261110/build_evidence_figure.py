"""Standalone scientific summary, generated only from sealed result files."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    x=json.loads((ROOT/'planning/test_frozen/ANALYSIS.json').read_text())
    c=json.loads((ROOT/'continuous_mismatch/formal_run_20261110/RESULTS.json').read_text())
    methods=['A_continuous_servo','A_myopic','entropy','fixed_two','B_two_step',
             'B_no_future_cap','B_frozen_belief']
    labels=['A: image servo','A: myopic task','Entropy','Fixed two-step',
            'B: contingent','B: no future capability','B: frozen belief']
    colors=['#22876f','#a5bdb5','#b9c2cc','#9eaec1','#3359ad','#b7c6e9','#d6dceb']
    fig,axes=plt.subplots(2,2,figsize=(13,8.7))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    y=np.arange(len(methods))
    for axis,key,multiplier,title in [(axes[0,0],'success',100,'Finite catalogue: correct public stop'),
                                      (axes[0,1],'RMT_s',1,'Finite catalogue: censored task time')]:
        vals=np.array([x['methods'][m][key]['mean'] for m in methods])*multiplier
        ci=np.array([x['methods'][m][key]['object_clustered_95pct'] for m in methods])*multiplier
        axis.barh(y,vals,color=colors,height=.65)
        axis.errorbar(vals,y,xerr=[vals-ci[:,0],ci[:,1]-vals],fmt='none',ecolor='#263443',capsize=3,lw=1)
        row_labels=[f"{label} ({x['methods'][m]['correct_count']}/32)" for label,m in zip(labels,methods)] if key=='success' else labels
        axis.set_yticks(y,row_labels);axis.invert_yaxis();axis.set_title(title,loc='left',fontweight='bold')
        axis.set_xlabel('Success (%)' if key=='success' else 'Seconds; failures assigned 60 s')
        axis.set_xlim(0,100 if key=='success' else 64)
        for j,m in enumerate(methods):
            count=x['methods'][m]['correct_count']
            if key!='success':
                axis.text(max(1,vals[j]/2),j,f'{vals[j]:.2f}',ha='center',va='center',fontsize=9,color='#102b3a')
        axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
    axis=axes[1,0]
    groups=c['summary']['by_condition_method']
    keys=['iid/A_continuous_servo','iid/B_two_step','shifted/A_continuous_servo','shifted/B_two_step']
    axis.bar(np.arange(4),[100*groups[k]['correct_public_stops']/groups[k]['n'] for k in keys],
             color=['#22876f','#3359ad']*2,width=.55)
    for j in range(4):axis.text(j,4,'0/8',ha='center',fontsize=12,fontweight='bold')
    axis.set_xticks(np.arange(4),['IID / A','IID / B','Shifted / A','Shifted / B'])
    axis.set_ylim(0,100);axis.set_ylabel('Correct public stop (%)')
    axis.set_title('Independent continuous anatomy: unresolved',loc='left',fontweight='bold')
    axis.text(.04,.75,'32 fixed hypotheses; GT is independent\nShift: +8 mm registration, +0.10 slope\nFinite-catalogue success does not transfer',
              transform=axis.transAxes,va='top',fontsize=10)
    axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    axis=axes[1,1]
    budgets=np.array([8,16,32]); locations=np.arange(3);width=.19
    for k,(name,vals,color) in enumerate([
        ('dIRD',[8,16,32],'#3359ad'),('Geometry',[8,16,29],'#22876f'),
        ('Manifest order',[5,10,23],'#9eaec1'),('Random mean',[5.21875,10.03125,20.09375],'#c5cbd1')]):
        axis.bar(locations+(k-1.5)*width,vals,width=width,label=name,color=color)
    axis.set_xticks(locations,[str(i) for i in budgets]);axis.set_xlabel('Full endpoint-search query budget')
    axis.set_ylabel('Checked legal endpoints found');axis.set_ylim(0,36)
    axis.set_title('dIRD: endpoint retrieval increment',loc='left',fontweight='bold')
    axis.legend(frameon=False,ncol=2,fontsize=9,loc='upper left')
    axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    for a in axes.flat:
        a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
    fig.suptitle('A/B module validation — 10 November 2026',fontsize=17,fontweight='bold',y=.995)
    fig.text(.02,.012,'Geometric L0 only. Six actions / 60 s; public verifier shared. Finite test: 16 anatomies × 2 robot contexts;\n'
             'error bars resample anatomy clusters. dIRD panel is a separate robot-only retrieval test, not task success.',fontsize=9,color='#344655')
    fig.tight_layout(rect=(0,.06,1,.97),h_pad=3,w_pad=2)
    fig.savefig(ROOT/'EVIDENCE_SUMMARY.png',dpi=200,facecolor='white')
    fig.savefig(ROOT/'EVIDENCE_SUMMARY.svg',facecolor='white')
    print(ROOT/'EVIDENCE_SUMMARY.png')
