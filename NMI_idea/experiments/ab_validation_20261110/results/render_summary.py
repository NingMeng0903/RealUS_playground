from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
load=lambda name:json.loads((ROOT/name).read_text())
a=load('core/a_results.json'); b=load('core/extension_results.json')
p=load('core/prior_factorial_results.json'); d=load('dird/summary.json')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
colors=['#67788a','#2e78b7','#e29734','#5c956c']
ax=axs[0,0]; methods=['analytic_geometry','canonical_learned_residual','online_local_jacobian_calibration','oracle_response_reference']
labels=['Analytic','Learned response','Online calibration','Oracle']
for j,(m,label) in enumerate(zip(methods,labels)):
 vals=[a['conditions'][c]['splits']['test']['methods'][m]['true_selected_error_mean'] for c in ['analytic_exact_negative_control','context_dependent_response_shift']]
 ax.bar(np.arange(2)+(j-1.5)*.18,vals,.18,label=label,color=colors[j])
ax.set_xticks([0,1],['Analytic-exact control','Context-dependent shift']);ax.set_ylabel('Post-action squared feature error');ax.set_title('A: 128 held-out synthetic worlds');ax.legend(fontsize=9)
ax.text(.02,.02,'Online calibration uses 4 additional probes.\nThese bars show error, before probe costs.',transform=ax.transAxes,fontsize=9,va='bottom',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
ax=axs[0,1]; ms=['learned_greedy','learned_open_loop','learned_B','learned_no_dird_proxy']
vals=[b['summary'][m]['true_expected_taskloss_plus_cost'] for m in ms]
bars=ax.bar(np.arange(4),vals,color=['#67788a','#8c9baa','#2e78b7','#5c956c'])
ax.set_xticks(range(4),['A greedy','A fixed sequence','B contingent','B without proxy']);ax.set_ylim(0,11);ax.bar_label(bars,fmt='%.3f',padding=3)
ax.set_ylabel('Expected modeled loss + cost (lower is better)');ax.set_title('Learned A → B: 72 finite scenarios')
ax.text(.02,.03,'Actual DMP + robot kinematics; synthetic observations.\nLearned B wrong-identity probability: 20.56%.',transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
ax=axs[1,0]
for j,(name,label) in enumerate([('broad_uniform:learned','Broad uniform prior'),('source_fitted:learned','Source-fitted prior')]):
 vals=[p['summary'][reg+':B'][name]['true_expected_taskloss_plus_cost'] for reg in ['source','shifted']]
 bars=ax.bar(np.arange(2)+(j-.5)*.3,vals,.3,color=colors[j],label=label);ax.bar_label(bars,fmt='%.2f',padding=3)
ax.set_xticks([0,1],['Source population','Shifted population']);ax.set_ylim(0,17);ax.set_ylabel('Expected modeled loss + cost');ax.set_title('Prior × response: fitting can hurt under shift');ax.legend(fontsize=9,loc='upper left')
ax=axs[1,1]
for j,(key,label) in enumerate([('proxy_pass','Positive reachability proxy'),('exact_sampled_pass','Sampled joint-path checks pass')]):
 vals=[d['methods'][m][key] for m in ['initial','no_dird','joint_guided']]
 bars=ax.bar(np.arange(3)+(j-.5)*.3,vals,.3,color=['#e29734','#2e78b7'][j],label=label);ax.bar_label(bars,padding=3)
ax.set_xticks([0,1,2],['Initial','Optimize, no dIRD','Joint guidance']);ax.set_ylim(0,7.5);ax.set_yticks(range(7));ax.set_ylabel('Paths passing / 6 proposals');ax.set_title('Trajectory ablation: proxy success ≠ executable path');ax.legend(fontsize=9,loc='upper left')
fig.suptitle('Offline evidence only — no physical ultrasound acquisition',fontsize=17,fontweight='bold')
fig.savefig(ROOT/'results/evidence_summary.png',dpi=170)
fig.savefig(ROOT/'results/evidence_summary.pdf')
