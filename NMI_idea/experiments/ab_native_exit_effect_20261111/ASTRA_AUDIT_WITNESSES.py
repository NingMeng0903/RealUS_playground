from pathlib import Path
import json,sys,hashlib,math
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;S=H.parent/'ab_continuing_rollout_20261111';sys.path.insert(0,str(S))
from native_adapter import _state_fingerprint,_state_from_payload
failures=[];late=[];retry=[]
def maxima(directory,route):
 ps=[];rs=[]
 for step in route['steps']:
  with np.load(directory/f"{step['record']}.npz",allow_pickle=False) as t:
   before=np.concatenate((np.asarray(step['entry_pose'])[None],t['actual'][:-1]),axis=0)
   pe=float(np.linalg.norm(before[:,:3,3]-t['desired'][:,:3,3],axis=1).max())
   re=float(Rotation.from_matrix(np.swapaxes(before[:,:3,:3],1,2)@t['desired'][:,:3,:3]).magnitude().max())
   if abs(pe-step['same_tick_position_error_max_m'])>1e-12:failures.append(str(directory)+':position_report')
   if abs(re-step['same_tick_rotation_error_max_rad'])>1e-12:failures.append(str(directory)+':rotation_report')
   ps.append(pe);rs.append(re)
 return max(ps),max(rs)
for p in sorted(H.glob('fixture_*/certified_retries/RETRY_VALIDATION.json')):
 d=json.loads(p.read_text());forecast=json.loads((H/Path(d['optimistic_forecasts'])).read_text());motion=json.loads((p.parent/'execution/MOTION.json').read_text());checks=[]
 cps=[json.loads(q.read_text()) for q in (p.parent/'planning/checkpoints').glob('*.json')]
 for cand in d['candidates']:
  for score in cand['scores']:
   rep=next(r for r in forecast['replays'] if r['branches']==score['branches']);values={r['action_id']:r['continuation']['actual']['value'] for r in rep['candidate_scores']};chosen=cand['probe'];margin=values[chosen]-max(v for a,v in values.items() if a!=chosen)
   if abs(margin-score['upper_bound_margin'])>1e-12 or margin<=0:failures.append(str(p)+':strict_margin')
   if not all(cand['retry_replays'][t]['verified'] for t in score['required']):failures.append(str(p)+':unverified_required')
   checks.append({'branches':score['branches'],'strict_margin':margin,'choice':chosen,'A_task':score['A_task'],'A_info':score['A_info'],'B_nominal':score['B_nominal']})
  for target,r in cand['retry_replays'].items():
   cp=next((c for c in cps if c['action_id']==target and c['input_fingerprint']==r['exact_input']),None)
   if cp is None:failures.append(str(p)+':retry_checkpoint_missing');continue
   state=_state_from_payload(cp['input_state']);out=_state_from_payload(cp['output_state'])
   if _state_fingerprint(state)!=r['exact_input'] or _state_fingerprint(out)!=r['actual_output'] or not out.valid:failures.append(str(p)+':retry_fullstate')
   route=next(x for x in motion['routes'] if x['route']==r['route']);pe,re=maxima(p.parent/'execution',route)
   if not route['valid'] or pe>.002 or re>math.radians(2):failures.append(str(p)+':retry_gate')
 retry.append({'file':str(p),'checks':checks,'retry_links':len(cand['retry_replays'])})
for p in sorted(H.glob('late_native_witness/after_*/RESULT.json')):
 d=json.loads(p.read_text());archive=json.loads(Path(d['source']).read_text());state=_state_from_payload(archive['input_state'])
 if not state.valid or _state_fingerprint(state)!=d['entry_fingerprint']:failures.append(str(p)+':late_input')
 motion=json.loads((p.parent/'execution/MOTION.json').read_text());attempts=[]
 for route in motion['routes']:
  pe,re=maxima(p.parent/'execution',route);attempts.append({'route':route['candidate'],'valid':route['valid'],'position_max_mm':pe*1000,'rotation_max_deg':math.degrees(re)})
  if route['valid'] and (pe>.002 or re>math.radians(2)):failures.append(str(p)+':late_pass_gate')
 if d['passed']!=any(a['valid'] for a in attempts):failures.append(str(p)+':late_event')
 late.append({'prefix':d['prefix'],'passed':d['passed'],'tcp_delta_mm':d['tcp_delta_mm'],'attempts':attempts,'source_sha256':hashlib.sha256(Path(d['source']).read_bytes()).hexdigest()})
dwell=[]
for p in sorted(H.glob('dwell_sensitivity/*/RESULT.json')):
 d=json.loads(p.read_text());motion=json.loads((p.parent/'execution/MOTION.json').read_text());attempts=[]
 state=d['root']
 for link in d['links']:
  if link['state_in']!=state:failures.append(str(p)+':dwell_chain')
  state=link['state_out']
 for route in motion['routes']:
  pe,re=maxima(p.parent/'execution',route);attempts.append({'capture':route['capture'],'route':route['candidate'],'valid':route['valid'],'position_max_mm':pe*1000})
  if route['valid'] and (pe>.002 or re>math.radians(2)):failures.append(str(p)+':dwell_gate')
 if d['complete'] and abs(sum(x['elapsed_s'] for x in d['links'])-d['motion_s'])>1e-12:failures.append(str(p)+':dwell_duration')
 dwell.append({'arm':d['arm'],'root':d['root'],'complete':d['complete'],'motion_s':d['motion_s'],'attempts':attempts})
if len({d['root'] for d in dwell})!=1:failures.append('dwell_not_matched_entry')
result={'dwell_regression':dwell,'retry_certification':retry,'late_retrospective_witness':late,'failures':failures,'passed':not failures,'scope':'independent stored-score ordering and native retry/late trace checks; no independent posterior quadrature recomputation; late pair is retrospective'}
(H/'ASTRA_WITNESS_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
