"""Public-belief two-observation planning on an exact-query restricted graph.
No realized hidden state enters any planning function. Evaluation alone receives h.
"""
from pathlib import Path
from itertools import product
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
LOSS=20.; SPEED=.02
COUNTERS={}
def count(key):COUNTERS[key]=COUNTERS.get(key,0)+1

def sensor():
 # Outcomes: no detection, marker 0, marker 1; persistent latent h=0..3.
 bits=np.array([[0,0,1,1],[0,1,0,1],[0,1,1,0]])
 L=np.zeros((3,4,3))
 for a,(nd,correct,flip) in enumerate([(.15,.8,.05),(.2,.75,.05),(.1,.82,.08)]):
  L[a,:,0]=nd
  for h in range(4): L[a,h,1+bits[a,h]]=correct;L[a,h,2-bits[a,h]]=flip
 assert np.allclose(L.sum(2),1.)
 return L

def beliefs():
 r=np.random.default_rng(41106); rows=[np.ones(4)/4]
 for h in range(4):
  b=np.full(4,.05);b[h]=.85;rows.append(b)
 rows.extend(r.dirichlet(np.full(4,.6),19))
 assert len(rows)==24 and np.min(rows)>0
 return rows

def branches(b,L,a):
 count('likelihood_branch_expansions')
 joint=b[:,None]*L[a]; p=joint.sum(0)
 return [(y,float(p[y]),joint[:,y]/p[y]) for y in range(3) if p[y]>0]

def ecost(e,mode='full',observation_extra=0.):
 count('cached_edge_cost_reads')
 return e['path_length_m']/SPEED+.4+observation_extra+(0. if mode in ('no_capability','no_proxy') else e['proxy_cost'])
def allowed(e,mode):
 count('cached_feasibility_reads')
 return mode=='no_capability' or e['exact_validated']
def terminal(g,node,b,mode='full',extra=0.):
 best=(LOSS,{'kind':'fail'})
 for hs,eid in g['nodes'][node]['terminal'].items():
  e=g['edges'][eid]
  if not allowed(e,mode):continue
  v=ecost(e,mode)+LOSS*(1-b[int(hs)])
  if v<best[0]-1e-12:best=(float(v),{'kind':'terminal','edge':eid,'target':int(hs)})
 return best

def solve(g,node,b,k,L,mode='full',extra=0.):
 best=terminal(g,node,b,mode,extra)
 if k:
  for ai,eid in g['nodes'][node]['inspect'].items():
   e=g['edges'][eid];a=int(ai)
   if not allowed(e,mode):continue
   sub={};v=ecost(e,mode,extra)
   for y,p,post in branches(b,L,a):
    vv,pol=solve(g,e['next'],post,k-1,L,mode,extra);v+=p*vv;sub[str(y)]=pol
   if v<best[0]-1e-12:best=(float(v),{'kind':'inspect','edge':eid,'action':a,'branches':sub})
 return best

def reference_joint(g,node,w,k,L,extra=0.):
 # Independent unnormalised joint-mass enumeration; minimize only after
 # worlds have been grouped by the same public observation y.
 mass=float(sum(w));best=(LOSS*mass,{'kind':'fail'})
 for hs,eid in g['nodes'][node]['terminal'].items():
  e=g['edges'][eid]
  if not e['exact_validated']:continue
  v=ecost(e)*mass+LOSS*(mass-w[int(hs)])
  if v<best[0]-1e-12:best=(float(v),{'kind':'terminal','edge':eid,'target':int(hs)})
 if k:
  for ai,eid in g['nodes'][node]['inspect'].items():
   e=g['edges'][eid];a=int(ai)
   if not e['exact_validated']:continue
   v=ecost(e,observation_extra=extra)*mass;sub={}
   for y in range(3):
    nextw=w*L[a,:,y]
    if sum(nextw)>0:
     vv,pp=reference_joint(g,e['next'],nextw,k-1,L,extra);v+=vv;sub[str(y)]=pp
   if v<best[0]-1e-12:best=(float(v),{'kind':'inspect','edge':eid,'action':a,'branches':sub})
 return best

def greedy(g,node,b,k,L,entropy=False):
 stop=terminal(g,node,b);best=stop;chosen=None
 if k:
  candidates=[]
  for ai,eid in g['nodes'][node]['inspect'].items():
   e=g['edges'][eid];a=int(ai)
   if not allowed(e,'full'):continue
   sub={};v=ecost(e);ig=0.
   H=lambda z:float(-np.sum(z[z>0]*np.log(z[z>0])))
   for y,p,post in branches(b,L,a):
    vv,pol=terminal(g,e['next'],post);v+=p*vv;ig+=p*(H(b)-H(post))
   candidates.append((a,eid,v,ig-ecost(e)/LOSS))
  if entropy and candidates:
   a,eid,v,utility=max(candidates,key=lambda x:x[3])
   if utility>0 and stop[0]>ecost(g['edges'][eid]):chosen=(a,eid)
  elif candidates:
   a,eid,v,utility=min(candidates,key=lambda x:x[2])
   if v<stop[0]-1e-12:chosen=(a,eid)
 if chosen:
  a,eid=chosen;e=g['edges'][eid];sub={};v=ecost(e)
  for y,p,post in branches(b,L,a):
   vv,pp=greedy(g,e['next'],post,k-1,L,entropy);v+=p*vv;sub[str(y)]=pp
  best=(float(v),{'kind':'inspect','edge':eid,'action':a,'branches':sub})
 return best

def fixed_sequence(g,node,b,seq,L):
 if not seq:return terminal(g,node,b)
 a=seq[0];eid=g['nodes'][node]['inspect'][str(a)];e=g['edges'][eid]
 if not e['exact_validated']:return float('inf'),{'kind':'fail'}
 v=ecost(e);sub={}
 for y,p,post in branches(b,L,a):
  vv,pp=fixed_sequence(g,e['next'],post,seq[1:],L);v+=p*vv;sub[str(y)]=pp
 return float(v),{'kind':'inspect','edge':eid,'action':a,'branches':sub}
def open_loop(g,b,L):
 best=terminal(g,'root',b)
 for k in [1,2]:
  for seq in product(range(3),repeat=k):
   candidate=fixed_sequence(g,'root',b,seq,L)
   if candidate[0]<best[0]-1e-12:best=candidate
 return best

def evaluate_hidden(g,pol,h,L):
 # [taskloss+allcost, false acceptance, distance, observations, invalid attempt]
 if pol['kind']=='fail':return np.array([LOSS,0.,0.,0.,0.])
 e=g['edges'][pol['edge']];v=np.array([ecost(e),0.,e['path_length_m'],1.,0.])
 if not e['exact_validated']:
  v[0]+=LOSS;v[4]=1.;return v
 if pol['kind']=='terminal':
  wrong=float(pol['target']!=h);v[0]+=LOSS*wrong;v[1]=wrong;return v
 for y,p in enumerate(L[pol['action'],h]):
  if p:v+=p*evaluate_hidden(g,pol['branches'][str(y)],h,L)
 return v

def evaluate(g,pol,b,L):return sum(b[h]*evaluate_hidden(g,pol,h,L) for h in range(4))
def signature(pol):
 if pol['kind']=='inspect':return [pol['edge'],{k:signature(v) for k,v in pol['branches'].items()}]
 return pol

def bootstrap(d):
 r=np.random.default_rng(41104);d=np.asarray(d);b=d[r.integers(0,len(d),(2000,len(d)))].mean(1)
 return {'mean':float(d.mean()),'ci95_descriptive_scenario_resampling':np.quantile(b,[.025,.975]).tolist(),'improved':int(np.sum(d< -1e-9)),'tied':int(np.sum(np.abs(d)<=1e-9)),'worse':int(np.sum(d>1e-9))}

def main():
 graph=json.loads((ROOT/'core/graph.json').read_text());L=sensor();bs=beliefs();rows=[];policies={};checks={'likelihood_normalized':bool(np.allclose(L.sum(2),1.)),'belief_positive_support':bool(np.min(bs)>0),'posterior_normalization_max_error':0.,'exact_reference_max_error':0.,'evaluation_max_error':0.,'hidden_noninterference':True,'uninformative_control_max_value_gain':0.,'high_observation_cost_control_max_gain':0.};controls=[]
 methods=['A_full_belief_greedy','A_same_horizon_open_loop','entropy_information','B_observation_contingent','B_without_capability','B_without_dird_proxy','B_collapsed_belief','full_history_exact_enumeration']
 for ci,g in graph['contexts'].items():
  for bi,b in enumerate(bs):
   record={'context':int(ci),'belief_id':bi,'belief':b.tolist(),'methods':{}};policies[f'{ci}:{bi}']={}
   for method in methods:
    COUNTERS.clear();started=time.perf_counter()
    if method=='A_full_belief_greedy':v,p=greedy(g,'root',b,2,L)
    elif method=='A_same_horizon_open_loop':v,p=open_loop(g,b,L)
    elif method=='entropy_information':v,p=greedy(g,'root',b,2,L,True)
    elif method=='B_without_capability':v,p=solve(g,'root',b,2,L,'no_capability')
    elif method=='B_without_dird_proxy':v,p=solve(g,'root',b,2,L,'no_proxy')
    elif method=='B_collapsed_belief':v,p=solve(g,'root',np.eye(4)[np.argmax(b)],2,L)
    elif method=='full_history_exact_enumeration':v,p=reference_joint(g,'root',b,2,L)
    else:v,p=solve(g,'root',b,2,L)
    wall=time.perf_counter()-started;work=dict(COUNTERS);metric=evaluate(g,p,b,L)
    record['methods'][method]={'planning_objective':v,'expected_taskloss_plus_cost':float(metric[0]),'false_acceptance_probability':float(metric[1]),'movement_distance_m':float(metric[2]),'acquired_observations_including_terminal':float(metric[3]),'infeasible_attempt_probability':float(metric[4]),'planner_wall_seconds':wall,'root_action':p.get('edge','fail'),'online_actual_capability_query_calls':0,'planning_cached_model_work':work}
    policies[f'{ci}:{bi}'][method]=p
    if method in ('A_full_belief_greedy','A_same_horizon_open_loop','entropy_information','B_observation_contingent','full_history_exact_enumeration'):checks['evaluation_max_error']=max(checks['evaluation_max_error'],abs(v-metric[0]))
   checks['exact_reference_max_error']=max(checks['exact_reference_max_error'],abs(record['methods']['B_observation_contingent']['planning_objective']-record['methods']['full_history_exact_enumeration']['planning_objective']))
   # Behavioral noninterference: compatible realized labels are evaluator-only;
   # reconstruct public belief from each same observed history and compare policy.
   for a in range(3):
    for y,prob,post in branches(b,L,a):
     checks['posterior_normalization_max_error']=max(checks['posterior_normalization_max_error'],abs(sum(post)-1))
     node=g['edges'][g['nodes']['root']['inspect'][str(a)]]['next'];expected=None
     for hidden in range(4):
      if L[a,hidden,y]>0:
       # hidden deliberately never passed to policy API.
       _,pp=solve(g,node,post,1,L);s=json.dumps(signature(pp),sort_keys=True)
       if expected is None:expected=s
       checks['hidden_noninterference'] &= s==expected
   # Controls use the same candidates/cost model, without retuning.
   if bi in (0,1,2,3,4):
    U=np.full_like(L,1/3);base,_=terminal(g,'root',b);vu,pu=solve(g,'root',b,2,U);vh,ph=solve(g,'root',b,2,L,extra=100.)
    # High-cost control raises only optional inspection observation costs;
    # terminal acquisition remains available at its original price.
    checks['uninformative_control_max_value_gain']=max(checks['uninformative_control_max_value_gain'],base-vu)
    checks['high_observation_cost_control_max_gain']=max(checks['high_observation_cost_control_max_gain'],base-vh)
    controls.append({'context':int(ci),'belief_id':bi,'uninformative_root':pu.get('edge','fail'),'uninformative_value':vu,'immediate_value':base,'highcost_root':ph.get('edge','fail'),'highcost_value':vh})
   rows.append(record)
 summary={m:{k:float(np.mean([r['methods'][m][k] for r in rows])) for k in ['expected_taskloss_plus_cost','false_acceptance_probability','movement_distance_m','acquired_observations_including_terminal','infeasible_attempt_probability','planner_wall_seconds']} for m in methods}
 paired={m:bootstrap([r['methods']['B_observation_contingent']['expected_taskloss_plus_cost']-r['methods'][m]['expected_taskloss_plus_cost'] for r in rows]) for m in methods if m!='B_observation_contingent'}
 for ci in graph['contexts']:
  subset=[r for r in rows if str(r['context'])==ci];summary[f'context_{ci}_paired_B_minus_open_loop']=bootstrap([r['methods']['B_observation_contingent']['expected_taskloss_plus_cost']-r['methods']['A_same_horizon_open_loop']['expected_taskloss_plus_cost'] for r in subset])
 # Evaluate unchanged policies under shifted truth distribution as robustness only.
 shift={m:[] for m in methods}
 for r in rows:
  g=graph['contexts'][str(r['context'])];truth=np.roll(np.array(r['belief']),1)
  for m in methods:shift[m].append(float(evaluate(g,policies[f'{r["context"]}:{r["belief_id"]}'][m],truth,L)[0]))
 result={'evidence':'SYNTHETIC MECHANISM + PRODUCTION KINEMATICS; no physical ultrasound','schema':'ab_B_results_v1','sensor_likelihood':L.tolist(),'sensor_outcomes':['no_detection','marker_0','marker_1'],'belief_seed':41106,'movement_speed_for_cost_m_s':SPEED,'loss_failure_or_wrong_identity':LOSS,'observation_budget':2,'terminal_observation_extra':1,'scope':'two inspection observations plus one synthetic terminal acquisition; restricted history graph, no continuous control','timing_model':'offline shared graph setup then asynchronous table planning; wall times separately reported','shared_setup_actual_capability_queries':graph['total_queries'],'shared_setup_seconds':graph['elapsed_seconds'],'shared_setup_exact_failures':sum(g['failures'] for g in graph['contexts'].values()),'shared_setup_reachable_edges':sum(e['source_node_reachable'] for g in graph['contexts'].values() for e in g['edges'].values()),'summary':summary,'paired_B_minus_baseline':paired,'checks':checks,'controls':controls,'source_prior_shift_robustness_mean_loss':{m:float(np.mean(v)) for m,v in shift.items()},'rows':rows,'limitations':['72 public scenarios are a designed finite set, not random patients; bootstrap is descriptive only.','No-capability ablation plans virtual descendants after infeasible edges; actual evaluator aborts at first infeasible attempt and charges failure. Such descendants have no claimed physical execution.','Exact feasibility uses sampled kinematics/self-collision only; no velocity/contact/environment certificate.','B optimality versus exact full-history reference is correctness, not novel general POMDP superiority.','No learned A response model is used to fabricate the B sensor; B uses a separately declared exact abstract likelihood.'],'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'core/b_planner.py',ROOT/'core/graph.json',ROOT/'CORE_PROTOCOL.json']}}
 (ROOT/'core/b_policies.json').write_text(json.dumps(policies,indent=2,allow_nan=False));(ROOT/'core/b_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'paired':paired,'checks':checks},indent=2))
 assert checks['exact_reference_max_error']<1e-9 and checks['evaluation_max_error']<1e-9 and checks['hidden_noninterference']
 assert checks['uninformative_control_max_value_gain']<1e-9 and checks['high_observation_cost_control_max_gain']<1e-9
if __name__=='__main__':main()
