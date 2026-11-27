"""History-specific, real DMP + actual dIRD/exact finite action graph."""
from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent.parent

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
 motion=module('core_motion',ROOT/'motion/adapter.py'); cap=module('core_cap',ROOT/'dird/adapter.py'); bridge=cap.CapabilityBridge()
 nominal=json.loads((ROOT/'dird/nominal_context.json').read_text()); T0=np.array(nominal['T_tcp_railbase']); q0=np.array(nominal['q_seed'])
 # Fixed offsets in robot rail_base, chosen geometrically before query outcomes.
 inspections=np.array([[-.025,-.035,0.],[0.,0.,0.],[.025,.035,0.]])
 targets=np.array([[-.03,-.035,0.],[-.03,.035,0.],[.03,-.035,0.],[.03,.035,0.]])
 config={'inspection_offsets_m':inspections.tolist(),'target_offsets_m':targets.tolist(),'placement_y_m':[0.,.04,-.04],'n_waypoints':21,'fixed_rail_m':float(q0[0]),'normal':'nominal TCP local z','terminal_loss':20.,'movement_speed_m_s':.05,'observation_cost':.4,'proxy_cost_coefficient':.05,'max_depth':2,'exact_max_joint_step_deg':12.,'schema':'finite_history_graph_v1'}
 (ROOT/'core/graph_config.json').write_text(json.dumps(config,indent=2))
 out={'config':config,'capability_manifest':bridge.manifest(),'primitive_source':str(ROOT/'motion/adapter.py'),'contexts':{},'started':time.time()}; start=time.perf_counter(); count=0
 for ci,dy in enumerate(config['placement_y_m']):
  context={'placement_translation_m':[0.,dy,0.],'nodes':{},'edges':{},'queries':0,'failures':0,'virtual_seed_nodes':0}; shift=np.array([0.,dy,0.])
  states={'root':(T0.copy(),q0.copy(),True)}
  for depth in range(3):
   keys=[k for k in states if (0 if k=='root' else len(k.split('.'))-1)==depth] # root=0, i0=1
   if depth: keys=[k for k in states if k!='root' and len(k.split('.'))==depth]
   for key in keys:
    Tseed,qseed,reachable=states[key]; context['nodes'][key]={'pose_exit':Tseed.tolist(),'q_exit':qseed.tolist(),'reachable_under_exact':reachable,'depth':depth,'inspect':{},'terminal':{}}
    for kind,offsets in [('terminal',targets)]+([('inspect',inspections)] if depth<2 else []):
     for ai,offset in enumerate(offsets):
      goal=T0[:3,3]+shift+offset
      row=motion.generate_candidates(Tseed[:3,3],goal,normal_xyz=T0[:3,2],rotation=T0[:3,:3],n=21,lateral_offsets_m=(0.,),methods=('dmp',),duration_s=max(.2,float(np.linalg.norm(goal-Tseed[:3,3]))/.05))[0]
      eid=f'{key}/{kind}{ai}'; qry=bridge.query_path(row['T_tcp'],np.full(21,q0[0]),qseed,exact=True,provenance=f'core graph context={ci} edge={eid}; actual DMP adapter')
      valid=bool(qry['exact']['validated']); exq=qry['exact'].get('q_ref'); exitq=np.asarray(exq[-1],dtype=float) if exq is not None and len(exq) and np.isfinite(np.asarray(exq,dtype=float)).all() else qseed.copy(); exitT=np.array(row['T_tcp'][-1]); proxycost=.05*float(np.logaddexp(0.,-qry['min_raw_score'])); move=row['metrics']['path_length_m']/.05
      edge={'id':eid,'kind':kind,'action':ai,'exact_validated':valid,'source_node_reachable':reachable,'next':None,'path_length_m':row['metrics']['path_length_m'],'movement_cost':move,'observation_cost':.4,'proxy_cost':proxycost,'cost':move+.4+proxycost,'cost_without_capability':move+.4,'T_tcp':np.asarray(row['T_tcp']).tolist(),'q_seed':qseed.tolist(),'q_exit':exitq.tolist(),'pose_exit':exitT.tolist(),'primitive_metrics':row['metrics'],'capability':qry}
      if kind=='inspect':
       nxt=f'i{ai}' if key=='root' else f'{key}.i{ai}'; edge['next']=nxt; states[nxt]=(exitT,exitq,reachable and valid)
       if not valid: context['virtual_seed_nodes']+=1
      context['edges'][eid]=edge; context['nodes'][key][kind][str(ai)]=eid; count+=1;context['queries']+=1;context['failures']+=not valid
   # Write progress with completed depth; rerun is explicit rather than silently mixed cache.
   out['contexts'][str(ci)]=context; out['elapsed_seconds']=time.perf_counter()-start
   (ROOT/'core/graph.json').write_text(json.dumps(out,indent=2,allow_nan=False)); print(f'context {ci} depth {depth}: queries={context["queries"]} failures={context["failures"]}',flush=True)
 out['total_queries']=count;out['elapsed_seconds']=time.perf_counter()-start;out['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'core/build_graph.py',ROOT/'motion/adapter.py',ROOT/'dird/adapter.py']};(ROOT/'core/graph.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('complete',count,out['elapsed_seconds'],flush=True)
if __name__=='__main__': main()
