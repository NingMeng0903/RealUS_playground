"""Frozen21 root-edge actual UltraPOC QP technical replay, not a B graph."""
from pathlib import Path
import argparse,hashlib,json,sys,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'dird'))
from qp_replay import QPReplayBridge
from common import clean_json
BRIDGE=None

def init(backend):
 global BRIDGE
 BRIDGE=QPReplayBridge(backend=backend,tcp_mode='urdf')
def gate(result,position=.002,rotation_deg=2.):
 trace=result.get('trace',[])
 if not trace:return False,{'missing_trace':True}
 rotation=max(float(r['rotation_error_rad']) for r in trace)
 finite=all(np.isfinite(np.array(r['q'])).all() and np.isfinite(np.array(r['qdot'])).all() for r in trace)
 flags={'no_failures':not result['failures'],'finite_state':finite,'no_latched_fault':not any(r['solver_fault_latched'] for r in trace),'tracking_position':result['position_error_max_m']<=position,'tracking_rotation':rotation<=np.deg2rad(rotation_deg),'joint_limits':min(result['joint_limit_min_margin_by_coordinate'])>=-1e-8,'self_collision':result['minimum_self_collision_distance_m']>=.01,'velocity':result['velocity_margin_min']>=-1e-5,'acceleration':result['acceleration_margin_min']>=-1e-4}
 return all(flags.values()),flags

def job(task):
 ci,eid,T,q=task
 out=BRIDGE.query_pose_path(T,q,speed_m_s=.02,settle_s=2.,provenance=f'fixed original core graph context={ci} {eid}; root technical test',return_trace=True)
 p,flags=gate(out);s,sflags=gate(out,.005,5.)
 trace=out.pop('trace',[])
 if trace:
  dest=HERE/'traces';dest.mkdir(exist_ok=True)
  np.savez_compressed(dest/f'c{ci}_{eid.replace("/","_")}.npz',**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
  out['rotation_error_max_rad']=max(float(r['rotation_error_rad']) for r in trace)
  out['qdot_exit']=trace[-1]['qdot']
 out.update(context=ci,edge=eid,primary_gate_pass=p,primary_flags=flags,sensitivity_gate_pass=s,sensitivity_flags=sflags)
 return clean_json(out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['native','python'],default='native');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 graph=json.loads((BASE/'core/graph.json').read_text());tasks=[]
 for ci,g in graph['contexts'].items():
  for eid in g['nodes']['root']['terminal'].values():
   e=g['edges'][eid];tasks.append((int(ci),eid,e['T_tcp'],g['nodes']['root']['q_exit']))
  for eid in g['nodes']['root']['inspect'].values():
   e=g['edges'][eid];tasks.append((int(ci),eid,e['T_tcp'],g['nodes']['root']['q_exit']))
 assert len(tasks)==21
 (HERE/'root_inputs.json').write_text(json.dumps(tasks,indent=2))
 manifest=QPReplayBridge(backend=a.backend,tcp_mode='urdf').manifest()
 (HERE/'qp_manifest.json').write_text(json.dumps(clean_json(manifest),indent=2))
 started=time.perf_counter();rows=[]
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.backend,),mp_context=multiprocessing.get_context('spawn')) as pool:
  pending=[pool.submit(job,t) for t in tasks]
  for future in as_completed(pending):
   out=future.result();rows.append(out);rows.sort(key=lambda r:(r['context'],r['edge']))
   result={'schema':'fixed_root_qp_replay_v1','continuous_B_graph_verified':False,'scope':'actual production motion QP, ideal measured plant; root edge technical test only','backend':a.backend,'tcp_mode':'urdf','workers':a.workers,'completed':len(rows),'expected':21,'elapsed_wall_seconds':time.perf_counter()-started,'primary_pass':sum(r['primary_gate_pass'] for r in rows),'sensitivity_pass':sum(r['sensitivity_gate_pass'] for r in rows),'rows':rows,'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.txt').read_bytes()).hexdigest(),'frozen_graph_sha256':hashlib.sha256((BASE/'core/graph.json').read_bytes()).hexdigest()}
   (HERE/'root_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(out['context'],out['edge'],out['primary_gate_pass'],out.get('position_error_max_m'),out.get('endpoint_position_error_m'),out.get('wall_seconds'),flush=True)
 print('complete',result['primary_pass'],result['sensitivity_pass'],result['elapsed_wall_seconds'],flush=True)
if __name__=='__main__':main()
