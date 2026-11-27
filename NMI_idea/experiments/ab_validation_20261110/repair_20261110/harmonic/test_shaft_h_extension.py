from pathlib import Path
from dataclasses import replace
import sys,json,time,numpy as np
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');OUT=Path(__file__).resolve().parent;TARGET=OUT/'shaft_h_extension_trials';TARGET.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import build_shaft_roi
plan=dict(h_min_candidates=[.10,.12,.13,.14],h_max_candidates=[.94,.95,.96,.98],station_count=72,theta_count=108,radial_count=3,inner_frac=.04,selection='largest interval passing every listed gate on both legs; exploratory only until final resolution/pose audits')
(TARGET/'frozen_plan.json').write_text(json.dumps(plan,indent=2));sources={};weights={}
for side in ['right','left']:
 base=load_leg_volume_atlas(OUT/f'rebuild_work/owned_base/atlas_{side}.npz');old=load_leg_volume_atlas(OUT/f'ownership_smooth/owned_smooth_{side}_skin.npz');sources[side]=replace(old,core_points=base.core_points,core_h=base.core_h);weights[side]=np.load(OUT/f'ownership_smooth/{side}_surface_lbs_weights.npz')['weights']
rows=[]
for low in plan['h_min_candidates']:
 for high in plan['h_max_candidates']:
  for side in ['right','left']:
   started=time.perf_counter();row=dict(side=side,h_min=low,h_max=high)
   try:
    _,_,metrics=build_shaft_roi(sources[side],weights[side],h_min=low,h_max=high,station_count=72,theta_count=108,radial_count=3);row.update(passed=True,metrics=metrics)
   except ValueError as e:row.update(passed=False,reason=str(e))
   row['wall_s']=time.perf_counter()-started;rows.append(row);(TARGET/'grid_results.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
passed=[]
for low in plan['h_min_candidates']:
 for high in plan['h_max_candidates']:
  if all(next(r for r in rows if r['side']==side and r['h_min']==low and r['h_max']==high)['passed'] for side in ['right','left']):passed.append((high-low,low,high))
selected=max(passed) if passed else None;(TARGET/'selected.json').write_text(json.dumps(dict(largest_both_leg_interval=selected,scope='exploratory geometry/graph gates only; no pose/publication acceptance'),indent=2));print('selected',selected,flush=True)
