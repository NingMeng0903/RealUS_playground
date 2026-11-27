"""Run a target-aware acquisition teacher and retain the coordinates of its actual echoes.

The teacher can verify past sensor echoes against simulator identity. The public readout
receives identical trajectory observations without the labels or the certification bits.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import numpy as np

from core import advance, band, get_world, grade, scan, terminal, visit
from experiments import OLD, SEEDS
from fan import proposals, visit_path
from observers import reader
from recovery import entry_trip
from verification import pulse_path
from anchor import reserve

PREFERRED_CONFIDENCE = .65


def true_observation(frames, public_probs, indices):
    """Keep an actual, separately resolved target echo with the best public score."""
    best = None
    for index in indices:
        origins = frames[index][3]
        for echo in np.flatnonzero(origins == 0):
            value = float(public_probs[index][0, echo])
            if best is None or value > best[2]:
                best = (index, int(echo), value)
    return best


def useful(choice):
    return choice is not None and choice[2] >= PREFERRED_CONFIDENCE


def exact_score(frames, choice, indices):
    if choice is None:
        return grade(frames, None, indices)
    index, echo, public_score = choice
    assert index in indices and int(frames[index][3][echo]) == 0
    return grade(frames, (index, echo, 1.), indices)


def summary(entry, fresh, bands):
    return dict(entry=entry, fresh=fresh, both=int(entry['correct'] and fresh['correct']),
                chain=int(all(p['correct'] for p in bands)),
                prefix_mm=5*next((i for i,p in enumerate(bands) if not p['correct']),3),
                wrong_entry=entry['wrong'], wrong_fresh=fresh['wrong'], bands=bands)


def balanced_retry(env, spec, node, trial, limits, phase, allowance, distance):
    path = visit_path(env,spec[1],spec[2],int(node.station),int(node.sector),phase)
    if path is None:
        return 0
    poses, pieces = path
    travel = 2 * len(pieces)
    if distance + 1.e-6 < 2*pieces.sum() or allowance <= travel:
        return 0
    dwell = min(4, allowance-travel)
    for pose,length in zip(poses[1:],pieces):
        node.tick += 1
        node.distance += float(length)
        env.scanner._observe(node,spec,trial,float(pose[0]),float(pose[1]),node.tick)
    for _ in range(dwell):
        node.tick += 1
        env.scanner._observe(node,spec,trial,float(poses[-1,0]),float(poses[-1,1]),node.tick)
    for pose,length in zip(poses[-2::-1],pieces[::-1]):
        node.tick += 1
        node.distance += float(length)
        env.scanner._observe(node,spec,trial,float(pose[0]),float(pose[1]),node.tick)
    return dwell+travel


def scan_request(env, spec, trial, lim, item, balanced=False, return_frames=False):
    _, entry, original, prefix_history, *_ = item
    n = original.fork()
    h = prefix_history.fork()
    frames = n.reader.frames
    entry_indices = [i for i,f in enumerate(frames) if entry-3.-1.e-8 <= f[0] <= entry+2.+1.e-8]
    selected_entry = true_observation(frames, h.online, entry_indices)
    count = 0
    entry_slack,_ = reserve(n,spec,lim)
    if not useful(selected_entry) and (not balanced or selected_entry is None or entry_slack>=90):
        visits = entry_trip(env,spec,n,h,lim,entry,10)
        if visits:
            preferred = max(visits, key=lambda trip:trip.score)
            start = len(frames)
            visit(env,spec,n,trial,preferred,True)
            h.append(frames)
            entry_indices.extend(i for i in range(start,len(frames)) if entry-3.-1.e-8<=frames[i][0]<=entry+2.+1.e-8)
            selected_entry = true_observation(frames,h.online,entry_indices)
            count = 1
    cut = len(frames)
    real_bands = []
    accepted_bands = []
    native_bands = []
    choices = []
    per_band = []
    for stage in range(3):
        start = len(frames)
        slack,_ = reserve(n,spec,lim)
        clock = max(3,min(9,int(slack/(3-stage))+1))
        public_field = reader(env).estimate(frames,anatomy=True).centers
        _, options = proposals(env,spec,n,h,lim,clock,trial.layout.centers,public_field)
        goal = max(options, key=lambda view:.4*view.target+.6*view.separation+.1*view.difference)
        advance(env,spec,n,trial,lim,goal.slot,None)
        h.append(frames)
        selected = true_observation(frames,h.online,range(start,len(frames)))
        moves = []
        # A missing downstream interval takes priority over polishing an already acquired image.
        # Allocate surplus only after reserving the baseline path to the original endpoint.
        remaining, spare_mm = reserve(n,spec,lim)
        allotted = max(0,int(remaining/(3-stage)))
        available_mm = max(0.,spare_mm/(3-stage))
        initial_tick, initial_mm = n.tick, n.distance
        phases = (0.,-.70,.70,-1.15,1.15)
        if balanced:
            phases = (0.,-.70,.70,0.,-1.15,1.15,0.,0.,0.,0.)
        for angle in phases:
            if useful(selected):
                break
            if balanced:
                allowance = allotted - (n.tick-initial_tick)
                mm_left = available_mm - (n.distance-initial_mm)
                if allowance <= 0 or (selected is not None and len(moves)>=1 and stage<2):
                    break
                acquired = balanced_retry(env,spec,n,trial,lim,angle,allowance,mm_left)
            else:
                acquired = pulse_path(env,spec,n,trial,lim,angle)
            if acquired:
                h.append(frames)
                selected = true_observation(frames,h.online,range(start,len(frames)))
                moves.append(angle)
        ids = range(start,len(frames))
        real_bands.append(exact_score(frames,selected,ids))
        accepted_bands.append(grade(frames,selected,ids))
        native_bands.append(band(frames,h,start))
        choices.append(selected)
        per_band.append(dict(slot=goal.slot, retries=moves, span=[start,len(frames)]))
    fresh_indices = range(cut,len(frames))
    selected_fresh = true_observation(frames,h.online,fresh_indices)
    result = {}
    for name, decide, bands in [('oracle',exact_score,real_bands),('oracle_public_floor',grade,accepted_bands)]:
        result[name] = summary(decide(frames,selected_entry,entry_indices),decide(frames,selected_fresh,fresh_indices),bands)
    result['anonymous'] = terminal(n,h,entry,cut,native_bands)
    result['images'] = dict(entry=selected_entry,fresh=selected_fresh,bands=choices)
    result['trajectory'] = dict(initial_revisit=count,bands=per_band,ticks=int(n.tick-original.tick),mm=n.distance-original.distance)
    if n.station!=original.station+3 or n.tick>lim.ticks or n.distance>lim.mm+1.e-6:
        raise RuntimeError('Acquisition exceeded the motion envelope')
    if return_frames:
        return result, frames, cut, [tuple(row['span']) for row in per_band]
    return result


def run(job):
    case,folder,split,balanced=job
    previous=json.loads((OLD/f'{case.world_seed}.json').read_text()) if split=='pilot' else None
    env,spec,t,lim,_,items=get_world(case,previous)
    row=dict(world=case.world_seed,layout=hashlib.sha256(np.nan_to_num(t.layout.centers).tobytes()+bytes([case.shape])).hexdigest(),
             profile=case.profile,requests={})
    for item in items:
        row['requests'][item[0]]=scan_request(env,spec,t,lim,item,balanced)
    (folder/f'{case.world_seed}.json').write_text(json.dumps(row,separators=(',',':')))
    return case.world_seed


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('split',choices=('pilot','test','validation'))
    p.add_argument('size',type=int)
    p.add_argument('folder',type=Path)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--balanced',action='store_true')
    a=p.parse_args();a.folder.mkdir(parents=True,exist_ok=True)
    cases=scan.cases('test',a.size,815601941 if a.split=='validation' else SEEDS[a.split])
    with ProcessPoolExecutor(a.workers) as pool:
        for j,_ in enumerate(pool.map(run,[(c,a.folder,a.split,a.balanced) for c in cases])):
            if j%16==0: print(j+1,a.size,flush=True)
