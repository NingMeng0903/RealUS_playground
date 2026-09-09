"""Replay a shared rest-field candidate and measure its changed soft material."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from ..consistent_runtime_v14 import load_compiled_subject
from ..shared_visceral_rest_fit_v16 import (
    _bone_mask, _edge_metrics, _mesh_info, _query_closed_surface,
    pair_collision_metrics_v16, skin_outside_metrics_v16,
)
from ..visceral_volume_v16 import _closed_oriented


def changed_material_bone_check(before, after, asset, changed):
    """Vertex-in-closed-bone audit; deliberately not a triangle crossing test."""
    names = list(asset.source_mesh_names)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=int)
    tissues = list(asset.source_tissues)
    owner = np.empty(len(before), dtype=int)
    for i, (s, e) in enumerate(ranges): owner[s:e] = i
    rows = []
    skipped = []
    for name, (s, e), tissue in zip(names, ranges, tissues):
        if str(tissue).strip().lower() != 'bone': continue
        bone = before[s:e]
        lo, hi = bone.min(axis=0)-.001, bone.max(axis=0)+.001
        overlap = (((before >= lo).all(axis=1) & (before <= hi).all(axis=1)) |
                   ((after >= lo).all(axis=1) & (after <= hi).all(axis=1))) & changed
        ids = np.flatnonzero(overlap)
        if not len(ids): continue
        _, _, faces = _mesh_info(asset, name)
        try: _closed_oriented(faces)
        except ValueError:
            skipped.append(str(name)); continue
        bi, bd, _, _ = _query_closed_surface(before[ids], bone, faces)
        ai, ad, _, _ = _query_closed_surface(after[ids], bone, faces)
        bd = np.where(bi, bd, 0); ad = np.where(ai, ad, 0)
        for own in np.unique(owner[ids]):
            selected = owner[ids] == own
            b, a = bd[selected], ad[selected]
            if max(b.max(initial=0), a.max(initial=0)) <= .0005: continue
            rows.append(dict(soft_mesh=str(names[own]), bone_mesh=str(name),
                before_max_mm=float(b.max(initial=0)*1000), after_max_mm=float(a.max(initial=0)*1000),
                before_over_05mm=int((b>.0005).sum()), after_over_05mm=int((a>.0005).sum()),
                vertices_worsened_over_01mm=int(((a-b)>.0001).sum())))
    return dict(rows=rows, skipped_open_or_inconsistent_bones=skipped,
        query='changed soft vertices inside closed bones', triangle_crossings_tested=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    before=load_compiled_subject(args.before); candidate=load_compiled_subject(args.candidate)
    for attr in ('betas','indices','weights','target_bind','reference_bind','translation_maps','rotation_maps'):
        if not np.array_equal(getattr(before,attr),getattr(candidate,attr)):
            raise ValueError('a rest-only comparison changed '+attr)
    asset=before.source_asset
    for attr in ('faces','source_mesh_names','source_vertex_ranges','source_tissues','source_bone_parents'):
        if not np.array_equal(getattr(asset,attr),getattr(candidate.source_asset,attr)):
            raise ValueError('source identity changed: '+attr)
    for attr in ('operator_runtime_digest','reference_digest','cache_key'):
        if getattr(before.source_pack,attr)!=getattr(candidate.source_pack,attr):
            raise ValueError('source package identity changed: '+attr)
    # A rest-only candidate must also retain the serialized motion response.
    for filename in ('motion_response.npz','pose_corrector.npz'):
        bpath, cpath = args.before/filename, args.candidate/filename
        if bpath.exists()!=cpath.exists(): raise ValueError('motion artifact changed: '+filename)
        if bpath.exists() and hashlib.sha256(bpath.read_bytes()).digest()!=hashlib.sha256(cpath.read_bytes()).digest():
            raise ValueError('motion artifact changed: '+filename)
    bone=_bone_mask(asset,len(before.target_rest))
    changed=np.linalg.norm(before.target_rest-candidate.target_rest,axis=1)>1e-10
    if np.any(changed & bone): raise ValueError('bone rest changed')
    args.output.mkdir(parents=True); geometry=args.output/'geometry';geometry.mkdir()
    report=dict(anatomical_passed=False,publishable=False,fit_run=False,
        before=str(args.before.resolve()),candidate=str(args.candidate.resolve()),
        changed_material_vertices=int(changed.sum()), expected_scene_count=len(args.input),
        completed=False, scenes=[])
    pairs=(('Diaphragm','L2'),('Liver','Rib_10R'),('Diaphragm','Rib_9R'),
           ('Large_Intestine','Rib_10R'),('Large_Intestine','Sacrum'))
    for path in args.input:
        with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
        b=before.apply_pose(d['pose'],d['transl']); c=candidate.apply_pose(d['pose'],d['transl'])
        if np.max(np.abs(b-d['before_vertices']))>1e-7: raise ValueError('baseline geometry mismatch')
        if not np.array_equal(b[bone],c[bone]): raise ValueError('posed bone changed')
        row=dict(input=str(path),bones_bit_exact=True,
            before_pairs=pair_collision_metrics_v16(b,asset,pairs),
            after_pairs=pair_collision_metrics_v16(c,asset,pairs),
            before_skin=skin_outside_metrics_v16(b,asset,d['skin_vertices'],d['skin_faces']),
            after_skin=skin_outside_metrics_v16(c,asset,d['skin_vertices'],d['skin_faces']),
            edges=_edge_metrics(b,c,asset,bone),
            changed_material_to_bones=changed_material_bone_check(b,c,asset,changed))
        d.update(before_vertices=b,candidate_vertices=c,source_vertices=b)
        np.savez_compressed(geometry/path.name,**d)
        report['scenes'].append(row)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(path.stem, 'evaluated; bones exact, changed-material/bone rows',
              len(row['changed_material_to_bones']['rows']),flush=True)
    report['completed']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
