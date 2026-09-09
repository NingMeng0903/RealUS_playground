"""Compile and inspect the bounded shared visceral response experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..chain_containment_v1 import _signed_distance
from ..consistent_runtime_v14 import load_compiled_subject
from ..soft_constraints import unique_mesh_edges
from ..visceral_volume_v16 import (
    VisceralRuntimeV16, compile_shared_field, load_visceral_runtime_v16, mesh_volume,
)


def evaluate(runtime, input_path, output):
    with np.load(input_path, allow_pickle=False) as data:
        d = {k: data[k] for k in data.files}
    before = runtime.base.apply_pose(d['pose'], d['transl'])
    candidate, response = runtime.apply_pose(d['pose'], d['transl'], return_report=True)
    # The supplied baseline must be an exact latest-package replay, not merely
    # a similarly named historical geometry archive.
    mismatch = float(np.max(np.abs(before - d['candidate_vertices'])))
    if mismatch > 1e-7:
        raise ValueError(f'baseline geometry does not match loaded base: {mismatch}')
    names = list(d['mesh_names'])
    ranges = d['mesh_ranges']
    tissues = d['mesh_tissues']
    faces = d['faces']
    field_active = np.linalg.norm(runtime.field, axis=1) > 0
    moved_meshes = []
    for name, (s, e), tissue in zip(names, ranges, tissues):
        own = faces[(faces[:, 0] >= s) & (faces[:, 0] < e)] - s
        if not field_active[s:e].any():
            if not np.array_equal(before[s:e], candidate[s:e]):
                raise ValueError(f'inactive material changed: {name}')
            continue
        b, c = before[s:e], candidate[s:e]
        bd = _signed_distance(b, d['skin_vertices'], d['skin_faces'])
        cd = _signed_distance(c, d['skin_vertices'], d['skin_faces'])
        edges = unique_mesh_edges(own)
        be = np.linalg.norm(b[edges[:, 0]] - b[edges[:, 1]], axis=1)
        ce = np.linalg.norm(c[edges[:, 0]] - c[edges[:, 1]], axis=1)
        good = be > 1e-8
        ratio = ce[good] / be[good]
        tb = b[own]; tc = c[own]
        nb = np.cross(tb[:, 1] - tb[:, 0], tb[:, 2] - tb[:, 0])
        nc = np.cross(tc[:, 1] - tc[:, 0], tc[:, 2] - tc[:, 0])
        active_faces = field_active[s:e][own].any(axis=1)
        flipped = np.sum(np.einsum('ij,ij->i', nb[active_faces], nc[active_faces]) < 0)
        row = dict(name=str(name), tissue=str(tissue),
            field_vertices=int(field_active[s:e].sum()),
            max_moved_mm=float(np.linalg.norm(c-b, axis=1).max()*1000),
            before_outside_max_mm=float(max(0, bd.max())*1000),
            after_outside_max_mm=float(max(0, cd.max())*1000),
            before_outside_count=int((bd > .001).sum()), after_outside_count=int((cd > .001).sum()),
            edge_ratio_min=float(ratio.min()), edge_ratio_max=float(ratio.max()),
            normals_over_90_degrees=int(flipped))
        if str(name) == runtime.metadata['mesh_name']:
            row.update(before_volume_ratio=mesh_volume(b, own)/runtime.volume.rest_volume,
                after_volume_ratio=mesh_volume(c, own)/runtime.volume.rest_volume)
        moved_meshes.append(row)
    d.update(before_vertices=before, source_vertices=before, candidate_vertices=candidate)
    np.savez_compressed(output / input_path.name, **d)
    return dict(input=str(input_path), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        response=response, meshes=moved_meshes, baseline_error_m=mismatch,
        bone_vertices_bit_exact=all(np.array_equal(before[s:e], candidate[s:e])
            for (s,e), t in zip(ranges,tissues) if t=='bone'),
        tetrahedral_inversion_tested=False, triangle_triangle_intersections_tested=False,
        anatomical_passed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-amplitude', type=float, default=.08)
    p.add_argument('--bone-guard-m', type=float, default=.04)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not 0 < args.max_amplitude <= .08: raise ValueError('amplitude must be in (0,.08]')
    base=load_compiled_subject(args.compiled)
    print('base loaded; compiling fixed cubic controller-volume coefficients',flush=True)
    if not .008 <= args.bone_guard_m <= .1: raise ValueError('bone guard must be in [.008,.1] m')
    volume, field, metadata=compile_shared_field(base, bone_guard_m=args.bone_guard_m)
    amplitude = min(args.max_amplitude, metadata['max_amplitude_for_15_percent_rest_edge_change'])
    metadata['amplitude_bound_derived_from_rest_only'] = True
    runtime=VisceralRuntimeV16(base, volume, field, metadata,amplitude)
    args.output.mkdir(parents=True)
    runtime.save(args.output/'compiled')
    loaded=load_visceral_runtime_v16(args.output/'compiled')
    geometry=args.output/'geometry'; geometry.mkdir()
    report=dict(artifact_kind='VisceralVolumeExperimentV16',publishable=False,anatomical_passed=False,
        base=str(args.compiled.resolve()), betas=base.betas.tolist(),
        compiled_metadata=metadata, original_weights_bit_exact=np.array_equal(base.weights,loaded.base.weights),
        original_indices_bit_exact=np.array_equal(base.indices,loaded.base.indices),
        original_faces_bit_exact=np.array_equal(base.source_asset.faces,loaded.base.source_asset.faces),
        inputs_used_for_fitting=False, exact_replay=True, scenes=[])
    for path in args.input:
        with np.load(path,allow_pickle=False) as z:
            expected=runtime.apply_pose(z['pose'],z['transl'])
            actual=loaded.apply_pose(z['pose'],z['transl'])
        if not np.array_equal(expected,actual): raise ValueError('saved replay differs')
        row=evaluate(loaded,path,geometry)
        report['scenes'].append(row)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        organ=next(m for m in row['meshes'] if m['name']=='Large_Intestine')
        print(path.stem, row['response'], 'colon ratios',organ['before_volume_ratio'],organ['after_volume_ratio'],flush=True)
    print('saved',args.output,flush=True)


if __name__=='__main__': main()
