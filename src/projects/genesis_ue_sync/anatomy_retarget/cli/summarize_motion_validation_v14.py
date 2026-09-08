"""Summarize frozen-motion measurements without loading or changing geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get('used_for_fit') is not False or report.get('runtime_optimization') is not False:
            raise ValueError(f'{path}: not a frozen, optimization-free motion evaluation')
        frames = report['frames']
        indices = [int(f['index']) for f in frames]
        if indices != list(range(len(frames))):
            raise ValueError(f'{path}: non-contiguous or duplicate frame indices')
        valid = [f for f in frames if f.get('status') == 'evaluated']
        if len(valid) != report.get('evaluated_frames') or len(frames)-len(valid) != report.get('unsupported_or_error_frames'):
            raise ValueError(f'{path}: final counters do not match frame records')
        if not valid:
            raise ValueError(f'{path}: no evaluated geometry')
        skin = {}
        for tissue in ('all_bones', 'left_arm_bones', 'vessels'):
            distance = np.array([f['skin'][tissue]['max_outside_m'] for f in valid], dtype=float)
            if not np.isfinite(distance).all() or np.any(distance < 0):
                raise ValueError(f'{path}: invalid distance measurements')
            worst = valid[int(np.argmax(distance))]
            skin[tissue] = dict(max_outside_mm=float(distance.max()*1000),
                p95_frame_max_outside_mm=float(np.quantile(distance, .95)*1000),
                worst_index=worst['index'], worst_source_frame=worst['source_frame'],
                frames_exceeding_1mm=int(np.count_nonzero(distance > .001)),
                skin_boundary_passed=bool(np.all(distance <= .001)))
        rows.append(dict(name=path.parent.name, report=str(path.resolve()),
            report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            compiled=report['compiled'], compiled_manifest_sha256=report['compiled_manifest_sha256'],
            motion_sha256=report['motion_sha256'], frame_count=len(frames),
            evaluated_frames=len(valid), unsupported_or_error_frames=len(frames)-len(valid),
            skin=skin, video=report.get('video'),
            median_pose_seconds=float(np.median([f['pose_seconds'] for f in valid]))))
    return dict(schema_version=1, artifact_kind='FrozenMotionValidationSummaryV14',
        reports=rows, report_count=len(rows),
        evaluated_frames=sum(r['evaluated_frames'] for r in rows),
        unsupported_or_error_frames=sum(r['unsupported_or_error_frames'] for r in rows),
        skin_boundary_passed=all(t['skin_boundary_passed'] for r in rows for t in r['skin'].values()),
        anatomical_passed=False, publishable=False, fitting_performed=False,
        note='Frame-wise skin distance is only one gate; execution success is not anatomical acceptance.',
        gates_not_measured_by_this_summary=['bone_bone_triangle_intersection','tube_bone_intersection',
            'posed_cap_shape','vessel_diameter','branch_continuity','material_volume_inversion'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; keep earlier evidence')
    result = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f"{result['evaluated_frames']} evaluated frames, skin_boundary_passed={result['skin_boundary_passed']}")


if __name__ == '__main__':
    main()
