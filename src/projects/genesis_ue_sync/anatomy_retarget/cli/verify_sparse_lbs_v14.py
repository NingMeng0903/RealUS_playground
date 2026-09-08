"""Compare sparse evaluation with original LBS and exact saved V14 geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..chain_rest_fit_v1 import _weighted_rest_correction
from ..consistent_runtime_v14 import load_compiled_subject
from ..sparse_lbs_v14 import SparseLBSV14


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', anatomical_passed=False, publishable=False, candidates=[])
    try:
        for root in args.candidates:
            c = load_compiled_subject(root/'compiled')
            sparse = SparseLBSV14(c.target_rest, c.indices, c.weights)
            row = dict(candidate=str(root.resolve()),
                compiled_manifest_sha256=hashlib.sha256((root/'compiled/manifest.json').read_bytes()).hexdigest(),
                full_slot_count=int(c.weights.size), positive_slot_count=int(len(sparse.rows)), frames=[])
            report['candidates'].append(row)
            for path in sorted(root.glob('subject_*.npz')):
                with np.load(path, allow_pickle=False) as data:
                    p = data['pose'].reshape(55, 3)
                    stored = data['candidate_vertices']
                    started = time.perf_counter()
                    sg = c.source_globals(p)
                    g = c.globals_from_source(sg)
                    motion_s = time.perf_counter()-started
                    transforms = g @ c.target_inverse
                    started = time.perf_counter()
                    before = _weighted_rest_correction(c.target_rest, c.indices, c.weights, transforms)
                    before_s = time.perf_counter()-started
                    started = time.perf_counter()
                    after = sparse(transforms)
                    after_s = time.perf_counter()-started
                    exact = bool(np.array_equal(before, after))
                    playback = (c.target_rest if not np.any(p) else after).astype(np.float32)
                    saved_exact = bool(np.array_equal(playback, stored))
                    row['frames'].append(dict(name=path.stem, float64_lbs_bitexact=exact,
                        stored_vertices_bitexact=saved_exact, motion_seconds=motion_s,
                        original_lbs_seconds=before_s, sparse_lbs_seconds=after_s))
                    if not exact or not saved_exact:
                        raise AssertionError(f'{path}: exact playback failed')
                    print(path.stem, before_s, after_s, 'bitexact', flush=True)
            if not row['frames']:
                raise ValueError(f'{root}: no saved geometry')
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['failure'] = repr(exc)
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
