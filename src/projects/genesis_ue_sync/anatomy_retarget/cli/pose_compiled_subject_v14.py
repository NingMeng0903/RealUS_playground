"""Drive one saved anatomy package from an explicit SMPL-X pose55 array.

Example:
  python -m projects.genesis_ue_sync.anatomy_retarget.cli.pose_compiled_subject_v14 \
    --compiled subject/compiled --pose-file frame.npz --pose-key pose \
    --output posed_anatomy.npz

This minimal driver loads no Blender scene or SMPL-X model. It evaluates the
saved reference frames, original weights and fixed response coefficients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..consistent_runtime_v14 import load_compiled_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--pose-file', type=Path, required=True)
    parser.add_argument('--pose-key', default='pose')
    parser.add_argument('--translation-key', help='Explicit optional [3] translation key; omitted means zero')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.pose_file, allow_pickle=False) as data:
        pose55 = np.asarray(data[args.pose_key], dtype=np.float64)
        if pose55.shape == (165,):
            pose55 = pose55.reshape(55, 3)
        if pose55.shape != (55, 3):
            raise ValueError('pose key must contain one explicit SMPL-X [55,3] or [165] axis-angle pose')
        translation = None if args.translation_key is None else np.asarray(data[args.translation_key], dtype=float)
    compiled = load_compiled_subject(args.compiled)
    vertices, globals_ = compiled.apply_pose(pose55, translation, return_globals=True)
    metadata = dict(anatomical_passed=False, publishable=False,
                    compiled_manifest_sha256=hashlib.sha256((args.compiled/'manifest.json').read_bytes()).hexdigest(),
                    pose_file_sha256=hashlib.sha256(args.pose_file.read_bytes()).hexdigest(),
                    pose_key=args.pose_key, translation_key=args.translation_key,
                    computational_support=compiled.support_report())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwrite if another process
    # produced the requested path after the initial existence check.
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, vertices=vertices, faces=compiled.source_asset.faces,
                            controller_globals=globals_, pose55=pose55, betas=compiled.betas,
                            translation=np.zeros(3) if translation is None else translation,
                            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    print(f'{len(vertices)} vertices; saved {args.output}; anatomical acceptance remains false')


if __name__ == '__main__':
    main()
