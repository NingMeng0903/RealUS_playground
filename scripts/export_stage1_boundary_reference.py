#!/usr/bin/env python3
"""Create a signed reusable Stage-1 cage-boundary initialization cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_obj_vertices(path: Path) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in line.split()[1:4]] for line in path.open() if line.startswith("v ")],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--registered-boundary-obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import _build_source_cage

    with np.load(args.asset, allow_pickle=True) as asset:
        source_skin = np.asarray(asset["source_skin_vertices"], dtype=np.float64)
        source_faces = np.asarray(asset["source_skin_faces"], dtype=np.int32)
    cage = _build_source_cage(
        source_skin,
        source_faces,
        args.canonical_dir / "source_skin_volume_cage_v18_subject_shell_full_domain.npz",
        dilation_iterations=1,
    )
    boundary = _load_obj_vertices(args.registered_boundary_obj)
    expected = np.asarray(cage["nodes"], dtype=np.float64)[np.asarray(cage["boundary"], dtype=np.int64)]
    if boundary.shape != expected.shape:
        raise ValueError(
            f"boundary shape {boundary.shape} does not match cage boundary {expected.shape}"
        )
    signature = str(np.asarray(cage["signature"]).reshape(-1)[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        cage_signature=np.asarray(signature),
        registered_boundary=boundary.astype(np.float32),
        source_boundary=expected.astype(np.float32),
        source_skin_vertices=source_skin.astype(np.float32),
        source_skin_faces=source_faces.astype(np.int32),
        nodes=np.asarray(cage["nodes"], dtype=np.float32),
        elements=np.asarray(cage["elements"], dtype=np.int32),
        boundary_indices=np.asarray(cage["boundary"], dtype=np.int32),
        boundary_faces=np.asarray(cage["boundary_faces"], dtype=np.int32),
        voxel_pitch=np.asarray(cage["voxel_pitch"], dtype=np.float32),
        meshing_backend=np.asarray(cage["meshing_backend"]),
        removed_degenerate_tetrahedra=np.asarray(cage["removed_degenerate_tetrahedra"], dtype=np.int32),
    )
    print(f"wrote {args.output} boundary_nodes={len(boundary)} signature={signature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
