#!/usr/bin/env python3
"""Verify RM75 vendor ``*.dae`` load as non-empty geometry (same path as Genesis ``trimesh.load``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

_THIS = Path(__file__).resolve()
_SRC = next(p for p in (_THIS.parent, *_THIS.parents) if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.project import project_paths


def _meshes_from_trimesh(loaded: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    out: list[trimesh.Trimesh] = []
    for node_name in loaded.graph.nodes_geometry:
        _t, geometry_name = loaded.graph[node_name]
        geom = loaded.geometry[geometry_name]
        if isinstance(geom, trimesh.Trimesh):
            m = geom.copy(include_cache=True)
            m.apply_transform(_t)
            out.append(m)
    return out


def main() -> int:
    root = project_paths(__file__).root
    default_dir = (
        root
        / "assets"
        / "robots"
        / "rm75_6f"
        / "vendor"
        / "rm_models"
        / "RM75"
        / "urdf"
        / "RM75-6F"
        / "meshes"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meshes-dir", type=Path, default=default_dir, help="Directory containing base_link.dae, link_*.dae")
    args = p.parse_args()
    meshes_dir = args.meshes_dir.expanduser().resolve()
    expected = ["base_link.dae"] + [f"link_{i}.dae" for i in range(1, 8)]
    failed = False
    for name in expected:
        path = meshes_dir / name
        if not path.is_file():
            print(f"MISSING {path}")
            failed = True
            continue
        loaded = trimesh.load(str(path), process=False)
        meshes = _meshes_from_trimesh(loaded)
        if not meshes:
            print(f"EMPTY_SCENE {path}")
            failed = True
            continue
        total_v = sum(len(m.vertices) for m in meshes)
        total_f = sum(len(m.faces) for m in meshes)
        print(f"OK {name} trimesh_meshes={len(meshes)} vertices={total_v} faces={total_f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
