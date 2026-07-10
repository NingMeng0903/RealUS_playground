"""Minimal Wavefront OBJ read/write helpers for anatomy retarget previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_obj(path: Path | str, vertices: np.ndarray, faces: np.ndarray, *, comment: str = "") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with out.open("w", encoding="utf-8") as handle:
        if comment:
            handle.write(f"# {comment}\n")
        for v in verts:
            handle.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for f in tris:
            handle.write(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")
    return out


def read_obj_vertices(path: Path | str) -> np.ndarray:
    verts: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float32)
