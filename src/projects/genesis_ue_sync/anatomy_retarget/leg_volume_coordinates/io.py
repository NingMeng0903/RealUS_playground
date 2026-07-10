"""Mesh and centerline I/O for leg volume coordinate baking."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_obj_mesh(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangular faces from a minimal OBJ."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                raw = [part.split("/")[0] for part in line.split()[1:]]
                idx = [int(v) - 1 for v in raw]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def read_centerline_obj(path: Path | str) -> dict[str, np.ndarray]:
    """Read multi-object polyline OBJ written by the vessel export."""
    out: dict[str, np.ndarray] = {}
    current: str | None = None
    pts: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("o "):
                if current is not None and pts:
                    out[current] = np.asarray(pts, dtype=np.float32)
                current = line[2:].strip()
                pts = []
            elif line.startswith("v "):
                parts = line.split()
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if current is not None and pts:
        out[current] = np.asarray(pts, dtype=np.float32)
    return out


def write_centerline_obj(path: Path | str, centerlines: dict[str, np.ndarray], *, comment: str = "") -> Path:
    """Write each centerline as an OBJ object with line elements."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    offset = 1
    with out.open("w", encoding="utf-8") as handle:
        if comment:
            handle.write(f"# {comment}\n")
        for label, line in centerlines.items():
            pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] == 0:
                continue
            handle.write(f"o {label}\n")
            for p in pts:
                handle.write(f"v {float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
            if pts.shape[0] >= 2:
                indices = " ".join(str(offset + i) for i in range(pts.shape[0]))
                handle.write(f"l {indices}\n")
            offset += pts.shape[0]
    return out
