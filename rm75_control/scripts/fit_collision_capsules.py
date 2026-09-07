"""Fit outer enclosing capsules to RM75 collision / visual meshes.

The capsule is a sphere-swept segment in the link frame: every mesh vertex
must satisfy dist(point, segment) <= radius.  A positive margin is added
so the primitive sits outside the mesh, not on it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "rm75_control" / "assets" / "robots" / "rm75_6f_8dof" / "meshes"
COLL = MESH / "collision"
OUT_YAML = (
    ROOT
    / "rm75_control"
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "collision_capsules.yaml"
)
OUT_URDF = (
    ROOT
    / "rm75_control"
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.collision.capsule.urdf"
)
OUT_JSON = COLL / "capsule_fit.json"

LINKS = (
    "base_link",
    "link_1",
    "link_2",
    "link_3",
    "link_4",
    "link_5",
    "link_6",
    "link_7",
    "probe45",
)
MARGIN_M = 0.002
SPLIT_GAIN = 0.72
SPLIT_SEP_M = 0.045


def _load_trimesh_vertices(path: Path) -> np.ndarray:
    import trimesh

    loaded = trimesh.load(str(path), force=None, skip_materials=True)
    geoms = (
        list(loaded.geometry.values())
        if isinstance(loaded, trimesh.Scene)
        else [loaded]
    )
    chunks = []
    for geom in geoms:
        verts = np.asarray(getattr(geom, "vertices", []), dtype=float)
        if verts.ndim == 2 and verts.shape[1] == 3 and len(verts):
            chunks.append(verts)
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    return np.unique(np.vstack(chunks), axis=0)


def _maybe_meters(points: np.ndarray) -> np.ndarray:
    """Collision STLs are meters. Visual DAE probe parts are millimeters."""
    if points.size == 0:
        return points
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if extent > 2.0:
        return points * 0.001
    return points


def load_vertices(name: str) -> tuple[np.ndarray, str]:
    stl = COLL / f"{name}.stl"
    if not stl.exists():
        raise FileNotFoundError(f"collision STL not found: {stl}")
    pts = _maybe_meters(_load_trimesh_vertices(stl))
    # Visual DAE is only merged when it lives in the same link frame and
    # the same metre scale as the collision STL.  probe45.dae is mm and
    # a multi-part assembly; mixing it raw produced 80 m capsules.
    return pts, stl.name


def _axis_to_rpy(axis: np.ndarray) -> np.ndarray:
    z = axis / max(float(np.linalg.norm(axis)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    xn = float(np.linalg.norm(x))
    if xn < 1e-12:
        x = np.array([0.0, 1.0, 0.0])
        xn = 1.0
    x = x / xn
    y = np.cross(z, x)
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    return Rsc.from_matrix(np.column_stack((x, y, z))).as_euler("xyz")


def _dist_to_segment(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    d = p1 - p0
    den = float(np.dot(d, d))
    if den < 1e-16:
        return np.linalg.norm(points - p0, axis=1)
    t = np.clip((points - p0) @ d / den, 0.0, 1.0)
    return np.linalg.norm(points - (p0 + t[:, None] * d), axis=1)


@dataclass
class Capsule:
    name: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    radius_m: float
    length_m: float
    p0: np.ndarray
    p1: np.ndarray
    n_points: int
    source: str
    bbox_mm: np.ndarray
    max_outside_mm: float
    min_clearance_mm: float

    @property
    def half_length_m(self) -> float:
        return 0.5 * float(self.length_m)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "origin_xyz": [round(float(x), 6) for x in self.origin_xyz],
            "origin_rpy": [round(float(x), 6) for x in self.origin_rpy],
            "radius_m": round(float(self.radius_m), 6),
            "length_m": round(float(self.length_m), 6),
            "radius_mm": round(1000.0 * float(self.radius_m), 2),
            "length_mm": round(1000.0 * float(self.length_m), 2),
            "segment_p0": [round(float(x), 6) for x in self.p0],
            "segment_p1": [round(float(x), 6) for x in self.p1],
            "n_points": int(self.n_points),
            "source": self.source,
            "bbox_mm": [round(float(x), 2) for x in self.bbox_mm],
            "max_outside_mm": round(float(self.max_outside_mm), 3),
            "min_clearance_mm": round(float(self.min_clearance_mm), 3),
        }


def fit_one(points: np.ndarray, *, name: str, source: str, margin: float) -> Capsule:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    center = pts.mean(axis=0)
    x = pts - center
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    axis = vt[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    t = x @ axis
    t0 = float(t.min())
    t1 = float(t.max())
    p0 = center + t0 * axis
    p1 = center + t1 * axis
    dist = _dist_to_segment(pts, p0, p1)
    radius = float(dist.max()) + float(margin)
    mid = 0.5 * (p0 + p1)
    length = float(np.linalg.norm(p1 - p0))
    slack = radius - dist
    bbox = pts.max(axis=0) - pts.min(axis=0)
    return Capsule(
        name=name,
        origin_xyz=mid,
        origin_rpy=_axis_to_rpy(axis),
        radius_m=radius,
        length_m=length,
        p0=p0,
        p1=p1,
        n_points=int(len(pts)),
        source=source,
        bbox_mm=1000.0 * bbox,
        max_outside_mm=1000.0 * float(np.min(slack)),
        min_clearance_mm=1000.0 * float(np.min(slack)),
    )


def _kmeans2(points: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    picks = rng.choice(len(pts), size=2, replace=False)
    c0, c1 = pts[picks[0]].copy(), pts[picks[1]].copy()
    lab = np.zeros(len(pts), dtype=int)
    for _ in range(24):
        d0 = np.linalg.norm(pts - c0, axis=1)
        d1 = np.linalg.norm(pts - c1, axis=1)
        lab = (d1 < d0).astype(int)
        if lab.min() == lab.max():
            break
        c0 = pts[lab == 0].mean(axis=0)
        c1 = pts[lab == 1].mean(axis=0)
    return pts[lab == 0], pts[lab == 1]


def fit_link(name: str, *, margin: float = MARGIN_M) -> list[Capsule]:
    pts, source = load_vertices(name)
    one = fit_one(pts, name=name, source=source, margin=margin)
    # Arm links are one bone each.  Only the probe assembly is allowed to
    # split: a single sausage around the camera+head is unnecessarily fat.
    if name not in {"probe45", "linear_probe"} or len(pts) < 32:
        return [one]
    a, b = _kmeans2(pts)
    if len(a) < 12 or len(b) < 12:
        return [one]
    ca = fit_one(a, name=f"{name}_a", source=source, margin=margin)
    cb = fit_one(b, name=f"{name}_b", source=source, margin=margin)
    sep = float(np.linalg.norm(ca.origin_xyz - cb.origin_xyz))
    thinner = max(ca.radius_m, cb.radius_m) <= SPLIT_GAIN * one.radius_m
    if thinner and sep >= SPLIT_SEP_M:
        union_out = []
        for cap in (ca, cb):
            slack = cap.radius_m - _dist_to_segment(pts, cap.p0, cap.p1)
            # A split is only kept if the union still encloses every vertex.
            union_out.append(slack)
        inside = np.maximum(union_out[0], union_out[1])
        if float(inside.min()) >= -1e-9:
            ca.min_clearance_mm = 1000.0 * float(inside.min())
            cb.min_clearance_mm = 1000.0 * float(inside.min())
            ca.max_outside_mm = ca.min_clearance_mm
            cb.max_outside_mm = cb.min_clearance_mm
            return [ca, cb]
    return [one]


def wrist_camera_capsule() -> Capsule:
    # Existing box: size 0.12 x 0.03 x 0.03 at xyz 0.02 0 0 in wrist_camera.
    # Outer capsule along +X, radius covers the square cross-section plus margin.
    radius = 0.5 * math.hypot(0.03, 0.03) + MARGIN_M
    return Capsule(
        name="wrist_camera",
        origin_xyz=np.array([0.02, 0.0, 0.0]),
        origin_rpy=np.array([0.0, math.pi / 2.0, 0.0]),
        radius_m=radius,
        length_m=0.12,
        p0=np.array([-0.04, 0.0, 0.0]),
        p1=np.array([0.08, 0.0, 0.0]),
        n_points=0,
        source="urdf_box",
        bbox_mm=np.array([120.0, 30.0, 30.0]),
        max_outside_mm=1000.0 * MARGIN_M,
        min_clearance_mm=1000.0 * MARGIN_M,
    )


import math


def dump_yaml(groups: dict[str, list[Capsule]]) -> str:
    lines = [
        "# Outer enclosing capsules for RM75-6F-8dof self-collision.",
        "# Primitive is a sphere-swept segment in the link frame.",
        "# URDF cylinder/capsule axis is +Z after origin rpy.",
        f"# Margin outside the source mesh: {1000.0 * MARGIN_M:.1f} mm.",
        "# length_m is the segment length (hemisphere caps are extra).",
        "margin_m: {:.6f}".format(MARGIN_M),
        "frame: link",
        "axis: local_z",
        "capsules:",
    ]
    for link, caps in groups.items():
        lines.append(f"  {link}:")
        for cap in caps:
            d = cap.as_dict()
            lines.append(f"    - name: {d['name']}")
            lines.append(
                "      origin_xyz: [{}, {}, {}]".format(*d["origin_xyz"])
            )
            lines.append(
                "      origin_rpy: [{}, {}, {}]".format(*d["origin_rpy"])
            )
            lines.append(f"      radius_m: {d['radius_m']}")
            lines.append(f"      length_m: {d['length_m']}")
            lines.append(f"      radius_mm: {d['radius_mm']}")
            lines.append(f"      length_mm: {d['length_mm']}")
            lines.append(f"      source: {d['source']}")
            lines.append(
                "      bbox_mm: [{}, {}, {}]".format(*d["bbox_mm"])
            )
            lines.append(f"      min_clearance_mm: {d['min_clearance_mm']}")
    return "\n".join(lines) + "\n"


def _collision_block(cap: Capsule) -> str:
    xyz = " ".join(f"{float(x):.6f}" for x in cap.origin_xyz)
    rpy = " ".join(f"{float(x):.6f}" for x in cap.origin_rpy)
    return (
        "    <collision>\n"
        f'      <origin xyz="{xyz}" rpy="{rpy}" />\n'
        "      <geometry>\n"
        f'        <cylinder radius="{cap.radius_m:.6f}" length="{cap.length_m:.6f}" />\n'
        "      </geometry>\n"
        "    </collision>\n"
    )


def dump_urdf(groups: dict[str, list[Capsule]]) -> str:
    src = (
        ROOT
        / "rm75_control"
        / "assets"
        / "robots"
        / "rm75_6f_8dof"
        / "RM75-6F-8dof.collision.urdf"
    )
    text = src.read_text(encoding="utf-8")
    text = text.replace(
        "arm mesh collision.",
        "outer enclosing capsules (cylinder primitive; see collision_capsules.yaml).",
    )
    import re

    def repl(link: str, caps: list[Capsule]) -> None:
        nonlocal text
        block = "".join(_collision_block(c) for c in caps)
        pattern = re.compile(
            rf'(<link name="{re.escape(link)}">.*?)(\s*<collision>.*?</collision>\s*)(</link>)',
            re.DOTALL,
        )
        new_text, n = pattern.subn(rf"\1\n{block}  \3", text, count=1)
        if n != 1:
            raise RuntimeError(f"could not replace collision for {link}")
        text = new_text

    for link, caps in groups.items():
        if link == "linear_probe":
            continue
        urdf_link = "link_8" if link == "probe45" else link
        repl(urdf_link, caps)
    return text


def main() -> None:
    groups: dict[str, list[Capsule]] = {}
    report = []
    for name in LINKS:
        caps = fit_link(name)
        groups[name] = caps
        report.append({name: [c.as_dict() for c in caps]})
        bits = []
        for c in caps:
            bits.append(
                f"{c.name}: r={1000*c.radius_m:.1f}mm L={1000*c.length_m:.1f}mm "
                f"clear={c.min_clearance_mm:.2f}mm n={c.n_points}"
            )
        print(f"{name:10s} {caps[0].source:28s} " + " | ".join(bits))
    extra = COLL / "linear_probe.stl"
    if extra.exists():
        groups["linear_probe"] = fit_link("linear_probe")
        c = groups["linear_probe"][0]
        print(
            f"{'linear_probe':10s} {c.source:28s} {c.name}: "
            f"r={1000*c.radius_m:.1f}mm L={1000*c.length_m:.1f}mm "
            f"clear={c.min_clearance_mm:.2f}mm n={c.n_points}"
        )
    groups["wrist_camera"] = [wrist_camera_capsule()]
    OUT_YAML.write_text(dump_yaml(groups), encoding="utf-8")
    OUT_URDF.write_text(dump_urdf(groups), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_YAML}")
    print(f"wrote {OUT_URDF}")


if __name__ == "__main__":
    main()
