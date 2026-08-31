"""Project R_SUPFEMV onto posed SMPL-X skin after a Y capture.

Reuses the Among_US / RealUS vessel-to-skin projector. Writes
``smplx_outputs/<run>/vessel_plan.json`` so Xbox B can start immediately.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from perception.capture_flow import repo_root, smplx_output_root

VESSEL_LABEL = "R_SUPFEMV"
WINDOW_M = 0.10
STANDOFF_M = 0.05
# Chart h: 0 = hip / proximal, 1 = ankle / distal. Mid-upper femoral trunk.
PROXIMAL_MID_H = (0.12, 0.42)
AMONG_US_DEFAULT = Path("/media/camp/EXT_DRIVE/Among_US")
PLAN_NAME = "vessel_plan.json"
DEFAULT_CANONICAL = Path("outputs/anatomy_retarget/latest_canonical")
CENTERLINES_REL = Path(
    "outputs/anatomy_retarget/limb_vessel_planning/centerlines/vessel_centerlines_rest.obj"
)
ATLAS_REL = Path("dataset/processed/anatomy_retarget/leg_volume_coordinates")


class VesselPlanError(RuntimeError):
    """Plan failed. ``reason`` is the one-line B refuse token."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = str(reason)
        super().__init__(detail or reason)


@dataclass(frozen=True)
class VesselPlan:
    path: Path
    label: str
    run_name: str
    world_xyz: np.ndarray
    scan_tangent: np.ndarray
    skin_normals: np.ndarray
    scan_poses: np.ndarray
    contact_pose: np.ndarray
    tcp_poses: np.ndarray
    tcp_contact: np.ndarray
    window_m: float
    standoff_m: float

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": True,
            "label": self.label,
            "run_name": self.run_name,
            "window_m": float(self.window_m),
            "standoff_m": float(self.standoff_m),
            "side": "right",
            "world_xyz": np.asarray(self.world_xyz, dtype=float).reshape(-1, 3).tolist(),
            "scan_tangent": np.asarray(self.scan_tangent, dtype=float).reshape(-1, 3).tolist(),
            "skin_normals": np.asarray(self.skin_normals, dtype=float).reshape(-1, 3).tolist(),
            "scan_poses": np.asarray(self.scan_poses, dtype=float).reshape(-1, 6).tolist(),
            "contact_pose": np.asarray(self.contact_pose, dtype=float).reshape(6).tolist(),
            "tcp_poses": np.asarray(self.tcp_poses, dtype=float).reshape(-1, 6).tolist(),
            "tcp_contact": np.asarray(self.tcp_contact, dtype=float).reshape(6).tolist(),
        }


def among_us_root() -> Path:
    return Path(os.environ.get("AMONG_US_ROOT", AMONG_US_DEFAULT))


def _first_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_atlas_path(side: str = "right", *, repo: Path | None = None) -> Path | None:
    """Production / layered atlas on RealUS, then the same relative path on Among_US."""
    root = repo if repo is not None else repo_root()
    try:
        from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.paths import (
            layered_atlas_path,
            production_atlas_path,
        )

        realus = [production_atlas_path(side), layered_atlas_path(side)]  # type: ignore[arg-type]
    except Exception:
        realus = [
            root / ATLAS_REL / "production" / "atlas" / f"atlas_{side}.npz",
            root / ATLAS_REL / "atlas_layered_laplace3d" / f"atlas_{side}.npz",
        ]
    among = among_us_root()
    extra = [
        among / ATLAS_REL / "production" / "atlas" / f"atlas_{side}.npz",
        among / ATLAS_REL / "atlas_layered_laplace3d" / f"atlas_{side}.npz",
        among / ATLAS_REL / "atlas" / f"atlas_{side}.npz",
        root / ATLAS_REL / "atlas" / f"atlas_{side}.npz",
    ]
    return _first_file([*realus, *extra])


def resolve_vessel_source(*, repo: Path | None = None) -> tuple[str, Path] | None:
    """Return ``("remap", npz)`` or ``("project", obj)`` if a source exists."""
    root = repo if repo is not None else repo_root()
    try:
        from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.paths import (
            production_vessel_material_path,
        )

        production = production_vessel_material_path()
    except Exception:
        production = root / ATLAS_REL / "production" / "vessels" / "vessel_material_coordinates.npz"
    among = among_us_root()
    npz = _first_file(
        [
            production,
            root / "outputs/anatomy_retarget/leg_volume_coordinates/vessel_skin_projection.npz",
            among / ATLAS_REL / "production" / "vessels" / "vessel_material_coordinates.npz",
            among / "outputs/anatomy_retarget/leg_volume_coordinates/vessel_skin_projection.npz",
        ]
    )
    if npz is not None:
        return "remap", npz
    obj = _first_file(
        [
            root / CENTERLINES_REL,
            among / CENTERLINES_REL,
        ]
    )
    if obj is not None:
        return "project", obj
    return None


def polyline_arclength(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if pts.shape[0] == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(points: np.ndarray, *, step_m: float = 0.005, length_m: float | None = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts.copy()
    s = polyline_arclength(pts)
    total = float(s[-1])
    target = float(total if length_m is None else min(max(length_m, 0.0), total))
    if target <= 1.0e-9:
        return pts[:1].copy()
    count = max(int(np.ceil(target / max(float(step_m), 1.0e-4))) + 1, 2)
    dst = np.linspace(0.0, target, count)
    out = np.zeros((count, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(dst, s, pts[:, dim])
    return out


def extract_vessel_window(
    points: np.ndarray,
    *,
    h: np.ndarray | None = None,
    window_m: float = WINDOW_M,
    h_band: tuple[float, float] = PROXIMAL_MID_H,
) -> np.ndarray:
    """Cut a ~10 cm mid-upper window. Prefer chart ``h``; else mid-proximal third."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise VesselPlanError("no R_SUPFEMV", "vessel polyline is too short")
    s = polyline_arclength(pts)
    total = float(s[-1])
    if total <= 1.0e-6:
        raise VesselPlanError("no R_SUPFEMV", "vessel polyline has zero length")
    start = None
    if h is not None and np.asarray(h).reshape(-1).size == pts.shape[0]:
        hh = np.asarray(h, dtype=float).reshape(-1)
        lo, hi = float(h_band[0]), float(h_band[1])
        mask = (hh >= lo) & (hh <= hi)
        if int(np.count_nonzero(mask)) >= 2:
            start = float(np.min(s[mask]))
    if start is None:
        start = total / 3.0 - 0.5 * float(window_m)
    start = float(np.clip(start, 0.0, max(0.0, total - float(window_m))))
    span = min(float(window_m), total)
    return _slice_arclength(pts, s, start, start + span)


def _slice_arclength(points: np.ndarray, s: np.ndarray, start: float, end: float) -> np.ndarray:
    span = max(float(end) - float(start), 0.0)
    count = max(int(np.ceil(span / 0.005)) + 1, 2)
    dst = np.linspace(float(start), float(end), count)
    out = np.zeros((count, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(dst, s, points[:, dim])
    return out


def _closest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0], dtype=float)
    bp = p - b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0], dtype=float)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-8)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0], dtype=float)
    cp = p - c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0], dtype=float)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-8)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w], dtype=float)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-8)
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w], dtype=float)
    denom = max(va + vb + vc, 1.0e-8)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w, np.asarray([1.0 - v - w, v, w], dtype=float)


def nearest_face_barycentric(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    candidate_k: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-face barycentric on ``vertices``/``faces`` for each query point."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces_i = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=float)
    k = min(int(candidate_k), verts.shape[0])
    face_of: dict[int, list[int]] = {}
    for fi, tri in enumerate(faces_i):
        for vi in tri.tolist():
            face_of.setdefault(int(vi), []).append(int(fi))
    face_idx = np.zeros((pts.shape[0],), dtype=np.int32)
    bary = np.zeros((pts.shape[0], 3), dtype=float)
    for i, p in enumerate(pts):
        d2 = np.sum(np.square(verts - p.reshape(1, 3)), axis=1)
        near = np.argpartition(d2, kth=k - 1)[:k]
        cand: list[int] = []
        for vi in near.tolist():
            cand.extend(face_of.get(int(vi), []))
        if not cand:
            cand = list(range(min(256, faces_i.shape[0])))
        best_d2 = float("inf")
        best_f = int(cand[0])
        best_b = np.asarray([1.0, 0.0, 0.0], dtype=float)
        for fi in set(cand):
            a, b, c = verts[faces_i[int(fi)]]
            closest, w = _closest_point_on_triangle(p, a, b, c)
            dist = float(np.sum(np.square(p - closest)))
            if dist < best_d2:
                best_d2 = dist
                best_f = int(fi)
                best_b = w
        face_idx[i] = best_f
        bary[i] = best_b
    return face_idx, bary


def apply_barycentric(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_idx: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces_i = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    fi = np.asarray(face_idx, dtype=np.int32).reshape(-1)
    w = np.asarray(bary, dtype=float).reshape(-1, 3)
    tri = verts[faces_i[fi]]
    return (w[:, 0:1] * tri[:, 0, :] + w[:, 1:2] * tri[:, 1, :] + w[:, 2:3] * tri[:, 2, :])


def bind_tpose_to_posed(
    tpose_points: np.ndarray,
    tpose_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pose-invariant skin transfer: same faces + barycentric, T-pose → posed.

    This is the SMPL-X surface-point invariance used by the vessel-to-skin
    figures (T-pose vs bent-leg): material points stay on the same triangles.

    Atlas ``full_vertex_indices`` go up to ~8725 (SMPL-X, 10475 verts). Classic
    SMPL only has 6890, so we never treat those IDs as SMPL. The projector
    returns remesh XYZ in the same Y-up T-pose frame; we snap by nearest face
    onto this subject's SMPL-X T-pose, then apply the same faces on the posed
    mesh. Do not mix a SMPL atlas with SMPL-X ``smplx_result`` vertex IDs.
    """
    src = np.asarray(tpose_vertices, dtype=float).reshape(-1, 3)
    dst = np.asarray(posed_vertices, dtype=float).reshape(-1, 3)
    if src.shape[0] != dst.shape[0]:
        raise VesselPlanError("no capture", "T-pose and posed meshes have different topology")
    face_idx, bary = nearest_face_barycentric(tpose_points, src, faces)
    world = apply_barycentric(dst, faces, face_idx, bary)
    return world, face_idx, bary


def _finite_tangents(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    tan = np.zeros_like(pts)
    if pts.shape[0] == 1:
        tan[0] = np.array([0.0, 1.0, 0.0])
        return tan
    tan[0] = pts[1] - pts[0]
    tan[-1] = pts[-1] - pts[-2]
    if pts.shape[0] > 2:
        tan[1:-1] = pts[2:] - pts[:-2]
    nrm = np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1.0e-9, None)
    return tan / nrm


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces_i = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    a, b, c = verts[faces_i[:, 0]], verts[faces_i[:, 1]], verts[faces_i[:, 2]]
    n = np.cross(b - a, c - a)
    nrm = np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-9, None)
    return n / nrm


def consistent_along_polyline(vectors: np.ndarray) -> np.ndarray:
    """Flip any sample that opposes its predecessor (silhouette / winding flips)."""
    out = np.asarray(vectors, dtype=float).reshape(-1, 3).copy()
    for i in range(1, out.shape[0]):
        if float(out[i] @ out[i - 1]) < 0.0:
            out[i] = -out[i]
    return out


def outward_skin_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    face_idx: np.ndarray,
) -> np.ndarray:
    """Face normals flipped so they point away from the mesh centroid."""
    fn = _face_normals(vertices, faces)
    nrm = fn[np.asarray(face_idx, dtype=np.int32).reshape(-1)]
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    center = np.asarray(vertices, dtype=float).reshape(-1, 3).mean(axis=0)
    for i in range(nrm.shape[0]):
        if float(nrm[i] @ (pts[i] - center)) < 0.0:
            nrm[i] = -nrm[i]
    return consistent_along_polyline(nrm)


def tool_axes_from_normal_tangent(
    normal: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Columns [X Y Z]: +Z into skin, +Y along scan, +X = Y×Z (perp to path)."""
    outward = np.asarray(normal, dtype=float).reshape(3)
    on = float(np.linalg.norm(outward))
    outward = outward / on if on > 1.0e-9 else np.array([0.0, 0.0, 1.0])
    z = -outward
    zn = float(np.linalg.norm(z))
    z = z / zn if zn > 1.0e-9 else np.array([0.0, 0.0, -1.0])
    y = np.asarray(tangent, dtype=float).reshape(3)
    y = y - z * float(y @ z)
    yn = float(np.linalg.norm(y))
    if yn < 1.0e-8:
        fallback = np.array([0.0, 1.0, 0.0]) if abs(float(z[1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = fallback - z * float(fallback @ z)
        yn = float(np.linalg.norm(y))
    y = y / max(yn, 1.0e-9)
    x = np.cross(y, z)
    xn = float(np.linalg.norm(x))
    x = x / max(xn, 1.0e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def poses_from_polyline(
    points: np.ndarray,
    normals: np.ndarray,
    tangents: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rsc

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=float).reshape(-1, 3)
    tan = np.asarray(tangents, dtype=float).reshape(-1, 3)
    out = np.zeros((pts.shape[0], 6), dtype=float)
    for i in range(pts.shape[0]):
        R = tool_axes_from_normal_tangent(nrm[i], tan[i])
        out[i, :3] = pts[i]
        out[i, 3:6] = Rsc.from_matrix(R).as_euler(euler_order, degrees=False)
    return out


def robot_world_yaml_path(*, repo: Path | None = None) -> Path:
    raw = (
        os.environ.get("REALUS_ROBOT_WORLD")
        or os.environ.get("CAMERA_CALIB_ROBOT_WORLD")
        or ""
    ).strip()
    if raw:
        return Path(raw)
    root = repo if repo is not None else repo_root()
    return root / "camera_calibration" / "calibration_results" / "robot_world.yaml"


def load_T_smplx_from_tcp(*, repo: Path | None = None) -> np.ndarray:
    """Robot TCP frame origin in the SMPL-X / camera world (fixed placement)."""
    path = robot_world_yaml_path(repo=repo)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    T = raw.get("T_world_railbase")
    if T is None:
        raise VesselPlanError("no capture", f"T_world_railbase missing in {path}")
    out = np.asarray(T, dtype=float).reshape(4, 4)
    if not np.isfinite(out).all():
        raise VesselPlanError("no capture", f"T_world_railbase in {path} is not finite")
    return out


def smplx_poses_to_tcp(
    poses: np.ndarray,
    *,
    T: np.ndarray | None = None,
    euler_order: str = "xyz",
    repo: Path | None = None,
) -> np.ndarray:
    """Same Cartesian polyline, rewritten into the TCP frame TRACK uses."""
    arr = np.asarray(poses, dtype=float)
    single = arr.ndim == 1
    rows = arr.reshape(-1, 6)
    Tw = np.asarray(T if T is not None else load_T_smplx_from_tcp(repo=repo), dtype=float).reshape(4, 4)
    R = Tw[:3, :3]
    t = Tw[:3, 3]
    out = np.zeros_like(rows)
    for i, pose in enumerate(rows):
        out[i, :3] = R.T @ (pose[:3] - t)
        Rw = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
        out[i, 3:6] = Rsc.from_matrix(R.T @ Rw).as_euler(euler_order, degrees=False)
    return out[0] if single else out


def contact_pose_from_window(
    points: np.ndarray,
    normals: np.ndarray,
    tangents: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    poses = poses_from_polyline(points[:1], normals[:1], tangents[:1], euler_order=euler_order)
    return poses[0]


def _load_posed_mesh(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.is_file():
        raise VesselPlanError("no capture", f"missing {npz_path}")
    with np.load(npz_path, allow_pickle=True) as payload:
        if "vertices" not in payload.files:
            raise VesselPlanError("no capture", "smplx_result.npz has no vertices")
        verts = np.asarray(payload["vertices"], dtype=np.float32).reshape(-1, 3)
        faces = (
            np.asarray(payload["faces"], dtype=np.int32).reshape(-1, 3)
            if "faces" in payload.files
            else np.zeros((0, 3), dtype=np.int32)
        )
    return verts, faces


def _load_tpose_mesh(canonical_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (
        load_canonical_smpl,
    )

    if not Path(canonical_dir).is_dir():
        raise VesselPlanError("no capture", f"missing canonical {canonical_dir}")
    vertices, faces, _skeleton = load_canonical_smpl(canonical_dir)
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3), np.asarray(faces, dtype=np.int32).reshape(-1, 3)


def _project_label(
    label: str,
    *,
    repo: Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    atlas_path = resolve_atlas_path("right", repo=repo)
    if atlas_path is None:
        raise VesselPlanError("atlas missing", "atlas_right.npz not found on RealUS or Among_US")
    source = resolve_vessel_source(repo=repo)
    if source is None:
        raise VesselPlanError("atlas missing", "no vessel_material_coordinates.npz or centerlines obj")
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates import (
        load_leg_volume_atlas,
        project_vessel_centerlines_to_skin,
        remap_vessel_projection_to_skin,
    )

    atlases = {"right": load_leg_volume_atlas(atlas_path)}
    kind, src = source
    if kind == "remap":
        projection, lines = remap_vessel_projection_to_skin(src, atlases)
    else:
        projection, lines = project_vessel_centerlines_to_skin(src, atlases)
    if label not in lines:
        raise VesselPlanError("no R_SUPFEMV", f"{label} missing from projection")
    pts = np.asarray(lines[label], dtype=np.float32).reshape(-1, 3)
    h = None
    labels = np.asarray(projection.labels, dtype=object)
    mask = np.array([str(v) == label for v in labels.reshape(-1)])
    if mask.any() and projection.xi_skin.size:
        h = np.asarray(projection.xi_skin, dtype=float).reshape(-1, 3)[mask, 1]
        if h.size != pts.shape[0]:
            h = None
    return pts, h


def build_vessel_plan(
    *,
    run_dir: Path,
    moment_dir: Path | None = None,
    canonical_dir: Path | None = None,
    repo: Path | None = None,
    label: str = VESSEL_LABEL,
    window_m: float = WINDOW_M,
    euler_order: str = "xyz",
) -> VesselPlan:
    root = repo if repo is not None else repo_root()
    run_dir = Path(run_dir)
    moment = Path(moment_dir) if moment_dir is not None else run_dir / "moment_0000"
    can_dir = Path(canonical_dir) if canonical_dir is not None else root / DEFAULT_CANONICAL
    npz_path = moment / "smplx_result.npz"
    posed, posed_faces = _load_posed_mesh(npz_path)
    tpose_pts, h = _project_label(label, repo=root)
    window = extract_vessel_window(tpose_pts, h=h, window_m=window_m)
    if float(polyline_arclength(window)[-1]) < 0.5 * float(window_m):
        raise VesselPlanError("no R_SUPFEMV", "window shorter than 5 cm")
    tpose_verts, tpose_faces = _load_tpose_mesh(can_dir)
    faces = posed_faces if posed_faces.size else tpose_faces
    if faces.size == 0:
        raise VesselPlanError("no capture", "mesh has no faces")
    if tpose_verts.shape[0] != posed.shape[0]:
        raise VesselPlanError("no capture", "canonical T-pose and smplx_result topology differ")
    world, face_idx, _bary = bind_tpose_to_posed(window, tpose_verts, posed, faces)
    tangents = _finite_tangents(world)
    normals = outward_skin_normals(posed, faces, world, face_idx)
    poses = poses_from_polyline(world, normals, tangents, euler_order=euler_order)
    contact = poses[0].copy()
    tcp = smplx_poses_to_tcp(poses, euler_order=euler_order, repo=root)
    return VesselPlan(
        path=run_dir / PLAN_NAME,
        label=label,
        run_name=run_dir.name,
        world_xyz=world,
        scan_tangent=tangents,
        skin_normals=normals,
        scan_poses=poses,
        contact_pose=contact,
        tcp_poses=tcp,
        tcp_contact=np.asarray(tcp[0], dtype=float).reshape(6),
        window_m=float(polyline_arclength(world)[-1]),
        standoff_m=STANDOFF_M,
    )


def write_plan_json(plan: VesselPlan, path: Path | None = None) -> Path:
    out = Path(path) if path is not None else plan.path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")
    return out


def write_plan_error(path: Path, reason: str, *, detail: str = "", run_name: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ok": False, "reason": str(reason), "detail": str(detail), "run_name": str(run_name)}
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def write_vessel_plan_for_run(
    *,
    run_dir: Path,
    moment_dir: Path | None = None,
    canonical_dir: Path | None = None,
    repo: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build or reuse ``vessel_plan.json``. Always writes a file (ok or refuse)."""
    run_dir = Path(run_dir)
    out = run_dir / PLAN_NAME
    if out.is_file() and not overwrite:
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, dict) and existing.get("ok"):
            return existing
    try:
        plan = build_vessel_plan(
            run_dir=run_dir,
            moment_dir=moment_dir,
            canonical_dir=canonical_dir,
            repo=repo,
        )
        write_plan_json(plan, out)
        return plan.to_json()
    except VesselPlanError as exc:
        write_plan_error(out, exc.reason, detail=str(exc), run_name=run_dir.name)
        return {"ok": False, "reason": exc.reason, "detail": str(exc), "path": str(out)}


def plan_from_json(raw: dict[str, Any], *, path: Path | None = None) -> VesselPlan:
    if not bool(raw.get("ok", True)):
        raise VesselPlanError(str(raw.get("reason") or "no capture"), str(raw.get("detail") or ""))
    xyz = np.asarray(raw["world_xyz"], dtype=float).reshape(-1, 3)
    poses = np.asarray(raw.get("scan_poses") or [], dtype=float)
    if poses.size == 0:
        nrm = np.asarray(raw["skin_normals"], dtype=float).reshape(-1, 3)
        tan = np.asarray(raw["scan_tangent"], dtype=float).reshape(-1, 3)
        poses = poses_from_polyline(xyz, nrm, tan)
    else:
        poses = poses.reshape(-1, 6)
        nrm = np.asarray(raw.get("skin_normals") or np.zeros_like(xyz), dtype=float).reshape(-1, 3)
        tan = np.asarray(raw.get("scan_tangent") or np.zeros_like(xyz), dtype=float).reshape(-1, 3)
    contact = np.asarray(raw.get("contact_pose") or poses[0], dtype=float).reshape(6)
    tcp_raw = np.asarray(raw.get("tcp_poses") or [], dtype=float)
    if tcp_raw.size:
        tcp = tcp_raw.reshape(-1, 6)
    else:
        tcp = smplx_poses_to_tcp(poses)
    tcp_contact = np.asarray(raw.get("tcp_contact") or tcp[0], dtype=float).reshape(6)
    return VesselPlan(
        path=Path(path) if path is not None else Path("."),
        label=str(raw.get("label") or VESSEL_LABEL),
        run_name=str(raw.get("run_name") or ""),
        world_xyz=xyz,
        scan_tangent=tan if tan.shape[0] == xyz.shape[0] else _finite_tangents(xyz),
        skin_normals=nrm if nrm.shape[0] == xyz.shape[0] else np.zeros_like(xyz),
        scan_poses=poses,
        contact_pose=contact,
        tcp_poses=tcp,
        tcp_contact=tcp_contact,
        window_m=float(raw.get("window_m") or polyline_arclength(xyz)[-1]),
        standoff_m=float(raw.get("standoff_m") or STANDOFF_M),
    )


def load_plan_file(path: Path) -> VesselPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return plan_from_json(raw, path=Path(path))


def latest_plan_path(repo: Path | None = None) -> Path | None:
    root = smplx_output_root(repo)
    if not root.is_dir():
        return None
    plans = [p for p in root.glob(f"*/{PLAN_NAME}") if p.is_file()]
    if not plans:
        return None
    plans.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return plans[0]


def latest_moment_dir(repo: Path | None = None) -> Path | None:
    root = smplx_output_root(repo)
    if not root.is_dir():
        return None
    npzs = [p for p in root.glob("*/moment_0000/smplx_result.npz") if p.is_file()]
    if not npzs:
        return None
    npzs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return npzs[0].parent


def diagnose_missing_plan(*, repo: Path | None = None) -> str:
    root = repo if repo is not None else repo_root()
    if latest_moment_dir(root) is None:
        return "no capture"
    if resolve_atlas_path("right", repo=root) is None:
        return "atlas missing"
    return "no capture"


def write_vessel_plan_after_fit(
    *,
    run_dir: Path,
    moment_dir: Path | None = None,
    canonical_dir: Path | None = None,
    repo: Path | None = None,
    gender: str = "male",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write the plan as soon as ``smplx_result.npz`` exists (before overlay PNGs)."""
    root = repo if repo is not None else repo_root()
    run_dir = Path(run_dir)
    moment = Path(moment_dir) if moment_dir is not None else run_dir / "moment_0000"
    can_dir = Path(canonical_dir) if canonical_dir is not None else root / DEFAULT_CANONICAL
    npz_path = moment / "smplx_result.npz"
    if not npz_path.is_file():
        write_plan_error(run_dir / PLAN_NAME, "no capture", detail="smplx_result.npz missing", run_name=run_dir.name)
        return {"ok": False, "reason": "no capture"}
    try:
        from projects.genesis_ue_sync.anatomy_retarget.canonical_export import (
            export_canonical_tpose,
            load_betas,
        )

        export_canonical_tpose(
            betas=load_betas(npz_path),
            output_dir=can_dir,
            staging_dir=None,
            gender=str(gender),
            device="cpu",
            source=str(run_dir.resolve()),
        )
    except Exception:
        pass
    return write_vessel_plan_for_run(
        run_dir=run_dir,
        moment_dir=moment,
        canonical_dir=can_dir,
        repo=root,
        overwrite=overwrite,
    )


def load_ready_plan(*, repo: Path | None = None, build_if_missing: bool = False) -> tuple[VesselPlan | None, str]:
    """Return ``(plan, "")`` or ``(None, refuse_reason)``."""
    path = latest_plan_path(repo)
    if path is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            raw = None
        if isinstance(raw, dict) and bool(raw.get("ok", True)):
            try:
                return plan_from_json(raw, path=path), ""
            except VesselPlanError as exc:
                return None, exc.reason
        if isinstance(raw, dict) and not bool(raw.get("ok", True)) and not build_if_missing:
            return None, str(raw.get("reason") or "no capture")
    if not build_if_missing:
        return None, diagnose_missing_plan(repo=repo)
    moment = latest_moment_dir(repo)
    if moment is None:
        return None, "no capture"
    info = write_vessel_plan_after_fit(run_dir=moment.parent, moment_dir=moment, repo=repo)
    if not bool(info.get("ok")):
        return None, str(info.get("reason") or "no capture")
    return load_ready_plan(repo=repo, build_if_missing=False)
