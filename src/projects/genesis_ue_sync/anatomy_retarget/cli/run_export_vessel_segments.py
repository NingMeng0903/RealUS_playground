"""Export artery/vein planning segments and skeleton OBJs from the retargeted anatomy asset.

Default output layout under ``outputs/anatomy_retarget/limb_vessel_planning/``:

  bone_segments/     per-bone rest + posed OBJ
  vessel_segments/   artery/vein segment OBJ (rest + posed)
  centerlines/       named centerline polylines
  figures/           overlap / leg zoom / body overlay PNG
  planning_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_points, skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.obj_io import read_obj_vertices, write_obj
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.viz_overlay import (
    draw_vein_on_body_pose_figure,
    pin_centerline_junctions,
    sparse_leg_bone_vertices,
)

DEFAULT_ASSET = Path("outputs/anatomy_retarget/latest_asset/anatomy_rigged.npz")
DEFAULT_MOTION = Path("outputs/offline_capture/20260611_180757/moment_0000/smplx_result.npz")
DEFAULT_OUTPUT = Path("outputs/anatomy_retarget/limb_vessel_planning")
DEFAULT_CANONICAL = Path("outputs/anatomy_retarget/latest_canonical")

# Thigh/calf bones shown as sparse interior markers in the bone-overlay figure.
LEG_BONE_VIZ_MESHES = frozenset(
    {"Femur_L", "Femur_R", "Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R", "Patella_L", "Patella_R"}
)

_SKELETON_MESH_EXACT = (
    {f"C{i}" for i in range(1, 8)}
    | {f"T{i}" for i in range(1, 13)}
    | {f"L{i}" for i in range(1, 6)}
    | {"C1_Atlas", "C2_Axis", "Sternum", "Sacrum", "Mandible", "Hyoid_Bone", "Upper_Skull"}
)

_SKELETON_MESH_KEYWORDS = (
    "Femur",
    "Tibia",
    "Fibula",
    "Patella",
    "Humerus",
    "Radius",
    "Ulna",
    "Clavicle",
    "Scapula",
    "Ilium",
    "Calcaneus",
    "Talus",
    "Navicular",
    "Cuboid",
    "Cuneiform",
    "Hamate",
    "Capitate",
    "Lunate",
    "Scaphoid",
    "Trapezium",
    "Trapezoid",
    "Triquetrum",
    "Pisiform",
    "Metacarpal",
    "Metatarsal",
    "Phalanx",
    "Phalanges",
    "Rib_",
)

_SKELETON_MESH_SKIP = (
    "UNCUT_",
    "Nerve",
    "Artery",
    "Vein",
    "Ligament",
    "Disc_",
    "Duct",
    "Gland",
    "Lobe",
    "Kidney",
    "Heart",
    "Lung",
    "Liver",
    "Stomach",
    "Intestine",
    "Canine",
    "Incisor",
    "Molar",
    "Premolar",
    "Cornea",
    "Iris",
    "Lens",
    "Cerebellum",
    "Callosum",
    "Amygdala",
    "Hippocampus",
    "Ventricles",
    "Bladder",
    "Esophagus",
    "Trachea",
    "Pharynx",
    "Larynx",
    "Diaphragm",
    "Appendix",
    "Pancreas",
    "Gallbladder",
    "Spleen",
    "Fornix",
    "Pons",
    "Thalamus",
    "Midbrain",
    "Optic",
    "Olfactory",
    "Pituitary",
    "Thyroid",
    "Autonomic",
    "Facial_Nerves",
    "Costal_Cartilage",
    "Parotid",
    "Sublingual",
    "Submandibular",
    "Ureter",
    "Urethra",
    "Spinal_Cord",
    "Adrenal",
    "Basal_Ganglia",
    "Frontal_Lobe",
    "Occipital_Lobe",
    "Parietal_Lobe",
    "Temporal_Lobe",
)

SEGMENT_COLORS: dict[str, tuple[int, int, int]] = {
    "ARTERY": (220, 32, 32),
    "VEIN_UNLABELED": (120, 150, 255),
    "L_COM_FEM_V": (0, 40, 255),
    "L_DEEP_FEM_V": (255, 85, 0),
    "L_SAPH_V": (255, 0, 180),
    "L_SUPFEMV": (0, 210, 255),
    "L_POPV": (30, 210, 110),
    "L_POST_TIB_V": (245, 200, 45),
    "L_PERONEAL_V": (210, 70, 230),
    "R_COM_FEM_V": (0, 25, 185),
    "R_DEEP_FEM_V": (190, 55, 0),
    "R_SAPH_V": (190, 0, 135),
    "R_SUPFEMV": (0, 155, 210),
    "R_POPV": (20, 150, 80),
    "R_POST_TIB_V": (190, 145, 25),
    "R_PERONEAL_V": (150, 45, 170),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-npz", type=Path, default=DEFAULT_ASSET)
    p.add_argument("--motion-npz", type=Path, default=DEFAULT_MOTION)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL)
    return p.parse_args()


def _source_range(raw: "np.lib.npyio.NpzFile", name: str) -> tuple[int, int]:
    names = [str(v) for v in raw["source_mesh_names"].reshape(-1).tolist()]
    if name not in names:
        raise KeyError(f"source mesh not found: {name}")
    idx = names.index(name)
    s, e = np.asarray(raw["source_vertex_ranges"], dtype=np.int64).reshape(-1, 2)[idx]
    return int(s), int(e)


def _faces_in_range(faces: np.ndarray, start: int, end: int) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    mask = np.all((f >= int(start)) & (f < int(end)), axis=1)
    return f[mask]


def _write_subset_obj(path: Path, vertices: np.ndarray, faces_global: np.ndarray, *, comment: str) -> int:
    if faces_global.size == 0:
        return 0
    unique = np.unique(faces_global.reshape(-1))
    remap = {int(v): i for i, v in enumerate(unique.tolist())}
    local_faces = np.vectorize(lambda x: remap[int(x)], otypes=[np.int32])(faces_global)
    write_obj(path, np.asarray(vertices, dtype=np.float32)[unique], local_faces, comment=comment)
    return int(local_faces.shape[0])


def _write_centerline_obj(path: Path, centerlines: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vessel centerlines; each l statement is one named branch\n")
        offset = 1
        for label, pts in centerlines.items():
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            if arr.shape[0] < 2:
                continue
            handle.write(f"o {label}\n")
            for p in arr:
                handle.write(f"v {float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
            handle.write("l " + " ".join(str(i) for i in range(offset, offset + arr.shape[0])) + "\n")
            offset += arr.shape[0]


def _side_prefix(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, "", dtype=object)
    out[x >= 0.0] = "L"
    out[x < 0.0] = "R"
    return out


def _joint(asset_joint_names: list[str], rest_joints: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(rest_joints, dtype=np.float32)[asset_joint_names.index(name)]


def _connected_components(faces: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Connected components over the mesh graph restricted to masked vertices."""
    vertex_count = int(mask.shape[0])
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for tri in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        a, b, c = [int(v) for v in tri.tolist()]
        if not (mask[a] and mask[b] and mask[c]):
            continue
        adjacency[a].extend((b, c))
        adjacency[b].extend((a, c))
        adjacency[c].extend((a, b))
    seen = np.zeros(vertex_count, dtype=bool)
    out: list[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        stack = [int(start)]
        seen[start] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        if comp:
            out.append(np.asarray(comp, dtype=np.int64))
    return out


def _nearest_core_labels(points: np.ndarray, sup_core: np.ndarray, deep_core: np.ndarray) -> np.ndarray:
    """Return True where points are closer to the sup-femoral branch core."""
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    if sup_core.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    if deep_core.size == 0:
        return np.ones(points.shape[0], dtype=bool)
    try:
        from scipy.spatial import cKDTree

        sup_dist, _ = cKDTree(sup_core).query(points, k=1)
        deep_dist, _ = cKDTree(deep_core).query(points, k=1)
    except Exception:
        sup_dist = np.sqrt(np.min(np.sum((points[:, None, :] - sup_core[None, :, :]) ** 2, axis=2), axis=1))
        deep_dist = np.sqrt(np.min(np.sum((points[:, None, :] - deep_core[None, :, :]) ** 2, axis=2), axis=1))
    return sup_dist <= deep_dist


def _nearest_three_way(
    points: np.ndarray,
    sup_core: np.ndarray,
    deep_core: np.ndarray,
    saph_core: np.ndarray,
) -> np.ndarray:
    """Nearest branch core index: 0=supfem, 1=deep fem, 2=saphenous."""
    cores = [sup_core, deep_core, saph_core]
    dists: list[np.ndarray] = []
    try:
        from scipy.spatial import cKDTree

        for core in cores:
            if core.size == 0:
                dists.append(np.full(points.shape[0], np.inf, dtype=np.float32))
            else:
                d, _ = cKDTree(core).query(points, k=1)
                dists.append(np.asarray(d, dtype=np.float32))
    except Exception:
        for core in cores:
            if core.size == 0:
                dists.append(np.full(points.shape[0], np.inf, dtype=np.float32))
            else:
                dists.append(
                    np.sqrt(np.min(np.sum((points[:, None, :] - core[None, :, :]) ** 2, axis=2), axis=1))
                )
    return np.argmin(np.stack(dists, axis=1), axis=1)


def _kmeans2(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(points, dtype=np.float32).reshape(-1, points.shape[-1])
    if X.shape[0] < 2:
        return np.zeros(X.shape[0], dtype=np.int64), X.copy()
    Xc = X - X.mean(axis=0)
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    score = Xc @ vh[0]
    centers = np.stack([X[int(np.argmin(score))], X[int(np.argmax(score))]])
    labels = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(30):
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = np.stack(
            [X[labels == idx].mean(axis=0) if np.any(labels == idx) else centers[idx] for idx in range(2)]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def _smooth_polyline(line: np.ndarray, *, window: int = 5, passes: int = 1, pin_ends: bool = False, pin_tail: int = 0) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 2:
        return pts.copy()
    w = max(3, int(window) | 1)
    half = w // 2
    tail = max(0, min(int(pin_tail), pts.shape[0] - 2))
    out = pts.copy()
    for _ in range(max(1, int(passes))):
        nxt = out.copy()
        stop = out.shape[0] - tail if tail else out.shape[0] - 1
        for i in range(1, stop):
            lo = max(0, i - half)
            hi = min(out.shape[0], i + half + 1)
            nxt[i] = out[lo:hi].mean(axis=0)
        if pin_ends:
            nxt[0] = pts[0]
            nxt[-1] = pts[-1]
        if tail:
            nxt[-tail:] = pts[-tail:]
        out = nxt
    return out.astype(np.float32)


def _resample_polyline(line: np.ndarray, *, target: int = 14, keep_distal: int = 0) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    keep = max(0, min(int(keep_distal), pts.shape[0] - 2))
    if keep:
        core = pts[:-keep]
        tail = pts[-keep:]
        core_target = max(2, int(target) - keep + 1)
        if core.shape[0] <= 2 or core.shape[0] <= core_target:
            core_rs = core.copy()
        else:
            core_rs = _resample_polyline(core, target=core_target)
        return np.vstack([core_rs, tail]).astype(np.float32)
    if pts.shape[0] <= 2 or pts.shape[0] <= int(target):
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1.0e-8:
        return pts.copy()
    samples = np.linspace(0.0, total, int(target))
    out: list[np.ndarray] = []
    j = 0
    for s in samples:
        while j + 1 < len(cum) and float(cum[j + 1]) < float(s):
            j += 1
        if j + 1 >= len(cum):
            out.append(pts[-1])
            continue
        alpha = (float(s) - float(cum[j])) / max(float(cum[j + 1] - cum[j]), 1.0e-8)
        out.append(((1.0 - alpha) * pts[j] + alpha * pts[j + 1]).astype(np.float32))
    return np.stack(out, axis=0).astype(np.float32)


def _distal_heel_tip(
    branch_pts: np.ndarray,
    *,
    proximal_ref: np.ndarray,
    ankle: np.ndarray,
) -> np.ndarray:
    """Distal vessel tip on the posterior heel from mesh (not the ankle joint)."""
    pts = np.asarray(branch_pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 4:
        return np.asarray(ankle, dtype=np.float32).reshape(3)
    o = np.asarray(proximal_ref, dtype=np.float32).reshape(3)
    a = np.asarray(ankle, dtype=np.float32).reshape(3)
    axis = a - o
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-6:
        axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        axis = axis / norm
    t = (pts - o) @ axis
    span = max(float(t.max() - t.min()), 1.0e-5)
    t_cut = float(np.quantile(t, 0.92))
    band = max(0.012, 0.05 * span)
    distal_mask = t >= t_cut - band
    distal = pts[distal_mask] if np.any(distal_mask) else pts[np.argsort(t)[-max(3, pts.shape[0] // 8):]]
    radial = distal - a.reshape(1, 3)
    radial = radial - np.outer(radial @ axis, axis)
    tip_idx = int(np.argmax(np.linalg.norm(radial, axis=1)))
    return distal[tip_idx].astype(np.float32)


def _order_polyline_along_axis(
    raw: np.ndarray,
    origin: np.ndarray,
    distal_ref: np.ndarray,
) -> np.ndarray:
    """Sort polyline samples monotonically from origin toward distal_ref."""
    pts = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts.copy()
    o = np.asarray(origin, dtype=np.float32).reshape(3)
    axis = np.asarray(distal_ref, dtype=np.float32).reshape(3) - o
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-6:
        return pts[np.argsort(pts[:, 1])]
    axis = axis / norm
    return pts[np.argsort((pts - o) @ axis)]


def _centerline_from_branch(points: np.ndarray, *, bins: int = 12) -> np.ndarray:
    """Collapse tube surface points to a smooth centerline along the branch axis."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] <= 6:
        return np.asarray([pts.mean(axis=0)], dtype=np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    t = centered @ axis
    t_min, t_max = float(t.min()), float(t.max())
    if abs(t_max - t_min) < 1.0e-5:
        return np.asarray([mean], dtype=np.float32)
    edges = np.linspace(t_min, t_max, max(3, int(bins)) + 1)
    raw: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (t >= lo) & (t <= hi if hi == edges[-1] else t < hi)
        if not np.any(mask):
            continue
        raw.append(np.median(pts[mask], axis=0).astype(np.float32))
    if len(raw) < 2:
        return np.stack(raw, axis=0) if raw else np.asarray([mean], dtype=np.float32)
    return np.stack(raw, axis=0)


def _branch_centerline(
    points: np.ndarray,
    *,
    bins: int,
    junction: np.ndarray | None = None,
    junction_at_start: bool = False,
    distal_ref: np.ndarray | None = None,
    distal_tip: np.ndarray | None = None,
    smooth_window: int = 5,
    smooth_passes: int = 2,
    resample_target: int = 14,
) -> np.ndarray:
    raw = _centerline_from_branch(points, bins=bins)
    if junction is None:
        line = _smooth_polyline(raw, window=smooth_window, passes=smooth_passes)
        line = _resample_polyline(line, target=max(6, min(20, resample_target)))
        if distal_tip is not None:
            line[-1] = np.asarray(distal_tip, dtype=np.float32).reshape(3)
        return line
    j = np.asarray(junction, dtype=np.float32).reshape(3)
    if raw.shape[0] >= 2:
        dref = (
            np.asarray(distal_ref, dtype=np.float32).reshape(3)
            if distal_ref is not None
            else raw[int(np.argmax(np.linalg.norm(raw - j, axis=1)))]
        )
        ordered = _order_polyline_along_axis(raw, j, dref)
        raw = ordered if junction_at_start else ordered[::-1].copy()
    line = _append_junction(raw, j, prepend=junction_at_start)
    if line.shape[0] >= 2:
        if junction_at_start:
            if float(np.linalg.norm(line[0] - j)) > float(np.linalg.norm(line[-1] - j)):
                line = line[::-1].copy()
            line[0] = j
        else:
            if float(np.linalg.norm(line[-1] - j)) > float(np.linalg.norm(line[0] - j)):
                line = line[::-1].copy()
            line[-1] = j
    line = _smooth_polyline(
        line,
        window=smooth_window,
        passes=smooth_passes,
        pin_ends=True,
        pin_tail=4 if distal_tip is not None else 0,
    )
    line = _resample_polyline(
        line,
        target=resample_target,
        keep_distal=4 if distal_tip is not None else 0,
    )
    if junction_at_start:
        line[0] = j
    else:
        line[-1] = j
    if distal_tip is not None:
        line[-1] = np.asarray(distal_tip, dtype=np.float32).reshape(3)
    return line


def _branch_centerline_two_junctions(
    points: np.ndarray,
    *,
    bins: int,
    start_j: np.ndarray,
    end_j: np.ndarray,
    smooth_window: int = 5,
    smooth_passes: int = 2,
    resample_target: int = 16,
) -> np.ndarray:
    """Centerline pinned at both ends (e.g. hip junction -> knee junction)."""
    sj = np.asarray(start_j, dtype=np.float32).reshape(3)
    ej = np.asarray(end_j, dtype=np.float32).reshape(3)
    raw = _centerline_from_branch(points, bins=bins)
    if raw.shape[0] == 0:
        return np.stack([sj, ej], axis=0)
    raw = _order_polyline_along_axis(raw, sj, ej)
    line = np.concatenate([[sj], raw, [ej]], axis=0)
    line = _smooth_polyline(line, window=smooth_window, passes=smooth_passes, pin_ends=True)
    line = _resample_polyline(line, target=resample_target)
    line[0] = sj
    line[-1] = ej
    return line


def _distance_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    line = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if line.shape[0] == 0:
        return np.full(pts.shape[0], np.inf, dtype=np.float32)
    if line.shape[0] == 1:
        return np.linalg.norm(pts - line[0], axis=1)
    best = np.full(pts.shape[0], np.inf, dtype=np.float32)
    for a, b in zip(line[:-1], line[1:], strict=True):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1.0e-10:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(pts - proj, axis=1)
        best = np.minimum(best, d)
    return best


def _append_junction(line: np.ndarray, junction: np.ndarray, *, prepend: bool) -> np.ndarray:
    arr = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    j = np.asarray(junction, dtype=np.float32).reshape(1, 3)
    if arr.shape[0] == 0:
        return j
    if prepend:
        return np.concatenate([j, arr], axis=0)
    return np.concatenate([arr, j], axis=0)


def _closest_on_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    poly = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if poly.shape[0] == 0:
        return pts.copy()
    if poly.shape[0] == 1:
        return np.repeat(poly, pts.shape[0], axis=0)
    out = np.zeros_like(pts)
    for i, p in enumerate(pts):
        best_d = np.inf
        best_q = poly[0]
        for a, b in zip(poly[:-1], poly[1:], strict=True):
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1.0e-10:
                q = a
            else:
                t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
                q = a + t * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d = d
                best_q = q
        out[i] = best_q
    return out.astype(np.float32)


def _trunk_arc_lengths(line: np.ndarray) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return np.zeros(pts.shape[0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)


def _trunk_point_at_arc(line: np.ndarray, arc_m: float) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    cum = _trunk_arc_lengths(pts)
    total = float(cum[-1])
    if total < 1.0e-8:
        return pts[0].copy()
    s = float(np.clip(arc_m, 0.0, total))
    j = int(np.searchsorted(cum, s, side="right") - 1)
    j = int(np.clip(j, 0, pts.shape[0] - 2))
    alpha = (s - float(cum[j])) / max(float(cum[j + 1] - cum[j]), 1.0e-8)
    return ((1.0 - alpha) * pts[j] + alpha * pts[j + 1]).astype(np.float32)


def _project_points_to_polyline_arc(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Project points onto a polyline and return nearest arc-length coordinates."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    poly = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if poly.shape[0] <= 1:
        return np.zeros(pts.shape[0], dtype=np.float32)
    cum = _trunk_arc_lengths(poly)
    best_d2 = np.full(pts.shape[0], np.inf, dtype=np.float32)
    best_arc = np.zeros(pts.shape[0], dtype=np.float32)
    for seg_idx, (a, b) in enumerate(zip(poly[:-1], poly[1:], strict=True)):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1.0e-10:
            t = np.zeros(pts.shape[0], dtype=np.float32)
            q = a.reshape(1, 3)
        else:
            t = np.clip(((pts - a.reshape(1, 3)) @ ab) / denom, 0.0, 1.0).astype(np.float32)
            q = a.reshape(1, 3) + t[:, None] * ab.reshape(1, 3)
        d2 = np.sum(np.square(pts - q), axis=1)
        update = d2 < best_d2
        best_d2[update] = d2[update]
        best_arc[update] = float(cum[seg_idx]) + t[update] * float(cum[seg_idx + 1] - cum[seg_idx])
    return best_arc.astype(np.float32)


def _centerline_from_deformed_label_vertices(
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    labels: np.ndarray,
    rest_centerlines: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract posed centerlines from the deformed Blender mesh using rest labels.

    Segment labels and ordering are defined in T-pose. For posed export, the
    corresponding labeled mesh vertices have already been skinned by Blender LBS
    weights, so we only collapse those posed tube vertices into center samples.
    """
    rest = np.asarray(rest_vertices, dtype=np.float32).reshape(-1, 3)
    posed = np.asarray(posed_vertices, dtype=np.float32).reshape(-1, 3)
    label_arr = np.asarray(labels, dtype=object).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for label, rest_line_raw in rest_centerlines.items():
        rest_line = np.asarray(rest_line_raw, dtype=np.float32).reshape(-1, 3)
        if rest_line.shape[0] < 2:
            continue
        idx = np.flatnonzero(label_arr == label)
        if idx.size < 4:
            continue
        rest_pts = rest[idx]
        posed_pts = posed[idx]
        vertex_arc = _project_points_to_polyline_arc(rest_pts, rest_line)
        line_arc = _trunk_arc_lengths(rest_line)
        samples: list[np.ndarray] = []
        for i, s in enumerate(line_arc.tolist()):
            if i == 0:
                lo = -np.inf
            else:
                lo = 0.5 * (float(line_arc[i - 1]) + float(s))
            if i + 1 == len(line_arc):
                hi = np.inf
            else:
                hi = 0.5 * (float(s) + float(line_arc[i + 1]))
            in_bin = (vertex_arc >= lo) & (vertex_arc <= hi)
            if np.count_nonzero(in_bin) < 3:
                nearest = np.argsort(np.abs(vertex_arc - float(s)))[: min(12, idx.size)]
                sample_pts = posed_pts[nearest]
            else:
                sample_pts = posed_pts[in_bin]
            samples.append(np.median(sample_pts, axis=0).astype(np.float32))
        line = np.stack(samples, axis=0).astype(np.float32)
        if line.shape[0] > 4:
            line = _smooth_polyline(line, window=3, passes=1, pin_ends=False)
        out[label] = line
    return out


def _smooth_posed_centerlines_for_export(centerlines: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove bin-to-bin jitter from posed mesh-derived lines without moving junctions."""
    out: dict[str, np.ndarray] = {}
    distal_preserve = ("POST_TIB_V", "PERONEAL_V")
    for label, line in centerlines.items():
        arr = np.asarray(line, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] < 5:
            out[label] = arr.copy()
            continue
        keep_tail = 3 if any(token in label for token in distal_preserve) and arr.shape[0] >= 8 else 0
        window = 5 if arr.shape[0] >= 8 else 3
        passes = 2 if arr.shape[0] >= 8 else 1
        out[label] = _smooth_polyline(
            arr,
            window=window,
            passes=passes,
            pin_ends=True,
            pin_tail=keep_tail,
        )
    return out


def _nearest_trunk_index(line: np.ndarray, point: np.ndarray) -> int:
    poly = np.asarray(line, dtype=np.float32).reshape(-1, 3)
    p = np.asarray(point, dtype=np.float32).reshape(3)
    return int(np.argmin(np.linalg.norm(poly - p[None, :], axis=1)))


def _branch_attachment_on_trunk(
    branch_pts: np.ndarray,
    trunk_line: np.ndarray,
    *,
    proximal_quantile: float = 0.88,
) -> np.ndarray | None:
    """Estimate where a side branch leaves the femoral trunk (mesh-derived)."""
    bp = np.asarray(branch_pts, dtype=np.float32).reshape(-1, 3)
    if bp.shape[0] < 4:
        return None
    y_cut = float(np.quantile(bp[:, 1], proximal_quantile))
    proximal = bp[bp[:, 1] >= y_cut]
    if proximal.shape[0] < 3:
        proximal = bp[bp[:, 1] >= float(np.quantile(bp[:, 1], 0.80))]
    if proximal.shape[0] < 3:
        return None
    attach = _closest_on_polyline(proximal, trunk_line)
    return np.median(attach, axis=0).astype(np.float32)


def _slice_branch_origins(
    verts: np.ndarray,
    *,
    trunk_mask: np.ndarray,
    saph_mask: np.ndarray,
    deep_mask: np.ndarray,
    confluence_y: float,
    deep_fem_end_y: float,
    thigh: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Scan Y slices on the femoral tube to find where SAPH then DEEP dominate."""
    band = 0.012 * thigh
    ys = np.linspace(float(confluence_y), float(deep_fem_end_y), 48)
    saph_scores: list[tuple[float, float]] = []
    deep_scores: list[tuple[float, float]] = []
    for y in ys:
        slab = trunk_mask & (np.abs(verts[:, 1] - y) < band)
        if int(np.count_nonzero(slab)) < 6:
            continue
        saph_n = int(np.count_nonzero(slab & saph_mask))
        deep_n = int(np.count_nonzero(slab & deep_mask))
        trunk_n = max(int(np.count_nonzero(slab)) - saph_n - deep_n, 1)
        saph_scores.append((float(y), float(saph_n) / float(trunk_n + saph_n + deep_n)))
        deep_scores.append((float(y), float(deep_n) / float(trunk_n + saph_n + deep_n)))

    def _pick_peak(scores: list[tuple[float, float]], *, above_y: float | None = None) -> np.ndarray | None:
        if not scores:
            return None
        filtered = [(y, s) for y, s in scores if above_y is None or y < above_y - band]
        if not filtered:
            filtered = scores
        y_peak = max(filtered, key=lambda item: item[1])[0]
        slab = trunk_mask & (np.abs(verts[:, 1] - y_peak) < band)
        if not np.any(slab):
            return np.asarray([0.0, y_peak, 0.0], dtype=np.float32)
        med = np.median(verts[slab], axis=0).astype(np.float32)
        med[1] = y_peak
        return med

    saph_y = _pick_peak(saph_scores)
    deep_y = _pick_peak(deep_scores, above_y=float(saph_y[1]) if saph_y is not None else None)
    return saph_y, deep_y


def _sequential_femoral_junctions(
    verts: np.ndarray,
    *,
    com_mask: np.ndarray,
    sup_mask: np.ndarray,
    saph_mask: np.ndarray,
    deep_mask: np.ndarray,
    trunk_mask: np.ndarray,
    confluence_y: float,
    deep_fem_end_y: float,
    thigh: float,
    hip: np.ndarray,
    knee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect two sequential hip branches: SAPH first, DEEP slightly distal on the trunk."""
    trunk_pts = verts[com_mask | sup_mask]
    if trunk_pts.shape[0] < 8:
        fb = np.asarray([float(hip[0]), confluence_y, float(hip[2])], dtype=np.float32)
        return fb, fb

    trunk_line = _centerline_from_branch(trunk_pts, bins=28)
    trunk_line = _order_polyline_along_axis(trunk_line, np.asarray(hip, dtype=np.float32).reshape(3), np.asarray(knee, dtype=np.float32).reshape(3))

    saph_j = _branch_attachment_on_trunk(verts[saph_mask], trunk_line)
    deep_j = None
    if saph_j is not None:
        deep_branch = verts[deep_mask & (verts[:, 1] < float(saph_j[1]))]
        deep_j = _branch_attachment_on_trunk(deep_branch, trunk_line, proximal_quantile=0.90)
    if deep_j is None:
        deep_j = _branch_attachment_on_trunk(verts[deep_mask], trunk_line)

    slice_saph, slice_deep = _slice_branch_origins(
        verts,
        trunk_mask=trunk_mask,
        saph_mask=saph_mask,
        deep_mask=deep_mask,
        confluence_y=confluence_y,
        deep_fem_end_y=deep_fem_end_y,
        thigh=thigh,
    )
    if saph_j is None:
        saph_j = slice_saph
    elif slice_saph is not None:
        saph_j = (0.55 * saph_j + 0.45 * slice_saph).astype(np.float32)
    if deep_j is None:
        deep_j = slice_deep
    elif slice_deep is not None:
        deep_j = (0.55 * deep_j + 0.45 * slice_deep).astype(np.float32)

    if saph_j is None:
        saph_j = trunk_line[0].copy()
    if deep_j is None:
        deep_j = trunk_line[min(len(trunk_line) - 1, max(2, len(trunk_line) // 5))].copy()

    saph_j = _closest_on_polyline(saph_j.reshape(1, 3), trunk_line)[0]
    deep_j = _closest_on_polyline(deep_j.reshape(1, 3), trunk_line)[0]

    if deep_j[1] >= saph_j[1]:
        idx_saph = _nearest_trunk_index(trunk_line, saph_j)
        arc = _trunk_arc_lengths(trunk_line)
        span = float(arc[-1] - arc[idx_saph])
        if span > 1.0e-5:
            deep_j = _trunk_point_at_arc(trunk_line, float(arc[idx_saph] + 0.28 * span))
        else:
            idx_deep = min(len(trunk_line) - 1, idx_saph + max(2, len(trunk_line) // 8))
            deep_j = trunk_line[idx_deep].copy()

    return saph_j.astype(np.float32), deep_j.astype(np.float32)


def _junction_at_y(
    verts: np.ndarray,
    mask: np.ndarray,
    junction_y: float,
    *,
    band_scale: float,
    fallback: np.ndarray,
    lock_y: bool = True,
) -> np.ndarray:
    """Shared junction from mesh vertices near a Y slice (stays inside the vessel tube)."""
    candidates = verts[mask]
    band = 0.040 * band_scale
    near = candidates[np.abs(candidates[:, 1] - junction_y) < band]
    if near.shape[0] >= 3:
        j = np.median(near, axis=0).astype(np.float32)
    elif candidates.shape[0] >= 3:
        j = np.median(candidates, axis=0).astype(np.float32)
    else:
        j = np.asarray(fallback, dtype=np.float32).reshape(3)
    if lock_y:
        j[1] = float(junction_y)
    return j


def _prelim_calf_side_masks(
    verts: np.ndarray,
    lower_limb: np.ndarray,
    *,
    prefix: str,
    knee: np.ndarray,
    ankle: np.ndarray,
    knee_y: float,
    ankle_y: float,
    thigh: float,
    calf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Medial/lateral calf split from mesh coordinates (search band only, not cut height)."""
    calf_search = lower_limb & (verts[:, 1] <= knee_y + 0.04 * thigh) & (verts[:, 1] >= ankle_y + 0.04 * calf)
    limb_x = (float(knee[0]) + float(ankle[0])) * 0.5
    if prefix == "L":
        pre_peroneal = calf_search & (verts[:, 0] > limb_x)
    else:
        pre_peroneal = calf_search & (verts[:, 0] < limb_x)
    pre_post = calf_search & ~pre_peroneal
    return pre_post, pre_peroneal


def _slice_calf_bifurcation_y(
    verts: np.ndarray,
    *,
    pre_post: np.ndarray,
    pre_peroneal: np.ndarray,
    lower_limb: np.ndarray,
    y_hi: float,
    y_lo: float,
    band: float,
) -> float | None:
    """Scan mesh Y slices for POP -> POST_TIB / PERONEAL separation near the knee."""
    ys = np.linspace(float(y_hi), float(y_lo), 40)
    scores: list[tuple[float, float]] = []
    for y in ys:
        post_s = verts[pre_post & (np.abs(verts[:, 1] - y) < band)]
        per_s = verts[pre_peroneal & (np.abs(verts[:, 1] - y) < band)]
        if post_s.shape[0] < 4 or per_s.shape[0] < 4:
            continue
        post_c = np.median(post_s, axis=0)
        per_c = np.median(per_s, axis=0)
        sep = float(np.linalg.norm(post_c - per_c))
        slab = verts[lower_limb & (np.abs(verts[:, 1] - y) < band)]
        if slab.shape[0] < 6:
            continue
        width = float(np.sqrt(np.var(slab[:, 0]) + np.var(slab[:, 2]))) + 1.0e-4
        scores.append((float(y), sep / width))
    if not scores:
        return None
    peak = max(s for _, s in scores)
    threshold = 0.32 * peak
    for y, score in scores:
        if score >= threshold:
            return y
    return max(scores, key=lambda item: item[1])[0]


def _slice_sup_pop_transition_y(
    verts: np.ndarray,
    *,
    lower_limb: np.ndarray,
    knee_y: float,
    thigh: float,
    calf: float,
    pre_post: np.ndarray,
    pre_peroneal: np.ndarray,
) -> float | None:
    """Scan mesh Y slices for SUPFEM -> POP transition in a tight knee-adjacent band."""
    band = 0.011 * max(thigh, calf)
    y_hi = float(knee_y) + 0.08 * thigh
    y_lo = float(knee_y) + 0.02 * thigh
    ys = np.linspace(y_hi, y_lo, 32)
    scores: list[tuple[float, float]] = []
    for y in ys:
        slab = verts[lower_limb & (np.abs(verts[:, 1] - y) < band)]
        if slab.shape[0] < 8:
            continue
        z_med = float(np.median(slab[:, 2]))
        posterior_mass = float(np.mean(slab[:, 2] >= z_med))
        below_post = int(np.count_nonzero(pre_post & (verts[:, 1] < y - band)))
        below_per = int(np.count_nonzero(pre_peroneal & (verts[:, 1] < y - band)))
        bifurcation_hint = 1.0 if (below_post > 12 and below_per > 12) else 0.0
        xz = slab[:, [0, 2]].astype(np.float32)
        sep_ratio = 0.0
        if xz.shape[0] >= 12:
            labels, centers = _kmeans2(xz)
            if np.count_nonzero(labels == 0) >= 4 and np.count_nonzero(labels == 1) >= 4:
                width = float(np.sqrt(np.var(slab[:, 0]) + np.var(slab[:, 2]))) + 1.0e-4
                sep_ratio = min(float(np.linalg.norm(centers[0] - centers[1])) / width, 1.5)
        score = posterior_mass * 0.55 + bifurcation_hint * 0.25 + sep_ratio * 0.20
        scores.append((float(y), score))
    if not scores:
        return None
    return max(scores, key=lambda item: item[1])[0]


def _clamp_pop_bounds(
    pop_upper_y: float,
    pop_lower_y: float,
    *,
    knee_y: float,
    thigh: float,
    calf: float,
) -> tuple[float, float]:
    """Keep popliteal segment short: mesh-derived Y clamped to a knee-local window."""
    upper_lo = float(knee_y) + 0.025 * thigh
    upper_hi = float(knee_y) + 0.075 * thigh
    lower_hi = float(knee_y) - 0.020 * calf
    lower_lo = float(knee_y) - 0.110 * calf
    upper = float(np.clip(pop_upper_y, upper_lo, upper_hi))
    lower = float(np.clip(pop_lower_y, lower_lo, lower_hi))
    min_span = 0.012 * max(thigh, calf)
    if lower >= upper - min_span:
        upper = float(knee_y) + 0.055 * thigh
        lower = float(knee_y) - 0.075 * calf
    return upper, lower


def _detect_knee_pop_bounds(
    verts: np.ndarray,
    *,
    lower_limb: np.ndarray,
    confluence_y: float,
    knee_y: float,
    ankle_y: float,
    thigh: float,
    calf: float,
    prefix: str,
    knee: np.ndarray,
    ankle: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    """Mesh-derived SUP/POP and POP/calf Y bounds; joint Y only defines the search window."""
    pre_post, pre_peroneal = _prelim_calf_side_masks(
        verts,
        lower_limb,
        prefix=prefix,
        knee=knee,
        ankle=ankle,
        knee_y=knee_y,
        ankle_y=ankle_y,
        thigh=thigh,
        calf=calf,
    )
    band = 0.011 * max(thigh, calf)
    pop_lower_y = _slice_calf_bifurcation_y(
        verts,
        pre_post=pre_post,
        pre_peroneal=pre_peroneal,
        lower_limb=lower_limb,
        y_hi=knee_y - 0.01 * calf,
        y_lo=knee_y - 0.16 * calf,
        band=band,
    )
    pop_upper_y = _slice_sup_pop_transition_y(
        verts,
        lower_limb=lower_limb,
        knee_y=knee_y,
        thigh=thigh,
        calf=calf,
        pre_post=pre_post,
        pre_peroneal=pre_peroneal,
    )
    meta = {
        "pop_upper_mesh_raw": float(pop_upper_y) if pop_upper_y is not None else float("nan"),
        "pop_lower_mesh_raw": float(pop_lower_y) if pop_lower_y is not None else float("nan"),
        "knee_joint_y": float(knee_y),
        "pop_upper_source": "mesh_slice" if pop_upper_y is not None else "fallback_joint_fraction",
        "pop_lower_source": "mesh_slice" if pop_lower_y is not None else "fallback_joint_fraction",
    }
    if pop_upper_y is None:
        pop_upper_y = knee_y + 0.055 * thigh
        meta["pop_upper_fallback"] = 1.0
    if pop_lower_y is None:
        pop_lower_y = knee_y - 0.075 * calf
        meta["pop_lower_fallback"] = 1.0
    raw_upper, raw_lower = float(pop_upper_y), float(pop_lower_y)
    pop_upper_y, pop_lower_y = _clamp_pop_bounds(
        float(pop_upper_y),
        float(pop_lower_y),
        knee_y=float(knee_y),
        thigh=float(thigh),
        calf=float(calf),
    )
    if abs(float(pop_upper_y) - raw_upper) > 1.0e-4:
        meta["pop_upper_clamped"] = 1.0
    if abs(float(pop_lower_y) - raw_lower) > 1.0e-4:
        meta["pop_lower_clamped"] = 1.0
    meta["pop_upper_mesh"] = float(pop_upper_y)
    meta["pop_lower_mesh"] = float(pop_lower_y)
    return float(pop_upper_y), float(pop_lower_y), meta


def _pop_calf_junction_from_branches(
    verts: np.ndarray,
    *,
    pop_mask: np.ndarray,
    post_mask: np.ndarray,
    peroneal_mask: np.ndarray,
    ankle: np.ndarray,
) -> np.ndarray | None:
    """Place POP/calf junction where POST_TIB and PERONEAL attach on the popliteal centerline."""
    pop_pts = verts[pop_mask]
    if pop_pts.shape[0] < 6:
        return None
    pop_line = _centerline_from_branch(pop_pts, bins=8)
    if pop_line.shape[0] < 2:
        return None
    pop_line = _order_polyline_along_axis(pop_line, pop_line[0], np.asarray(ankle, dtype=np.float32).reshape(3))
    attachments: list[np.ndarray] = []
    for branch_mask in (post_mask, peroneal_mask):
        bp = verts[branch_mask]
        if bp.shape[0] < 4:
            continue
        attach = _branch_attachment_on_trunk(bp, pop_line, proximal_quantile=0.82)
        if attach is not None:
            attachments.append(attach)
    if not attachments:
        return None
    return np.median(np.stack(attachments, axis=0), axis=0).astype(np.float32)


def _knee_junction_from_branches(
    verts: np.ndarray,
    *,
    pop_mask: np.ndarray,
    sup_mask: np.ndarray,
    ankle: np.ndarray,
) -> np.ndarray | None:
    """Knee junction from posed/rest mesh where SUP meets POP (not a rest Y slice)."""
    pop_pts = verts[pop_mask]
    if pop_pts.shape[0] < 4:
        return None
    pop_line = _centerline_from_branch(pop_pts, bins=6)
    if pop_line.shape[0] < 2:
        return None
    pop_line = _order_polyline_along_axis(pop_line, pop_line[0], np.asarray(ankle, dtype=np.float32).reshape(3))
    sup_pts = verts[sup_mask]
    if sup_pts.shape[0] >= 4:
        attach = _branch_attachment_on_trunk(sup_pts, pop_line, proximal_quantile=0.22)
        if attach is not None:
            return attach
    return np.median(pop_pts, axis=0).astype(np.float32)


def _calf_junction_from_branches(
    verts: np.ndarray,
    *,
    pop_mask: np.ndarray,
    post_mask: np.ndarray,
    peroneal_mask: np.ndarray,
    ankle: np.ndarray,
) -> np.ndarray | None:
    """Calf junction from mesh branch attachment on the popliteal trunk."""
    branch_j = _pop_calf_junction_from_branches(
        verts,
        pop_mask=pop_mask,
        post_mask=post_mask,
        peroneal_mask=peroneal_mask,
        ankle=ankle,
    )
    if branch_j is not None:
        return branch_j
    calf_pts = verts[post_mask | peroneal_mask]
    if calf_pts.shape[0] >= 3:
        return np.median(calf_pts, axis=0).astype(np.float32)
    pop_pts = verts[pop_mask]
    if pop_pts.shape[0] >= 3:
        return np.median(pop_pts, axis=0).astype(np.float32)
    return None


def _extract_leg_vein_centerlines_for_side(
    verts: np.ndarray,
    labels: np.ndarray,
    *,
    prefix: str,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    thigh: float,
    calf: float,
    confluence_y: float,
    deep_fem_end_y: float,
    pop_upper_y: float,
    pop_lower_y: float,
    common_mask: np.ndarray,
    thigh_after_confluence: np.ndarray,
    hip_junction: np.ndarray | None,
    use_y_slice_junctions: bool,
) -> dict[str, np.ndarray]:
    """Build named centerlines for one leg side from labeled mesh vertices."""
    centerlines: dict[str, np.ndarray] = {}
    limb_scale = max(thigh, calf)
    com_label = f"{prefix}_COM_FEM_V"
    sup_label = f"{prefix}_SUPFEMV"
    deep_label = f"{prefix}_DEEP_FEM_V"
    saph_label = f"{prefix}_SAPH_V"
    pop_label = f"{prefix}_POPV"
    post_label = f"{prefix}_POST_TIB_V"
    peroneal_label = f"{prefix}_PERONEAL_V"
    femoral_labels = (com_label, sup_label, deep_label, saph_label)

    saph_j: np.ndarray | None = hip_junction
    deep_j: np.ndarray | None = None
    if np.sum(labels == saph_label) >= 4 and np.sum(labels == deep_label) >= 4:
        saph_j, deep_j = _sequential_femoral_junctions(
            verts,
            com_mask=labels == com_label,
            sup_mask=labels == sup_label,
            saph_mask=labels == saph_label,
            deep_mask=labels == deep_label,
            trunk_mask=common_mask | thigh_after_confluence,
            confluence_y=float(confluence_y),
            deep_fem_end_y=float(deep_fem_end_y),
            thigh=float(thigh),
            hip=hip,
            knee=knee,
        )
        if hip_junction is not None:
            saph_j = hip_junction

    if use_y_slice_junctions:
        knee_j = _junction_at_y(
            verts,
            (labels == sup_label) | (labels == pop_label),
            float(pop_upper_y),
            band_scale=limb_scale,
            fallback=np.asarray([float(knee[0]), pop_upper_y, float(knee[2])], dtype=np.float32),
            lock_y=False,
        )
        calf_j = _junction_at_y(
            verts,
            (labels == pop_label) | (labels == post_label) | (labels == peroneal_label),
            float(pop_lower_y),
            band_scale=limb_scale,
            fallback=np.asarray([float(knee[0]), pop_lower_y, float(knee[2])], dtype=np.float32),
            lock_y=False,
        )
        branch_calf_j = _pop_calf_junction_from_branches(
            verts,
            pop_mask=labels == pop_label,
            post_mask=labels == post_label,
            peroneal_mask=labels == peroneal_label,
            ankle=ankle,
        )
        pop_band = 0.045 * limb_scale
        if branch_calf_j is not None and abs(float(branch_calf_j[1]) - float(pop_lower_y)) <= pop_band:
            calf_j = branch_calf_j
    else:
        knee_j = _knee_junction_from_branches(
            verts,
            pop_mask=labels == pop_label,
            sup_mask=labels == sup_label,
            ankle=ankle,
        )
        if knee_j is None:
            knee_j = np.asarray([float(knee[0]), float(knee[1]), float(knee[2])], dtype=np.float32)
        calf_j = _calf_junction_from_branches(
            verts,
            pop_mask=labels == pop_label,
            post_mask=labels == post_label,
            peroneal_mask=labels == peroneal_label,
            ankle=ankle,
        )
        if calf_j is None:
            calf_j = np.asarray([float(knee[0]), float(knee[1]), float(knee[2])], dtype=np.float32)

    if np.sum(labels == com_label) >= 4 and saph_j is not None:
        centerlines[com_label] = _branch_centerline(
            verts[labels == com_label],
            bins=8,
            junction=saph_j,
            junction_at_start=False,
            smooth_window=5,
            resample_target=12,
        )
    if np.sum(labels == sup_label) >= 4:
        sup_pts = verts[labels == sup_label]
        if saph_j is not None:
            centerlines[sup_label] = _branch_centerline_two_junctions(
                sup_pts,
                bins=10,
                start_j=saph_j,
                end_j=knee_j,
            )
        else:
            centerlines[sup_label] = _branch_centerline(
                sup_pts,
                bins=10,
                junction=knee_j,
                junction_at_start=False,
            )
    if np.sum(labels == saph_label) >= 4 and saph_j is not None:
        centerlines[saph_label] = _branch_centerline(
            verts[labels == saph_label],
            bins=14,
            junction=saph_j,
            junction_at_start=True,
            smooth_window=7,
            smooth_passes=3,
            resample_target=18,
        )
    if np.sum(labels == deep_label) >= 4 and deep_j is not None:
        centerlines[deep_label] = _branch_centerline(
            verts[labels == deep_label],
            bins=10,
            junction=deep_j,
            junction_at_start=True,
            smooth_window=6,
            smooth_passes=2,
            resample_target=16,
        )

    pop_pts = verts[labels == pop_label]
    if pop_pts.shape[0] >= 4:
        centerlines[pop_label] = _branch_centerline_two_junctions(
            pop_pts,
            bins=4,
            start_j=knee_j,
            end_j=calf_j,
            smooth_window=3,
            smooth_passes=1,
            resample_target=6,
        )

    post_pts = verts[labels == post_label]
    if post_pts.shape[0] >= 4:
        post_heel = _distal_heel_tip(post_pts, proximal_ref=calf_j, ankle=ankle)
        centerlines[post_label] = _branch_centerline(
            post_pts,
            bins=12,
            junction=calf_j,
            junction_at_start=True,
            distal_ref=post_heel,
            distal_tip=post_heel,
            smooth_window=3,
            smooth_passes=1,
            resample_target=16,
        )

    peroneal_pts = verts[labels == peroneal_label]
    if peroneal_pts.shape[0] >= 4:
        per_heel = _distal_heel_tip(peroneal_pts, proximal_ref=calf_j, ankle=ankle)
        centerlines[peroneal_label] = _branch_centerline(
            peroneal_pts,
            bins=12,
            junction=calf_j,
            junction_at_start=True,
            distal_ref=per_heel,
            distal_tip=per_heel,
            smooth_window=3,
            smooth_passes=1,
            resample_target=16,
        )
    return centerlines


def _extract_leg_vein_centerlines_from_labels(
    asset_joint_names: list[str],
    rest_joints: np.ndarray,
    vertices: np.ndarray,
    labels: np.ndarray,
    bounds_meta: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Re-extract centerlines from LBS-posed mesh vertices using rest segment labels."""
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    labels = np.asarray(labels, dtype=object).reshape(-1)
    centerlines: dict[str, np.ndarray] = {}
    side = _side_prefix(verts[:, 0])
    for prefix, hip_name, knee_name, ankle_name in (
        ("L", "left_hip", "left_knee", "left_ankle"),
        ("R", "right_hip", "right_knee", "right_ankle"),
    ):
        if prefix not in bounds_meta:
            continue
        hip = _joint(asset_joint_names, rest_joints, hip_name)
        knee = _joint(asset_joint_names, rest_joints, knee_name)
        ankle = _joint(asset_joint_names, rest_joints, ankle_name)
        hip_y, knee_y, ankle_y = float(hip[1]), float(knee[1]), float(ankle[1])
        thigh = max(hip_y - knee_y, 1.0e-4)
        calf = max(knee_y - ankle_y, 1.0e-4)
        sign_mask = side == prefix
        lower_limb = sign_mask & (labels != "VEIN_UNLABELED")
        if not np.any(lower_limb):
            continue
        bounds = bounds_meta[prefix]
        confluence_y = knee_y + 0.90 * thigh
        deep_fem_end_y = knee_y + 0.50 * thigh
        pop_upper_y = float(bounds.get("pop_upper_y", knee_y + 0.055 * thigh))
        pop_lower_y = float(bounds.get("pop_lower_y", knee_y - 0.075 * calf))
        common_mask = lower_limb & (labels == f"{prefix}_COM_FEM_V")
        thigh_mask = lower_limb & np.isin(
            labels,
            [f"{prefix}_SUPFEMV", f"{prefix}_DEEP_FEM_V", f"{prefix}_SAPH_V"],
        )
        centerlines.update(
            _extract_leg_vein_centerlines_for_side(
                verts,
                labels,
                prefix=prefix,
                hip=hip,
                knee=knee,
                ankle=ankle,
                thigh=thigh,
                calf=calf,
                confluence_y=confluence_y,
                deep_fem_end_y=deep_fem_end_y,
                pop_upper_y=pop_upper_y,
                pop_lower_y=pop_lower_y,
                common_mask=common_mask,
                thigh_after_confluence=thigh_mask,
                hip_junction=None,
                use_y_slice_junctions=False,
            )
        )
    return centerlines


def _classify_leg_veins(
    asset_joint_names: list[str],
    rest_joints: np.ndarray,
    vertices_rest: np.ndarray,
    faces_local: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, float]]]:
    verts = np.asarray(vertices_rest, dtype=np.float32).reshape(-1, 3)
    labels = np.full(verts.shape[0], "VEIN_UNLABELED", dtype=object)
    centerlines: dict[str, np.ndarray] = {}
    bounds_meta: dict[str, dict[str, float]] = {}
    side = _side_prefix(verts[:, 0])
    for prefix, hip_name, knee_name, ankle_name in (
        ("L", "left_hip", "left_knee", "left_ankle"),
        ("R", "right_hip", "right_knee", "right_ankle"),
    ):
        hip = _joint(asset_joint_names, rest_joints, hip_name)
        knee = _joint(asset_joint_names, rest_joints, knee_name)
        ankle = _joint(asset_joint_names, rest_joints, ankle_name)
        hip_y, knee_y, ankle_y = float(hip[1]), float(knee[1]), float(ankle[1])
        thigh = max(hip_y - knee_y, 1.0e-4)
        calf = max(knee_y - ankle_y, 1.0e-4)
        sign_mask = side == prefix
        lower_limb = sign_mask & (verts[:, 1] <= hip_y + 0.12) & (verts[:, 1] >= ankle_y - 0.16)
        if not np.any(lower_limb):
            continue

        # Keep common femoral as a short proximal confluence only. Below this
        # point the asset branches into the femoral/superficial trunk and the
        # profunda/deep femoral branch.
        confluence_y = knee_y + 0.90 * thigh
        deep_fem_end_y = knee_y + 0.50 * thigh
        pop_upper_y = knee_y + 0.055 * thigh
        pop_lower_y = knee_y - 0.075 * calf
        side_bounds = {
            "pop_upper_y": float(pop_upper_y),
            "pop_lower_y": float(pop_lower_y),
            "knee_joint_y": float(knee_y),
            "pop_upper_source": "knee_local_band",
            "pop_lower_source": "knee_local_band",
        }
        bounds_meta[prefix] = side_bounds

        common = lower_limb & (verts[:, 1] > confluence_y)
        thigh_after_confluence = lower_limb & (verts[:, 1] <= confluence_y) & (verts[:, 1] > pop_upper_y)
        pop = lower_limb & (verts[:, 1] <= pop_upper_y) & (verts[:, 1] > pop_lower_y)
        calf_mask = lower_limb & (verts[:, 1] <= pop_lower_y)

        labels[common] = f"{prefix}_COM_FEM_V"
        hip_junction: np.ndarray | None = None

        proximal_branch = thigh_after_confluence & (verts[:, 1] > deep_fem_end_y)
        sup_core_idx = np.flatnonzero(thigh_after_confluence & (verts[:, 1] <= deep_fem_end_y))
        prox_idx = np.flatnonzero(proximal_branch & (verts[:, 1] < confluence_y - 0.035 * thigh))
        if sup_core_idx.size >= 8 and prox_idx.size >= 12:
            # Split the proximal fork into two anatomical branches. Saphenous is
            # the more medial/superficial fork; deep femoral is the other proximal
            # fork. SupFem is anchored by the distal femoral trunk.
            features = np.stack(
                [
                    np.abs(verts[prox_idx, 0]),
                    verts[prox_idx, 2] * 2.0,
                    verts[prox_idx, 1] * 0.35,
                ],
                axis=1,
            )
            fork_labels, _ = _kmeans2(features)
            c0 = verts[prox_idx[fork_labels == 0]].mean(axis=0)
            c1 = verts[prox_idx[fork_labels == 1]].mean(axis=0)
            # Medial means closer to the body midline (smaller abs(x)).
            saph_cluster = 0 if abs(float(c0[0])) <= abs(float(c1[0])) else 1
            saph_core_idx = prox_idx[fork_labels == saph_cluster]
            deep_core_idx = prox_idx[fork_labels != saph_cluster]

            assign_idx = np.flatnonzero(thigh_after_confluence)
            branch = _nearest_three_way(
                verts[assign_idx],
                verts[sup_core_idx],
                verts[deep_core_idx],
                verts[saph_core_idx],
            )
            labels[assign_idx[branch == 0]] = f"{prefix}_SUPFEMV"
            labels[assign_idx[branch == 1]] = f"{prefix}_DEEP_FEM_V"
            labels[assign_idx[branch == 2]] = f"{prefix}_SAPH_V"

            com_label = f"{prefix}_COM_FEM_V"
            sup_label = f"{prefix}_SUPFEMV"
            deep_label = f"{prefix}_DEEP_FEM_V"
            saph_label = f"{prefix}_SAPH_V"
            femoral_labels = (com_label, sup_label, deep_label, saph_label)
            prelim = dict(
                zip(
                    femoral_labels,
                    (common, labels == sup_label, labels == deep_label, labels == saph_label),
                    strict=True,
                )
            )
            saph_j, deep_j = _sequential_femoral_junctions(
                verts,
                com_mask=prelim[com_label],
                sup_mask=prelim[sup_label],
                saph_mask=prelim[saph_label],
                deep_mask=prelim[deep_label],
                trunk_mask=common | thigh_after_confluence,
                confluence_y=float(confluence_y),
                deep_fem_end_y=float(deep_fem_end_y),
                thigh=float(thigh),
                hip=hip,
                knee=knee,
            )
            hip_junction = saph_j

            centerlines[com_label] = _branch_centerline(
                verts[prelim[com_label]],
                bins=8,
                junction=saph_j,
                junction_at_start=False,
                smooth_window=5,
                resample_target=12,
            )
            centerlines[sup_label] = _branch_centerline(
                verts[prelim[sup_label]],
                bins=12,
                junction=saph_j,
                junction_at_start=True,
                smooth_window=5,
                resample_target=16,
            )
            centerlines[saph_label] = _branch_centerline(
                verts[prelim[saph_label]],
                bins=14,
                junction=saph_j,
                junction_at_start=True,
                smooth_window=7,
                smooth_passes=3,
                resample_target=18,
            )
            centerlines[deep_label] = _branch_centerline(
                verts[prelim[deep_label]],
                bins=10,
                junction=deep_j,
                junction_at_start=True,
                smooth_window=6,
                smooth_passes=2,
                resample_target=16,
            )

            # Re-assign the femoral trifurcation tube surface by nearest centerline.
            # This makes branch labels meet exactly at the shared junction instead
            # of splitting the two sides of one tube by raw surface coordinates.
            femoral_mask = common | thigh_after_confluence
            femoral_idx = np.flatnonzero(femoral_mask)
            distances = np.stack([_distance_to_polyline(verts[femoral_idx], centerlines[label]) for label in femoral_labels], axis=1)
            nearest = np.argmin(distances, axis=1)
            for idx_label, label in enumerate(femoral_labels):
                labels[femoral_idx[nearest == idx_label]] = label
        else:
            deep_candidates = thigh_after_confluence & (verts[:, 1] > deep_fem_end_y)
            deep_threshold = float(np.percentile(verts[deep_candidates, 2], 62.0)) if np.any(deep_candidates) else 0.0
            deep_fem = deep_candidates & (verts[:, 2] >= deep_threshold)
            labels[thigh_after_confluence & ~deep_fem] = f"{prefix}_SUPFEMV"
            labels[deep_fem] = f"{prefix}_DEEP_FEM_V"

        labels[pop] = f"{prefix}_POPV"

        sup_label = f"{prefix}_SUPFEMV"
        pop_label = f"{prefix}_POPV"
        post_label = f"{prefix}_POST_TIB_V"
        peroneal_label = f"{prefix}_PERONEAL_V"

        # Posterior tibial is medial (closer to the body midline); peroneal is lateral.
        knee_x, ankle_x = float(knee[0]), float(ankle[0])
        limb_x = (knee_x + ankle_x) * 0.5
        if prefix == "L":
            peroneal = calf_mask & (verts[:, 0] > limb_x)
        else:
            peroneal = calf_mask & (verts[:, 0] < limb_x)
        post_tib = calf_mask & ~peroneal
        labels[post_tib] = post_label
        labels[peroneal] = peroneal_label

        side_lines = _extract_leg_vein_centerlines_for_side(
            verts,
            labels,
            prefix=prefix,
            hip=hip,
            knee=knee,
            ankle=ankle,
            thigh=thigh,
            calf=calf,
            confluence_y=confluence_y,
            deep_fem_end_y=deep_fem_end_y,
            pop_upper_y=pop_upper_y,
            pop_lower_y=pop_lower_y,
            common_mask=common,
            thigh_after_confluence=thigh_after_confluence,
            hip_junction=hip_junction,
            use_y_slice_junctions=True,
        )
        centerlines.update(side_lines)
        bounds_meta[prefix]["pop_upper_y"] = float(pop_upper_y)
        bounds_meta[prefix]["pop_lower_y"] = float(pop_lower_y)
        if pop_label in side_lines and side_lines[pop_label].shape[0] >= 2:
            bounds_meta[prefix]["knee_j_y"] = float(side_lines[pop_label][0][1])
            bounds_meta[prefix]["calf_j_y"] = float(side_lines[pop_label][-1][1])
            bounds_meta[prefix]["calf_junction_source"] = "mesh_band_median"
    return labels, centerlines, bounds_meta


def _segment_faces_by_vertex_labels(faces_global: np.ndarray, labels_global: np.ndarray, label: str) -> np.ndarray:
    if faces_global.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    face_labels = labels_global[faces_global]
    mask = np.all(face_labels == label, axis=1)
    return faces_global[mask]


def _mesh_obj_name(mesh_name: str) -> str:
    return str(mesh_name).lower().replace(" ", "_")


def _is_skeleton_mesh(mesh_name: str) -> bool:
    name = str(mesh_name)
    if any(token in name for token in _SKELETON_MESH_SKIP):
        return False
    if name in _SKELETON_MESH_EXACT:
        return True
    return any(key in name for key in _SKELETON_MESH_KEYWORDS)


def _export_source_mesh_objs(
    raw: "np.lib.npyio.NpzFile",
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
    *,
    mesh_filter,
    report: dict[str, object],
    report_key: str,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(v) for v in raw["source_mesh_names"].reshape(-1).tolist()]
    ranges = np.asarray(raw["source_vertex_ranges"], dtype=np.int64).reshape(-1, 2)
    exported: list[str] = []
    entries: dict[str, object] = {}
    for mesh_name, (start, end) in zip(names, ranges, strict=True):
        if not mesh_filter(mesh_name):
            continue
        src_faces = _faces_in_range(faces, int(start), int(end))
        if src_faces.shape[0] == 0:
            continue
        slug = _mesh_obj_name(mesh_name)
        _write_subset_obj(out_dir / f"{slug}_rest.obj", rest_vertices, src_faces, comment=f"{mesh_name} rest")
        _write_subset_obj(out_dir / f"{slug}_posed.obj", posed_vertices, src_faces, comment=f"{mesh_name} posed")
        exported.append(mesh_name)
        entries[mesh_name] = {"vertices": int(end - start), "faces": int(src_faces.shape[0])}
    report[report_key] = {"count": len(exported), "meshes": entries}
    return exported


def _vein_points_by_label(
    vertices: np.ndarray,
    labels: np.ndarray,
    *,
    include_unlabeled: bool = False,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    unique = sorted({str(v) for v in labels.tolist() if str(v)})
    for label in unique:
        if label == "VEIN_UNLABELED" and not include_unlabeled:
            continue
        if label in ("ARTERY", ""):
            continue
        idx = np.flatnonzero(labels == label)
        if idx.size == 0:
            continue
        out[label] = np.asarray(vertices[idx], dtype=np.float32)
    return out


def _load_smpl_tpose_vertices(canonical_dir: Path) -> np.ndarray | None:
    obj_path = Path(canonical_dir) / "smpl_canonical_tpose.obj"
    if not obj_path.is_file():
        return None
    return read_obj_vertices(obj_path)


def _load_posed_vertices(asset_npz: Path, motion_npz: Path) -> np.ndarray:
    asset = load_rigged_asset(asset_npz)
    data = np.load(motion_npz)
    pose = easymocap_fit_to_smplx55(data["Rh"], data["poses"]).reshape(-1)
    transl = easymocap_drive_translation(data["Rh"], data["Th"], np.asarray(asset.rest_joints, dtype=np.float32)[0])
    return skin_vertices(asset, pose, transl=transl)


def _draw_overlap(path: Path, smpl_vertices: np.ndarray, points_by_label: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    views = [(0, 1, "XY front"), (2, 1, "ZY side"), (0, 2, "XZ top")]
    smpl = np.asarray(smpl_vertices, dtype=np.float32).reshape(-1, 3)
    for ax, (i, j, title) in zip(axes, views, strict=True):
        ax.scatter(smpl[::4, i], smpl[::4, j], s=0.4, c="#d0a000", alpha=0.28, label="SMPL fit")
        for label, pts in points_by_label.items():
            if pts.size == 0:
                continue
            rgb = SEGMENT_COLORS.get(label, (170, 170, 170))
            color = tuple(v / 255.0 for v in rgb)
            step = max(1, pts.shape[0] // 900)
            ax.scatter(pts[::step, i], pts[::step, j], s=1.0, color=color, alpha=0.9, label=label)
        ax.set_aspect("equal")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    plt.tight_layout(rect=(0, 0.18, 1, 1))
    plt.savefig(path, dpi=130)
    plt.close(fig)


def _draw_leg_zoom(
    path: Path,
    points_by_label: dict[str, np.ndarray],
    *,
    centerlines: dict[str, np.ndarray] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    labels = [
        label
        for label in (centerlines or points_by_label)
        if any(k in label for k in ("COM_FEM", "DEEP_FEM", "SAPH", "SUPFEM", "POPV"))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    views = [(0, 1, "XY front"), (2, 1, "ZY side")]
    for row, (i, j, view_title) in enumerate(views):
        for col, side in enumerate(("L_", "R_")):
            ax = axes[row, col]
            for label in labels:
                if not label.startswith(side):
                    continue
                pts = (centerlines or points_by_label)[label]
                if pts.size == 0:
                    continue
                rgb = SEGMENT_COLORS.get(label, (170, 170, 170))
                color = tuple(v / 255.0 for v in rgb)
                if centerlines is not None:
                    ax.plot(
                        pts[:, i],
                        pts[:, j],
                        "-",
                        linewidth=2.6,
                        color=color,
                        alpha=0.95,
                        label=label if row == 0 and col == 0 else None,
                    )
                else:
                    ax.scatter(pts[:, i], pts[:, j], s=12.0, color=color, alpha=0.95, label=label)
            ax.set_aspect("equal")
            ax.set_title(f"{side[0]} {view_title}")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bone_out = out / "bone_segments"
    vessel_out = out / "vessel_segments"
    figures_out = out / "figures"
    centerlines_out = out / "centerlines"
    for sub in (bone_out, vessel_out, figures_out, centerlines_out):
        sub.mkdir(parents=True, exist_ok=True)

    asset = load_rigged_asset(args.asset_npz)
    raw = np.load(args.asset_npz, allow_pickle=True)
    rest_vertices = np.asarray(raw["vertices_rest"], dtype=np.float32)
    posed_vertices = _load_posed_vertices(args.asset_npz, args.motion_npz)
    faces = np.asarray(raw["faces"], dtype=np.int32).reshape(-1, 3)

    labels_global = np.full(rest_vertices.shape[0], "", dtype=object)
    report: dict[str, object] = {}

    _export_source_mesh_objs(
        raw,
        rest_vertices,
        posed_vertices,
        faces,
        bone_out,
        mesh_filter=_is_skeleton_mesh,
        report=report,
        report_key="BONE_SEGMENTS",
    )

    for source_name, label in (("Artery", "ARTERY"), ("Vein", "VEIN_UNLABELED")):
        s, e = _source_range(raw, source_name)
        src_faces = _faces_in_range(faces, s, e)
        _write_subset_obj(vessel_out / f"{label.lower()}_rest.obj", rest_vertices, src_faces, comment=f"{source_name} rest")
        _write_subset_obj(vessel_out / f"{label.lower()}_posed.obj", posed_vertices, src_faces, comment=f"{source_name} posed")
        labels_global[s:e] = label
        report[label] = {"vertices": int(e - s), "faces": int(src_faces.shape[0])}

    vein_s, vein_e = _source_range(raw, "Vein")
    vein_faces = _faces_in_range(faces, vein_s, vein_e)
    vein_labels, centerlines_local, pop_bounds_meta = _classify_leg_veins(
        asset.joint_names,
        asset.rest_joints,
        rest_vertices[vein_s:vein_e],
        vein_faces - int(vein_s),
    )
    centerlines_rest = {label: line for label, line in centerlines_local.items() if line.shape[0] >= 2}
    junction_report: dict[str, object] = {}
    for prefix in ("L", "R"):
        entry: dict[str, object] = {}
        com_key = f"{prefix}_COM_FEM_V"
        sup_key = f"{prefix}_SUPFEMV"
        deep_key = f"{prefix}_DEEP_FEM_V"
        saph_key = f"{prefix}_SAPH_V"
        pop_key = f"{prefix}_POPV"
        if all(k in centerlines_rest for k in (com_key, sup_key, deep_key, saph_key)):
            pts = np.stack(
                [
                    centerlines_rest[com_key][-1],
                    centerlines_rest[sup_key][0],
                    centerlines_rest[saph_key][0],
                ],
                axis=0,
            )
            entry["hip_saph"] = {
                "xyz": [float(v) for v in pts[0]],
                "max_delta_m": float(np.max(np.linalg.norm(pts - pts[0], axis=1))),
            }
            deep_start = centerlines_rest[deep_key][0]
            saph_xyz = pts[0]
            entry["hip_deep"] = {
                "xyz": [float(v) for v in deep_start],
                "max_delta_m": float(np.linalg.norm(deep_start - saph_xyz)),
            }
            entry["hip_branch_sep_m"] = float(np.linalg.norm(deep_start - saph_xyz))
        if sup_key in centerlines_rest and pop_key in centerlines_rest:
            pts = np.stack([centerlines_rest[sup_key][-1], centerlines_rest[pop_key][0]], axis=0)
            entry["knee"] = {
                "xyz": [float(v) for v in pts[0]],
                "max_delta_m": float(np.max(np.linalg.norm(pts - pts[0], axis=1))),
            }
        post_key = f"{prefix}_POST_TIB_V"
        peroneal_key = f"{prefix}_PERONEAL_V"
        if pop_key in centerlines_rest and post_key in centerlines_rest and peroneal_key in centerlines_rest:
            pts = np.stack(
                [
                    centerlines_rest[pop_key][-1],
                    centerlines_rest[post_key][0],
                    centerlines_rest[peroneal_key][0],
                ],
                axis=0,
            )
            entry["calf"] = {
                "xyz": [float(v) for v in pts[0]],
                "max_delta_m": float(np.max(np.linalg.norm(pts - pts[0], axis=1))),
            }
        if entry:
            if prefix in pop_bounds_meta:
                entry["pop_mesh_bounds"] = pop_bounds_meta[prefix]
                if "calf_junction_source" in pop_bounds_meta[prefix]:
                    entry["calf_junction_source"] = pop_bounds_meta[prefix]["calf_junction_source"]
            junction_report[prefix] = entry
    labels_global[vein_s:vein_e] = vein_labels
    for label in sorted({str(v) for v in vein_labels.tolist()} - {"VEIN_UNLABELED"}):
        seg_faces = _segment_faces_by_vertex_labels(vein_faces, labels_global, label)
        if seg_faces.shape[0] == 0:
            continue
        _write_subset_obj(vessel_out / f"{label.lower()}_rest.obj", rest_vertices, seg_faces, comment=f"{label} rest")
        _write_subset_obj(vessel_out / f"{label.lower()}_posed.obj", posed_vertices, seg_faces, comment=f"{label} posed")
        report[label] = {
            "vertices": int(np.sum(vein_labels == label)),
            "faces": int(seg_faces.shape[0]),
            "color_rgb": SEGMENT_COLORS.get(label, (170, 170, 170)),
        }

    point_mask = labels_global != ""

    smpl_posed = np.asarray(np.load(args.motion_npz)["vertices"], dtype=np.float32).reshape(-1, 3)
    smpl_tpose_loaded = _load_smpl_tpose_vertices(args.canonical_dir)
    smpl_tpose = smpl_tpose_loaded if smpl_tpose_loaded is not None else rest_vertices

    centerlines_tpose = {k: np.asarray(v, dtype=np.float32).copy() for k, v in centerlines_rest.items()}
    pin_centerline_junctions(centerlines_tpose)

    motion = np.load(args.motion_npz)
    pose55 = easymocap_fit_to_smplx55(motion["Rh"], motion["poses"]).reshape(-1)
    transl = easymocap_drive_translation(motion["Rh"], motion["Th"], np.asarray(asset.rest_joints, dtype=np.float32)[0])

    # Posed centerlines: collapse the already Blender-LBS-deformed vein mesh,
    # using the T-pose segment labels and centerline ordering.
    centerlines_posed = _centerline_from_deformed_label_vertices(
        rest_vertices[vein_s:vein_e],
        posed_vertices[vein_s:vein_e],
        vein_labels,
        centerlines_tpose,
    )
    centerlines_posed = {k: np.asarray(v, dtype=np.float32) for k, v in centerlines_posed.items() if np.asarray(v).shape[0] >= 2}
    pin_centerline_junctions(centerlines_posed)
    centerlines_posed = _smooth_posed_centerlines_for_export(centerlines_posed)
    pin_centerline_junctions(centerlines_posed)
    _write_centerline_obj(centerlines_out / "vessel_centerlines_posed.obj", centerlines_posed)

    bone_tpose = sparse_leg_bone_vertices(rest_vertices, raw, LEG_BONE_VIZ_MESHES)
    bone_posed = skin_points(asset, bone_tpose, pose_axis_angle=pose55, transl=transl, neighbor_k=6)

    draw_vein_on_body_pose_figure(
        figures_out / "vessel_veins_on_body.png",
        smpl_tpose=smpl_tpose,
        smpl_posed=smpl_posed,
        tpose_centerlines=centerlines_tpose,
        posed_centerlines=centerlines_posed,
        segment_colors=SEGMENT_COLORS,
        title="Leg vein centerlines on SMPL body (T-pose vs posed)",
    )
    draw_vein_on_body_pose_figure(
        figures_out / "vessel_veins_on_body_with_bones.png",
        smpl_tpose=smpl_tpose,
        smpl_posed=smpl_posed,
        tpose_centerlines=centerlines_tpose,
        posed_centerlines=centerlines_posed,
        segment_colors=SEGMENT_COLORS,
        leg_bones_tpose=bone_tpose,
        leg_bones_posed=bone_posed,
        title="Leg vein centerlines on SMPL body + thigh/calf bone markers",
    )
    report["BODY_REFERENCE"] = {
        "tpose": "smpl_canonical_tpose" if smpl_tpose_loaded is not None else "anatomy_rest",
        "posed": "smpl_fit_posed",
        "leg_bone_markers": sorted(LEG_BONE_VIZ_MESHES),
        "leg_bone_marker_points_tpose": int(bone_tpose.shape[0]),
        "leg_bone_marker_points_posed": int(bone_posed.shape[0]),
    }

    points_by_label = {
        label: posed_vertices[np.flatnonzero(labels_global == label)]
        for label in sorted({str(v) for v in labels_global[point_mask].tolist()})
        if label != "VEIN_UNLABELED"
    }
    _draw_overlap(figures_out / "vessel_segments_overlap.png", smpl_posed, points_by_label)
    _draw_leg_zoom(figures_out / "vessel_segments_leg_zoom.png", points_by_label, centerlines=centerlines_rest)
    _write_centerline_obj(centerlines_out / "vessel_centerlines_rest.obj", centerlines_tpose)

    report["CENTERLINE_JUNCTIONS"] = junction_report
    report["OUTPUT_LAYOUT"] = {
        "root": str(out),
        "bone_segments": str(bone_out),
        "vessel_segments": str(vessel_out),
        "centerlines": str(centerlines_out),
        "figures": str(figures_out),
    }
    (out / "planning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"INFO vessel segments exported -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
