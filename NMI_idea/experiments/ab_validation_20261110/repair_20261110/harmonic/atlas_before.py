"""SMPL canonical left/right leg volumetric coordinate atlases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .io import read_obj_mesh

LegSide = Literal["left", "right"]


@dataclass(frozen=True)
class LegVolumeConfig:
    """Configuration for SMPL leg volume harmonic field bake."""

    proximal_station: float = 0.02
    distal_station: float = 1.0
    max_radius_m: float = 0.26
    radial_quantile: float = 0.985
    station_count: int = 48
    d_levels: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    skin_sample_stride: int = 1
    inner_core_radius_frac: float = 0.14
    interior_station_count: int = 24
    interior_theta_count: int = 18
    interior_radial_count: int = 8
    proximal_band: float = 0.04
    distal_band: float = 0.04

    @classmethod
    def fast_preview(cls) -> "LegVolumeConfig":
        """Small bake preset for smoke tests and quick diagnostic figures."""
        return cls(
            station_count=16,
            skin_sample_stride=4,
            interior_station_count=8,
            interior_theta_count=8,
            interior_radial_count=4,
            d_levels=(0.0, 0.5, 1.0),
        )


@dataclass(frozen=True)
class LegVolumeAtlas:
    side: LegSide
    skin_vertices: np.ndarray
    skin_faces: np.ndarray
    full_vertex_indices: np.ndarray
    skin_theta: np.ndarray
    skin_h: np.ndarray
    skin_d: np.ndarray
    skin_normals: np.ndarray
    core_points: np.ndarray
    core_h: np.ndarray
    volume_points: np.ndarray
    volume_xi: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    pelvis: np.ndarray
    seam_theta: float
    harmonic_vertices: np.ndarray
    harmonic_tets: np.ndarray
    harmonic_h: np.ndarray
    harmonic_theta: np.ndarray
    harmonic_d: np.ndarray
    metadata: dict[str, object]

    @property
    def skin_frames(self) -> np.ndarray:
        """On-demand local frames at skin vertices."""
        from .pose_bundle import estimate_local_frames

        xi = np.stack([self.skin_theta, self.skin_h, self.skin_d], axis=1).astype(np.float32)
        return estimate_local_frames(self, self.skin_vertices, xi)

    @property
    def volume_frames(self) -> np.ndarray:
        """On-demand local frames at stored volume samples."""
        from .pose_bundle import estimate_local_frames

        return estimate_local_frames(self, self.volume_points, self.volume_xi)


@dataclass(frozen=True)
class VesselSkinProjection:
    labels: np.ndarray
    original_points: np.ndarray
    projected_points: np.ndarray
    xi_skin: np.ndarray
    side: np.ndarray


def load_canonical_smpl(canonical_dir: Path | str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load canonical SMPL vertices/faces/joints from an anatomy canonical directory."""
    root = Path(canonical_dir)
    manifest_path = root / "source_manifest.json"
    obj_path = root / "smpl_canonical_tpose.obj"
    skeleton_path = root / "smpl_canonical_skeleton.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        obj_path = Path(str(manifest.get("subject_obj", obj_path)))
        skeleton_path = Path(str(manifest.get("skeleton_json", skeleton_path)))
    vertices, faces = read_obj_mesh(obj_path)
    if faces.size == 0:
        weights_path = root / "smpl_canonical_weights.npz"
        if weights_path.is_file():
            with np.load(weights_path, allow_pickle=True) as payload:
                if "faces" in payload.files:
                    faces = np.asarray(payload["faces"], dtype=np.int32)
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    return vertices.astype(np.float32), faces.astype(np.int32), skeleton


def save_leg_volume_atlas(path: Path | str, atlas: LegVolumeAtlas) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        side=np.asarray(atlas.side),
        skin_vertices=np.asarray(atlas.skin_vertices, dtype=np.float32),
        skin_faces=np.asarray(atlas.skin_faces, dtype=np.int32),
        full_vertex_indices=np.asarray(atlas.full_vertex_indices, dtype=np.int32),
        skin_theta=np.asarray(atlas.skin_theta, dtype=np.float32),
        skin_h=np.asarray(atlas.skin_h, dtype=np.float32),
        skin_d=np.asarray(atlas.skin_d, dtype=np.float32),
        skin_normals=np.asarray(atlas.skin_normals, dtype=np.float32),
        core_points=np.asarray(atlas.core_points, dtype=np.float32),
        core_h=np.asarray(atlas.core_h, dtype=np.float32),
        volume_points=np.asarray(atlas.volume_points, dtype=np.float32),
        volume_xi=np.asarray(atlas.volume_xi, dtype=np.float32),
        hip=np.asarray(atlas.hip, dtype=np.float32),
        knee=np.asarray(atlas.knee, dtype=np.float32),
        ankle=np.asarray(atlas.ankle, dtype=np.float32),
        pelvis=np.asarray(atlas.pelvis, dtype=np.float32),
        seam_theta=np.asarray(float(atlas.seam_theta), dtype=np.float32),
        harmonic_vertices=np.asarray(atlas.harmonic_vertices, dtype=np.float32),
        harmonic_tets=np.asarray(atlas.harmonic_tets, dtype=np.int32),
        harmonic_h=np.asarray(atlas.harmonic_h, dtype=np.float32),
        harmonic_theta=np.asarray(atlas.harmonic_theta, dtype=np.float32),
        harmonic_d=np.asarray(atlas.harmonic_d, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(atlas.metadata, ensure_ascii=True)),
    )
    return out


def load_leg_volume_atlas(path: Path | str) -> LegVolumeAtlas:
    payload = np.load(Path(path), allow_pickle=False)
    side_raw = payload["side"]
    side = str(side_raw.item() if side_raw.shape == () else side_raw.reshape(-1)[0])
    metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload.files else {}
    return LegVolumeAtlas(
        side=side,  # type: ignore[arg-type]
        skin_vertices=np.asarray(payload["skin_vertices"], dtype=np.float32),
        skin_faces=np.asarray(payload["skin_faces"], dtype=np.int32),
        full_vertex_indices=np.asarray(payload["full_vertex_indices"], dtype=np.int32),
        skin_theta=np.asarray(payload["skin_theta"], dtype=np.float32),
        skin_h=np.asarray(payload["skin_h"], dtype=np.float32),
        skin_d=np.asarray(payload["skin_d"], dtype=np.float32),
        skin_normals=np.asarray(payload["skin_normals"], dtype=np.float32)
        if "skin_normals" in payload.files
        else np.zeros_like(np.asarray(payload["skin_vertices"], dtype=np.float32)),
        core_points=np.asarray(payload["core_points"], dtype=np.float32),
        core_h=np.asarray(payload["core_h"], dtype=np.float32),
        volume_points=np.asarray(payload["volume_points"], dtype=np.float32),
        volume_xi=np.asarray(payload["volume_xi"], dtype=np.float32),
        hip=np.asarray(payload["hip"], dtype=np.float32),
        knee=np.asarray(payload["knee"], dtype=np.float32),
        ankle=np.asarray(payload["ankle"], dtype=np.float32),
        pelvis=np.asarray(payload["pelvis"], dtype=np.float32),
        seam_theta=float(np.asarray(payload["seam_theta"]).reshape(-1)[0]),
        harmonic_vertices=np.asarray(payload["harmonic_vertices"], dtype=np.float32)
        if "harmonic_vertices" in payload.files
        else np.zeros((0, 3), dtype=np.float32),
        harmonic_tets=np.asarray(payload["harmonic_tets"], dtype=np.int32)
        if "harmonic_tets" in payload.files
        else np.zeros((0, 4), dtype=np.int32),
        harmonic_h=np.asarray(payload["harmonic_h"], dtype=np.float32)
        if "harmonic_h" in payload.files
        else np.zeros((0,), dtype=np.float32),
        harmonic_theta=np.asarray(payload["harmonic_theta"], dtype=np.float32)
        if "harmonic_theta" in payload.files
        else np.zeros((0,), dtype=np.float32),
        harmonic_d=np.asarray(payload["harmonic_d"], dtype=np.float32)
        if "harmonic_d" in payload.files
        else np.zeros((0,), dtype=np.float32),
        metadata=metadata,
    )


def _joint(skeleton: dict[str, object], name: str) -> np.ndarray:
    names = [str(v) for v in skeleton["joint_names"]]  # type: ignore[index]
    if name not in names:
        raise KeyError(f"SMPL joint not found: {name}")
    idx = names.index(name)
    joints = np.asarray(skeleton["rest_joints_subject"], dtype=np.float32)  # type: ignore[index]
    return joints[idx].astype(np.float32)


def _side_sign(side: LegSide) -> float:
    return 1.0 if side == "left" else -1.0


def _piecewise_station(points: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    h = np.asarray(hip, dtype=np.float32).reshape(3)
    k = np.asarray(knee, dtype=np.float32).reshape(3)
    a = np.asarray(ankle, dtype=np.float32).reshape(3)
    hk = k - h
    ka = a - k
    l1 = max(float(np.linalg.norm(hk)), 1.0e-8)
    l2 = max(float(np.linalg.norm(ka)), 1.0e-8)
    t1 = np.clip(((pts - h.reshape(1, 3)) @ hk) / (l1 * l1), 0.0, 1.0)
    t2 = np.clip(((pts - k.reshape(1, 3)) @ ka) / (l2 * l2), 0.0, 1.0)
    q1 = h.reshape(1, 3) + t1[:, None] * hk.reshape(1, 3)
    q2 = k.reshape(1, 3) + t2[:, None] * ka.reshape(1, 3)
    d1 = np.linalg.norm(pts - q1, axis=1)
    d2 = np.linalg.norm(pts - q2, axis=1)
    use_calf = d2 < d1
    station = np.where(use_calf, (l1 + t2 * l2) / (l1 + l2), t1 * l1 / (l1 + l2))
    closest = np.where(use_calf[:, None], q2, q1)
    return station.astype(np.float32), closest.astype(np.float32)


def _piecewise_station_unclipped(points: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    """Station along the hip-knee-ankle axis before clamping at the ankle cut."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    h = np.asarray(hip, dtype=np.float32).reshape(3)
    k = np.asarray(knee, dtype=np.float32).reshape(3)
    a = np.asarray(ankle, dtype=np.float32).reshape(3)
    hk = k - h
    ka = a - k
    l1 = max(float(np.linalg.norm(hk)), 1.0e-8)
    l2 = max(float(np.linalg.norm(ka)), 1.0e-8)
    t1_raw = ((pts - h.reshape(1, 3)) @ hk) / (l1 * l1)
    t2_raw = ((pts - k.reshape(1, 3)) @ ka) / (l2 * l2)
    t1 = np.clip(t1_raw, 0.0, 1.0)
    t2 = np.clip(t2_raw, 0.0, 1.0)
    q1 = h.reshape(1, 3) + t1[:, None] * hk.reshape(1, 3)
    q2 = k.reshape(1, 3) + t2[:, None] * ka.reshape(1, 3)
    d1 = np.linalg.norm(pts - q1, axis=1)
    d2 = np.linalg.norm(pts - q2, axis=1)
    use_calf = d2 < d1
    station = np.where(use_calf, (l1 + t2_raw * l2) / (l1 + l2), t1_raw * l1 / (l1 + l2))
    return station.astype(np.float32)


def _axis_point_and_tangent(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, station: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(hip, dtype=np.float32).reshape(3)
    k = np.asarray(knee, dtype=np.float32).reshape(3)
    a = np.asarray(ankle, dtype=np.float32).reshape(3)
    l1 = max(float(np.linalg.norm(k - h)), 1.0e-8)
    l2 = max(float(np.linalg.norm(a - k)), 1.0e-8)
    distance = float(np.clip(station, 0.0, 1.0)) * (l1 + l2)
    if distance <= l1:
        tangent = (k - h) / l1
        point = h + distance * tangent
    else:
        tangent = (a - k) / l2
        point = k + (distance - l1) * tangent
    return point.astype(np.float32), tangent.astype(np.float32)


def _theta_for_points(points: np.ndarray, atlas: LegVolumeAtlas | None, *, side: LegSide, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    station, axis_pts = _piecewise_station(pts, hip, knee, ankle)
    medial = np.asarray(pelvis, dtype=np.float32).reshape(3) - np.asarray(hip, dtype=np.float32).reshape(3)
    theta = np.zeros(pts.shape[0], dtype=np.float32)
    for i, s in enumerate(station.tolist()):
        _axis_pt, tangent = _axis_point_and_tangent(hip, knee, ankle, float(s))
        e1 = medial - float(medial @ tangent) * tangent
        if float(np.linalg.norm(e1)) < 1.0e-8:
            e1 = np.asarray([-_side_sign(side), 0.0, 0.0], dtype=np.float32)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
        e2 = np.cross(tangent, e1)
        e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)
        rel = pts[i] - axis_pts[i]
        rel = rel - float(rel @ tangent) * tangent
        ang = float(np.arctan2(float(rel @ e2), float(rel @ e1)))
        if ang < 0.0:
            ang += 2.0 * np.pi
        theta[i] = ang
    del atlas
    return theta


def _remap_faces(faces: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_idx = np.flatnonzero(mask).astype(np.int32)
    local = np.full(mask.shape[0], -1, dtype=np.int32)
    local[full_idx] = np.arange(full_idx.shape[0], dtype=np.int32)
    tri_mask = np.all(mask[np.asarray(faces, dtype=np.int32)], axis=1)
    local_faces = local[np.asarray(faces, dtype=np.int32)[tri_mask]]
    return full_idx, local_faces.astype(np.int32)


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    normals = np.zeros_like(verts, dtype=np.float32)
    if tris.size == 0:
        return normals
    for tri in tris:
        a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(b - a, c - a)
        area2 = float(np.linalg.norm(n))
        if area2 <= 1.0e-12:
            continue
        normals[tri] += n.reshape(1, 3)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norm[:, 0] > 1.0e-8
    normals[valid] /= norm[valid]
    return normals.astype(np.float32)


def _hermite_skin_to_core(
    skin: np.ndarray,
    core: np.ndarray,
    normals: np.ndarray,
    d_value: float,
) -> np.ndarray:
    """Curve from skin to core with derivative at skin along inward normal."""
    p0 = np.asarray(skin, dtype=np.float32).reshape(-1, 3)
    p1 = np.asarray(core, dtype=np.float32).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    d = float(np.clip(d_value, 0.0, 1.0))
    length = np.linalg.norm(p1 - p0, axis=1, keepdims=True).clip(min=1.0e-6)
    m0 = -n * length
    m1 = p1 - p0
    h00 = 2.0 * d**3 - 3.0 * d**2 + 1.0
    h10 = d**3 - 2.0 * d**2 + d
    h01 = -2.0 * d**3 + 3.0 * d**2
    h11 = d**3 - d**2
    return (h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1).astype(np.float32)


def _build_side_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton: dict[str, object],
    *,
    side: LegSide,
    config: LegVolumeConfig,
) -> LegVolumeAtlas:
    from .harmonic import LegHarmonicFields, sample_volume_xi_points, solve_leg_harmonic_fields

    pelvis = _joint(skeleton, "pelvis")
    hip = _joint(skeleton, f"{side}_hip")
    knee = _joint(skeleton, f"{side}_knee")
    ankle = _joint(skeleton, f"{side}_ankle")
    station, axis_pts = _piecewise_station(vertices, hip, knee, ankle)
    station_for_cut = _piecewise_station_unclipped(vertices, hip, knee, ankle)
    radial = np.linalg.norm(vertices - axis_pts, axis=1)
    sign_mask = (vertices[:, 0] - float(pelvis[0])) * _side_sign(side) >= -0.015
    station_mask = (station_for_cut >= float(config.proximal_station)) & (station_for_cut <= float(config.distal_station))
    candidate = sign_mask & station_mask & (radial <= float(config.max_radius_m))
    if np.count_nonzero(candidate) > 16:
        radius_limit = min(float(config.max_radius_m), float(np.quantile(radial[candidate], float(config.radial_quantile))))
        candidate &= radial <= radius_limit
    full_idx, local_faces = _remap_faces(faces, candidate)
    skin_vertices = vertices[full_idx].astype(np.float32)
    skin_station, skin_core = _piecewise_station(skin_vertices, hip, knee, ankle)
    skin_station = skin_station.astype(np.float32)
    skin_normals = _compute_vertex_normals(skin_vertices, local_faces)
    radial_vec = skin_vertices - skin_core
    radial_norm = np.linalg.norm(radial_vec, axis=1, keepdims=True).clip(min=1.0e-8)
    radial_unit = radial_vec / radial_norm
    missing_normals = np.linalg.norm(skin_normals, axis=1) <= 1.0e-8
    skin_normals[missing_normals] = radial_unit[missing_normals]
    flip = np.sum(skin_normals * radial_unit, axis=1) < 0.0
    skin_normals[flip] *= -1.0

    harmonic: LegHarmonicFields = solve_leg_harmonic_fields(
        skin_vertices,
        local_faces,
        skin_station,
        side=side,
        hip=hip,
        knee=knee,
        ankle=ankle,
        pelvis=pelvis,
        proximal_station=float(config.proximal_station),
        distal_station=float(config.distal_station),
        proximal_band=float(config.proximal_band),
        distal_band=float(config.distal_band),
        inner_core_radius_frac=float(config.inner_core_radius_frac),
        interior_station_count=int(config.interior_station_count),
        interior_theta_count=int(config.interior_theta_count),
        interior_radial_count=int(config.interior_radial_count),
        medial_station_count=int(config.station_count),
    )
    skin_theta = harmonic.skin_theta.astype(np.float32)
    skin_h = harmonic.skin_h.astype(np.float32)
    skin_d = harmonic.skin_d.astype(np.float32)

    core_h = np.linspace(float(config.proximal_station), float(config.distal_station), int(config.station_count), dtype=np.float32)
    from .harmonic import medial_point_at_station

    core_points = np.stack(
        [medial_point_at_station(harmonic.medial_curve_h, harmonic.medial_curve_points, float(s)) for s in core_h],
        axis=0,
    ).astype(np.float32)
    volume_points, volume_xi = sample_volume_xi_points(
        harmonic,
        d_levels=tuple(float(v) for v in config.d_levels),
        seed=17 if side == "left" else 23,
    )

    metadata: dict[str, object] = {
        "method": "harmonic_dirichlet_tet_fem",
        "topology": "single_leg_cylindrical_chart_without_foot",
        "proximal_station": float(config.proximal_station),
        "distal_station": float(config.distal_station),
        "distal_boundary": "ankle_cut",
        "foot_policy": "excluded_from_leg_chart",
        "max_radius_m": float(config.max_radius_m),
        "skin_vertex_count": int(skin_vertices.shape[0]),
        "skin_face_count": int(local_faces.shape[0]),
        "volume_sample_count": int(volume_points.shape[0]),
        **harmonic.metadata,
    }
    return LegVolumeAtlas(
        side=side,
        skin_vertices=skin_vertices,
        skin_faces=local_faces,
        full_vertex_indices=full_idx,
        skin_theta=skin_theta,
        skin_h=skin_h,
        skin_d=skin_d,
        skin_normals=skin_normals,
        core_points=core_points,
        core_h=core_h,
        volume_points=volume_points.astype(np.float32),
        volume_xi=volume_xi.astype(np.float32),
        hip=hip,
        knee=knee,
        ankle=ankle,
        pelvis=pelvis,
        seam_theta=0.0,
        harmonic_vertices=harmonic.volume_mesh.vertices.astype(np.float32),
        harmonic_tets=harmonic.volume_mesh.tets.astype(np.int32),
        harmonic_h=harmonic.vol_h.astype(np.float32),
        harmonic_theta=harmonic.vol_theta.astype(np.float32),
        harmonic_d=harmonic.vol_d.astype(np.float32),
        metadata=metadata,
    )


def bake_leg_volume_atlases(
    canonical_dir: Path | str,
    *,
    config: LegVolumeConfig | None = None,
) -> dict[str, LegVolumeAtlas]:
    cfg = config or LegVolumeConfig()
    vertices, faces, skeleton = load_canonical_smpl(canonical_dir)
    return {
        "left": _build_side_atlas(vertices, faces, skeleton, side="left", config=cfg),
        "right": _build_side_atlas(vertices, faces, skeleton, side="right", config=cfg),
    }


def query_atlas_coordinates(atlas: LegVolumeAtlas, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return xi and nearest skin projection for arbitrary points in canonical space."""
    from .harmonic import HarmonicVolumeMesh, LegHarmonicFields, interpolate_volume_field, medial_point_at_station

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if atlas.harmonic_vertices.size and atlas.harmonic_tets.size:
        fields = LegHarmonicFields(
            skin_h=atlas.skin_h.astype(np.float32),
            skin_theta=atlas.skin_theta.astype(np.float32),
            skin_d=atlas.skin_d.astype(np.float32),
            vol_h=atlas.harmonic_h.astype(np.float32),
            vol_theta=atlas.harmonic_theta.astype(np.float32),
            vol_d=atlas.harmonic_d.astype(np.float32),
            volume_mesh=HarmonicVolumeMesh(
                vertices=atlas.harmonic_vertices.astype(np.float32),
                tets=atlas.harmonic_tets.astype(np.int32),
                skin_vertex_indices=np.arange(int(atlas.skin_vertices.shape[0]), dtype=np.int32),
                medial_vertex_indices=np.zeros(0, dtype=np.int32),
            ),
            medial_curve_h=atlas.core_h.astype(np.float32),
            medial_curve_points=atlas.core_points.astype(np.float32),
            metadata={"atlas_id": id(atlas), "side": atlas.side},
        )
        h, theta, d = interpolate_volume_field(fields, pts)
        xi = np.stack([theta, h, d], axis=1).astype(np.float32)
    else:
        h, _axis_core = _piecewise_station(pts, atlas.hip, atlas.knee, atlas.ankle)
        core = np.stack(
            [medial_point_at_station(atlas.core_h, atlas.core_points, float(hi)) for hi in h.tolist()],
            axis=0,
        ).astype(np.float32)
        theta = _theta_for_points(pts, atlas, side=atlas.side, hip=atlas.hip, knee=atlas.knee, ankle=atlas.ankle, pelvis=atlas.pelvis)
        query = np.stack([np.cos(theta), np.sin(theta), h], axis=1)
        skin_feat = np.stack([np.cos(atlas.skin_theta), np.sin(atlas.skin_theta), atlas.skin_h], axis=1)
        try:
            from scipy.spatial import cKDTree

            _dist, idx = cKDTree(skin_feat).query(query, k=1)
        except Exception:
            dist = np.linalg.norm(skin_feat[:, None, :] - query[None, :, :], axis=2).T
            idx = np.argmin(dist, axis=1)
        p_skin = atlas.skin_vertices[np.asarray(idx, dtype=np.int64)]
        dist_to_core = np.linalg.norm(pts - core, axis=1)
        dist_skin_to_core = np.linalg.norm(p_skin - core, axis=1)
        d = 1.0 - dist_to_core / np.maximum(dist_skin_to_core, 1.0e-8)
        xi = np.stack([theta, h, np.clip(d, 0.0, 1.0)], axis=1).astype(np.float32)
    query = np.stack([np.cos(xi[:, 0]), np.sin(xi[:, 0]), xi[:, 1]], axis=1)
    skin_feat = np.stack([np.cos(atlas.skin_theta), np.sin(atlas.skin_theta), atlas.skin_h], axis=1)
    try:
        from scipy.spatial import cKDTree

        _dist, idx = cKDTree(skin_feat).query(query, k=1)
    except Exception:
        dist = np.linalg.norm(skin_feat[:, None, :] - query[None, :, :], axis=2).T
        idx = np.argmin(dist, axis=1)
    p_skin = atlas.skin_vertices[np.asarray(idx, dtype=np.int64)]
    return xi, p_skin.astype(np.float32)
