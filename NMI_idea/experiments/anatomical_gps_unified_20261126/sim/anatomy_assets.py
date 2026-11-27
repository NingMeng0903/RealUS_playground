"""Training-only, provenance-preserving vascular anatomy inputs.

All public geometry uses neutral SMPL-X world XYZ in millimetres; ``y`` is the
longitudinal/material coordinate. No evaluation-only assets are loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import hashlib
from typing import Any, Mapping

import numpy as np


CT_GEOMETRY_PATH = Path(
    "/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/"
    "leg_scan_study/topology/ct_geometry.npz"
)
NATIVE_GRAPH_PATH = Path(
    "/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/"
    "leg_scan_study/topology/medseg_native_graph.json"
)
CT_LINEAGE_PATH = CT_GEOMETRY_PATH.with_name("ct_reference_topology.json.gz")
ATLAS_SURFACES_PATH = Path(
    "/media/camp/PEI_T7/SMPLX_Vascular_Probability_Cloud/aligned_vessel_surfaces.npz"
)
OPEN3D_REGISTERED_PATH = ATLAS_SURFACES_PATH.parent / (
    "right_leg_joint/shared_field_test/right_leg_open3dmodel_shared_field.npz"
)
_OPEN3D_REGISTERED_SHA256 = "9a4744cc37b65207e00ba63b16b2f34b94ba2ef3ae5a380726893210bafe69a0"
NEUTRAL_SMPLX_SURFACE_PATH = ATLAS_SURFACES_PATH.with_name("neutral_smplx_surface_mm.npz")
_NEUTRAL_SMPLX_SURFACE_SHA256 = "882b54e9bfefaa6d5329fcb056a81a3750b4b01e4de456f088d39eade337c6ba"
BONE_OCCUPANCY_PATH = Path(
    "/media/camp/PEI_T7/SMPLX_Vascular_Probability_Cloud/right_leg_joint/"
    "bonehub_right_leg_occupancy.npz"
)
THIGH_NYSORA_Y_BOUNDS_MM = (-875.0, -480.0)
FULL_LEG_Y_BOUNDS_MM = (-1280.0, -480.0)


def verify_neutral_smplx_surface_sha256(data: bytes) -> str:
    """Verify the exact archive bytes the geometry consumer will parse."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != _NEUTRAL_SMPLX_SURFACE_SHA256:
        raise ValueError("neutral skin SHA-256 does not match registered beta=0 training geometry")
    return digest


class VesselRole(IntEnum):
    MAIN_VEIN = 0
    MAIN_ARTERY = 1
    GSV = 2
    DFV = 3
    SSV_TRIB = 4
    DFA = 5


# These are role-set constraints, not y-only station classifiers.
NYSORA_STATIONS: Mapping[str, frozenset[VesselRole]] = {
    "S1_CFV_CFA": frozenset((VesselRole.MAIN_VEIN, VesselRole.MAIN_ARTERY)),
    "S2_SFJ_MICKEY": frozenset((VesselRole.MAIN_VEIN, VesselRole.MAIN_ARTERY, VesselRole.GSV)),
    "S3_ART_BIF": frozenset((VesselRole.MAIN_VEIN, VesselRole.MAIN_ARTERY, VesselRole.DFA)),
    "S4_VEN_BIF": frozenset((VesselRole.MAIN_VEIN, VesselRole.MAIN_ARTERY, VesselRole.DFV, VesselRole.DFA)),
    "S5_POPLITEAL": frozenset((VesselRole.MAIN_VEIN, VesselRole.MAIN_ARTERY)),
}

BODY_PARTS3D_RIGHT = "bodyparts3d_right"
BODY_PARTS3D_LEFT_MIRRORED = "bodyparts3d_left_mirrored"
OPEN3DMODEL_RIGHT = "open3dmodel_right"
MEDSEG_RIGHT = "medseg_right"
HOLDOUT_SOURCE_ID = "blender_holdout"


@dataclass(frozen=True)
class AnatomySplit:
    train_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]
    relationship: str


LOAO_OPEN3D = AnatomySplit(
    train_source_ids=(BODY_PARTS3D_RIGHT, BODY_PARTS3D_LEFT_MIRRORED, MEDSEG_RIGHT),
    test_source_ids=(OPEN3DMODEL_RIGHT,),
    relationship="cross-atlas, same atlas family; not an independent anatomy",
)
LOAO_BODYPARTS = AnatomySplit(
    train_source_ids=(OPEN3DMODEL_RIGHT, MEDSEG_RIGHT),
    test_source_ids=(BODY_PARTS3D_RIGHT, BODY_PARTS3D_LEFT_MIRRORED),
    relationship="cross-atlas, same atlas family; not an independent anatomy",
)
FAMILY_HELD_OUT_ATLASES = AnatomySplit(
    train_source_ids=(MEDSEG_RIGHT,),
    test_source_ids=(BODY_PARTS3D_RIGHT, BODY_PARTS3D_LEFT_MIRRORED, OPEN3DMODEL_RIGHT),
    relationship="keeps atlas family together; CT-only training lacks distal support",
)
FULL_TO_BLENDER = AnatomySplit(
    train_source_ids=(BODY_PARTS3D_RIGHT, BODY_PARTS3D_LEFT_MIRRORED, OPEN3DMODEL_RIGHT, MEDSEG_RIGHT),
    test_source_ids=(HOLDOUT_SOURCE_ID,),
    relationship="independent evaluation source",
)
SPLIT_PROTOCOLS: Mapping[str, AnatomySplit] = {
    "LOAO_OPEN3D": LOAO_OPEN3D,
    "LOAO_BODYPARTS": LOAO_BODYPARTS,
    "FAMILY_HELD_OUT_ATLASES": FAMILY_HELD_OUT_ATLASES,
    "FULL_TO_BLENDER": FULL_TO_BLENDER,
}

_ARCHIVE_SOURCE_NAMES = {
    BODY_PARTS3D_RIGHT: "BodyParts3D right",
    BODY_PARTS3D_LEFT_MIRRORED: "BodyParts3D mirrored left",
    OPEN3DMODEL_RIGHT: "Open3DModel right",
    MEDSEG_RIGHT: "MedSeg",
}
_DFA_SURFACE = "Open3DModel|Deep_artery_of_the_thigh.r"


@dataclass(frozen=True)
class AnatomySource:
    source_id: str
    h_mm: np.ndarray
    centers_mm: np.ndarray  # (N, 6, 3), world XYZ
    supported: np.ndarray  # true exactly where all center coordinates are finite
    links: np.ndarray  # (N - 1, 6), source-object adjacency only
    mesh_names: np.ndarray  # (N, 6), empty string where unsupported
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CTSkinGeometry:
    case_id: int
    h_mm: np.ndarray
    station_y_mm: np.ndarray
    origin_xz_mm: np.ndarray
    radii_mm: np.ndarray

    def query(self, h_mm: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate the stored CT skin profile; never extrapolate beyond it.

        Returns ``(origin_xz_mm, radii_mm, station_y_mm)`` with the query shape
        prepended to the coordinate/radius dimensions. The 24 radii retain the
        asset's original circumferential direction ordering.
        """
        query = np.asarray(h_mm, dtype=np.float64)
        flat = query.reshape(-1)
        order = np.argsort(self.h_mm)
        grid = self.h_mm[order]
        outside = (flat < grid[0]) | (flat > grid[-1])

        def interp(values: np.ndarray) -> np.ndarray:
            v = values[order]
            out = np.stack(
                [np.interp(flat, grid, v[:, j]) for j in range(v.shape[1])], axis=-1
            )
            out[outside] = np.nan
            return out

        origin = interp(self.origin_xz_mm).reshape(query.shape + (2,))
        radii = interp(self.radii_mm).reshape(query.shape + (self.radii_mm.shape[1],))
        station_y = np.interp(flat, grid, self.station_y_mm[order])
        station_y[outside] = np.nan
        return origin, radii, station_y.reshape(query.shape)


@dataclass(frozen=True)
class BoneOccupancy:
    count: np.ndarray
    cohort_size: int
    x_mm: np.ndarray
    y_mm: np.ndarray
    z_mm: np.ndarray
    _interpolator: Any = field(repr=False, compare=False)

    def probability_at(self, xyz_mm: np.ndarray | tuple[float, float, float]) -> np.ndarray | float:
        """Trilinearly query empirical bone occupancy at world XYZ millimetres."""
        points = np.asarray(xyz_mm, dtype=np.float64)
        if points.shape == () or points.shape[-1] != 3:
            raise ValueError("xyz_mm must have final dimension 3 in world XYZ order")
        out = np.asarray(self._interpolator(points.reshape(-1, 3)), dtype=np.float64)
        out = (out / float(self.cohort_size)).reshape(points.shape[:-1])
        return float(out) if points.ndim == 1 else out


@dataclass(frozen=True)
class TrainingAnatomyBank:
    _sources: Mapping[str, AnatomySource]
    native_graph: dict[str, Any]
    bone_occupancy: BoneOccupancy
    _ct_skin_cases: tuple[CTSkinGeometry, ...]
    provenance: Mapping[str, Any]

    @classmethod
    def load(cls) -> "TrainingAnatomyBank":
        """Load the four training sources plus native graph, CT profiles and bone field."""
        import gzip
        import json

        from scipy.interpolate import RegularGridInterpolator
        import trimesh

        neutral_skin_hash = verify_neutral_smplx_surface_sha256(NEUTRAL_SMPLX_SURFACE_PATH.read_bytes())

        with np.load(CT_GEOMETRY_PATH, allow_pickle=False) as pack:
            neutral_grid = np.array(pack["neutral_grid"], dtype=np.float64, copy=True)
            ct_ys = np.array(pack["ys"], dtype=np.float64, copy=True)
            ct_origins = np.array(pack["origins"], dtype=np.float64, copy=True)
            ct_radii = np.array(pack["radii"], dtype=np.float64, copy=True)
            archived_centers = np.array(pack["centers"][0], dtype=np.float64, copy=True)
            archived_status = np.array(pack["status"], copy=True)
            archived_links = np.array(pack["links"], dtype=bool, copy=True)
            archived_node_ids = np.array(pack["selected_node_ids"], copy=True)
            ct_bounds = np.array(pack["ct_bounds"], copy=True)

        with np.load(ATLAS_SURFACES_PATH, allow_pickle=False) as pack:
            legacy_names = pack["names"].tolist()
            if _DFA_SURFACE not in legacy_names:
                raise ValueError("original surface inventory lacks the required Open3D DFA")
        file_hash = hashlib.sha256(OPEN3D_REGISTERED_PATH.read_bytes()).hexdigest()
        if file_hash != _OPEN3D_REGISTERED_SHA256:
            raise ValueError("Open3D surface registration does not match archived training geometry")
        with np.load(OPEN3D_REGISTERED_PATH, allow_pickle=False) as pack:
            names = pack["names"].tolist()
            vertices = pack["vertices"]
            triangles = pack["triangles"]
            vertex_offsets = pack["vertex_offsets"]
            triangle_offsets = pack["triangle_offsets"]

        with np.load(BONE_OCCUPANCY_PATH, allow_pickle=False) as pack:
            bone_count = np.array(pack["count"], copy=True)
            bone_cohort = int(pack["cohort_size"])
            bone_x = np.array(pack["x_mm"], dtype=np.float64, copy=True)
            bone_y = np.array(pack["y_mm"], dtype=np.float64, copy=True)
            bone_z = np.array(pack["z_mm"], dtype=np.float64, copy=True)
        with NATIVE_GRAPH_PATH.open("r", encoding="utf-8") as stream:
            native_graph = json.load(stream)
        with gzip.open(CT_LINEAGE_PATH, "rt", encoding="utf-8") as stream:
            lineage = json.load(stream)["sources"]

        if bone_cohort <= 0:
            raise ValueError("bone occupancy cohort_size must be positive")
        if bone_count.shape != (len(bone_x), len(bone_y), len(bone_z)):
            raise ValueError("bone occupancy axes do not match count array")
        if ct_origins.shape[:2] != ct_ys.shape or ct_radii.shape[:2] != ct_ys.shape:
            raise ValueError("CT skin profile arrays have incompatible shapes")

        n, s = len(neutral_grid), len(_ARCHIVE_SOURCE_NAMES)
        if (
            archived_centers.shape != (s, n, 6, 2)
            or archived_status.shape != (s, n, 6)
            or archived_node_ids.shape != archived_status.shape
            or archived_links.shape != (s, n - 1, 6)
            or not np.array_equal(ct_ys[0], neutral_grid)
        ):
            raise ValueError("malformed neutral training topology")

        # Reuse the full, already registered atlas/CT centerlines and links.
        # Reslicing all roles would discard the original gap/edge audit and
        # accidentally truncate the calf. Slice only the named missing DFA.
        dfa_mesh_index = names.index(_DFA_SURFACE.split("|", 1)[1])
        v0, v1 = vertex_offsets[dfa_mesh_index : dfa_mesh_index + 2]
        f0, f1 = triangle_offsets[dfa_mesh_index : dfa_mesh_index + 2]
        dfa_mesh = trimesh.Trimesh(vertices=vertices[v0:v1], faces=triangles[f0:f1], process=False)
        dfa = np.full((n, 3), np.nan, dtype=np.float64)
        for i in np.flatnonzero((neutral_grid >= -880.) & (neutral_grid <= -480.)):
            center = _horizontal_section_center(dfa_mesh, float(neutral_grid[i]))
            if center is not None:
                dfa[i] = center
        dfa_supported = np.isfinite(dfa).all(axis=1)

        sources: dict[str, AnatomySource] = {}
        for slot, (source_id, lineage_name) in enumerate(_ARCHIVE_SOURCE_NAMES.items()):
            archived = lineage[lineage_name]
            node_names = {node["id"]: node["name"] for node in archived["nodes"]}
            supported = archived_status[slot] > 0
            xyz = np.full((n, 6, 3), np.nan, dtype=np.float64)
            xyz[..., (0, 2)] = archived_centers[slot]
            xyz[..., 1] = neutral_grid[:, None]
            if not np.isfinite(xyz[supported]).all():
                raise ValueError(f"supported archived row is not finite: {source_id}")
            xyz[~supported] = np.nan
            chosen_names = np.full((n, 6), "", dtype=object)
            chosen_names[supported] = [node_names[i] for i in archived_node_ids[slot][supported]]
            links = archived_links[slot].copy()
            if np.any(links & ~(supported[:-1] & supported[1:])):
                raise ValueError(f"archived link without both endpoints: {source_id}")
            if source_id == OPEN3DMODEL_RIGHT:
                if supported[:, VesselRole.DFA].any():
                    raise ValueError("Open3D DFA already present; do not overwrite original evidence")
                xyz[:, VesselRole.DFA] = dfa
                supported[:, VesselRole.DFA] = dfa_supported
                chosen_names[dfa_supported, VesselRole.DFA] = _DFA_SURFACE
                links[:, VesselRole.DFA] = dfa_supported[:-1] & dfa_supported[1:]

            mirror_x = source_id == BODY_PARTS3D_LEFT_MIRRORED
            source_provenance = {
                "source_family": "medseg" if source_id == MEDSEG_RIGHT else "bodyparts3d_open3dmodel",
                "input_laterality": "left" if mirror_x else "right",
                "canonical_laterality": "right",
                "coordinate_frame": "neutral SMPL-X world XYZ mm",
                "geometry_file": str(CT_GEOMETRY_PATH),
                "lineage_file": str(CT_LINEAGE_PATH),
                "archive_source": lineage_name,
                "archive_source_slot": slot,
                "archived_role_identities": tuple(map(tuple, archived["role_identities"])),
                "left_transform": "already mirrored by archive" if mirror_x else "identity",
                "longitudinal_grid_mm": 5.0,
                "support_policy": "archived sampled/interpolated centers and links; gaps remain NaN/False",
                "added_role_surface": _DFA_SURFACE if source_id == OPEN3DMODEL_RIGHT else None,
                "added_role_registration_file": (
                    str(OPEN3D_REGISTERED_PATH) if source_id == OPEN3DMODEL_RIGHT else None
                ),
            }
            sources[source_id] = AnatomySource(
                source_id=source_id,
                h_mm=_readonly(neutral_grid.copy()),
                centers_mm=_readonly(xyz),
                supported=_readonly(supported),
                links=_readonly(links),
                mesh_names=_readonly(chosen_names),
                provenance=source_provenance,
            )

        skin_cases = tuple(
            CTSkinGeometry(
                case_id=case,
                h_mm=_readonly(neutral_grid.copy()),
                station_y_mm=_readonly(ct_ys[case].copy()),
                origin_xz_mm=_readonly(ct_origins[case].copy()),
                radii_mm=_readonly(ct_radii[case].copy()),
            )
            for case in range(ct_ys.shape[0])
        )
        bone_interpolator = RegularGridInterpolator(
            (bone_x, bone_y, bone_z),
            bone_count.astype(np.float32),
            bounds_error=False,
            fill_value=np.nan,
        )
        bone_occupancy = BoneOccupancy(
            count=_readonly(bone_count),
            cohort_size=bone_cohort,
            x_mm=_readonly(bone_x),
            y_mm=_readonly(bone_y),
            z_mm=_readonly(bone_z),
            _interpolator=bone_interpolator,
        )
        bank_provenance = {
            "coordinate_frame": "neutral SMPL-X world XYZ mm; y is longitudinal",
            "ct_geometry_file": str(CT_GEOMETRY_PATH),
            "ct_bounds_archive_rows": tuple(map(int, ct_bounds)),
            "ct_centerline_lineage_file": str(CT_LINEAGE_PATH),
            "native_graph_file": str(NATIVE_GRAPH_PATH),
            "native_graph_counts": {
                "nodes": len(native_graph["nodes"]),
                "sampled_edges": len(native_graph["sampled_edges"]),
                "verified_junction_edges": len(native_graph["verified_junction_edges"]),
            },
            "aligned_surface_file": str(ATLAS_SURFACES_PATH),
            "open3d_registered_surface_file": str(OPEN3D_REGISTERED_PATH),
            "open3d_registered_surface_sha256": file_hash,
            "neutral_smplx_surface_file": str(NEUTRAL_SMPLX_SURFACE_PATH),
            "neutral_smplx_surface_sha256": neutral_skin_hash,
            "bone_occupancy_file": str(BONE_OCCUPANCY_PATH),
            "ct_scope": "one labelled CT participant with seven stored SMPL-X shape profiles",
            "source_ids": tuple(sources),
        }
        return cls(
            _sources=sources,
            native_graph=native_graph,
            bone_occupancy=bone_occupancy,
            _ct_skin_cases=skin_cases,
            provenance=bank_provenance,
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(self._sources)

    @property
    def ct_case_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self._ct_skin_cases)))

    def source(self, source_id: str) -> AnatomySource:
        """Return one atlas family/laterality source; shape profiles are separate CT cases."""
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown training source_id {source_id!r}; choose from {self.source_ids}") from exc

    def ct_skin(self, case_id: int = 0) -> CTSkinGeometry:
        """Return one stored CT-case skin profile (seven synthetic shape profiles exist)."""
        if isinstance(case_id, bool) or not isinstance(case_id, (int, np.integer)):
            raise TypeError("case_id must be an integer CT profile index")
        if not 0 <= int(case_id) < len(self._ct_skin_cases):
            raise IndexError(f"CT profile case_id {case_id} outside {self.ct_case_ids}")
        return self._ct_skin_cases[int(case_id)]


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _horizontal_section_center(mesh: Any, h_mm: float) -> np.ndarray | None:
    """Return the largest cross-sectional contour centroid in world XYZ."""
    section = mesh.section(
        plane_origin=np.asarray((0.0, h_mm, 0.0), dtype=np.float64),
        plane_normal=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
    )
    if section is None:
        return None
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in section.discrete:
        points = np.asarray(contour, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            continue
        xz = points[:, (0, 2)]
        closed = np.linalg.norm(xz[0] - xz[-1]) <= 1e-5
        if closed and len(xz) >= 4:
            ring = xz[:-1]
            next_ring = np.roll(ring, -1, axis=0)
            cross = ring[:, 0] * next_ring[:, 1] - next_ring[:, 0] * ring[:, 1]
            area2 = float(cross.sum())
            if abs(area2) > 1e-10:
                centroid = np.sum((ring + next_ring) * cross[:, None], axis=0) / (3.0 * area2)
                candidates.append((abs(area2) * 0.5, centroid))
                continue
        segments = np.diff(xz, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        total_length = float(lengths.sum())
        if total_length > 1e-10:
            midpoint = 0.5 * (xz[:-1] + xz[1:])
            centroid = np.sum(midpoint * lengths[:, None], axis=0) / total_length
            candidates.append((total_length, centroid))
    if not candidates:
        return None
    _, center_xz = max(candidates, key=lambda item: item[0])
    return np.asarray((center_xz[0], h_mm, center_xz[1]), dtype=np.float64)
