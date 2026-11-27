"""Evaluation-only holdout assets behind an authorization check at every load."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import gzip
import json
import os

import numpy as np

from .anatomy_assets import HOLDOUT_SOURCE_ID, NYSORA_STATIONS, VesselRole


_HOLDOUT_ALLOWED_ENV = "AGPS_EVAL_HOLDOUT_ALLOWED"
_GEOMETRY_PATH = Path(
    "/media/camp/PEI_T7/1_team meeting/atlas_ppt_build/assets/"
    "blender_topology_holdout/geometry.npz"
)
_LINKS_PATH = Path(
    "/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/"
    "leg_scan_study/topology/blender_holdout_geometry_links.npz"
)
_TOPOLOGY_PATH = Path(
    "/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/"
    "leg_scan_study/topology/blender_holdout_topology.json.gz"
)
_ARTERIAL_PATHS_PATH = Path(
    "/media/camp/PEI_T7/SMPLX_Vascular_Retarget_Audit/evidence/arterial_paths.json"
)

# Exact legacy scan tasks from the frozen scanner. The route slices are inclusive.
_LEGACY_TASKS: Mapping[str, tuple[float, float, tuple[int, ...]]] = {
    "saphenofemoral": (-525.0, -590.0, (0, 1, 2, 3)),
    "femoral": (-595.0, -790.0, (0, 1, 2, 3)),
    "popliteal": (-835.0, -875.0, (0, 1, 2, 3)),
    "calf": (-980.0, -1255.0, (0, 1, 2, 3)),
    "groin_to_distal_calf": (-525.0, -1255.0, (0, 1, 2, 3, 4)),
}
_ARTERIAL_KEYS = ("R_CFA", "R_FA", "R_DFA", "R_PA")


@dataclass(frozen=True)
class RetargetedArterialPath:
    name: str
    neutral_xyz_mm: np.ndarray
    source_frame: str
    neutral_frame: str
    provenance: str


@dataclass(frozen=True)
class LegacyRoute:
    name: str
    h_mm: np.ndarray
    global_row_indices: np.ndarray
    role_ids: tuple[int, ...]
    centers_mm: np.ndarray
    supported: np.ndarray
    links: np.ndarray
    physically_reachable_mask: np.ndarray


@dataclass(frozen=True)
class BlenderLegacyRoutes:
    source_id: str
    h_mm: np.ndarray
    centers_mm: np.ndarray  # legacy archive's six slots in world XYZ
    supported: np.ndarray
    links: np.ndarray
    status: np.ndarray
    edge_status: np.ndarray
    inside_bone: np.ndarray
    physically_reachable_mask: np.ndarray  # exact elementwise complement of inside_bone
    station_physically_reachable_mask: np.ndarray  # true only if no role at that row is inside bone
    contacts_mm: np.ndarray  # (N, 24, 3), world XYZ
    valid_contact: np.ndarray
    bone_distance_mm: np.ndarray
    routes: Mapping[str, LegacyRoute]
    arterial_paths: Mapping[str, RetargetedArterialPath]
    role_identities: tuple[tuple[str, ...], ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class BlenderNysoraThigh:
    source_id: str
    h_mm: np.ndarray
    centers_mm: np.ndarray  # canonical role order; DFA is an explicit inferred path
    supported: np.ndarray
    directly_supported: np.ndarray
    support_source: np.ndarray  # 0 unsupported, 1 holdout geometry, 2 retargeted arterial path
    station_role_support: np.ndarray  # (N, 5), role-set evidence only, not station labels
    station_named_role_support: np.ndarray  # stricter identity match; still not clinical labels
    named_roles: np.ndarray  # canonical (N,6), empty string for missing rows
    raw_geometry_centers_mm: np.ndarray  # original holdout slots, retained without remapping
    raw_geometry_supported: np.ndarray
    raw_geometry_links: np.ndarray
    inside_bone: np.ndarray  # canonical order; valid only where reachability_known
    physically_reachable_mask: np.ndarray  # known and outside the legacy bone exclusion
    reachability_known: np.ndarray
    contacts_mm: np.ndarray
    valid_contact: np.ndarray
    bone_distance_mm: np.ndarray  # (N,24,6), NaN where canonical reachability is unknown
    arterial_paths: Mapping[str, RetargetedArterialPath]
    arterial_path_centers_mm: np.ndarray  # (N, 4, 3), y-resampled neutral path samples
    arterial_path_supported: np.ndarray  # (N, 4)
    role_identities: tuple[tuple[str, ...], ...]
    provenance: Mapping[str, Any]


def _require_holdout_authorization() -> None:
    if os.environ.get(_HOLDOUT_ALLOWED_ENV) != "1":
        raise RuntimeError("Holdout assets forbidden during training")


_require_holdout_authorization()


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _load_inputs() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, RetargetedArterialPath]]:
    _require_holdout_authorization()
    with np.load(_GEOMETRY_PATH, allow_pickle=False) as pack:
        geometry = {name: np.array(pack[name], copy=True) for name in pack.files}
    with np.load(_LINKS_PATH, allow_pickle=False) as pack:
        links = {name: np.array(pack[name], copy=True) for name in pack.files}
    with gzip.open(_TOPOLOGY_PATH, "rt", encoding="utf-8") as stream:
        topology = json.load(stream)
    with _ARTERIAL_PATHS_PATH.open("r", encoding="utf-8") as stream:
        arterial_json = json.load(stream)

    topology_source = topology.get("source", {})
    if topology_source.get("source") != "Blender right held-out":
        raise ValueError("holdout topology metadata does not describe the expected source")
    role_identities = tuple(
        tuple(str(name) for name in role)
        for role in topology_source.get("role_identities", ())
    )
    if len(role_identities) != 6:
        raise ValueError("holdout topology must declare six legacy role identity groups")

    paths: dict[str, RetargetedArterialPath] = {}
    for name in _ARTERIAL_KEYS:
        record = arterial_json.get(name)
        if not isinstance(record, dict) or "neutral" not in record:
            raise ValueError(f"missing neutral arterial path {name}")
        neutral = np.asarray(record["neutral"], dtype=np.float64)
        if neutral.ndim != 2 or neutral.shape[1] != 3:
            raise ValueError(f"arterial path {name} must contain XYZ points")
        paths[name] = RetargetedArterialPath(
            name=name,
            neutral_xyz_mm=_readonly(neutral.copy()),
            source_frame="source-model coordinates (not simulator world frame)",
            neutral_frame="neutral SMPL-X world XYZ mm",
            provenance="arterial_paths.json neutral retargeted centerline; inferred holdout evidence",
        )
    metadata = {
        "role_identities": role_identities,
        "topology_input_sha256": topology.get("input_sha256", {}),
        "coordinate_frame": topology.get("coordinates", "neutral SMPL-X mm"),
        "node_names_by_id": {row["id"]: row["name"] for row in topology_source["nodes"]},
        "station_step_mm": topology.get("station_step_mm"),
    }
    return {**geometry, **{f"links_{key}": value for key, value in links.items()}}, metadata, paths


def _xyz_centers(xz_mm: np.ndarray, h_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xz = np.asarray(xz_mm, dtype=np.float64)
    h = np.asarray(h_mm, dtype=np.float64)
    if xz.ndim != 3 or xz.shape[-1] != 2 or xz.shape[0] != len(h):
        raise ValueError("cross-section centers must have shape (N, roles, 2)")
    supported = np.isfinite(xz).all(axis=-1)
    centers = np.empty(xz.shape[:-1] + (3,), dtype=np.float64)
    centers[..., 0] = xz[..., 0]
    centers[..., 1] = h[:, None]
    centers[..., 2] = xz[..., 1]
    centers[~supported] = np.nan
    return centers, supported


def _world_contacts(contacts_xz_mm: np.ndarray, h_mm: np.ndarray) -> np.ndarray:
    contacts = np.asarray(contacts_xz_mm, dtype=np.float64)
    if contacts.ndim != 3 or contacts.shape[0] != len(h_mm) or contacts.shape[-1] != 2:
        raise ValueError("contact points must have shape (N, contacts, 2)")
    xyz = np.empty(contacts.shape[:-1] + (3,), dtype=np.float64)
    xyz[..., 0] = contacts[..., 0]
    xyz[..., 1] = np.asarray(h_mm, dtype=np.float64)[:, None]
    xyz[..., 2] = contacts[..., 1]
    xyz[~np.isfinite(contacts).all(axis=-1)] = np.nan
    return xyz




def _build_arterial_path_samples(
    h_mm: np.ndarray, paths: Mapping[str, RetargetedArterialPath]
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.full((len(h_mm), len(_ARTERIAL_KEYS), 3), np.nan, dtype=np.float64)
    supported = np.zeros((len(h_mm), len(_ARTERIAL_KEYS)), dtype=bool)
    for path_idx, name in enumerate(_ARTERIAL_KEYS):
        points = np.asarray(paths[name].neutral_xyz_mm, dtype=np.float64)
        order = np.argsort(points[:, 1])
        points = points[order]
        y, unique_index = np.unique(points[:, 1], return_index=True)
        points = points[unique_index]
        if len(y) < 2:
            continue
        for row, h in enumerate(h_mm):
            if h < y[0] or h > y[-1]:
                continue
            upper = int(np.searchsorted(y, h, side="left"))
            if upper < len(y) and abs(y[upper] - h) <= 1e-8:
                samples[row, path_idx] = points[upper]
                supported[row, path_idx] = True
                continue
            if upper == 0 or upper >= len(y):
                continue
            lower = upper - 1
            gap = y[upper] - y[lower]
            # Regrid only locally sampled path segments; a large gap is not support.
            if gap <= 0.0 or gap > 8.0:
                continue
            weight = (h - y[lower]) / gap
            samples[row, path_idx] = (1.0 - weight) * points[lower] + weight * points[upper]
            supported[row, path_idx] = True
    samples[~supported] = np.nan
    return samples, supported


def load_blender_legacy_routes() -> BlenderLegacyRoutes:
    """Load raw legacy route geometry; authorization is checked on every call."""
    _require_holdout_authorization()
    data, metadata, arterial_paths = _load_inputs()
    h_mm = np.asarray(data["ys"], dtype=np.float64)
    xz = np.asarray(data["centers"], dtype=np.float64)[3]
    centers, supported = _xyz_centers(xz, h_mm)
    links = np.asarray(data["links_links"], dtype=bool)
    status = np.asarray(data["links_status"], dtype=np.uint8)
    edge_status = np.asarray(data["links_edge_status"], dtype=np.uint8)
    if links.shape != (len(h_mm) - 1, 6) or status.shape != supported.shape:
        raise ValueError("heldout topology links have incompatible dimensions")
    inside_bone = np.asarray(data["inside_bone"], dtype=bool)
    reachable = ~inside_bone
    station_reachable = ~inside_bone.any(axis=1)
    contacts = _world_contacts(data["contacts"], h_mm)
    valid_contact = np.asarray(data["valid_contact"], dtype=bool)
    bone_distance = np.asarray(data["bone_distance"], dtype=np.float64)

    routes: dict[str, LegacyRoute] = {}
    for name, (start_y, end_y, role_ids) in _LEGACY_TASKS.items():
        low, high = sorted((start_y, end_y))
        rows = np.flatnonzero((h_mm >= low) & (h_mm <= high))
        if len(rows):
            link_rows = links[rows[:-1]] if len(rows) > 1 else links[:0]
        else:
            link_rows = links[:0]
        routes[name] = LegacyRoute(
            name=name,
            h_mm=_readonly(h_mm[rows].copy()),
            global_row_indices=_readonly(rows.astype(np.int64, copy=True)),
            role_ids=role_ids,
            centers_mm=_readonly(centers[rows].copy()),
            supported=_readonly(supported[rows].copy()),
            links=_readonly(link_rows.copy()),
            physically_reachable_mask=_readonly(reachable[rows].copy()),
        )
    provenance = {
        "source_id": HOLDOUT_SOURCE_ID,
        "geometry_index": 3,
        "geometry_file": str(_GEOMETRY_PATH),
        "links_file": str(_LINKS_PATH),
        "topology_file": str(_TOPOLOGY_PATH),
        "arterial_paths_file": str(_ARTERIAL_PATHS_PATH),
        "coordinate_frame": "neutral SMPL-X world XYZ mm; source center arrays store XZ",
        "array_selection": "geometry.centers[3] as specified by the holdout asset",
        "route_definitions": "inclusive longitudinal bounds and legacy slot ids copied from frozen TASKS",
        "physical_reachability": "elementwise complement of inside_bone; row-level mask requires every role to be outside bone",
        "arterial_paths": "neutral retargeted paths are exposed separately and do not overwrite raw route centers",
        "topology_metadata": metadata,
    }
    return BlenderLegacyRoutes(
        source_id=HOLDOUT_SOURCE_ID,
        h_mm=_readonly(h_mm.copy()),
        centers_mm=_readonly(centers),
        supported=_readonly(supported),
        links=_readonly(links.copy()),
        status=_readonly(status.copy()),
        edge_status=_readonly(edge_status.copy()),
        inside_bone=_readonly(inside_bone.copy()),
        physically_reachable_mask=_readonly(reachable.copy()),
        station_physically_reachable_mask=_readonly(station_reachable.copy()),
        contacts_mm=_readonly(contacts),
        valid_contact=_readonly(valid_contact.copy()),
        bone_distance_mm=_readonly(bone_distance.copy()),
        routes=routes,
        arterial_paths=arterial_paths,
        role_identities=metadata["role_identities"],
        provenance=provenance,
    )


def load_blender_nysora_thigh() -> BlenderNysoraThigh:
    """Load the anatomically supported thigh subset with direct and inferred evidence separated."""
    _require_holdout_authorization()
    data, metadata, arterial_paths = _load_inputs()
    full_h = np.asarray(data["ys"], dtype=np.float64)
    row_mask = (full_h >= -875.0) & (full_h <= -510.0)
    rows = np.flatnonzero(row_mask)
    h_mm = full_h[rows]
    raw_xz = np.asarray(data["centers"], dtype=np.float64)[3, rows]
    raw_centers, raw_supported = _xyz_centers(raw_xz, h_mm)
    raw_links = np.asarray(data["links_links"], dtype=bool)
    links = raw_links[rows[:-1]] if len(rows) > 1 else raw_links[:0]

    path_centers, path_supported = _build_arterial_path_samples(h_mm, arterial_paths)
    centers = np.full((len(h_mm), 6, 3), np.nan, dtype=np.float64)
    support_source = np.zeros((len(h_mm), 6), dtype=np.uint8)
    # The legacy geometry slots 0..4 align with the thigh role identities here;
    # its slot 5 is R_CALF_L and is deliberately not re-labelled as DFA.
    centers[:, :5] = raw_centers[:, :5]
    support_source[:, :5][raw_supported[:, :5]] = 1
    dfa_path_index = _ARTERIAL_KEYS.index("R_DFA")
    centers[:, int(VesselRole.DFA)] = path_centers[:, dfa_path_index]
    support_source[:, int(VesselRole.DFA)][path_supported[:, dfa_path_index]] = 2
    supported = support_source != 0
    directly_supported = support_source == 1

    station_support = np.zeros((len(h_mm), len(NYSORA_STATIONS)), dtype=bool)
    for station_index, (station_id, role_set) in enumerate(NYSORA_STATIONS.items()):
        role_ids = np.asarray(sorted(int(role) for role in role_set), dtype=np.int64)
        station_support[:, station_index] = supported[:, role_ids].all(axis=1)
        if station_id == "S5_POPLITEAL":
            station_support[:, station_index] &= (h_mm >= -875.0) & (h_mm <= -825.0)

    node_ids = data["links_selected_node_ids"][rows]
    names_by_id = metadata["node_names_by_id"]
    names = np.full(supported.shape, "", dtype="<U32")
    for row, role in zip(*np.nonzero(directly_supported), strict=True):
        names[row, role] = names_by_id[int(node_ids[row, role])]
    names[path_supported[:, dfa_path_index], VesselRole.DFA] = "R_DFA"
    expected_names = {
        "S1_CFV_CFA": {0: "R_COM_FEM_V", 1: "R_CFA"},
        "S2_SFJ_MICKEY": {0: "R_COM_FEM_V", 1: "R_CFA", 2: "R_SAPH_V"},
        "S3_ART_BIF": {0: "R_COM_FEM_V", 1: "R_FA", 5: "R_DFA"},
        "S4_VEN_BIF": {0: "R_SUPFEMV", 1: "R_FA", 3: "R_DEEP_FEM_V", 5: "R_DFA"},
        "S5_POPLITEAL": {0: "R_POPV", 1: "R_PA"},
    }
    named_support = station_support.copy()
    for i, station_id in enumerate(NYSORA_STATIONS):
        for role, name in expected_names[station_id].items():
            named_support[:, i] &= names[:, role] == name

    contacts = _world_contacts(data["contacts"][rows], h_mm)
    reachability_known = directly_supported.copy()
    raw_inside_bone = np.asarray(data["inside_bone"], dtype=bool)[rows]
    inside_bone = raw_inside_bone & reachability_known
    bone_distance = np.where(
        reachability_known[:, None, :], np.asarray(data["bone_distance"], dtype=np.float64)[rows], np.nan,
    )
    provenance = {
        "source_id": HOLDOUT_SOURCE_ID,
        "geometry_index": 3,
        "geometry_file": str(_GEOMETRY_PATH),
        "links_file": str(_LINKS_PATH),
        "topology_file": str(_TOPOLOGY_PATH),
        "arterial_paths_file": str(_ARTERIAL_PATHS_PATH),
        "coordinate_frame": "neutral SMPL-X world XYZ mm; y=h",
        "longitudinal_bounds_mm": (-875.0, -510.0),
        "raw_role_identities": metadata["role_identities"],
        "canonical_mapping": "raw geometry slots 0..4 retained; raw slot 5 is R_CALF_L and never mapped to DFA; DFA comes only from R_DFA neutral retargeted path",
        "support_source_codes": {0: "unsupported", 1: "raw holdout geometry", 2: "retargeted arterial path"},
        "station_support_rule": "all roles in NYSORA_STATIONS role set must be supported; this is role-set evidence, not an automatic station label; S5 is additionally restricted to its declared y band",
        "station_named_support_rule": "raw named anatomy must also match; no clinical label or junction-window annotation is implied",
        "arterial_path_regrid": "linear interpolation only between neutral path samples with <=8 mm y gap; no extrapolation",
        "raw_main_artery_gap_y_mm": (-545.0, -520.0),
        "raw_support_extent_y_mm": {"gsv": (-690.0, -545.0), "dfv": (-570.0, -535.0)},
        "reachability": "known only for direct raw roles 0..4; a retargeted path is not a reachability certificate",
        "topology_metadata": metadata,
    }
    return BlenderNysoraThigh(
        source_id=HOLDOUT_SOURCE_ID,
        h_mm=_readonly(h_mm.copy()),
        centers_mm=_readonly(centers),
        supported=_readonly(supported),
        directly_supported=_readonly(directly_supported),
        support_source=_readonly(support_source),
        station_named_role_support=_readonly(named_support),
        named_roles=_readonly(names),
        station_role_support=_readonly(station_support),
        raw_geometry_centers_mm=_readonly(raw_centers),
        raw_geometry_supported=_readonly(raw_supported),
        raw_geometry_links=_readonly(links.copy()),
        inside_bone=_readonly(inside_bone.copy()),
        physically_reachable_mask=_readonly(reachability_known & ~inside_bone),
        reachability_known=_readonly(reachability_known),
        contacts_mm=_readonly(contacts),
        valid_contact=_readonly(np.asarray(data["valid_contact"], dtype=bool)[rows].copy()),
        bone_distance_mm=_readonly(bone_distance),
        arterial_paths=arterial_paths,
        arterial_path_centers_mm=_readonly(path_centers),
        arterial_path_supported=_readonly(path_supported),
        role_identities=(*metadata["role_identities"][:5], ("R_DFA (inferred)",)),
        provenance=provenance,
    )
