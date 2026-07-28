"""V7 acceptance evidence-pack generator (schema_version = 7).

Produces the anatomical PNG views and JSON sidecars required by
``MD/v7_acceptance_spec.md`` section 7.  Fail closed: a view that cannot be
produced is recorded in ``missing_views`` and never replaced by a placeholder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from projects.genesis_ue_sync.anatomy_retarget.acceptance_matrix_v7 import (
    MatrixPoseSpecV7,
    MatrixSubjectSpecV7,
    body_surface_for_cell_v7,
    synthetic_knee_sweep_poses_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.compound_gates_v7 import (
    COSTAL_CARTILAGE_NAMES,
    RIB_MESH_NAMES,
    STERNUM_MESH_NAME,
    THORACIC_VERTEBRA_NAMES,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
    fit_sphere_fixed_radius_v7,
    fit_sphere_v7,
    patellofemoral_trajectory_metrics_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
    patella_oracle_sweep_v7,
    solve_patella_contact_corrections_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_pose_hash
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import apply_subject_pose
from projects.genesis_ue_sync.anatomy_retarget.vessel_gates_v7 import (
    vessel_tissue_vertex_ids_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.viz_overlay import (
    anatomy_cloud,
    dense_body_cloud,
)


EVIDENCE_PACK_SCHEMA_VERSION = 7
_SIDES = ("left", "right")
_DPI = 150

REQUIRED_VIEWS_V7: tuple[str, ...] = (
    "surface_front",
    "surface_side",
    "hip_section",
    "knee_section",
    "elbow_section",
    "hip_contact_heatmap",
    "knee_condyle_heatmap",
    "patella_track",
    "vessel_centerline",
    "rib_connection",
)

_SIDEBAR_KEYS = (
    "operator_digest",
    "subject_digest",
    "beta",
    "pose_digest",
    "asset_file_digest",
    "view",
    "command",
    "geometry_source",
)


def evidence_file_stem_v7(
    *,
    operator_digest: str,
    beta: str,
    pose: str,
    view: str,
) -> str:
    prefix = str(operator_digest).strip().lower()[:8]
    if len(prefix) != 8:
        raise ValueError(
            f"operator_digest prefix must be exactly 8 hex characters, got {prefix!r}"
        )
    return f"{prefix}_{beta}_{pose}_{view}"


def write_evidence_sidecar_v7(png_path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(png_path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"evidence sidecar requires a .png path, got {path}")
    missing = [key for key in _SIDEBAR_KEYS if key not in payload]
    if missing:
        raise ValueError(f"evidence sidecar missing required keys: {missing}")
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sidecar


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_distance(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(pts) or not len(ref):
        return np.full((len(pts),), np.inf, dtype=np.float64)
    distance, _ = cKDTree(ref).query(pts, k=1)
    return np.asarray(distance, dtype=np.float64)


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{label} is degenerate")
    return value / norm


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _unit(normal, "slab normal")
    helper = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(n, helper))) > 0.9:
        helper = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    u = _unit(np.cross(helper, n), "slab u")
    v = _unit(np.cross(n, u), "slab v")
    return n, u, v


def _slab_points(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    half_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, u, v = _orthonormal_basis(normal)
    centered = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    depth = centered @ n
    mask = np.abs(depth) <= float(half_thickness_m)
    selected = centered[mask]
    return selected @ u, selected @ v


def _mesh_lookup(asset: Any) -> dict[str, tuple[int, int, int]]:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("source_vertex_ranges must be [mesh_count,2]")
    result: dict[str, tuple[int, int, int]] = {}
    for index, (name, limits) in enumerate(zip(asset.source_mesh_names, ranges)):
        result[str(name)] = (int(index), int(limits[0]), int(limits[1]))
    return result


def _mesh_ids(lookup: Mapping[str, tuple[int, int, int]], name: str) -> np.ndarray:
    if name not in lookup:
        raise ValueError(f"required mesh {name!r} is absent")
    _index, start, stop = lookup[name]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _closest_subset(
    indices: np.ndarray,
    vertices: np.ndarray,
    target: np.ndarray,
    *,
    fraction: float,
    minimum: int,
    maximum: int = 512,
) -> np.ndarray:
    if not len(indices):
        return indices
    count = min(
        len(indices),
        maximum,
        max(minimum, int(np.ceil(fraction * len(indices)))),
    )
    distance = _nearest_distance(vertices[indices], target)
    order = np.argpartition(distance, count - 1)[:count]
    selected = np.asarray(indices[order], dtype=np.int64)
    selected.sort()
    return selected


def _union_mesh_ids(
    lookup: Mapping[str, tuple[int, int, int]],
    names: Sequence[str],
) -> np.ndarray:
    chunks = [_mesh_ids(lookup, name) for name in names if name in lookup]
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def _resolve_posed_vertices(
    *,
    subject_spec: MatrixSubjectSpecV7,
    pose_spec: MatrixPoseSpecV7,
    posed_dir: Path | None,
) -> tuple[np.ndarray, str]:
    subject = subject_spec.subject
    digest = str(subject.content_digest())
    if posed_dir is not None:
        cached = Path(posed_dir) / f"{subject_spec.label}_{pose_spec.label}.npz"
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as data:
                if "vertices" not in data.files or "subject_digest" not in data.files:
                    raise ValueError(f"posed NPZ missing required arrays: {cached}")
                stored = str(np.asarray(data["subject_digest"]).reshape(-1)[0])
                if stored == digest:
                    vertices = np.asarray(data["vertices"], dtype=np.float64)
                    return vertices, f"posed_cache:{cached}"
    vertices = np.asarray(
        apply_subject_pose(
            subject,
            pose_axis_angle=pose_spec.pose_axis_angle,
            transl=pose_spec.transl,
            validate=False,
        ),
        dtype=np.float64,
    )
    return vertices, "apply_subject_pose"


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    fig.clf()


def _title(beta: str, pose: str, view: str) -> str:
    return f"beta={beta}  pose={pose}  view={view}"


def _render_surface(
    *,
    path: Path,
    vertices: np.ndarray,
    body_surface: tuple[np.ndarray, np.ndarray] | None,
    beta: str,
    pose: str,
    view: str,
    projection: str,
) -> None:
    plt = _configure_matplotlib()
    if body_surface is None:
        raise ValueError("SMPL-X body surface unavailable for surface evidence view")
    body_v, body_f = body_surface
    from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance

    anatomy = anatomy_cloud(vertices, step=12)
    signed, _closest, _normals = signed_distance(anatomy, body_v, body_f)
    inside = signed <= 0.0
    body = dense_body_cloud(body_v, step=4)

    if projection == "front":
        ax_x, ax_y = 0, 1
        xlabel, ylabel = "x (mm)", "y (mm)"
    elif projection == "side":
        ax_x, ax_y = 2, 1
        xlabel, ylabel = "z (mm)", "y (mm)"
    else:
        raise ValueError(f"unknown surface projection {projection!r}")

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    ax.scatter(
        body[:, ax_x] * 1000.0,
        body[:, ax_y] * 1000.0,
        s=0.4,
        c="#7a7a7a",
        alpha=0.25,
        linewidths=0.0,
        label="SMPL-X skin",
    )
    ax.scatter(
        anatomy[inside, ax_x] * 1000.0,
        anatomy[inside, ax_y] * 1000.0,
        s=0.7,
        c="#2f6fed",
        alpha=0.55,
        linewidths=0.0,
        label="anatomy inside",
    )
    ax.scatter(
        anatomy[~inside, ax_x] * 1000.0,
        anatomy[~inside, ax_y] * 1000.0,
        s=1.2,
        c="#cc3333",
        alpha=0.85,
        linewidths=0.0,
        label="anatomy outside",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(_title(beta, pose, view))
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    outside_mm = float(np.max(signed[~inside])) * 1000.0 if np.any(~inside) else 0.0
    ax.text(
        0.02,
        0.02,
        f"outside={int(np.count_nonzero(~inside))}/{len(anatomy)}  "
        f"max_out={outside_mm:.2f} mm",
        transform=ax.transAxes,
        fontsize=8,
    )
    _save_figure(fig, path)
    plt.close(fig)
    del body_f  # faces retained only for signed_distance


def _circle_uv(radius_m: float, n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    return radius_m * np.cos(theta), radius_m * np.sin(theta)


def _render_hip_section(
    *,
    path: Path,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5))
    half_thickness = 0.0025
    for ax, side in zip(axes, _SIDES):
        head_ids = domains.require(f"{side}/femoral_head")
        socket_ids = domains.require(f"{side}/acetabulum")
        head_fit = fit_sphere_v7(vertices[head_ids])
        if not head_fit["available"]:
            raise ValueError(f"{side} femoral-head sphere fit failed: {head_fit.get('reason')}")
        socket_fit = fit_sphere_fixed_radius_v7(
            vertices[socket_ids],
            radius_m=float(head_fit["radius_m"]),
            initial_center=np.asarray(head_fit["center"], dtype=np.float64),
        )
        if not socket_fit["available"]:
            raise ValueError(
                f"{side} acetabulum sphere fit failed: {socket_fit.get('reason')}"
            )
        head_c = np.asarray(head_fit["center"], dtype=np.float64)
        socket_c = np.asarray(socket_fit["center"], dtype=np.float64)
        offset_mm = float(np.linalg.norm(head_c - socket_c)) * 1000.0
        # Medial-lateral axis: world X with a consistent left->right sign.
        ml = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        hx, hy = _slab_points(
            vertices[head_ids], origin=head_c, normal=ml, half_thickness_m=half_thickness
        )
        sx, sy = _slab_points(
            vertices[socket_ids],
            origin=head_c,
            normal=ml,
            half_thickness_m=half_thickness,
        )
        ax.scatter(hx * 1000.0, hy * 1000.0, s=2.0, c="#cc3333", alpha=0.7, label="femoral head")
        ax.scatter(sx * 1000.0, sy * 1000.0, s=2.0, c="#2f6fed", alpha=0.7, label="acetabulum")
        # Project sphere centres into the slab plane (depth along ML is zeroed at head_c).
        n, u, v = _orthonormal_basis(ml)
        head_uv = ((head_c - head_c) @ u, (head_c - head_c) @ v)
        socket_uv = ((socket_c - head_c) @ u, (socket_c - head_c) @ v)
        cx, cy = _circle_uv(float(head_fit["radius_m"]))
        ax.plot(
            (cx + head_uv[0]) * 1000.0,
            (cy + head_uv[1]) * 1000.0,
            color="#8b0000",
            lw=1.2,
            label="head sphere",
        )
        cx, cy = _circle_uv(float(socket_fit["radius_m"]))
        ax.plot(
            (cx + socket_uv[0]) * 1000.0,
            (cy + socket_uv[1]) * 1000.0,
            color="#003399",
            lw=1.2,
            label="socket sphere",
        )
        ax.plot(
            [head_uv[0] * 1000.0, socket_uv[0] * 1000.0],
            [head_uv[1] * 1000.0, socket_uv[1] * 1000.0],
            "k--",
            lw=1.0,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_title(f"{side}  centre offset={offset_mm:.2f} mm")
        ax.legend(fontsize=7, loc="best")
        del n
    fig.suptitle(_title(beta, pose, "hip_section"))
    _save_figure(fig, path)
    plt.close(fig)


def _render_knee_section(
    *,
    path: Path,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5))
    half_thickness = 0.0035
    for ax, side in zip(axes, _SIDES):
        medial_c = domains.require(f"{side}/femoral_condyle_medial")
        lateral_c = domains.require(f"{side}/femoral_condyle_lateral")
        medial_p = domains.require(f"{side}/tibial_plateau_medial")
        lateral_p = domains.require(f"{side}/tibial_plateau_lateral")
        patella = domains.require(f"{side}/patella")
        ml = np.mean(vertices[lateral_c], axis=0) - np.mean(vertices[medial_c], axis=0)
        origin = 0.5 * (
            np.mean(vertices[np.concatenate((medial_c, lateral_c))], axis=0)
            + np.mean(vertices[np.concatenate((medial_p, lateral_p))], axis=0)
        )
        gaps = {
            "medial": float(
                np.min(_nearest_distance(vertices[medial_c], vertices[medial_p]))
            ),
            "lateral": float(
                np.min(_nearest_distance(vertices[lateral_c], vertices[lateral_p]))
            ),
        }
        # Also annotate plateau-to-condyle reverse so reviewers see all four
        # compartment measurements (condyle->plateau and plateau->condyle).
        gaps["medial_rev"] = float(
            np.min(_nearest_distance(vertices[medial_p], vertices[medial_c]))
        )
        gaps["lateral_rev"] = float(
            np.min(_nearest_distance(vertices[lateral_p], vertices[lateral_c]))
        )
        series = (
            (medial_c, "#cc3333", "femoral condyle M"),
            (lateral_c, "#aa2222", "femoral condyle L"),
            (medial_p, "#2f6fed", "tibial plateau M"),
            (lateral_p, "#1a4fb8", "tibial plateau L"),
            (patella, "#d0a000", "patella"),
        )
        for ids, color, label in series:
            x, y = _slab_points(
                vertices[ids],
                origin=origin,
                normal=ml,
                half_thickness_m=half_thickness,
            )
            ax.scatter(x * 1000.0, y * 1000.0, s=2.0, c=color, alpha=0.75, label=label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_title(
            f"{side}  gaps M={gaps['medial']*1000:.2f}/{gaps['medial_rev']*1000:.2f} "
            f"L={gaps['lateral']*1000:.2f}/{gaps['lateral_rev']*1000:.2f} mm"
        )
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(_title(beta, pose, "knee_section"))
    _save_figure(fig, path)
    plt.close(fig)


def _elbow_contact_ids(
    lookup: Mapping[str, tuple[int, int, int]],
    reference: np.ndarray,
    side: str,
) -> dict[str, np.ndarray]:
    suffix = "L" if side == "left" else "R"
    humerus = _mesh_ids(lookup, f"Humerus_{suffix}")
    ulna = _mesh_ids(lookup, f"Ulna_{suffix}")
    radius = _mesh_ids(lookup, f"Radius_{suffix}")
    forearm = np.unique(np.concatenate((ulna, radius)))
    humerus_contact = _closest_subset(
        humerus, reference, reference[forearm], fraction=0.20, minimum=8, maximum=512
    )
    ulna_contact = _closest_subset(
        ulna, reference, reference[humerus], fraction=0.20, minimum=8, maximum=512
    )
    radius_contact = _closest_subset(
        radius, reference, reference[humerus], fraction=0.20, minimum=8, maximum=512
    )
    if min(len(humerus_contact), len(ulna_contact), len(radius_contact)) < 8:
        raise ValueError(f"{side} elbow contact domains are too small")
    return {
        "humerus": humerus_contact,
        "ulna": ulna_contact,
        "radius": radius_contact,
        "humerus_all": humerus,
        "ulna_all": ulna,
        "radius_all": radius,
    }


def _render_elbow_section(
    *,
    path: Path,
    vertices: np.ndarray,
    rest: np.ndarray,
    asset: Any,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    lookup = _mesh_lookup(asset)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5))
    half_thickness = 0.0035
    for ax, side in zip(axes, _SIDES):
        ids = _elbow_contact_ids(lookup, rest, side)
        origin = np.mean(vertices[ids["humerus"]], axis=0)
        ml = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        gap_hu = float(
            np.min(_nearest_distance(vertices[ids["humerus"]], vertices[ids["ulna"]]))
        )
        gap_hr = float(
            np.min(_nearest_distance(vertices[ids["humerus"]], vertices[ids["radius"]]))
        )
        for key, color, label in (
            ("humerus", "#cc3333", "humerus"),
            ("ulna", "#2f6fed", "ulna"),
            ("radius", "#d0a000", "radius"),
        ):
            x, y = _slab_points(
                vertices[ids[key]],
                origin=origin,
                normal=ml,
                half_thickness_m=half_thickness,
            )
            ax.scatter(x * 1000.0, y * 1000.0, s=2.5, c=color, alpha=0.8, label=label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_title(
            f"{side}  H-U={gap_hu*1000:.2f} mm  H-R={gap_hr*1000:.2f} mm"
        )
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(_title(beta, pose, "elbow_section"))
    _save_figure(fig, path)
    plt.close(fig)


def _sphere_azimuth_elevation(
    points: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    radius = np.linalg.norm(delta, axis=1)
    safe = np.maximum(radius, 1.0e-12)
    unit = delta / safe[:, None]
    azimuth = np.degrees(np.arctan2(unit[:, 2], unit[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(unit[:, 1], -1.0, 1.0)))
    return azimuth, elevation, radius


def _render_hip_contact_heatmap(
    *,
    path: Path,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    last_mappable = None
    for ax, side in zip(axes, _SIDES):
        head_ids = domains.require(f"{side}/femoral_head")
        socket_ids = domains.require(f"{side}/acetabulum")
        head_fit = fit_sphere_v7(vertices[head_ids])
        if not head_fit["available"]:
            raise ValueError(f"{side} femoral-head sphere fit failed")
        socket_fit = fit_sphere_fixed_radius_v7(
            vertices[socket_ids],
            radius_m=float(head_fit["radius_m"]),
            initial_center=np.asarray(head_fit["center"], dtype=np.float64),
        )
        if not socket_fit["available"]:
            raise ValueError(f"{side} acetabulum sphere fit failed")
        head_pts = vertices[head_ids]
        unsigned = _nearest_distance(head_pts, vertices[socket_ids])
        socket_c = np.asarray(socket_fit["center"], dtype=np.float64)
        socket_r = float(socket_fit["radius_m"])
        radial = np.linalg.norm(head_pts - socket_c, axis=1)
        # Negative when the head vertex sits inside the fitted acetabulum sphere.
        signed = np.where(radial < socket_r, -unsigned, unsigned)
        az, el, _r = _sphere_azimuth_elevation(head_pts, head_fit["center"])
        last_mappable = ax.scatter(
            az,
            el,
            c=signed * 1000.0,
            s=8.0,
            cmap="coolwarm",
            vmin=-3.0,
            vmax=3.0,
        )
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("elevation (deg)")
        ax.set_title(f"{side}  med={np.median(signed)*1000:.2f} mm")
    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("signed clearance (mm)")
    fig.suptitle(_title(beta, pose, "hip_contact_heatmap"))
    _save_figure(fig, path)
    plt.close(fig)


def _render_knee_condyle_heatmap(
    *,
    path: Path,
    vertices: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    panels = (
        ("left", "medial", axes[0, 0]),
        ("left", "lateral", axes[0, 1]),
        ("right", "medial", axes[1, 0]),
        ("right", "lateral", axes[1, 1]),
    )
    last_mappable = None
    for side, compartment, ax in panels:
        condyle = domains.require(f"{side}/femoral_condyle_{compartment}")
        plateau = domains.require(f"{side}/tibial_plateau_{compartment}")
        pts = vertices[condyle]
        dist = _nearest_distance(pts, vertices[plateau])
        center = np.mean(pts, axis=0)
        # Flatten about the condyle centroid onto a local PCA plane.
        centered = pts - center
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        u_axis, v_axis = vt[0], vt[1]
        x = centered @ u_axis
        y = centered @ v_axis
        last_mappable = ax.scatter(
            x * 1000.0,
            y * 1000.0,
            c=dist * 1000.0,
            s=10.0,
            cmap="viridis",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_title(f"{side}/{compartment}  min={np.min(dist)*1000:.2f} mm")
    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("nearest plateau distance (mm)")
    fig.suptitle(_title(beta, pose, "knee_condyle_heatmap"))
    _save_figure(fig, path)
    plt.close(fig)


def _patella_centroid_path(
    frames: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    *,
    side: str,
) -> np.ndarray:
    patella = domains.require(f"{side}/patella")
    trochlea = domains.require(f"{side}/trochlea")
    offsets = []
    for frame in frames:
        offsets.append(np.mean(frame[patella], axis=0) - np.mean(frame[trochlea], axis=0))
    path = np.stack(offsets, axis=0)
    return path - path[0]


def _build_patella_sweep(
    *,
    subject: Any,
    domains: FrozenJointMaterialDomainsV7,
    law: Any,
    sweep_count: int,
) -> dict[str, Any]:
    asset = subject.rigged_asset
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(asset.faces)
    sweep_poses = synthetic_knee_sweep_poses_v7(count=int(sweep_count))
    flexion_rad = np.asarray(
        [float(pose.pose_axis_angle[4, 0]) for pose in sweep_poses],
        dtype=np.float64,
    )
    candidate = []
    for pose_spec in sweep_poses:
        frame = apply_subject_pose(
            subject,
            pose_axis_angle=pose_spec.pose_axis_angle,
            transl=pose_spec.transl,
            validate=False,
        )
        candidate.append(np.asarray(frame, dtype=np.float64))
    candidate_frames = np.stack(candidate, axis=0)

    contact_tables: dict[str, np.ndarray] = {}
    for side in _SIDES:
        table, _report = solve_patella_contact_corrections_v7(
            law,
            vertices=rest,
            faces=faces,
            domains=domains,
            asset=asset,
            side=side,
            knee_axis_local=law.axis_knee_local[side],
        )
        contact_tables[side] = np.asarray(table, dtype=np.float64)

    # Interpolate contact tables onto the sweep flexion samples.
    knots = np.asarray(law.knots_deg, dtype=np.float64).reshape(-1)
    side_contact: dict[str, np.ndarray] = {}
    for side in _SIDES:
        table = contact_tables[side]
        rows = []
        for angle in flexion_rad:
            deg = float(abs(np.degrees(angle)))
            rows.append(
                np.asarray(
                    [np.interp(deg, knots, table[:, axis]) for axis in range(3)],
                    dtype=np.float64,
                )
            )
        side_contact[side] = np.stack(rows, axis=0)

    oracle_frames, _oracle_report = patella_oracle_sweep_v7(
        law,
        asset=asset,
        domains=domains,
        flexion_rad=flexion_rad,
        side_contact_translations=side_contact,
        knee_axis_local=law.axis_knee_local,
        base_vertices=rest,
    )
    oracle_frames = np.asarray(oracle_frames, dtype=np.float64)
    return {
        "flexion_rad": flexion_rad,
        "candidate": candidate_frames,
        "oracle": oracle_frames,
        "faces": faces,
        "axis_knee_local": {
            side: np.asarray(law.axis_knee_local[side], dtype=np.float64)
            for side in _SIDES
        },
    }


def _render_patella_track(
    *,
    path: Path,
    sweep: Mapping[str, Any],
    domains: FrozenJointMaterialDomainsV7,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    candidate = np.asarray(sweep["candidate"], dtype=np.float64)
    oracle = np.asarray(sweep["oracle"], dtype=np.float64)
    faces = np.asarray(sweep["faces"])
    for ax, side in zip(axes, _SIDES):
        cand_path = _patella_centroid_path(candidate, domains, side=side)
        ora_path = _patella_centroid_path(oracle, domains, side=side)
        metrics = patellofemoral_trajectory_metrics_v7(
            domains,
            posed_vertices=candidate,
            oracle_vertices=oracle,
            faces=faces,
            side=side,
        )
        if not metrics.get("available", False):
            raise ValueError(
                f"{side} patella trajectory unavailable: {metrics.get('reason')}"
            )
        # Project into the plane perpendicular to the hinge axis (femur-local).
        axis = _unit(sweep["axis_knee_local"][side], f"{side} knee axis")
        helper = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        u = _unit(np.cross(helper, axis), "patella u")
        v = _unit(np.cross(axis, u), "patella v")
        ax.plot(
            (ora_path @ u) * 1000.0,
            (ora_path @ v) * 1000.0,
            color="#7a7a7a",
            lw=2.0,
            label="oracle",
        )
        ax.plot(
            (cand_path @ u) * 1000.0,
            (cand_path @ v) * 1000.0,
            color="#cc3333",
            lw=2.0,
            label="candidate",
        )
        ax.scatter(
            [(cand_path @ u)[0] * 1000.0],
            [(cand_path @ v)[0] * 1000.0],
            c="k",
            s=20,
            zorder=5,
        )
        # Mark hinge axis as the origin normal (visual tick).
        ax.axhline(0.0, color="#3366cc", lw=0.8, ls="--", label="hinge plane")
        ax.axvline(0.0, color="#3366cc", lw=0.8, ls=":")
        rms_mm = float(metrics["trajectory_rms_m"]) * 1000.0
        max_mm = float(metrics["trajectory_max_m"]) * 1000.0
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_title(f"{side}  RMS={rms_mm:.2f} mm  max={max_mm:.2f} mm")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(_title(beta, pose, "patella_track"))
    _save_figure(fig, path)
    plt.close(fig)


def _local_component_faces(faces: np.ndarray, global_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    if not len(ids):
        return np.empty((0, 3), dtype=np.int64)
    remap = -np.ones(int(np.max(ids)) + 1, dtype=np.int64)
    remap[ids] = np.arange(len(ids), dtype=np.int64)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if int(np.max(triangles)) >= len(remap):
        # Faces may reference vertices beyond this component's max id.
        keep = np.all(triangles < len(remap), axis=1)
        triangles = triangles[keep]
    mapped = remap[triangles]
    mask = np.all(mapped >= 0, axis=1)
    return mapped[mask]


def _edge_weight_adjacency(
    vertex_count: int, faces: np.ndarray, points: np.ndarray
):
    from scipy import sparse

    if vertex_count <= 0 or not len(faces):
        return sparse.csr_matrix((vertex_count, vertex_count))
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    a = edges[:, 0]
    b = edges[:, 1]
    lengths = np.linalg.norm(points[a] - points[b], axis=1)
    data = np.concatenate((lengths, lengths))
    rows = np.concatenate((a, b))
    cols = np.concatenate((b, a))
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(vertex_count, vertex_count)
    )


def _centerline_samples_for_component(
    *,
    global_ids: np.ndarray,
    faces: np.ndarray,
    reference_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    bin_count: int = 32,
) -> dict[str, Any] | None:
    """Minimal geodesic-bin centerline for plotting.

    ``vessel_gates_v7`` keeps its centerline builder private; this reimplements
    only the per-component sampling needed for the evidence figure.
    """
    from scipy.sparse.csgraph import connected_components, dijkstra

    ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    if len(ids) < 24:
        return None
    local_faces = _local_component_faces(faces, ids)
    if not len(local_faces):
        return None
    reference_local = np.asarray(reference_vertices[ids], dtype=np.float64)
    posed_local = np.asarray(posed_vertices[ids], dtype=np.float64)
    graph = _edge_weight_adjacency(len(ids), local_faces, reference_local)
    component_count, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    best: dict[str, Any] | None = None
    for component_id in range(int(component_count)):
        members = np.flatnonzero(labels == component_id)
        if len(members) < 24:
            continue
        from scipy import sparse

        sub = graph[members][:, members]
        # Diameter endpoints via double BFS on hop metric.
        from scipy.sparse.csgraph import breadth_first_order

        degrees = np.asarray(sub.getnnz(axis=1)).reshape(-1)
        seeds = np.flatnonzero(degrees > 0)
        if not len(seeds):
            continue
        order, _ = breadth_first_order(
            sub, i_start=int(seeds[0]), directed=False, return_predecessors=True
        )
        if not len(order):
            continue
        u = int(order[-1])
        order_u, _ = breadth_first_order(
            sub, i_start=u, directed=False, return_predecessors=True
        )
        if not len(order_u):
            continue
        v = int(order_u[-1])
        distances, predecessors = dijkstra(
            sub, directed=False, indices=u, return_predecessors=True
        )
        if not np.isfinite(distances[v]):
            continue
        # Reconstruct shortest path.
        path: list[int] = []
        node = int(v)
        seen: set[int] = set()
        while node != u and node >= 0:
            if node in seen:
                path = []
                break
            seen.add(node)
            path.append(node)
            node = int(predecessors[node])
        if not path or node < 0:
            continue
        path.append(u)
        path.reverse()
        path_arr = np.asarray(path, dtype=np.int64)
        path_points = reference_local[members][path_arr]
        seg = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        span = float(arc[-1])
        if span <= 1.0e-9:
            continue
        # Geodesic distance from start for all members, then bin.
        member_dist = distances
        finite = np.isfinite(member_dist)
        main_vertices = np.flatnonzero(finite)
        if len(main_vertices) < 8:
            continue
        raw = member_dist[main_vertices] / span * float(bin_count)
        bin_ids = np.clip(np.floor(raw).astype(np.int64), 0, bin_count - 1)
        ref_samples = []
        posed_samples = []
        for bin_index in range(int(bin_count)):
            hit = main_vertices[bin_ids == bin_index]
            if not len(hit):
                continue
            # hit indexes into the subgraph member list.
            global_members = members[hit]
            ref_samples.append(np.mean(reference_local[global_members], axis=0))
            posed_samples.append(np.mean(posed_local[global_members], axis=0))
        if len(ref_samples) < 3:
            continue
        reference_samples = np.stack(ref_samples, axis=0)
        posed_samples_arr = np.stack(posed_samples, axis=0)

        def _turns(samples: np.ndarray) -> np.ndarray:
            segments = np.diff(samples, axis=0)
            norms = np.linalg.norm(segments, axis=1)
            valid = norms > 1.0e-12
            if np.count_nonzero(valid) < 2:
                return np.empty((0,), dtype=np.float64)
            directions = segments[valid] / norms[valid, None]
            cosine = np.clip(
                np.einsum("ij,ij->i", directions[:-1], directions[1:]), -1.0, 1.0
            )
            return np.degrees(np.arccos(cosine))

        ref_turns = _turns(reference_samples)
        posed_turns = _turns(posed_samples_arr)
        if not len(ref_turns) or not len(posed_turns):
            continue
        # Arc position of each turn is midway along the sample polyline.
        sample_seg = np.linalg.norm(np.diff(posed_samples_arr, axis=0), axis=1)
        sample_arc = np.concatenate([[0.0], np.cumsum(sample_seg)])
        turn_arc = sample_arc[1:-1] if len(sample_arc) >= 3 else sample_arc[: len(posed_turns)]
        turn_arc = turn_arc[: len(posed_turns)]
        max_increase = float(np.max(posed_turns) - np.max(ref_turns))
        candidate = {
            "reference_turns": ref_turns,
            "posed_turns": posed_turns,
            "turn_arc_m": np.asarray(turn_arc, dtype=np.float64),
            "reference_arc_m": sample_arc[1 : 1 + len(ref_turns)],
            "max_turn_increase_deg": max_increase,
            "posed_max_turn_deg": float(np.max(posed_turns)),
            "reference_max_turn_deg": float(np.max(ref_turns)),
            "vertex_count": int(len(members)),
        }
        if best is None or max_increase > float(best["max_turn_increase_deg"]):
            best = candidate
        del sparse
    return best


def _render_vessel_centerline(
    *,
    path: Path,
    vertices: np.ndarray,
    rest: np.ndarray,
    asset: Any,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    faces = np.asarray(asset.faces, dtype=np.int64)
    by_mesh = vessel_tissue_vertex_ids_v7(asset)
    measured: list[tuple[str, dict[str, Any]]] = []
    for name, ids in sorted(by_mesh.items()):
        sample = _centerline_samples_for_component(
            global_ids=ids,
            faces=faces,
            reference_vertices=rest,
            posed_vertices=vertices,
        )
        if sample is None:
            continue
        measured.append((name, sample))
    if not measured:
        raise ValueError("no measurable vessel/nerve centerline components")
    measured.sort(key=lambda item: float(item[1]["max_turn_increase_deg"]), reverse=True)
    worst = measured[:6]
    cols = 2
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.0, 3.2 * rows), squeeze=False)
    for index, (name, sample) in enumerate(worst):
        ax = axes[index // cols][index % cols]
        ref_arc = np.asarray(sample["reference_arc_m"], dtype=np.float64)
        posed_arc = np.asarray(sample["turn_arc_m"], dtype=np.float64)
        # Align lengths if turn counts differ slightly.
        ref_turns = np.asarray(sample["reference_turns"], dtype=np.float64)
        posed_turns = np.asarray(sample["posed_turns"], dtype=np.float64)
        n_ref = min(len(ref_arc), len(ref_turns))
        n_posed = min(len(posed_arc), len(posed_turns))
        ax.plot(
            ref_arc[:n_ref] * 1000.0,
            ref_turns[:n_ref],
            color="#7a7a7a",
            lw=1.5,
            label="rest",
        )
        ax.plot(
            posed_arc[:n_posed] * 1000.0,
            posed_turns[:n_posed],
            color="#cc3333",
            lw=1.5,
            label="posed",
        )
        ax.set_xlabel("arc (mm)")
        ax.set_ylabel("turn (deg)")
        ax.set_title(
            f"{name}\nΔmax={float(sample['max_turn_increase_deg']):.2f} deg"
        )
        ax.legend(fontsize=7, loc="best")
    for index in range(len(worst), rows * cols):
        axes[index // cols][index % cols].axis("off")
    fig.suptitle(_title(beta, pose, "vessel_centerline"))
    _save_figure(fig, path)
    plt.close(fig)


def _render_rib_connection(
    *,
    path: Path,
    vertices: np.ndarray,
    rest: np.ndarray,
    asset: Any,
    beta: str,
    pose: str,
) -> None:
    plt = _configure_matplotlib()
    lookup = _mesh_lookup(asset)
    present_ribs = [name for name in RIB_MESH_NAMES if name in lookup]
    if len(present_ribs) != 24:
        raise ValueError(f"expected 24 ribs, found {len(present_ribs)}")
    present_vertebrae = [name for name in THORACIC_VERTEBRA_NAMES if name in lookup]
    present_costal = [name for name in COSTAL_CARTILAGE_NAMES if name in lookup]
    if not present_vertebrae or not present_costal:
        raise ValueError("rib connection targets (vertebrae/costal) are missing")
    vertebra = _union_mesh_ids(lookup, present_vertebrae)
    costal = _union_mesh_ids(lookup, present_costal)
    sternum = _mesh_ids(lookup, STERNUM_MESH_NAME) if STERNUM_MESH_NAME in lookup else None
    sternal_target = (
        np.unique(np.concatenate((costal, sternum))) if sternum is not None else costal
    )

    vertebral_rest: list[float] = []
    vertebral_posed: list[float] = []
    sternal_rest: list[float] = []
    sternal_posed: list[float] = []
    labels: list[str] = []
    for name in present_ribs:
        rib = _mesh_ids(lookup, name)
        vertebral_end = _closest_subset(
            rib, rest, rest[vertebra], fraction=0.08, minimum=8, maximum=512
        )
        sternal_end = _closest_subset(
            rib, rest, rest[sternal_target], fraction=0.08, minimum=8, maximum=512
        )
        if min(len(vertebral_end), len(sternal_end)) < 8:
            raise ValueError(f"{name} connection end needs at least eight vertices")
        v_rest = float(np.min(_nearest_distance(rest[vertebral_end], rest[vertebra])))
        v_posed = float(
            np.min(_nearest_distance(vertices[vertebral_end], vertices[vertebra]))
        )
        s_rest = float(
            np.min(_nearest_distance(rest[sternal_end], rest[sternal_target]))
        )
        s_posed = float(
            np.min(_nearest_distance(vertices[sternal_end], vertices[sternal_target]))
        )
        vertebral_rest.append(v_rest)
        vertebral_posed.append(v_posed)
        sternal_rest.append(s_rest)
        sternal_posed.append(s_posed)
        labels.append(name)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    for ax, rest_vals, posed_vals, title in (
        (axes[0], vertebral_rest, vertebral_posed, "vertebral end distance"),
        (axes[1], sternal_rest, sternal_posed, "sternal/costal end distance"),
    ):
        ax.plot(x, np.asarray(rest_vals) * 1000.0, "o-", color="#7a7a7a", label="rest")
        ax.plot(x, np.asarray(posed_vals) * 1000.0, "s-", color="#cc3333", label="posed")
        for index, (a, b) in enumerate(zip(rest_vals, posed_vals)):
            ax.plot([index, index], [a * 1000.0, b * 1000.0], color="#bbbbbb", lw=0.8)
        ax.set_ylabel("distance (mm)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_xlabel("rib")
    fig.suptitle(_title(beta, pose, "rib_connection"))
    _save_figure(fig, path)
    plt.close(fig)


def _render_one_view(
    *,
    view: str,
    path: Path,
    vertices: np.ndarray,
    rest: np.ndarray,
    subject: Any,
    domains: FrozenJointMaterialDomainsV7,
    body_surface: tuple[np.ndarray, np.ndarray] | None,
    sweep: Mapping[str, Any] | None,
    beta: str,
    pose: str,
) -> None:
    asset = subject.rigged_asset
    if view == "surface_front":
        _render_surface(
            path=path,
            vertices=vertices,
            body_surface=body_surface,
            beta=beta,
            pose=pose,
            view=view,
            projection="front",
        )
    elif view == "surface_side":
        _render_surface(
            path=path,
            vertices=vertices,
            body_surface=body_surface,
            beta=beta,
            pose=pose,
            view=view,
            projection="side",
        )
    elif view == "hip_section":
        _render_hip_section(
            path=path, vertices=vertices, domains=domains, beta=beta, pose=pose
        )
    elif view == "knee_section":
        _render_knee_section(
            path=path, vertices=vertices, domains=domains, beta=beta, pose=pose
        )
    elif view == "elbow_section":
        _render_elbow_section(
            path=path,
            vertices=vertices,
            rest=rest,
            asset=asset,
            beta=beta,
            pose=pose,
        )
    elif view == "hip_contact_heatmap":
        _render_hip_contact_heatmap(
            path=path, vertices=vertices, domains=domains, beta=beta, pose=pose
        )
    elif view == "knee_condyle_heatmap":
        _render_knee_condyle_heatmap(
            path=path, vertices=vertices, domains=domains, beta=beta, pose=pose
        )
    elif view == "patella_track":
        if sweep is None:
            raise ValueError("patella sweep data is unavailable")
        _render_patella_track(
            path=path, sweep=sweep, domains=domains, beta=beta, pose=pose
        )
    elif view == "vessel_centerline":
        _render_vessel_centerline(
            path=path,
            vertices=vertices,
            rest=rest,
            asset=asset,
            beta=beta,
            pose=pose,
        )
    elif view == "rib_connection":
        _render_rib_connection(
            path=path,
            vertices=vertices,
            rest=rest,
            asset=asset,
            beta=beta,
            pose=pose,
        )
    else:
        raise ValueError(f"unknown evidence view {view!r}")


def generate_evidence_pack_v7(
    *,
    subjects: Sequence[MatrixSubjectSpecV7],
    poses: Sequence[MatrixPoseSpecV7],
    domains: FrozenJointMaterialDomainsV7,
    law: Any,
    operator_digest: str,
    output_dir: str | Path,
    posed_dir: str | Path | None = None,
    sweep_count: int = 13,
    command: str = "",
) -> dict[str, Any]:
    if not subjects:
        raise ValueError("subjects must be non-empty")
    if not poses:
        raise ValueError("poses must be non-empty")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    posed_path = (
        None if posed_dir is None else Path(posed_dir).expanduser().resolve()
    )
    digest = str(operator_digest).strip()
    if len(digest) < 8:
        raise ValueError("operator_digest must contain at least 8 characters")

    files: list[dict[str, Any]] = []
    missing_views: list[dict[str, Any]] = []
    sweep_by_subject: dict[str, Mapping[str, Any] | None] = {}
    sweep_error_by_subject: dict[str, str] = {}

    for subject_spec in subjects:
        subject = subject_spec.subject
        rest = np.asarray(subject.rigged_asset.vertices_rest, dtype=np.float64)
        domains.validate_topology(rest, np.asarray(subject.rigged_asset.faces))
        asset_file_digest = _file_sha256(subject_spec.path)
        subject_digest = str(subject.content_digest())

        if subject_spec.label not in sweep_by_subject:
            try:
                sweep_by_subject[subject_spec.label] = _build_patella_sweep(
                    subject=subject,
                    domains=domains,
                    law=law,
                    sweep_count=int(sweep_count),
                )
            except Exception as exc:
                sweep_by_subject[subject_spec.label] = None
                sweep_error_by_subject[subject_spec.label] = str(exc)

        for pose_spec in poses:
            pose_digest = smplx_pose_hash(
                pose_spec.pose_axis_angle, pose_spec.transl
            )
            try:
                vertices, geometry_source = _resolve_posed_vertices(
                    subject_spec=subject_spec,
                    pose_spec=pose_spec,
                    posed_dir=posed_path,
                )
            except Exception as exc:
                for view in REQUIRED_VIEWS_V7:
                    missing_views.append(
                        {
                            "beta": subject_spec.label,
                            "pose": pose_spec.label,
                            "view": view,
                            "reason": f"posed geometry unavailable: {exc}",
                        }
                    )
                continue

            body_surface, _body_meta = body_surface_for_cell_v7(
                subject=subject, pose_spec=pose_spec
            )
            sweep = sweep_by_subject.get(subject_spec.label)

            for view in REQUIRED_VIEWS_V7:
                stem = evidence_file_stem_v7(
                    operator_digest=digest,
                    beta=subject_spec.label,
                    pose=pose_spec.label,
                    view=view,
                )
                png_path = out / f"{stem}.png"
                if view == "patella_track" and sweep is None:
                    missing_views.append(
                        {
                            "beta": subject_spec.label,
                            "pose": pose_spec.label,
                            "view": view,
                            "reason": (
                                "patella sweep unavailable: "
                                + sweep_error_by_subject.get(
                                    subject_spec.label, "unknown"
                                )
                            ),
                        }
                    )
                    continue
                try:
                    _render_one_view(
                        view=view,
                        path=png_path,
                        vertices=vertices,
                        rest=rest,
                        subject=subject,
                        domains=domains,
                        body_surface=body_surface,
                        sweep=sweep,
                        beta=subject_spec.label,
                        pose=pose_spec.label,
                    )
                    if not png_path.is_file() or png_path.stat().st_size < 64:
                        raise ValueError("renderer produced an empty or tiny PNG")
                    payload = {
                        "operator_digest": digest,
                        "subject_digest": subject_digest,
                        "beta": subject_spec.label,
                        "pose_digest": str(pose_digest),
                        "asset_file_digest": asset_file_digest,
                        "view": view,
                        "command": str(command),
                        "geometry_source": geometry_source,
                        "schema_version": EVIDENCE_PACK_SCHEMA_VERSION,
                    }
                    sidecar = write_evidence_sidecar_v7(png_path, payload)
                    files.append(
                        {
                            "png": str(png_path),
                            "sidecar": str(sidecar),
                            "beta": subject_spec.label,
                            "pose": pose_spec.label,
                            "view": view,
                            "payload": payload,
                        }
                    )
                except Exception as exc:
                    if png_path.exists():
                        png_path.unlink()
                    sidecar_path = png_path.with_suffix(".json")
                    if sidecar_path.exists():
                        sidecar_path.unlink()
                    missing_views.append(
                        {
                            "beta": subject_spec.label,
                            "pose": pose_spec.label,
                            "view": view,
                            "reason": str(exc),
                        }
                    )

    expected = len(subjects) * len(poses) * len(REQUIRED_VIEWS_V7)
    complete = bool(not missing_views and len(files) == expected)
    return {
        "schema_version": EVIDENCE_PACK_SCHEMA_VERSION,
        "operator_digest": digest,
        "output_dir": str(out),
        "files": files,
        "missing_views": missing_views,
        "complete": complete,
        "required_views": list(REQUIRED_VIEWS_V7),
        "expected_file_count": expected,
        "generated_file_count": len(files),
    }


__all__ = [
    "EVIDENCE_PACK_SCHEMA_VERSION",
    "REQUIRED_VIEWS_V7",
    "evidence_file_stem_v7",
    "write_evidence_sidecar_v7",
    "generate_evidence_pack_v7",
]
