"""Independent material review for source/candidate anatomy cells.

This renderer intentionally consumes a small, explicit per-cell contract instead
of rebuilding a Blender scene.  A cell is an ``.npz`` containing

``faces, skin_faces, source_vertices, candidate_vertices, skin_vertices,
smplx_joints, pose``

and either ``vertex_tissue`` or ``tissue`` (one label per anatomy vertex).
The three-dimensional panels are PyVista off-screen renders.  The camera is
constructed only from the posed SMPL-X joints and the SMPL-X skin bounding box;
neither anatomy mesh is allowed to influence framing.  The root axis-angle is
removed once and the same inverse rotation is applied to every point array.

Each input produces:

* ``whole_3d.png`` plus left/right elbow, knee, ankle and a feet-context view.
  Every image has source and candidate panels with exactly the same camera.
* Left/right forearm and shank section plots.  Each section plane is the
  midpoint of the corresponding SMPL-X joint pair and its normal is the
  joint-to-joint axis.  Lines are obtained with
  :func:`trimesh.intersections.mesh_plane` and shown in millimetres.

The module does not edit or overwrite an existing output directory.  PyVista
and VTK are imported lazily so schema/contract checks and ``--help`` remain
usable on machines that do not have the renderer installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# The section plots are rendered in a headless worker as well as interactively.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


SMPLX_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)

# Material colors are deliberately stable across all review packs.
TISSUE_COLORS: dict[str, tuple[float, float, float]] = {
    # Keep bone pale and nerve yellow so the two internal line families remain
    # separable in both the 3-D panel and the section plot.
    "bone": (0.88, 0.86, 0.72),
    "vessel": (0.86, 0.12, 0.08),
    "nerve": (0.98, 0.82, 0.12),
    "other": (0.48, 0.38, 0.56),
}
DEFAULT_INTEGER_TISSUE_MAP: dict[str, str] = {
    "0": "bone",
    "1": "vessel",
    "2": "nerve",
    "3": "organ",
    "4": "heart",
    "5": "connective",
}
SECTION_COLORS: dict[str, str] = {
    "skin": "#4f83cc",
    "bone": "#b9a35e",
    "vessel": "#d53b2f",
    "nerve": "#f0c419",
}
SECTION_ORDER: tuple[str, ...] = ("skin", "bone", "vessel", "nerve")

# SMPL-X 55-joint ids used by the camera and the section planes.
REGION_JOINTS: dict[str, tuple[int, ...]] = {
    "whole": tuple(range(0, 22)),
    "left_elbow": (16, 18, 20),
    "right_elbow": (17, 19, 21),
    "left_knee": (1, 4, 7),
    "right_knee": (2, 5, 8),
    "left_ankle": (4, 7, 10),
    "right_ankle": (5, 8, 11),
    "feet": (7, 8, 10, 11),
}
SECTION_JOINTS: dict[str, tuple[int, int]] = {
    "forearm": (18, 20),
    "right_forearm": (19, 21),
    "shank": (4, 7),
    "right_shank": (5, 8),
}


class MaterialReviewError(ValueError):
    """Raised when a per-cell review contract is malformed."""


@dataclass(frozen=True)
class ReviewCell:
    path: Path
    faces: np.ndarray
    skin_faces: np.ndarray
    source_vertices: np.ndarray
    candidate_vertices: np.ndarray
    skin_vertices: np.ndarray
    smplx_joints: np.ndarray
    pose: np.ndarray
    tissue_labels: np.ndarray
    tissue_names: tuple[str, ...]
    source_sha256: str

    @property
    def stem(self) -> str:
        return self.path.stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays into manifest-safe values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _label_text(value: Any) -> str:
    value = _scalar(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _canonical_tissue(value: Any) -> str:
    """Map common source labels to the four review categories."""

    text = _label_text(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"bone", "bones", "skeletal", "skeleton", "osseous"}:
        return "bone"
    if any(token in text for token in ("vessel", "artery", "vein", "vascular")):
        return "vessel"
    if any(token in text for token in ("nerve", "neural", "nervous")):
        return "nerve"
    return "other"


def _parse_tissue_map(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        raw = candidate.read_text(encoding="utf-8")
    else:
        raw = spec
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MaterialReviewError(f"--tissue-map must be JSON or a JSON file: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterialReviewError("--tissue-map must decode to an object")
    return {str(key): _label_text(item) for key, item in value.items()}


def _array_from_npz(data: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        return np.asarray(data[key])
    except KeyError as exc:
        raise MaterialReviewError(f"cell is missing required array {key!r}") from exc


def _tissue_names_from_npz(data: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("tissue_names", "tissues", "tissue_labels"):
        if key not in data:
            continue
        value = np.asarray(data[key], dtype=object).reshape(-1)
        return tuple(_label_text(item) for item in value)
    return ()


def _load_cell(path: Path, tissue_map: Mapping[str, str]) -> ReviewCell:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        faces = _array_from_npz(data, "faces").astype(np.int64, copy=False)
        skin_faces = _array_from_npz(data, "skin_faces").astype(np.int64, copy=False)
        source_vertices = _array_from_npz(data, "source_vertices").astype(np.float64, copy=False)
        candidate_vertices = _array_from_npz(data, "candidate_vertices").astype(np.float64, copy=False)
        skin_vertices = _array_from_npz(data, "skin_vertices").astype(np.float64, copy=False)
        smplx_joints = _array_from_npz(data, "smplx_joints").astype(np.float64, copy=False)
        pose = _array_from_npz(data, "pose").astype(np.float64, copy=False)
        tissue_key = "vertex_tissue" if "vertex_tissue" in keys else "tissue"
        tissue_labels = _array_from_npz(data, tissue_key).reshape(-1)
        tissue_names = _tissue_names_from_npz(data)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise MaterialReviewError(f"{path}: faces must have shape (F,3), got {faces.shape}")
    if skin_faces.ndim != 2 or skin_faces.shape[1] != 3:
        raise MaterialReviewError(
            f"{path}: skin_faces must have shape (F,3), got {skin_faces.shape}"
        )
    for name, points in (
        ("source_vertices", source_vertices),
        ("candidate_vertices", candidate_vertices),
        ("skin_vertices", skin_vertices),
    ):
        if points.ndim != 2 or points.shape[1] != 3:
            raise MaterialReviewError(f"{path}: {name} must have shape (N,3), got {points.shape}")
        if not np.isfinite(points).all():
            raise MaterialReviewError(f"{path}: {name} contains non-finite values")
    if source_vertices.shape[0] != candidate_vertices.shape[0]:
        raise MaterialReviewError(
            f"{path}: source/candidate vertex count differs: "
            f"{source_vertices.shape[0]} != {candidate_vertices.shape[0]}"
        )
    if tissue_labels.shape[0] != source_vertices.shape[0]:
        raise MaterialReviewError(
            f"{path}: tissue label count {tissue_labels.shape[0]} does not match "
            f"anatomy vertex count {source_vertices.shape[0]}"
        )
    if smplx_joints.shape != (55, 3):
        raise MaterialReviewError(f"{path}: smplx_joints must have shape (55,3), got {smplx_joints.shape}")
    if pose.shape not in {(55, 3), (3,), (3, 3)}:
        raise MaterialReviewError(
            f"{path}: pose must be axis-angle (55,3), root axis-angle (3,), "
            f"or root rotation matrix (3,3); got {pose.shape}"
        )
    if faces.size and (faces.min() < 0 or faces.max() >= source_vertices.shape[0]):
        raise MaterialReviewError(f"{path}: anatomy faces index outside vertex array")
    if skin_faces.size and (skin_faces.min() < 0 or skin_faces.max() >= skin_vertices.shape[0]):
        raise MaterialReviewError(f"{path}: skin_faces index outside skin vertex array")

    labels = tissue_labels.copy()
    effective_tissue_map = dict(tissue_map)
    if not effective_tissue_map and np.issubdtype(labels.dtype, np.integer):
        effective_tissue_map = DEFAULT_INTEGER_TISSUE_MAP
    if effective_tissue_map:
        labels = np.asarray(
            [effective_tissue_map.get(_label_text(item), _label_text(item)) for item in labels],
            dtype=object,
        )
    elif tissue_names:
        # Numeric per-vertex labels commonly index a compact tissue-name table.
        mapped: list[Any] = []
        for item in labels:
            scalar = _scalar(item)
            if isinstance(scalar, (int, np.integer)) and 0 <= int(scalar) < len(tissue_names):
                mapped.append(tissue_names[int(scalar)])
            else:
                mapped.append(scalar)
        labels = np.asarray(mapped, dtype=object)

    return ReviewCell(
        path=path,
        faces=faces,
        skin_faces=skin_faces,
        source_vertices=source_vertices,
        candidate_vertices=candidate_vertices,
        skin_vertices=skin_vertices,
        smplx_joints=smplx_joints,
        pose=pose,
        tissue_labels=labels,
        tissue_names=tissue_names,
        source_sha256=_sha256(path),
    )


def _root_rotation(pose: np.ndarray) -> Rotation:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (55, 3):
        return Rotation.from_rotvec(pose[0])
    if pose.shape == (3,):
        return Rotation.from_rotvec(pose)
    return Rotation.from_matrix(pose)


def _root_unrotate(
    points: np.ndarray,
    root: Rotation,
    *,
    pivot: np.ndarray,
) -> np.ndarray:
    """Apply one inverse root rotation around the SMPL-X pelvis pivot.

    SMPL-X's root transform is translated to the rest pelvis before the root
    rotation is applied.  Rotating around world zero would move the pelvis and
    make a capture look translated while it is being made upright.  The same
    pivot and inverse are used for anatomy, skin, and joints so source and
    candidate stay in one frame.
    """

    value = np.asarray(points, dtype=np.float64)
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    return np.asarray(root.inv().apply(value - center) + center, dtype=np.float64)


def _unrotated_cell(cell: ReviewCell) -> dict[str, Any]:
    root = _root_rotation(cell.pose)
    pivot = np.asarray(cell.smplx_joints[0], dtype=np.float64)
    return {
        "source_vertices": _root_unrotate(cell.source_vertices, root, pivot=pivot),
        "candidate_vertices": _root_unrotate(cell.candidate_vertices, root, pivot=pivot),
        "skin_vertices": _root_unrotate(cell.skin_vertices, root, pivot=pivot),
        "smplx_joints": _root_unrotate(cell.smplx_joints, root, pivot=pivot),
        "root_rotvec": root.as_rotvec(),
        "root_pivot": pivot,
    }


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise MaterialReviewError(f"cannot normalize degenerate {name}")
    return vector / norm


def _body_axes(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, front axes derived only from SMPL-X joints."""

    up = _normalize(joints[15] - joints[0], name="pelvis-to-head axis")
    # left minus right gives a stable anatomical left-right axis independent of
    # the candidate geometry.  If a capture has collapsed shoulders, hips are
    # the deterministic fallback.
    right = joints[16] - joints[17]
    if np.linalg.norm(right) < 1e-10:
        right = joints[1] - joints[2]
    right = _normalize(right, name="SMPL-X left-right axis")
    front = np.cross(right, up)
    if np.linalg.norm(front) < 1e-10:
        raise MaterialReviewError("SMPL-X joints do not define a camera plane")
    front = _normalize(front, name="SMPL-X front axis")
    # Re-orthogonalize right so numerical drift cannot change the section axes.
    right = _normalize(np.cross(up, front), name="orthogonal left-right axis")
    return right, up, front


def _camera_for_region(
    *,
    region: str,
    joints: np.ndarray,
    skin: np.ndarray,
) -> dict[str, np.ndarray | float]:
    right, up, front = _body_axes(joints)
    bbox_min = skin.min(axis=0)
    bbox_max = skin.max(axis=0)
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    indices = np.asarray(REGION_JOINTS[region], dtype=np.int64)
    joint_points = joints[indices]
    target_indices = {
        "whole": 0,
        "left_elbow": 18,
        "right_elbow": 19,
        "left_knee": 4,
        "right_knee": 5,
        "left_ankle": 7,
        "right_ankle": 8,
        "feet": 7,
    }
    joint_center = joints[target_indices[region]]
    if region == "whole":
        target = (joint_center + bbox_center) * 0.5
        projected_up = np.abs(np.dot(skin - bbox_center, up))
        projected_right = np.abs(np.dot(skin - bbox_center, right))
        parallel_scale = max(float(projected_up.max() * 2.0 * 1.08), float(projected_right.max() * 2.0 * 1.08))
        radius = parallel_scale * 0.5
    else:
        target = joint_center
        segment_lengths = np.linalg.norm(np.diff(joint_points, axis=0), axis=1)
        local_span = float(np.max(segment_lengths)) if len(segment_lengths) else 0.0
        # Keep the focal region around the named joint while retaining a small
        # skin-bbox-derived floor for unusually compact shapes.
        radius = max(local_span * 0.72, bbox_diag * 0.018)
        parallel_scale = radius * 2.0
    distance = max(parallel_scale * 1.8, bbox_diag * 0.12)
    position = target + front * distance
    return {
        "position": position,
        "target": target,
        "up": up,
        "right": right,
        "front": front,
        "radius": float(radius),
        "parallel_scale": float(parallel_scale),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def _pv_polydata(pv: Any, points: np.ndarray, faces: np.ndarray) -> Any:
    if faces.size == 0:
        return None
    packed = np.column_stack(
        (np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64, copy=False))
    ).reshape(-1)
    return pv.PolyData(np.asarray(points, dtype=np.float64), packed)


def _face_codes(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    """Use the majority label for a face, avoiding object arrays in VTK."""

    vertex_codes = np.asarray([_canonical_tissue(item) for item in vertex_labels], dtype=object)
    a = vertex_codes[faces[:, 0]]
    b = vertex_codes[faces[:, 1]]
    c = vertex_codes[faces[:, 2]]
    return np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))


def _require_pyvista() -> Any:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - exercised on render hosts
        raise RuntimeError(
            "render_material_review_v13 requires PyVista and VTK; install "
            "pyvista in the render environment"
        ) from exc
    pv.OFF_SCREEN = True
    return pv


def _add_review_mesh(
    plotter: Any,
    pv: Any,
    *,
    points: np.ndarray,
    faces: np.ndarray,
    face_codes: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    label: str,
) -> None:
    skin_mesh = _pv_polydata(pv, skin, skin_faces)
    if skin_mesh is not None:
        plotter.add_mesh(
            skin_mesh,
            color=(0.34, 0.52, 0.76),
            opacity=0.12,
            smooth_shading=False,
            pickable=False,
        )
    for category in ("other", "bone", "vessel", "nerve"):
        selected = faces[face_codes == category]
        mesh = _pv_polydata(pv, points, selected)
        if mesh is None:
            continue
        opacity = {"other": 0.25, "bone": 0.86, "vessel": 0.78, "nerve": 0.92}[category]
        plotter.add_mesh(
            mesh,
            color=TISSUE_COLORS[category],
            opacity=opacity,
            smooth_shading=False,
            pickable=False,
        )
    plotter.add_text(label, position="upper_left", font_size=11, color="white")
    plotter.add_text(
        "skin  bone  vessel  nerve | camera: SMPL-X joints + skin bbox",
        position="lower_left",
        font_size=8,
        color="white",
    )


def _render_3d(
    *,
    cell: ReviewCell,
    arrays: Mapping[str, Any],
    region: str,
    output: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    pv = _require_pyvista()
    joints = np.asarray(arrays["smplx_joints"], dtype=np.float64)
    skin = np.asarray(arrays["skin_vertices"], dtype=np.float64)
    camera = _camera_for_region(region=region, joints=joints, skin=skin)
    face_codes = _face_codes(cell.faces, cell.tissue_labels)
    plotter = pv.Plotter(shape=(1, 2), window_size=(width, height), off_screen=True)
    plotter.set_background("#11151c")
    camera_position = (
        np.asarray(camera["position"]),
        np.asarray(camera["target"]),
        np.asarray(camera["up"]),
    )
    for column, (label, key) in enumerate((("source", "source_vertices"), ("candidate", "candidate_vertices"))):
        plotter.subplot(0, column)
        _add_review_mesh(
            plotter,
            pv,
            points=np.asarray(arrays[key]),
            faces=cell.faces,
            face_codes=face_codes,
            skin=skin,
            skin_faces=cell.skin_faces,
            label=label,
        )
        plotter.camera_position = camera_position
        plotter.camera.SetParallelProjection(True)
        plotter.camera.parallel_scale = float(camera["parallel_scale"])
    plotter.link_views()
    # link_views may synchronize the active renderer by fitting all actors;
    # restore the SMPL-X-only camera after linking so neither anatomy mesh can
    # change the framing.
    for column in range(2):
        plotter.subplot(0, column)
        plotter.camera_position = camera_position
        plotter.camera.SetParallelProjection(True)
        plotter.camera.parallel_scale = float(camera["parallel_scale"])
    plotter.show(screenshot=str(output), auto_close=True)
    return {
        "path": str(output),
        "camera": {key: _jsonable(value) for key, value in camera.items()},
        "source_sha256": cell.source_sha256,
    }


def _plane_basis(normal: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.cross(normal, up)
    if np.linalg.norm(u) < 1e-10:
        u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    u = _normalize(u, name="section plane basis")
    v = _normalize(np.cross(normal, u), name="section plane second basis")
    return u, v


def _mesh_plane_segments(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0:
        return (
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    segments, face_index = trimesh.intersections.mesh_plane(
        mesh,
        plane_normal=np.asarray(plane_normal, dtype=np.float64),
        plane_origin=np.asarray(plane_origin, dtype=np.float64),
        return_faces=True,
    )
    if segments is None:
        return (
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    value = np.asarray(segments, dtype=np.float64)
    if value.size == 0:
        return (
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    return value.reshape(-1, 2, 3), np.asarray(face_index, dtype=np.int64).reshape(-1)


def _project_segments(
    segments: np.ndarray,
    *,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    unit_scale: float,
) -> np.ndarray:
    if segments.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    delta = segments - origin.reshape(1, 1, 3)
    x = np.einsum("...i,i->...", delta, u) * unit_scale
    y = np.einsum("...i,i->...", delta, v) * unit_scale
    return np.stack((x, y), axis=-1)


def _section_segments(
    *,
    points: np.ndarray,
    skin: np.ndarray,
    faces: np.ndarray,
    skin_faces: np.ndarray,
    face_codes: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    display_radius: float,
) -> dict[str, np.ndarray]:
    skin_lines, _skin_faces_hit = _mesh_plane_segments(
        skin, skin_faces, plane_origin=origin, plane_normal=normal
    )
    sections: dict[str, np.ndarray] = {"skin": skin_lines}
    anatomy_lines, anatomy_faces_hit = _mesh_plane_segments(
        points, faces, plane_origin=origin, plane_normal=normal
    )
    # ``mesh_plane`` correctly intersects the infinite plane.  Restrict only
    # the displayed lines to the local joint neighbourhood so a vertical leg
    # plane does not also show the opposite leg or torso.  The plane itself,
    # origin and normal remain exact and are shared by source/candidate.
    def local(lines: np.ndarray) -> np.ndarray:
        if lines.size == 0:
            return lines
        delta = lines - origin.reshape(1, 1, 3)
        axial = np.einsum("...i,i->...", delta, normal)
        radial_sq = np.sum(delta * delta, axis=-1) - axial * axial
        keep = np.max(radial_sq, axis=1) <= float(display_radius) ** 2
        return lines[keep]

    sections["skin"] = local(sections["skin"])
    if anatomy_lines.size:
        delta = anatomy_lines - origin.reshape(1, 1, 3)
        axial = np.einsum("...i,i->...", delta, normal)
        radial_sq = np.sum(delta * delta, axis=-1) - axial * axial
        keep = np.max(radial_sq, axis=1) <= float(display_radius) ** 2
        anatomy_face_codes = face_codes[anatomy_faces_hit][keep]
        anatomy_lines = anatomy_lines[keep]
    else:
        anatomy_face_codes = np.empty((0,), dtype=object)
    for category in ("bone", "vessel", "nerve"):
        sections[category] = anatomy_lines[anatomy_face_codes == category]
    return sections


def _plot_section(
    *,
    cell: ReviewCell,
    arrays: Mapping[str, Any],
    section_name: str,
    output: Path,
    unit_scale: float,
) -> dict[str, Any]:
    joint_a, joint_b = SECTION_JOINTS[section_name]
    joints = np.asarray(arrays["smplx_joints"], dtype=np.float64)
    origin = (joints[joint_a] + joints[joint_b]) * 0.5
    normal = _normalize(joints[joint_b] - joints[joint_a], name=f"{section_name} axis")
    _right, up, _front = _body_axes(joints)
    u, v = _plane_basis(normal, up)
    skin_extent = np.ptp(np.asarray(arrays["skin_vertices"], dtype=np.float64), axis=0)
    display_radius = max(float(np.linalg.norm(joints[joint_b] - joints[joint_a])) * 0.70,
                         float(np.linalg.norm(skin_extent)) * 0.025)
    face_codes = _face_codes(cell.faces, cell.tissue_labels)
    projected: dict[str, dict[str, np.ndarray]] = {}
    for label, key in (("source", "source_vertices"), ("candidate", "candidate_vertices")):
        raw = _section_segments(
            points=np.asarray(arrays[key]),
            skin=np.asarray(arrays["skin_vertices"]),
            faces=cell.faces,
            skin_faces=cell.skin_faces,
            face_codes=face_codes,
            origin=origin,
            normal=normal,
            display_radius=display_radius,
        )
        projected[label] = {
            category: _project_segments(
                segments,
                origin=origin,
                u=u,
                v=v,
                unit_scale=unit_scale,
            )
            for category, segments in raw.items()
        }

    all_xy = [
        values
        for per_view in projected.values()
        for values in per_view.values()
        if values.size
    ]
    if all_xy:
        bounds = np.concatenate([value.reshape(-1, 2) for value in all_xy], axis=0)
        lo = bounds.min(axis=0)
        hi = bounds.max(axis=0)
    else:
        lo = np.array([-10.0, -10.0])
        hi = np.array([10.0, 10.0])
    span = max(float(np.max(hi - lo)), 1.0)
    margin = max(span * 0.10, 1.0)
    lo -= margin
    hi += margin

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    for axis, label in zip(axes, ("source", "candidate")):
        for category in SECTION_ORDER:
            values = projected[label][category]
            if values.size == 0:
                continue
            for line in values:
                axis.plot(
                    line[:, 0],
                    line[:, 1],
                    color=SECTION_COLORS[category],
                    linewidth=0.8 if category == "skin" else 1.05,
                    alpha=0.88,
                    solid_capstyle="round",
                )
        axis.set_title(label)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(float(lo[0]), float(hi[0]))
        axis.set_ylim(float(lo[1]), float(hi[1]))
        axis.set_xlabel("section u (mm)")
        axis.grid(True, linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("section v (mm)")
    handles = [
        Line2D([0], [0], color=SECTION_COLORS[name], linewidth=1.2, label=name)
        for name in SECTION_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"{section_name} mid-section | joints {SMPLX_JOINT_NAMES[joint_a]} → "
        f"{SMPLX_JOINT_NAMES[joint_b]} | plane origin "
        f"({origin[0] * unit_scale:.1f}, {origin[1] * unit_scale:.1f}, "
        f"{origin[2] * unit_scale:.1f}) mm"
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {
        "path": str(output),
        "plane_origin": origin,
        "plane_normal": normal,
        "basis_u": u,
        "basis_v": v,
        "display_radius_m": display_radius,
        "segment_counts": {
            label: {category: int(values.shape[0]) for category, values in per_view.items()}
            for label, per_view in projected.items()
        },
    }


def _iter_input_paths(inputs: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for value in inputs:
        path = value.expanduser()
        if path.is_dir():
            found.extend(sorted(path.glob("*.npz")))
        else:
            found.append(path)
    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise FileNotFoundError("--input did not resolve to any .npz cells")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="one or more per-cell .npz files, or directories containing them",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tissue-map",
        help=(
            "JSON object or JSON file mapping integer/string vertex labels to "
            "bone/vessel/nerve/other, e.g. '{\"0\":\"bone\",\"1\":\"vessel\"}'"
        ),
    )
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1000.0,
        help="input-unit to millimetre scale; anatomy inputs are metres by default",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="only make cross-section plots (useful for a VTK-free contract check)",
    )
    parser.add_argument(
        "--skip-sections",
        action="store_true",
        help="only make PyVista 3-D panels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.width < 400 or args.height < 300:
        raise SystemExit("--width/--height are too small for a comparative review")
    if args.unit_scale <= 0:
        raise SystemExit("--unit-scale must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    tissue_map = _parse_tissue_map(args.tissue_map)
    inputs = _iter_input_paths(args.input)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "MaterialReviewV13",
        "publishable": False,
        "camera_source": "unrotated_smplx_joints_and_skin_bbox_only",
        "root_rotation_policy": "inverse_pose_root_applied_identically_to_all_point_arrays",
        "section_method": "trimesh.intersections.mesh_plane",
        "unit_scale_to_mm": float(args.unit_scale),
        "tissue_map": dict(tissue_map),
        "cells": {},
    }
    for path in inputs:
        cell = _load_cell(path, tissue_map)
        arrays = _unrotated_cell(cell)
        cell_dir = output / cell.stem
        cell_dir.mkdir(parents=True, exist_ok=False)
        record: dict[str, Any] = {
            "input": str(cell.path),
            "input_sha256": cell.source_sha256,
            "vertex_count": int(cell.source_vertices.shape[0]),
            "face_count": int(cell.faces.shape[0]),
            "tissue_labels": sorted({_canonical_tissue(item) for item in cell.tissue_labels}),
            "root_rotvec_removed": arrays["root_rotvec"],
            "root_rotation_pivot_smplx_pelvis": arrays["root_pivot"],
            "renders": {},
            "sections": {},
        }
        if not args.skip_3d:
            for region in (
                "whole",
                "left_elbow",
                "right_elbow",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "feet",
            ):
                path_out = cell_dir / f"{region}_3d.png"
                record["renders"][region] = _render_3d(
                    cell=cell,
                    arrays=arrays,
                    region=region,
                    output=path_out,
                    width=int(args.width),
                    height=int(args.height),
                )
        if not args.skip_sections:
            for section_name in (
                "forearm",
                "right_forearm",
                "shank",
                "right_shank",
            ):
                path_out = cell_dir / f"{section_name}_section.png"
                record["sections"][section_name] = _plot_section(
                    cell=cell,
                    arrays=arrays,
                    section_name=section_name,
                    output=path_out,
                    unit_scale=float(args.unit_scale),
                )
        manifest["cells"][cell.stem] = _jsonable(record)
    (output / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
