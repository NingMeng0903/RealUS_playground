"""Genesis triangle-mesh evidence pack for V8.14 anatomical bone review."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

from .anatomy_lbs import (
    joint_global_transforms,
    skin_vertices,
    source_bone_skinning_transforms,
)
from .bone_segment_diagnostics import write_bone_segment_diagnostics
from .containment import signed_distance
from .functional_joint_v8 import FUNCTIONAL_FRAME_NAMES_V8
from .intersection_diagnostics import _intersection_pairs
from .pose_adapter import smplx_pose_hash
from .rigged_asset import AnatomyRiggedAsset
from .smplx_body_surface_v7 import smplx_body_surface_v7
from .v8_artifacts import ResidentPoseEvaluatorV8, SubjectRuntimePackV8


BONE_REVIEW_PACK_SCHEMA_VERSION = 8
REVIEW_MODES = ("bones_only", "bones+tubes", "signed_distance", "joint_sections")
COLOR_BONE = (0.86, 0.82, 0.69, 1.0)
COLOR_ARTERY = (0.78, 0.04, 0.04, 1.0)
COLOR_VEIN = (0.03, 0.24, 0.78, 1.0)
COLOR_NERVE = (0.94, 0.72, 0.03, 1.0)
COLOR_SKIN = (0.88, 0.55, 0.42, 0.28)
COLOR_STATION = (0.92, 0.03, 0.72, 1.0)
COLOR_PIVOT = (0.02, 0.82, 0.86, 1.0)
COLOR_AXIS = (0.96, 0.76, 0.02, 1.0)
COLOR_RESIDUAL = (0.95, 0.03, 0.02, 1.0)
COLOR_NEAR = (1.0, 0.68, 0.02, 1.0)
COLOR_OUTSIDE = (0.95, 0.02, 0.01, 1.0)

_REVIEW_REGION_TOKENS = {
    "pelvis": ("ilium", "ischium", "pubis", "sacrum"),
    "leg": ("femur", "patella", "tibia", "fibula"),
    "foot": (
        "talus",
        "calcaneus",
        "navicular",
        "cuboid",
        "cuneiform",
        "metatars",
        "phalanx_foot",
    ),
    "shoulder_girdle": ("scapula", "clavicle"),
    "arm": ("humerus", "radius", "ulna"),
    "hand": (
        "carpal",
        "scaphoid",
        "lunate",
        "triquetr",
        "pisiform",
        "trapez",
        "capitate",
        "hamate",
        "metacarp",
        "phalanges_hand",
    ),
}


@dataclass(frozen=True)
class ReviewPoseV8:
    label: str
    pose_axis_angle: np.ndarray
    transl: np.ndarray
    source: str

    def validate(self) -> None:
        pose = np.asarray(self.pose_axis_angle, dtype=np.float32)
        translation = np.asarray(self.transl, dtype=np.float32)
        if pose.shape != (55, 3) or translation.shape != (3,):
            raise ValueError("review pose must contain [55,3] pose and [3] translation")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(translation)):
            raise ValueError("review pose contains non-finite values")
        if not self.label or "/" in self.label or "\\" in self.label:
            raise ValueError("review pose label is invalid")


@dataclass(frozen=True)
class ReviewCameraV8:
    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float]
    group: str
    fov: float = 38.0


@dataclass(frozen=True)
class SyntheticSweepStateV8:
    label: str
    pose_axis_angle: np.ndarray
    joint_pair: str | None
    diagnostic_joints: tuple[str, ...]
    motion: str
    value_deg: float


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.unique(np.asarray(face_rows, dtype=np.int64).reshape(-1))
    if not len(rows):
        raise ValueError("review mesh selection contains no triangle faces")
    selected = np.asarray(faces, dtype=np.int64).reshape(-1, 3)[rows]
    used, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    return (
        np.asarray(vertices, dtype=np.float32).reshape(-1, 3)[used],
        inverse.reshape(-1, 3).astype(np.int32),
    )


def _export_triangle_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_rows: np.ndarray | None = None,
) -> Path:
    import trimesh

    if face_rows is not None:
        vertices, faces = _compact_mesh(vertices, faces, face_rows)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"cannot export empty review mesh {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def _mesh_face_rows(
    asset: AnatomyRiggedAsset,
    predicate: Any,
) -> np.ndarray:
    faces = np.asarray(asset.faces, dtype=np.int64)
    selected = np.zeros(len(faces), dtype=bool)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = list(asset.source_tissues or ())
    names = list(asset.source_mesh_names or ())
    if len(ranges) != len(names) or len(tissues) != len(names):
        raise ValueError("source mesh metadata is incomplete")
    for name, tissue, (start, stop) in zip(names, tissues, ranges):
        if predicate(str(name), str(tissue).strip().lower()):
            selected |= np.all((faces >= int(start)) & (faces < int(stop)), axis=1)
    return np.flatnonzero(selected)


def _review_region_face_rows(asset: AnatomyRiggedAsset) -> dict[str, np.ndarray]:
    """Return disjoint, named appendicular review domains."""

    result: dict[str, np.ndarray] = {}
    for region, tokens in _REVIEW_REGION_TOKENS.items():
        def selected(name: str, tissue: str, *, _region: str = region) -> bool:
            lower = name.lower()
            if tissue != "bone":
                return False
            if _region == "leg" and any(
                token in lower for token in _REVIEW_REGION_TOKENS["foot"]
            ):
                return False
            if _region == "arm" and any(
                token in lower for token in _REVIEW_REGION_TOKENS["hand"]
            ):
                return False
            return any(token in lower for token in tokens)

        rows = _mesh_face_rows(asset, selected)
        if not len(rows):
            raise ValueError(f"review containment region {region!r} is empty")
        result[region] = rows
    flattened = np.concatenate(list(result.values()))
    if len(np.unique(flattened)) != len(flattened):
        raise ValueError("review containment regions overlap")
    return result


def _primitive_mesh(
    *,
    spheres: Sequence[np.ndarray] = (),
    segments: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    radius: float,
) -> Any:
    import trimesh

    pieces: list[Any] = []
    for center in spheres:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
        mesh.apply_translation(np.asarray(center, dtype=np.float64))
        pieces.append(mesh)
    for first, second in segments:
        start = np.asarray(first, dtype=np.float64).reshape(3)
        stop = np.asarray(second, dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(stop - start))
        if length <= 1.0e-8:
            continue
        pieces.append(
            trimesh.creation.cylinder(
                radius=float(radius),
                segment=np.stack((start, stop)),
                sections=12,
            )
        )
    if not pieces:
        raise ValueError("debug primitive set is empty")
    return trimesh.util.concatenate(pieces)


def _export_primitive_mesh(
    path: Path,
    *,
    spheres: Sequence[np.ndarray] = (),
    segments: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    radius: float,
) -> Path:
    mesh = _primitive_mesh(spheres=spheres, segments=segments, radius=radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def _functional_debug_geometry(
    asset: AnatomyRiggedAsset,
    pose: ReviewPoseV8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = dict(asset.metadata or {}).get("functional_joint_frames_v8")
    if not isinstance(metadata, dict):
        raise ValueError("subject is missing functional_joint_frames_v8 metadata")
    centers = np.asarray(metadata["centers_m"], dtype=np.float64).reshape(-1, 3)
    axes = np.asarray(metadata["axes"], dtype=np.float64).reshape(-1, 3, 3)
    joints = np.asarray(metadata["smplx_joint_ids"], dtype=np.int64).reshape(-1)
    bones = np.asarray(metadata["controller_bone_ids"], dtype=np.int64).reshape(-1)
    if len(centers) != len(FUNCTIONAL_FRAME_NAMES_V8):
        raise ValueError("functional frame count changed")
    pose_global = joint_global_transforms(
        pose_axis_angle=pose.pose_axis_angle,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    source_delta = source_bone_skinning_transforms(
        asset, pose.pose_axis_angle
    ).astype(np.float64)
    translation = np.asarray(pose.transl, dtype=np.float64).reshape(3)
    stations = pose_global[joints, :3, 3] + translation
    pivots = np.einsum(
        "nij,nj->ni", source_delta[bones, :3, :3], centers
    ) + source_delta[bones, :3, 3] + translation
    posed_axes = np.einsum(
        "nij,nj->ni", source_delta[bones, :3, :3], axes[:, :, 0]
    )
    axis_half_length = 0.035
    axis_segments = [
        (pivot - axis_half_length * axis, pivot + axis_half_length * axis)
        for pivot, axis in zip(pivots, posed_axes)
    ]
    residuals = [(station, pivot) for station, pivot in zip(stations, pivots)]
    report = {
        name: {
            "smplx_station_m": stations[index].tolist(),
            "anatomical_pivot_m": pivots[index].tolist(),
            "station_pivot_residual_m": float(
                np.linalg.norm(stations[index] - pivots[index])
            ),
            "functional_axis": posed_axes[index].tolist(),
        }
        for index, name in enumerate(FUNCTIONAL_FRAME_NAMES_V8)
    }
    arrays = {
        "stations": stations,
        "pivots": pivots,
        "axis_segments": np.asarray(axis_segments, dtype=np.float64),
        "residual_segments": np.asarray(residuals, dtype=np.float64),
    }
    return report, arrays


def _camera_specs(
    vertices: np.ndarray,
    debug: Mapping[str, np.ndarray],
    *,
    section_only: bool,
) -> list[ReviewCameraV8]:
    points = np.asarray(vertices, dtype=np.float64)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent = points.max(axis=0) - points.min(axis=0)
    whole_distance = max(2.0, 1.55 * float(np.linalg.norm(extent)))
    cameras: list[ReviewCameraV8] = []
    if not section_only:
        cameras.extend(
            (
                ReviewCameraV8(
                    "whole_front",
                    tuple(center + (0.0, 0.0, whole_distance)),
                    tuple(center),
                    (0.0, -1.0, 0.0),
                    "whole",
                ),
                ReviewCameraV8(
                    "whole_side",
                    tuple(center + (whole_distance, 0.0, 0.0)),
                    tuple(center),
                    (0.0, -1.0, 0.0),
                    "whole",
                ),
                ReviewCameraV8(
                    "whole_top",
                    tuple(center + (0.0, whole_distance, 0.0)),
                    tuple(center),
                    (0.0, 0.0, 1.0),
                    "whole",
                ),
            )
        )
    pivots = np.asarray(debug["pivots"], dtype=np.float64)
    frame_index = {name: index for index, name in enumerate(FUNCTIONAL_FRAME_NAMES_V8)}
    groups = {
        f"{side}_{group}": f"{side}_{frame}"
        for side in ("left", "right")
        for group, frame in (
            ("hip", "hip"),
            ("knee", "knee"),
            ("ankle_foot", "ankle"),
            ("shoulder", "shoulder"),
            ("elbow", "elbow"),
            ("wrist_hand", "wrist"),
        )
    }
    distance_by_kind = {
        "hip": 0.68,
        "shoulder": 0.58,
        "knee": 0.42,
        "elbow": 0.38,
        "ankle_foot": 0.36,
        "wrist_hand": 0.32,
    }
    prefix = "section" if section_only else "joint"
    for group, frame_name in groups.items():
        target = pivots[frame_index[frame_name]]
        kind = group.removeprefix("left_").removeprefix("right_")
        distance = distance_by_kind[kind]
        if section_only:
            direction = np.asarray((0.70, 0.20, 0.68), dtype=np.float64)
            direction /= np.linalg.norm(direction)
            cameras.append(
                ReviewCameraV8(
                    f"{prefix}_{group}",
                    tuple(target + distance * direction),
                    tuple(target),
                    (0.0, -1.0, 0.0),
                    group,
                    fov=32.0,
                )
            )
        else:
            oblique = np.asarray((0.65, 0.28, 0.70), dtype=np.float64)
            oblique /= np.linalg.norm(oblique)
            cameras.extend(
                (
                    ReviewCameraV8(
                        f"{prefix}_{group}_ortho",
                        tuple(target + (0.0, 0.0, distance)),
                        tuple(target),
                        (0.0, -1.0, 0.0),
                        group,
                        fov=32.0,
                    ),
                    ReviewCameraV8(
                        f"{prefix}_{group}_oblique",
                        tuple(target + distance * oblique),
                        tuple(target),
                        (0.0, -1.0, 0.0),
                        group,
                        fov=32.0,
                    ),
                )
            )
    return cameras


def _section_skin_faces(
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    pivots: np.ndarray,
    *,
    radius_m: float = 0.095,
) -> np.ndarray:
    vertices = np.asarray(skin_vertices, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int64)
    distance = np.min(
        np.linalg.norm(vertices[:, None, :] - np.asarray(pivots)[None, :, :], axis=2),
        axis=1,
    )
    return np.flatnonzero(np.all(distance[faces] >= float(radius_m), axis=1))


def _joint_intersection_report(
    asset: AnatomyRiggedAsset,
    vertices: np.ndarray,
    *,
    baseline_vertices: np.ndarray,
) -> dict[str, Any]:
    names = list(asset.source_mesh_names or ())
    tissues = list(asset.source_tissues or ())
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    faces = np.asarray(asset.faces, dtype=np.int64)

    def rows(tokens: tuple[str, ...], side: str) -> np.ndarray:
        selected_ranges: list[tuple[int, int]] = []
        suffix = "_l" if side == "left" else "_r"
        for name, tissue, limits in zip(names, tissues, ranges):
            lower = str(name).lower()
            if (
                str(tissue).lower() == "bone"
                and lower.endswith(suffix)
                and any(token in lower for token in tokens)
            ):
                selected_ranges.append((int(limits[0]), int(limits[1])))
        mask = np.zeros(len(faces), dtype=bool)
        for start, stop in selected_ranges:
            mask |= np.all((faces >= start) & (faces < stop), axis=1)
        return np.flatnonzero(mask)

    definitions = {
        "hip": (("ilium", "ischium", "pubis"), ("femur",)),
        "knee": (("femur",), ("tibia", "patella")),
        "ankle": (("tibia", "fibula"), ("talus",)),
        "shoulder": (("scapula",), ("humerus",)),
        "elbow": (("humerus",), ("radius", "ulna")),
        "wrist": (("radius", "ulna"), ("carpal", "scaphoid", "lunate")),
    }
    pairs: dict[str, Any] = {}
    total = 0
    baseline_total = 0
    positive_net_new_total = 0
    maximum_depth_increase = 0.0

    def penetration_depth(first: np.ndarray, second: np.ndarray, xyz: np.ndarray) -> float:
        first_vertices, first_faces = _compact_mesh(xyz, faces, first)
        second_vertices, second_faces = _compact_mesh(xyz, faces, second)
        first_signed, _first_closest, _first_normals = signed_distance(
            first_vertices, second_vertices, second_faces
        )
        second_signed, _second_closest, _second_normals = signed_distance(
            second_vertices, first_vertices, first_faces
        )
        return float(
            max(
                0.0,
                float(np.max(-first_signed)),
                float(np.max(-second_signed)),
            )
        )

    for side in ("left", "right"):
        for joint, (first_tokens, second_tokens) in definitions.items():
            first = rows(first_tokens, side)
            second = rows(second_tokens, side)
            label = f"{side}_{joint}"
            if not len(first) or not len(second):
                pairs[label] = {
                    "available": False,
                    "pass": False,
                    "reason": "one or both triangle domains are absent",
                }
                continue
            intersections = _intersection_pairs(
                np.asarray(vertices, dtype=np.float64), faces, first, second
            )
            baseline_intersections = _intersection_pairs(
                np.asarray(baseline_vertices, dtype=np.float64),
                faces,
                first,
                second,
            )
            count = len(intersections)
            baseline_count = len(baseline_intersections)
            net_new = count - baseline_count
            depth = penetration_depth(first, second, vertices)
            baseline_depth = penetration_depth(first, second, baseline_vertices)
            depth_increase = max(0.0, depth - baseline_depth)
            total += count
            baseline_total += baseline_count
            positive_net_new_total += max(0, net_new)
            maximum_depth_increase = max(maximum_depth_increase, depth_increase)
            pairs[label] = {
                "available": True,
                "triangle_pair_count": int(count),
                "baseline_triangle_pair_count": int(baseline_count),
                "net_new_triangle_pair_count": int(net_new),
                "introduced_face_pair_count": int(
                    len(intersections - baseline_intersections)
                ),
                "penetration_depth_m": depth,
                "baseline_penetration_depth_m": baseline_depth,
                "penetration_depth_increase_m": depth_increase,
                "maximum_penetration_depth_increase_m": 0.0005,
                "pass": bool(depth_increase <= 0.0005),
            }
    available = bool(pairs) and all(value.get("available") for value in pairs.values())
    return {
        "available": available,
        "backend": "exact_triangle_aabb_plus_moller_trumbore",
        "triangle_pair_count": int(total),
        "baseline_triangle_pair_count": int(baseline_total),
        "positive_net_new_triangle_pair_count": int(positive_net_new_total),
        "maximum_penetration_depth_increase_m": maximum_depth_increase,
        "allowed_penetration_depth_increase_m": 0.0005,
        "pairs": pairs,
        "pass": bool(available and maximum_depth_increase <= 0.0005),
    }


def synthetic_sweep_states_v8(
    asset: AnatomyRiggedAsset,
) -> tuple[SyntheticSweepStateV8, ...]:
    """Build deterministic bilateral sweeps around the baked anatomical axes."""

    metadata = (asset.metadata or {}).get("functional_joint_frames_v8")
    if not isinstance(metadata, dict):
        raise ValueError("functional joint frames are required for synthetic sweeps")
    joints = np.asarray(metadata.get("smplx_joint_ids", []), dtype=np.int64)
    axes = np.asarray(metadata.get("axes", []), dtype=np.float64)
    if axes.shape != (len(joints), 3, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("functional joint sweep axes are invalid")

    def frame_axis(joint: int, column: int = 0) -> np.ndarray:
        matches = np.flatnonzero(joints == int(joint))
        if len(matches) != 1:
            raise ValueError(f"functional sweep joint {joint} is unavailable")
        axis = np.asarray(axes[int(matches[0]), :, int(column)], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            raise ValueError(f"functional sweep joint {joint} has a zero axis")
        return axis / norm

    states: list[SyntheticSweepStateV8] = []

    def append(
        *,
        side: str,
        motion: str,
        joint: int,
        axis: np.ndarray,
        values_deg: Sequence[float],
        pair: str | None,
        diagnostics: tuple[str, ...],
    ) -> None:
        for value in values_deg:
            pose = np.zeros((55, 3), dtype=np.float32)
            pose[int(joint)] = np.asarray(
                np.radians(float(value)) * axis,
                dtype=np.float32,
            )
            states.append(
                SyntheticSweepStateV8(
                    label=f"{side}_{motion}_{float(value):+06.1f}",
                    pose_axis_angle=pose,
                    joint_pair=pair,
                    diagnostic_joints=diagnostics,
                    motion=motion,
                    value_deg=float(value),
                )
            )

    for side, knee, ankle, elbow, wrist, hand_start in (
        ("left", 4, 7, 18, 20, 25),
        ("right", 5, 8, 19, 21, 40),
    ):
        append(
            side=side,
            motion="knee_flexion",
            joint=knee,
            axis=frame_axis(knee),
            values_deg=(0.0, 30.0, 60.0, 90.0, 120.0),
            pair=f"{side}_knee",
            diagnostics=(f"knee_{side}",),
        )
        append(
            side=side,
            motion="elbow_flexion",
            joint=elbow,
            axis=frame_axis(elbow),
            values_deg=(0.0, 35.0, 70.0, 105.0, 140.0),
            pair=f"{side}_elbow",
            diagnostics=(f"elbow_{side}",),
        )
        append(
            side=side,
            motion="ankle_flexion",
            joint=ankle,
            axis=frame_axis(ankle),
            values_deg=(-30.0, -15.0, 0.0, 10.0, 20.0),
            pair=f"{side}_ankle",
            diagnostics=(f"ankle_{side}",),
        )
        append(
            side=side,
            motion="wrist_flexion",
            joint=wrist,
            axis=frame_axis(wrist),
            values_deg=(-60.0, -30.0, 0.0, 30.0, 60.0),
            pair=f"{side}_wrist",
            diagnostics=(f"wrist_{side}",),
        )
        append(
            side=side,
            motion="wrist_deviation",
            joint=wrist,
            axis=frame_axis(wrist, 2),
            values_deg=(-20.0, 20.0),
            pair=f"{side}_wrist",
            diagnostics=(f"wrist_{side}",),
        )
        for fraction in (0.0, 0.5, 1.0):
            pose = np.zeros((55, 3), dtype=np.float32)
            pose[hand_start : hand_start + 15, 0] = np.radians(
                70.0 * fraction
            )
            states.append(
                SyntheticSweepStateV8(
                    label=f"{side}_finger_flexion_{fraction:.1f}",
                    pose_axis_angle=pose,
                    joint_pair=None,
                    diagnostic_joints=(
                        f"index_proximal_{side}",
                        f"thumb_proximal_{side}",
                        f"thumb_middle_{side}",
                        f"thumb_distal_{side}",
                    ),
                    motion="finger_flexion",
                    value_deg=70.0 * fraction,
                )
            )
    return tuple(states)


def write_bone_review_sweep_v8(
    *,
    subject_label: str,
    subject: SubjectRuntimePackV8,
    baseline_subject: SubjectRuntimePackV8,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Evaluate synthetic joint sweeps without any runtime collision solve."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = subject.rigged_asset
    baseline_asset = baseline_subject.rigged_asset
    evaluator = ResidentPoseEvaluatorV8(subject, validate=False)
    evaluator.apply_pose(np.zeros((55, 3), dtype=np.float32))
    states = synthetic_sweep_states_v8(asset)
    samples: list[dict[str, Any]] = []
    pose_seconds: list[float] = []
    # Benchmark the resident hot path contiguously. Interleaving each pose
    # with signed-distance diagnostics would measure cache eviction caused by
    # the offline reviewer, not the runtime pose evaluator.
    for state in states:
        started = time.perf_counter()
        evaluator.apply_pose(state.pose_axis_angle)
        pose_seconds.append(time.perf_counter() - started)
    for state in states:
        posed = evaluator.apply_pose(state.pose_axis_angle)
        baseline_posed = skin_vertices(
            baseline_asset,
            state.pose_axis_angle,
            validate=False,
        )
        intersections = _joint_intersection_report(
            asset,
            posed,
            baseline_vertices=baseline_posed,
        )
        diagnostic_path = out / "diagnostics" / f"{state.label}.json"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = write_bone_segment_diagnostics(
            asset,
            pose_axis_angle=state.pose_axis_angle,
            transl=np.zeros(3, dtype=np.float32),
            output_path=diagnostic_path,
        )
        selected_diagnostics = {
            name: diagnostics["joints"][name]
            for name in state.diagnostic_joints
        }
        pair = (
            None
            if state.joint_pair is None
            else intersections["pairs"][state.joint_pair]
        )
        passed = bool(
            (pair is None or pair["pass"])
            and all(value["pass"] for value in selected_diagnostics.values())
        )
        samples.append(
            {
                "label": state.label,
                "motion": state.motion,
                "value_deg": state.value_deg,
                "joint_pair": state.joint_pair,
                "diagnostic_joints": list(state.diagnostic_joints),
                "exact_intersection": pair,
                "selected_diagnostics": selected_diagnostics,
                "diagnostic_path": str(diagnostic_path),
                "pass": passed,
            }
        )
    failures = [sample["label"] for sample in samples if not sample["pass"]]
    report = {
        "schema_version": BONE_REVIEW_PACK_SCHEMA_VERSION,
        "artifact_kind": "BoneReviewSweepV8",
        "subject": str(subject_label),
        "sample_count": int(len(samples)),
        "motions": sorted({sample["motion"] for sample in samples}),
        "resident_pose_p95_seconds": float(np.percentile(pose_seconds, 95.0)),
        "pose_cli_limit_seconds": 1.0,
        "resident_pose_limit_seconds": 0.2,
        "runtime_spatial_queries": False,
        "samples": samples,
        "failures": failures,
        "pass": bool(
            not failures
            and float(np.percentile(pose_seconds, 95.0)) <= 0.2
        ),
        "publishable": False,
        "human_signature": "pending",
    }
    _json_write(out / "sweep.json", report)
    return report


def _save_modalities(
    output_dir: Path,
    rendered: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    records: list[dict[str, Any]] = []
    rgb_paths: list[Path] = []
    for camera, payload in rendered.items():
        rgb = np.asarray(payload["rgb"])
        if rgb.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth = np.asarray(payload["depth"], dtype=np.float32).squeeze()
        segmentation = np.asarray(payload["segmentation"])
        foreground = np.any(rgb > 12, axis=-1)
        depth_valid = np.isfinite(depth) & (depth > 0.0)
        segmentation_nonzero = (
            np.any(segmentation != 0, axis=-1)
            if segmentation.ndim == 3
            else segmentation != 0
        )
        coverage = float(np.mean(segmentation_nonzero))
        height, width = segmentation_nonzero.shape[:2]
        half_window = max(4, int(round(0.04 * min(height, width))))
        center_y, center_x = height // 2, width // 2
        center_window = segmentation_nonzero[
            max(0, center_y - half_window) : min(height, center_y + half_window + 1),
            max(0, center_x - half_window) : min(width, center_x + half_window + 1),
        ]
        center_hit = bool(np.any(center_window))
        passed = bool(
            np.any(foreground)
            and np.any(depth_valid)
            and 0.001 <= coverage <= 0.98
            and center_hit
        )
        rgb_path = output_dir / "rgb" / f"{camera}.png"
        depth_path = output_dir / "depth" / f"{camera}.npy"
        depth_png = output_dir / "depth" / f"{camera}.png"
        seg_path = output_dir / "segmentation" / f"{camera}.png"
        for path in (rgb_path, depth_path, depth_png, seg_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(rgb_path, rgb)
        np.save(depth_path, depth, allow_pickle=False)
        visible_depth = np.zeros_like(depth, dtype=np.uint8)
        if np.any(depth_valid):
            low, high = np.quantile(depth[depth_valid], (0.01, 0.99))
            span = max(float(high - low), 1.0e-8)
            visible_depth[depth_valid] = np.clip(
                255.0 * (1.0 - (depth[depth_valid] - low) / span), 0.0, 255.0
            ).astype(np.uint8)
        imageio.imwrite(depth_png, visible_depth)
        imageio.imwrite(seg_path, segmentation_nonzero.astype(np.uint8) * 255)
        rgb_paths.append(rgb_path)
        records.append(
            {
                "camera": camera,
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "segmentation": str(seg_path),
                "foreground_fraction": float(np.mean(foreground)),
                "depth_valid_fraction": float(np.mean(depth_valid)),
                "segmentation_coverage": coverage,
                "lookat_center_hit": center_hit,
                "pass": passed,
            }
        )
    return records, rgb_paths


def _contact_sheet(paths: Sequence[Path], output: Path, *, columns: int = 4) -> Path:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        raise ValueError("contact sheet needs at least one RGB render")
    thumb = (320, 240)
    rows = (len(images) + int(columns) - 1) // int(columns)
    sheet = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + 24)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (image, path) in enumerate(zip(images, paths)):
        image.thumbnail(thumb)
        x = (index % columns) * thumb[0]
        y = (index // columns) * (thumb[1] + 24)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb[1] + 3), path.stem, fill=(235, 235, 235))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def _render_genesis_scene(
    *,
    mesh_entities: Sequence[tuple[str, Path, tuple[float, float, float, float]]],
    cameras: Sequence[ReviewCameraV8],
    output_dir: Path,
    backend: str,
    resolution: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[Path]]:
    from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
        GenesisPlatformRuntime,
        GenesisRuntimeConfig,
        MeshEntityConfig,
        StaticCameraConfig,
    )

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=str(backend),
            show_viewer=False,
            show_fps=False,
            enable_collision=False,
            gravity=(0.0, 0.0, 0.0),
            plane_reflection=False,
            ambient_light=(0.38, 0.38, 0.38),
        )
    )
    try:
        runtime.initialize()
        for name, path, color in mesh_entities:
            runtime.add_mesh_entity(
                MeshEntityConfig(
                    name=name,
                    file=path,
                    color=color,
                    fixed=True,
                    collision=False,
                )
            )
        for camera in cameras:
            runtime.add_camera(
                StaticCameraConfig(
                    name=camera.name,
                    res=resolution,
                    pos=camera.pos,
                    lookat=camera.lookat,
                    up=camera.up,
                    fov=camera.fov,
                    near=0.01,
                    far=10.0,
                    gui=False,
                )
            )
        runtime.build()
        rendered = runtime.render_all_cameras(
            modalities=("rgb", "depth", "segmentation"),
            force_render=True,
        )
        return _save_modalities(output_dir, rendered)
    finally:
        runtime.close()


def _bed_alignment(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a proper rigid transform that lays the body along the bed X axis."""

    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    center = np.mean(points, axis=0)
    _u, _singular, axes = np.linalg.svd(points - center, full_matrices=False)
    basis = np.asarray(axes.T, dtype=np.float64)
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    rotation = basis.T
    aligned = (points - center) @ rotation.T
    if np.mean(aligned[points[:, 1] > np.median(points[:, 1]), 0]) < 0.0:
        rotation[0] *= -1.0
        rotation[1] *= -1.0
        aligned = (points - center) @ rotation.T
    bed_top = 0.31
    translation = np.asarray(
        (0.0, 0.0, bed_top + 0.012 - float(np.min(aligned[:, 2]))),
        dtype=np.float64,
    )
    return rotation, translation - rotation @ center


def _render_bed_robot_scene(
    *,
    posed: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: AnatomyRiggedAsset,
    bone_rows: np.ndarray,
    artery_rows: np.ndarray,
    vein_rows: np.ndarray,
    nerve_rows: np.ndarray,
    output_dir: Path,
    backend: str,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    """Render one final Genesis context scene with the project bed and RM75."""

    from projects.genesis_ue_sync.sim_platform.scenes.common_scene import (
        _load_robot_spec,
    )
    from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import (
        RobotRegistry,
    )
    from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
        add_robots_to_runtime,
        init_robots_after_build,
    )
    from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
        BoxEntityConfig,
        GenesisPlatformRuntime,
        GenesisRuntimeConfig,
        MeshEntityConfig,
        StaticCameraConfig,
    )

    rotation, translation = _bed_alignment(skin)

    def transformed(values: np.ndarray) -> np.ndarray:
        xyz = np.asarray(values, dtype=np.float64)
        return xyz @ rotation.T + translation

    assets = output_dir / "mesh_assets"
    scene_paths: list[tuple[str, Path, tuple[float, float, float, float]]] = [
        (
            "patient_skin",
            _export_triangle_mesh(
                assets / "patient_skin.obj", transformed(skin), skin_faces
            ),
            COLOR_SKIN,
        ),
        (
            "patient_bones",
            _export_triangle_mesh(
                assets / "patient_bones.obj",
                transformed(posed),
                asset.faces,
                bone_rows,
            ),
            COLOR_BONE,
        ),
    ]
    for name, rows, color in (
        ("arteries", artery_rows, COLOR_ARTERY),
        ("veins", vein_rows, COLOR_VEIN),
        ("nerves", nerve_rows, COLOR_NERVE),
    ):
        if len(rows):
            scene_paths.append(
                (
                    name,
                    _export_triangle_mesh(
                        assets / f"{name}.obj",
                        transformed(posed),
                        asset.faces,
                        rows,
                    ),
                    color,
                )
            )

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=str(backend),
            show_viewer=False,
            show_fps=False,
            enable_collision=False,
            gravity=(0.0, 0.0, 0.0),
            plane_reflection=False,
            ambient_light=(0.42, 0.42, 0.42),
        )
    )
    try:
        runtime.initialize()
        runtime.add_box(
            BoxEntityConfig(
                name="bed_surface",
                pos=(0.0, 0.0, 0.25),
                size=(1.9158, 0.7039, 0.12),
                color=(0.46, 0.50, 0.55, 1.0),
                fixed=True,
                collision=False,
            )
        )
        for name, path, color in scene_paths:
            runtime.add_mesh_entity(
                MeshEntityConfig(
                    name=name,
                    file=path,
                    color=color,
                    fixed=True,
                    collision=False,
                )
            )
        robot_spec = _load_robot_spec(
            {
                "model_id": "rm75_6f_8dof",
                "name": "robot_main",
                "base_pos": [0.0, 0.62, 0.0],
                "base_quat_wxyz": [
                    -0.7071067811865477,
                    0.0,
                    0.0,
                    0.7071067811865475,
                ],
                "joint_positions": [0.0] * 8,
                "use_collision_geometry": False,
            }
        )
        registry = RobotRegistry()
        robot_names = add_robots_to_runtime(
            runtime,
            [robot_spec],
            registry,
            enable_collision=False,
            repo_root=Path(__file__).resolve().parents[5],
        )
        cameras = (
            StaticCameraConfig(
                name="bed_robot_oblique",
                res=resolution,
                pos=(2.25, -1.75, 1.55),
                lookat=(0.0, 0.05, 0.55),
                up=(0.0, 0.0, 1.0),
                fov=42.0,
                gui=False,
            ),
            StaticCameraConfig(
                name="bed_robot_overhead",
                res=resolution,
                pos=(0.15, -0.15, 2.65),
                lookat=(0.0, 0.05, 0.40),
                up=(1.0, 0.0, 0.0),
                fov=39.0,
                gui=False,
            ),
        )
        for camera in cameras:
            runtime.add_camera(camera)
        runtime.build()
        init_robots_after_build(runtime, registry, [robot_spec], robot_names)
        rendered = runtime.render_all_cameras(
            modalities=("rgb", "depth", "segmentation"),
            force_render=True,
        )
        records, rgb_paths = _save_modalities(output_dir, rendered)
        contact_sheet = _contact_sheet(
            rgb_paths,
            output_dir / "contact_sheet.png",
            columns=2,
        )
        return {
            "available": True,
            "bed_source": "configs/scenes/realus_bed_rail_scene.yaml",
            "robot_model": "rm75_6f_8dof",
            "checks": records,
            "pass": bool(records and all(value["pass"] for value in records)),
            "contact_sheet": str(contact_sheet),
        }
    finally:
        runtime.close()


def write_bone_review_cell_v8(
    *,
    subject_label: str,
    subject: SubjectRuntimePackV8,
    baseline_subject: SubjectRuntimePackV8,
    pose: ReviewPoseV8,
    body_model: Mapping[str, np.ndarray],
    output_dir: Path | str,
    backend: str = "cpu",
    resolution: tuple[int, int] = (640, 640),
    include_bed_robot_scene: bool = False,
) -> dict[str, Any]:
    pose.validate()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = subject.rigged_asset
    posed = np.asarray(
        skin_vertices(asset, pose.pose_axis_angle, transl=pose.transl),
        dtype=np.float64,
    )
    baseline_posed = np.asarray(
        skin_vertices(
            baseline_subject.rigged_asset,
            pose.pose_axis_angle,
            transl=pose.transl,
        ),
        dtype=np.float64,
    )
    if baseline_posed.shape != posed.shape:
        raise ValueError("142 baseline and bone-review subject topology disagree")
    skin, skin_faces = smplx_body_surface_v7(
        body_model,
        betas=subject.betas,
        pose_axis_angle=pose.pose_axis_angle,
        transl=pose.transl,
    )
    debug_report, debug = _functional_debug_geometry(asset, pose)
    bone_rows = _mesh_face_rows(asset, lambda _name, tissue: tissue == "bone")
    region_rows = _review_region_face_rows(asset)
    scoped_bone_rows = np.sort(np.concatenate(list(region_rows.values())))
    context_bone_rows = np.setdiff1d(bone_rows, scoped_bone_rows)
    artery_rows = _mesh_face_rows(
        asset, lambda name, tissue: tissue == "vessel" and "arter" in name.lower()
    )
    vein_rows = _mesh_face_rows(
        asset, lambda name, tissue: tissue == "vessel" and "vein" in name.lower()
    )
    nerve_rows = _mesh_face_rows(asset, lambda _name, tissue: tissue == "nerve")
    bone_ids = np.unique(
        np.asarray(asset.faces, dtype=np.int64)[scoped_bone_rows].reshape(-1)
    )
    bone_signed, _closest, _normals = signed_distance(
        posed[bone_ids], skin, skin_faces
    )
    baseline_bone_signed, _baseline_closest, _baseline_normals = signed_distance(
        baseline_posed[bone_ids], skin, skin_faces
    )
    signed_all = np.full(len(posed), -np.inf, dtype=np.float64)
    signed_all[bone_ids] = bone_signed
    baseline_signed_all = np.full(len(posed), -np.inf, dtype=np.float64)
    baseline_signed_all[bone_ids] = baseline_bone_signed
    bone_face_signed = signed_all[
        np.asarray(asset.faces, dtype=np.int64)[scoped_bone_rows]
    ]
    outside_rows = scoped_bone_rows[np.any(bone_face_signed > 0.0, axis=1)]
    near_rows = scoped_bone_rows[
        ~np.any(bone_face_signed > 0.0, axis=1)
        & np.any(bone_face_signed > -0.002, axis=1)
    ]
    safe_rows = np.setdiff1d(
        scoped_bone_rows, np.concatenate((outside_rows, near_rows))
    )
    containment = {
        "backend": "igl_signed_triangle_distance",
        "scope": "pelvis_and_appendicular_bones_only",
        "bone_vertex_count": int(len(bone_ids)),
        "inside_fraction": float(np.mean(bone_signed <= 0.0)),
        "outside_vertex_count": int(np.count_nonzero(bone_signed > 0.0)),
        "near_surface_vertex_count": int(
            np.count_nonzero((bone_signed <= 0.0) & (bone_signed > -0.002))
        ),
        "maximum_outside_m": float(max(0.0, float(np.max(bone_signed)))),
        "baseline_inside_fraction": float(np.mean(baseline_bone_signed <= 0.0)),
        "baseline_maximum_outside_m": float(
            max(0.0, float(np.max(baseline_bone_signed)))
        ),
        "regions": {},
    }
    for region, rows in region_rows.items():
        region_ids = np.unique(
            np.asarray(asset.faces, dtype=np.int64)[rows].reshape(-1)
        )
        region_signed = signed_all[region_ids]
        region_baseline = baseline_signed_all[region_ids]
        region_report = {
            "bone_vertex_count": int(len(region_ids)),
            "inside_fraction": float(np.mean(region_signed <= 0.0)),
            "outside_vertex_count": int(np.count_nonzero(region_signed > 0.0)),
            "near_surface_vertex_count": int(
                np.count_nonzero(
                    (region_signed <= 0.0) & (region_signed > -0.002)
                )
            ),
            "maximum_outside_m": float(
                max(0.0, float(np.max(region_signed)))
            ),
            "baseline_inside_fraction": float(
                np.mean(region_baseline <= 0.0)
            ),
            "baseline_maximum_outside_m": float(
                max(0.0, float(np.max(region_baseline)))
            ),
        }
        rigid_translation_budget = 0.002
        rigid_translation_lower_bound = max(
            0.0,
            region_report["maximum_outside_m"] - rigid_translation_budget,
        )
        region_report["bounded_rigid_translation_feasibility"] = {
            "translation_budget_m": rigid_translation_budget,
            "signed_distance_lipschitz_lower_bound_m": rigid_translation_lower_bound,
            "can_reach_maximum_outside_gate": bool(
                rigid_translation_lower_bound <= 0.002
            ),
            "rotation_or_scaling_assumed": False,
        }
        region_report["pass"] = bool(
            region_report["inside_fraction"] >= 0.995
            and region_report["maximum_outside_m"] <= 0.002
        )
        containment["regions"][region] = region_report
    containment["pass"] = bool(
        containment["inside_fraction"] >= 0.995
        and containment["maximum_outside_m"] <= 0.002
        and all(value["pass"] for value in containment["regions"].values())
    )

    assets = out / "mesh_assets"
    mesh_paths = {
        "bones": _export_triangle_mesh(assets / "bones.obj", posed, asset.faces, bone_rows),
        "skin": _export_triangle_mesh(assets / "skin.obj", skin, skin_faces),
        "station": _export_primitive_mesh(
            assets / "stations.obj", spheres=debug["stations"], radius=0.007
        ),
        "pivot": _export_primitive_mesh(
            assets / "pivots.obj", spheres=debug["pivots"], radius=0.006
        ),
        "axis": _export_primitive_mesh(
            assets / "hinge_axes.obj",
            segments=[tuple(value) for value in debug["axis_segments"]],
            radius=0.0022,
        ),
        "residual": _export_primitive_mesh(
            assets / "station_residuals.obj",
            segments=[tuple(value) for value in debug["residual_segments"]],
            radius=0.0016,
        ),
    }
    if len(artery_rows):
        mesh_paths["artery"] = _export_triangle_mesh(
            assets / "arteries.obj", posed, asset.faces, artery_rows
        )
    if len(vein_rows):
        mesh_paths["vein"] = _export_triangle_mesh(
            assets / "veins.obj", posed, asset.faces, vein_rows
        )
    if len(nerve_rows):
        mesh_paths["nerve"] = _export_triangle_mesh(
            assets / "nerves.obj", posed, asset.faces, nerve_rows
        )
    heatmap_parts = (("safe", safe_rows), ("near", near_rows), ("outside", outside_rows))
    for name, rows in heatmap_parts:
        if len(rows):
            mesh_paths[name] = _export_triangle_mesh(
                assets / f"heatmap_{name}.obj", posed, asset.faces, rows
            )
    if len(context_bone_rows):
        mesh_paths["context"] = _export_triangle_mesh(
            assets / "heatmap_out_of_scope_context.obj",
            posed,
            asset.faces,
            context_bone_rows,
        )
    section_rows = _section_skin_faces(
        skin,
        skin_faces,
        debug["pivots"],
    )
    mesh_paths["section_skin"] = _export_triangle_mesh(
        assets / "section_skin.obj", skin, skin_faces, section_rows
    )

    base_debug = [
        ("smplx_stations", mesh_paths["station"], COLOR_STATION),
        ("anatomical_pivots", mesh_paths["pivot"], COLOR_PIVOT),
        ("functional_axes", mesh_paths["axis"], COLOR_AXIS),
        ("station_residuals", mesh_paths["residual"], COLOR_RESIDUAL),
    ]
    render_records: dict[str, Any] = {}
    all_rgb: list[Path] = []
    for mode in REVIEW_MODES:
        section = mode == "joint_sections"
        skin_key = "section_skin" if section else "skin"
        entities: list[tuple[str, Path, tuple[float, float, float, float]]] = [
            ("smplx_skin", mesh_paths[skin_key], COLOR_SKIN)
        ]
        if mode == "signed_distance":
            if "context" in mesh_paths:
                entities.append(
                    ("bones_out_of_scope_context", mesh_paths["context"], COLOR_BONE)
                )
            for key, color in (
                ("safe", COLOR_BONE),
                ("near", COLOR_NEAR),
                ("outside", COLOR_OUTSIDE),
            ):
                if key in mesh_paths:
                    entities.append((f"bones_{key}", mesh_paths[key], color))
        else:
            entities.append(("bones", mesh_paths["bones"], COLOR_BONE))
        if mode == "bones+tubes":
            for key, color in (
                ("artery", COLOR_ARTERY),
                ("vein", COLOR_VEIN),
                ("nerve", COLOR_NERVE),
            ):
                if key in mesh_paths:
                    entities.append((key, mesh_paths[key], color))
        entities.extend(base_debug)
        cameras = _camera_specs(posed, debug, section_only=section)
        mode_dir = out / "renders" / mode.replace("+", "_plus_")
        records, rgb_paths = _render_genesis_scene(
            mesh_entities=entities,
            cameras=cameras,
            output_dir=mode_dir,
            backend=backend,
            resolution=resolution,
        )
        contact_sheet = _contact_sheet(
            rgb_paths,
            mode_dir / "contact_sheet.png",
        )
        render_records[mode] = {
            "cameras": [camera.__dict__ for camera in cameras],
            "checks": records,
            "pass": bool(records and all(record["pass"] for record in records)),
            "contact_sheet": str(contact_sheet),
        }
        all_rgb.extend(rgb_paths)

    diagnostic_path = out / "bone_segment_diagnostics.json"
    bone_diagnostic = write_bone_segment_diagnostics(
        asset,
        pose_axis_angle=pose.pose_axis_angle,
        transl=pose.transl,
        output_path=diagnostic_path,
    )
    intersections = _joint_intersection_report(
        asset,
        posed,
        baseline_vertices=baseline_posed,
    )
    bed_robot_scene = (
        _render_bed_robot_scene(
            posed=posed,
            skin=skin,
            skin_faces=skin_faces,
            asset=asset,
            bone_rows=bone_rows,
            artery_rows=artery_rows,
            vein_rows=vein_rows,
            nerve_rows=nerve_rows,
            output_dir=out / "renders" / "bed_robot_scene",
            backend=backend,
            resolution=resolution,
        )
        if include_bed_robot_scene
        else {"available": False, "reason": "representative cell only"}
    )
    cell = {
        "schema_version": BONE_REVIEW_PACK_SCHEMA_VERSION,
        "artifact_kind": "BoneReviewCellV8",
        "subject": subject_label,
        "pose": pose.label,
        "pose_source": pose.source,
        "pose_digest": smplx_pose_hash(pose.pose_axis_angle, pose.transl),
        "subject_runtime_digest": subject.runtime_digest(validate=False),
        "functional_frames": debug_report,
        "containment": containment,
        "exact_joint_triangle_intersections": intersections,
        "bone_segment_diagnostics": {
            "path": str(diagnostic_path),
            "passed": bool(bone_diagnostic["passed"]),
            "failures": list(bone_diagnostic["failures"]),
        },
        "render_modes": render_records,
        "bed_robot_scene": bed_robot_scene,
        "geometry_source": "real_triangle_meshes_only",
        "point_cloud_release_evidence": False,
    }
    cell["automatic_pass"] = bool(
        containment["pass"]
        and intersections["pass"]
        and bone_diagnostic["passed"]
        and all(value["pass"] for value in render_records.values())
        and (not bed_robot_scene["available"] or bed_robot_scene["pass"])
    )
    cell["publishable"] = False
    cell["human_signature"] = "pending"
    _json_write(out / "cell.json", cell)
    return cell


def write_bone_review_pack_manifest_v8(
    *,
    output_dir: Path | str,
    operator_runtime_digest: str,
    operator_manifest: Path,
    cells: Sequence[Mapping[str, Any]],
    sweeps: Sequence[Mapping[str, Any]] = (),
) -> Path:
    out = Path(output_dir)
    manifest = {
        "schema_version": BONE_REVIEW_PACK_SCHEMA_VERSION,
        "artifact_kind": "BoneReviewPackV8",
        "operator_runtime_digest": str(operator_runtime_digest),
        "operator_manifest": str(operator_manifest),
        "operator_manifest_sha256": _sha256(operator_manifest),
        "matrix": "2_subjects_x_3_poses",
        "cell_count": int(len(cells)),
        "cells": [
            {
                "subject": str(cell["subject"]),
                "pose": str(cell["pose"]),
                "automatic_pass": bool(cell["automatic_pass"]),
                "publishable": False,
            }
            for cell in cells
        ],
        "sweeps": [
            {
                "subject": str(sweep["subject"]),
                "sample_count": int(sweep["sample_count"]),
                "pass": bool(sweep["pass"]),
                "failure_count": int(len(sweep["failures"])),
            }
            for sweep in sweeps
        ],
        "all_sweeps_passed": bool(
            len(sweeps) == 2 and all(bool(sweep["pass"]) for sweep in sweeps)
        ),
        "all_automatic_gates_passed": bool(
            len(cells) == 6 and all(bool(cell["automatic_pass"]) for cell in cells)
            and len(sweeps) == 2
            and all(bool(sweep["pass"]) for sweep in sweeps)
        ),
        "publishable": False,
        "human_signature": "pending",
        "latest_asset_updated": False,
        "scope": "pelvis_and_appendicular_bones_only",
        "vessel_geometry_repair": "not_run",
    }
    path = out / "bone_review_pack_v8.json"
    _json_write(path, manifest)
    return path


__all__ = [
    "BONE_REVIEW_PACK_SCHEMA_VERSION",
    "REVIEW_MODES",
    "ReviewPoseV8",
    "SyntheticSweepStateV8",
    "synthetic_sweep_states_v8",
    "write_bone_review_cell_v8",
    "write_bone_review_pack_manifest_v8",
    "write_bone_review_sweep_v8",
]
