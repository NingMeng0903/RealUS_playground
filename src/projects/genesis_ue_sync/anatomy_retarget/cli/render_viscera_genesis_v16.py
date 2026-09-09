"""Genesis review renderer for the internal anatomy layers.

This module is deliberately a review tool.  It does not fit, rebind, or
modify a retarget package.  It consumes the already evaluated geometry cells
used by the retarget validation scripts and exports only the source meshes
selected by their authenticated mesh metadata.  In particular, a source
``organ`` label is not by itself treated as a thoraco-abdomino-pelvic organ:
the regional list below is explicit and is recorded in the manifest.

The renderer uses the existing Genesis runtime and the private helpers from
the established review renderer for OBJ export, cameras' RGB/depth/segmentation
output, and source hashing.  Every mesh is exported with local vertices and
localised faces; passing the full 394k vertex array to each small object would
make the resulting review both slow and unnecessarily large.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .render_alignment_truth_genesis_v1 import _render_layer, COLORS as BASE_COLORS
from .render_chain_rest_fit_genesis_v1 import _export, _sha256


# These are names observed in the source inventory.  Keeping the selection
# explicit prevents cranial organs, oral structures, and other labelled organ
# meshes from silently entering a torso review.
THORACO_ABDOMINO_PELVIC_ORGANS = frozenset(
    {
        "Adrenal_Gland_L",
        "Adrenal_Gland_R",
        "Appendix",
        "Bladder",
        "Diaphragm",
        "Esophagus",
        "Gallbladder",
        "Kidney_L",
        "Kidney_R",
        "Large_Intestine",
        "Liver",
        "Lung_L",
        "Lung_R",
        "Pancreas",
        "Small_Intestine",
        "Spleen",
        "Stomach",
        "Trachea",
        "Ureter_L",
        "Ureter_R",
        "Urethra_Short",
    }
)

# UNCUT meshes are references to a larger, overlapping segmentation.  They
# are intentionally kept out of ``viscera`` and rendered only in their own
# layer, so overlap cannot be mistaken for a retarget error.
UNCUT_REFERENCE_MESHES = frozenset(
    {"UNCUT_Digestive_Tract", "UNCUT_Cerebrum_L", "UNCUT_Cerebrum_R"}
)

ARTERY_MESHES = frozenset({"Artery"})
VEIN_MESHES = frozenset({"Vein"})

# This is the nerve set relevant to a torso/pelvis check.  Facial and optic
# nerves remain in the inventory but are excluded from these regional views.
TRUNK_NERVE_MESHES = frozenset(
    {
        "Autonomic",
        "Cervical_Nerves_L",
        "Cervical_Nerves_R",
        "Coccygeal_Nerve_L",
        "Coccygeal_Nerve_R",
        "Lumbar_Nerves_L",
        "Lumbar_Nerves_R",
        "Sacral_Nerves_L",
        "Sacral_Nerves_R",
        "Spinal_Cord",
        "Thoracic_Nerves_L",
        "Thoracic_Nerves_R",
    }
)

AXIAL_BONE_NAMES = frozenset(
    {
        "C1_Atlas",
        "C2_Axis",
        *(f"C{i}" for i in range(1, 8)),
        *(f"T{i}" for i in range(1, 13)),
        *(f"L{i}" for i in range(1, 6)),
        *(f"Disc_{a}_{b}" for a, b in (
            ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "C6"),
            ("C6", "C7"), ("C7", "T1"), ("T1", "T2"), ("T2", "T3"),
            ("T3", "T4"), ("T4", "T5"), ("T5", "T6"), ("T6", "T7"),
            ("T7", "T8"), ("T8", "T9"), ("T9", "T10"), ("T10", "T11"),
            ("T11", "T12"), ("T12", "L1"), ("L1", "L2"), ("L2", "L3"),
            ("L3", "L4"), ("L4", "L5"), ("L5", "S1"),
        )),
        *(f"Rib_{i}{side}" for i in range(1, 13) for side in ("L", "R")),
        "Sacrum",
        "Sternum",
        "Ilium_L",
        "Ilium_R",
        "Clavicle_L",
        "Clavicle_R",
        "Scapula_L",
        "Scapula_R",
        "Femur_L",
        "Femur_R",
    }
)


# Colors are Genesis RGBA values.  The artery/vein/nerve colors are stable
# across variants and runs, which is necessary for visual before/after review.
ARTERY_COLOR = (0.92, 0.025, 0.018, 0.95)
VEIN_COLOR = (0.045, 0.22, 0.94, 0.95)
NERVE_COLOR = (0.98, 0.78, 0.035, 0.95)
# A requested focus mesh that is outside the default regional allowlist (for
# example a connective ligament) gets its own diagnostic material.  This
# group is empty unless the caller explicitly names such a mesh.
FOCUS_EXTRA_COLOR = (0.86, 0.38, 0.06, 1.0)
BONE_COLOR = (0.78, 0.80, 0.84, 0.42)
BONE_CONTEXT_COLOR = (0.78, 0.80, 0.84, 0.58)
SKIN_CONTEXT_COLOR = (0.90, 0.58, 0.43, 0.12)
SKIN_VISCERA_COLOR = (0.90, 0.58, 0.43, 0.10)
SKIN_UNCUT_COLOR = (0.90, 0.58, 0.43, 0.055)
SKIN_OPAQUE_COLOR = (0.90, 0.58, 0.43, 1.0)


# A palette gives neighbouring organs visual separation while preserving
# anatomical identity.  The exact source mesh name is retained in the
# Genesis entity name and in the manifest.
ORGAN_COLORS: dict[str, tuple[float, float, float, float]] = {
    "Adrenal_Gland_L": (0.98, 0.40, 0.08, 0.92),
    "Adrenal_Gland_R": (0.98, 0.52, 0.08, 0.92),
    "Appendix": (0.80, 0.60, 0.12, 0.92),
    "Bladder": (0.34, 0.66, 0.96, 0.92),
    "Diaphragm": (0.88, 0.42, 0.70, 0.70),
    "Esophagus": (0.82, 0.25, 0.54, 0.92),
    "Gallbladder": (0.68, 0.88, 0.08, 0.92),
    "Kidney_L": (0.68, 0.16, 0.12, 0.92),
    "Kidney_R": (0.82, 0.22, 0.12, 0.92),
    "Large_Intestine": (0.65, 0.34, 0.75, 0.88),
    "Liver": (0.62, 0.12, 0.055, 0.94),
    "Lung_L": (0.95, 0.38, 0.54, 0.76),
    "Lung_R": (0.98, 0.50, 0.61, 0.76),
    "Pancreas": (0.98, 0.69, 0.26, 0.92),
    "Small_Intestine": (0.86, 0.54, 0.13, 0.88),
    "Spleen": (0.38, 0.13, 0.45, 0.92),
    "Stomach": (0.90, 0.27, 0.36, 0.92),
    "Trachea": (0.28, 0.78, 0.78, 0.90),
    "Ureter_L": (0.90, 0.78, 0.38, 0.92),
    "Ureter_R": (0.96, 0.84, 0.42, 0.92),
    "Urethra_Short": (0.74, 0.84, 0.22, 0.92),
}
UNCUT_COLORS = {
    "UNCUT_Digestive_Tract": (0.50, 0.82, 0.35, 0.38),
    "UNCUT_Cerebrum_L": (0.40, 0.60, 0.92, 0.32),
    "UNCUT_Cerebrum_R": (0.54, 0.66, 0.98, 0.32),
}


TISSUE_CODE_NAMES = {
    0: "bone",
    1: "vessel",
    2: "nerve",
    3: "organ",
    4: "heart",
    5: "connective_tissue",
}


def _as_string_list(value: Any, *, field: str) -> list[str]:
    values = np.asarray(value).reshape(-1)
    result: list[str] = []
    for item in values.tolist():
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _normalise_tissues(value: Any, *, field: str) -> list[str]:
    values = np.asarray(value).reshape(-1)
    result: list[str] = []
    for item in values.tolist():
        if isinstance(item, (int, np.integer)):
            try:
                result.append(TISSUE_CODE_NAMES[int(item)])
            except KeyError as exc:
                raise ValueError(f"{field} has unknown tissue code {item}") from exc
        else:
            text = str(item).strip().lower()
            aliases = {
                "vessels": "vessel",
                "artery": "vessel",
                "vein": "vessel",
                "nerves": "nerve",
                "organs": "organ",
                "heart": "heart",
                "connective": "connective_tissue",
                "connective tissue": "connective_tissue",
                "connective_tissue": "connective_tissue",
            }
            result.append(aliases.get(text, text))
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _load_compiled_metadata(path: Path) -> tuple[list[str], np.ndarray, list[str], str]:
    """Read authenticated source metadata from a V8 subject pack once."""

    from ..v8_artifacts import load_subject_runtime
    from ..rigged_asset import load_rigged_asset

    root = path.resolve()
    candidates = [root]
    if root.is_dir() and (root / "source_pack").is_dir():
        candidates.insert(0, root / "source_pack")
    errors: list[str] = []
    for candidate in candidates:
        try:
            if candidate.is_dir() and (candidate / "manifest.json").exists():
                subject = load_subject_runtime(candidate, mmap=True)
                asset = subject.rigged_asset
            else:
                asset = load_rigged_asset(candidate)
            return (
                [str(v) for v in asset.source_mesh_names],
                np.asarray(asset.source_vertex_ranges, dtype=np.int64).copy(),
                [str(v) for v in (asset.source_tissues or [])],
                str(getattr(subject, "runtime_digest", lambda **_: "")(
                    validate=False
                )) if "subject" in locals() else "",
            )
        except Exception as exc:  # try the next accepted compiled form
            errors.append(f"{candidate}: {exc}")
    raise ValueError(
        "--compiled did not contain a readable SubjectRuntimePackV8 or rigged asset: "
        + " | ".join(errors)
    )


def _read_metadata(
    data: Mapping[str, Any],
    *,
    compiled: Path | None,
) -> tuple[list[str], np.ndarray, list[str], str]:
    fields = ("mesh_names", "mesh_ranges", "mesh_tissues")
    present = [field in data for field in fields]
    if all(present):
        names = _as_string_list(data["mesh_names"], field="mesh_names")
        ranges = np.asarray(data["mesh_ranges"], dtype=np.int64).reshape(-1, 2)
        tissues = _normalise_tissues(data["mesh_tissues"], field="mesh_tissues")
        if len(names) != len(ranges) or len(names) != len(tissues):
            raise ValueError("mesh_names, mesh_ranges, mesh_tissues must have equal length")
        return names, ranges, tissues, "input_npz"
    if any(present):
        missing = [field for field, present_one in zip(fields, present) if not present_one]
        raise ValueError(
            "geometry contains only part of the mesh metadata; missing "
            + ", ".join(missing)
            + ". Pass the matching --compiled source pack instead."
        )
    if compiled is None:
        raise ValueError(
            "geometry has no mesh_names/mesh_ranges/mesh_tissues; pass --compiled "
            "to authenticate the source inventory"
        )
    return (*_load_compiled_metadata(compiled)[:3], "compiled_source_pack")


def _validate_metadata(
    *,
    names: Sequence[str],
    ranges: np.ndarray,
    tissues: Sequence[str],
    vertex_count: int,
) -> None:
    if len(names) != len(ranges) or len(names) != len(tissues):
        raise ValueError("source metadata arrays have inconsistent lengths")
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"mesh_ranges must be [M,2], got {ranges.shape}")
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError("mesh_ranges must be non-empty and non-negative")
    if np.any(ranges[1:, 0] != ranges[:-1, 1]):
        raise ValueError("mesh_ranges must be sorted contiguous source ranges")
    if int(ranges[0, 0]) != 0 or int(ranges[-1, 1]) != int(vertex_count):
        raise ValueError(
            "mesh_ranges must cover the complete geometry vertex array; "
            f"got [{ranges[0,0]}, {ranges[-1,1]}) for {vertex_count} vertices"
        )


def _mesh_faces(
    faces: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"faces must be [F,3], got {triangles.shape}")
    mask = np.all((triangles >= int(start)) & (triangles < int(stop)), axis=1)
    selected = triangles[mask]
    return (selected - int(start)).astype(np.int32, copy=False)


def _slice_export(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Path | None:
    """Export the positive half-space of one display-only mesh.

    The clipping plane is intentionally uncapped.  A cap would introduce
    synthetic topology and could be mistaken for an authored organ surface in
    the review.  ``trimesh`` performs the triangle clipping while retaining
    the original material-independent geometry; an empty result is reported
    as ``None`` and is never replaced by the uncut mesh.
    """

    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    clipped = mesh.slice_plane(
        plane_origin=np.asarray(plane_origin, dtype=np.float64),
        plane_normal=np.asarray(plane_normal, dtype=np.float64),
        cap=False,
    )
    if clipped is None or len(clipped.faces) == 0 or len(clipped.vertices) == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped.export(path)
    return path


def _merge_meshes(
    vertices: np.ndarray,
    faces: np.ndarray,
    ranges: np.ndarray,
    mesh_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Merge selected contiguous source meshes into one local OBJ mesh."""

    vertex_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    included: list[int] = []
    offset = 0
    for raw_index in mesh_ids:
        index = int(raw_index)
        start, stop = (int(v) for v in ranges[index])
        local_faces = _mesh_faces(faces, start=start, stop=stop)
        if len(local_faces) == 0:
            continue
        vertex_parts.append(np.asarray(vertices[start:stop], dtype=np.float32))
        face_parts.append(local_faces + offset)
        included.append(index)
        offset += stop - start
    if not vertex_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32), []
    return (
        np.concatenate(vertex_parts, axis=0),
        np.concatenate(face_parts, axis=0).astype(np.int32, copy=False),
        included,
    )


def _selected_ids(
    names: Sequence[str], tissues: Sequence[str], predicate: Any
) -> list[int]:
    return [index for index, (name, tissue) in enumerate(zip(names, tissues)) if predicate(name, tissue)]


def _selection_manifest(
    names: Sequence[str],
    tissues: Sequence[str],
    ranges: np.ndarray,
    faces: np.ndarray,
    focus_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    def entries(ids: Iterable[int], reason: str) -> list[dict[str, Any]]:
        result = []
        for index in ids:
            start, stop = (int(v) for v in ranges[index])
            result.append(
                {
                    "index": int(index),
                    "name": str(names[index]),
                    "tissue": str(tissues[index]),
                    "vertex_range": [start, stop],
                    "face_count": int(len(_mesh_faces(faces, start=start, stop=stop))),
                    "reason": reason,
                }
            )
        return result

    ids = list(range(len(names)))
    organ_ids = _selected_ids(
        names,
        tissues,
        lambda name, tissue: str(name) in THORACO_ABDOMINO_PELVIC_ORGANS
        and str(tissue) == "organ",
    )
    heart_ids = _selected_ids(names, tissues, lambda name, tissue: str(name) == "Heart" and str(tissue) == "heart")
    artery_ids = _selected_ids(names, tissues, lambda name, tissue: str(name) in ARTERY_MESHES and str(tissue) == "vessel")
    vein_ids = _selected_ids(names, tissues, lambda name, tissue: str(name) in VEIN_MESHES and str(tissue) == "vessel")
    nerve_ids = _selected_ids(names, tissues, lambda name, tissue: str(name) in TRUNK_NERVE_MESHES and str(tissue) == "nerve")
    uncut_ids = _selected_ids(names, tissues, lambda name, tissue: str(name) in UNCUT_REFERENCE_MESHES and str(tissue) == "organ")
    bone_ids = _selected_ids(names, tissues, lambda name, tissue: str(name) in AXIAL_BONE_NAMES and str(tissue) == "bone")
    standard_ids = organ_ids + heart_ids + artery_ids + vein_ids + nerve_ids + uncut_ids + bone_ids
    selected = set(standard_ids)
    focus_extra_ids: list[int] = []
    # Focus extras are deliberately opt-in.  This preserves the authenticated
    # regional selection for normal full-body renders while allowing a caller
    # to inspect an exact connective/other mesh by its source name.
    for requested_name in focus_names or ():
        requested_text = str(requested_name)
        matching = [index for index, name in enumerate(names) if str(name) == requested_text]
        if not matching:
            # The main CLI reports this as an unknown focus mesh after the
            # selection is built; do not manufacture an inventory entry here.
            continue
        index = int(matching[0])
        if index not in selected and index not in focus_extra_ids:
            focus_extra_ids.append(index)
    selected.update(focus_extra_ids)
    excluded = [index for index in ids if index not in selected]
    return {
        "selection_policy": {
            "regional_organs": "explicit mesh-name allowlist; tissue must be organ",
            "heart": "exact mesh name Heart; tissue must be heart",
            "artery": "exact mesh name Artery; tissue must be vessel",
            "vein": "exact mesh name Vein; tissue must be vessel",
            "trunk_nerves": "explicit thoracic/abdominal/pelvic nerve allowlist; tissue must be nerve",
            "uncut_reference": "exact UNCUT names; separate reference layer",
            "axial_bones": "explicit spine/rib-support/pelvic bone allowlist; tissue must be bone",
            "focus_extra": "exact --focus-meshes source names outside the default regional allowlist; opt-in only",
        },
        "included_by_group": {
            "organs": entries(organ_ids, "regional_organ_allowlist"),
            "heart": entries(heart_ids, "exact_heart_mesh"),
            "artery": entries(artery_ids, "exact_artery_mesh"),
            "vein": entries(vein_ids, "exact_vein_mesh"),
            "nerves": entries(nerve_ids, "trunk_nerve_allowlist"),
            "uncut_reference": entries(uncut_ids, "separate_uncut_reference_layer"),
            "bones": entries(bone_ids, "axial_pelvic_context_allowlist"),
            "focus_extra": entries(focus_extra_ids, "explicit_focus_mesh_outside_default_allowlist"),
        },
        "excluded": entries(excluded, "outside_v16_regional_review_selection"),
        "counts": {
            "source_meshes": len(names),
            "included_meshes": len(selected),
            "excluded_meshes": len(excluded),
        },
    }


def _internal_ids(selection: Mapping[str, Any]) -> list[int]:
    groups = selection["included_by_group"]
    names = ("organs", "heart", "artery", "vein", "nerves")
    return [int(entry["index"]) for name in names for entry in groups[name]]


def _group_ids(selection: Mapping[str, Any], group: str) -> list[int]:
    return [int(entry["index"]) for entry in selection["included_by_group"].get(group, [])]


def _centre_and_span(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("camera points must be non-empty [N,3]")
    lower, upper = points.min(axis=0), points.max(axis=0)
    return (lower + upper) * 0.5, np.maximum(upper - lower, 1.0e-4)


def _camera(
    center: np.ndarray,
    span: np.ndarray,
    direction: Sequence[float],
    *,
    fov: float = 34.0,
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> dict[str, Any]:
    direction_array = np.asarray(direction, dtype=np.float64)
    direction_array /= max(float(np.linalg.norm(direction_array)), 1.0e-8)
    # The camera distance is computed from the selected interior extent, not
    # from the full body.  This keeps the torso and pelvis views useful while
    # leaving a generous margin for the skin shell.
    radius = max(float(span[0]), float(span[1]) * 720.0 / 540.0, float(span[2])) * 0.62
    distance = max(0.18, radius / np.tan(np.deg2rad(fov * 0.5)))
    position = np.asarray(center, dtype=np.float64) + direction_array * distance
    return {
        "pos": position.tolist(),
        "lookat": np.asarray(center, dtype=np.float64).tolist(),
        "up": list(up),
        "fov": float(fov),
        "near": max(0.01, distance - max(radius * 1.8, 0.2)),
        "far": distance + max(radius * 1.8, 0.2),
    }


def _cameras(
    skin: np.ndarray,
    vertices: np.ndarray,
    ranges: np.ndarray,
    selection: Mapping[str, Any],
    focus_ids: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    # Artery, vein, and the long spinal/lower-limb nerve meshes span nearly
    # the whole body.  They are useful entities in a render, but using them
    # to derive a torso camera would put the camera several metres away and
    # make the viscera unreadable.  Use the explicitly selected regional
    # organs (and Heart) for the torso framing instead.
    regional_ids = _group_ids(selection, "organs") + _group_ids(selection, "heart")
    if focus_ids is not None:
        # A focus camera is derived only from the requested source meshes. The
        # complete skin is still loaded by every skin layer, so this changes
        # framing rather than the target body surface.
        regional_ids = list(dict.fromkeys(int(index) for index in focus_ids))
    regional_parts = [
        vertices[int(ranges[index, 0]) : int(ranges[index, 1])]
        for index in regional_ids
    ]
    if not regional_parts:
        raise ValueError("selection has no regional organ vertices for cameras")
    regional_points = np.concatenate(regional_parts, axis=0)
    torso_center, torso_span = _centre_and_span(regional_points)
    pelvis_names = {
        "Bladder",
        "Kidney_L",
        "Kidney_R",
        "Ureter_L",
        "Ureter_R",
        "Urethra_Short",
        "Appendix",
        "Large_Intestine",
        "Small_Intestine",
    }
    pelvis_ids = [
        int(entry["index"])
        for entry in selection["included_by_group"]["organs"]
        if str(entry["name"]) in pelvis_names
    ]
    pelvis_bone_names = {"Ilium_L", "Ilium_R", "Sacrum", "Femur_L", "Femur_R"}
    pelvis_ids += [
        int(entry["index"])
        for entry in selection["included_by_group"]["bones"]
        if str(entry["name"]) in pelvis_bone_names
    ]
    pelvis_parts = [vertices[int(ranges[index, 0]) : int(ranges[index, 1])] for index in pelvis_ids]
    pelvis_points = np.concatenate(pelvis_parts, axis=0) if pelvis_parts else regional_points
    pelvis_center, pelvis_span = _centre_and_span(pelvis_points)
    skin_center, skin_span = _centre_and_span(skin)
    cameras = {
        "whole_ap": _camera(skin_center, skin_span, (0.0, 0.0, 1.0), fov=34.0),
        "torso_anterior": _camera(torso_center, torso_span, (0.0, 0.0, 1.0)),
        "torso_posterior": _camera(torso_center, torso_span, (0.0, 0.0, -1.0)),
        "torso_oblique": _camera(torso_center, torso_span, (0.72, 0.16, 0.68)),
        "torso_left": _camera(torso_center, torso_span, (1.0, 0.0, 0.0)),
        "torso_right": _camera(torso_center, torso_span, (-1.0, 0.0, 0.0)),
        "pelvis_ap": _camera(pelvis_center, pelvis_span, (0.0, 0.0, 1.0), fov=36.0),
        "pelvis_oblique": _camera(pelvis_center, pelvis_span, (0.72, 0.16, 0.68), fov=36.0),
        "pelvis_posterior": _camera(pelvis_center, pelvis_span, (0.0, 0.0, -1.0), fov=36.0),
    }
    return cameras


def _write_contact_sheet(
    paths: Sequence[Path | Image.Image], labels: Sequence[str], output: Path
) -> Path:
    # The layer renderer supplies file paths, while the before/after branch
    # already has two PIL images open.  Accept both forms so comparison
    # generation cannot fail after a successful Genesis render.
    images = [
        value.convert("RGB") if isinstance(value, Image.Image)
        else Image.open(value).convert("RGB")
        for value in paths
    ]
    if not images:
        raise ValueError("cannot write empty comparison sheet")
    columns = min(4, len(images))
    thumb_width, thumb_height = 360, 270
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 28)), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + 28)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + thumb_height + 5), str(label), fill=(240, 240, 240))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def _mesh_entity_paths(
    root: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    names: Sequence[str],
    ranges: np.ndarray,
    selection: Mapping[str, Any],
    focus_names: set[str] | None = None,
    clip_plane: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    geometry_root = root / "geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {
        "organs": {}, "heart": {}, "artery": {}, "vein": {},
        "nerves": {}, "uncut_reference": {}, "focus_extra": {},
        "clipped_groups": {
            "organs": {}, "heart": {}, "artery": {}, "vein": {},
            "nerves": {}, "uncut_reference": {}, "focus_extra": {},
        },
        # This is metadata for the renderer only.  Keeping the plane beside
        # the exported paths lets the layer selector choose clipped copies
        # without changing the authored geometry or the input arrays.
        "clip_plane": clip_plane,
    }

    for group in (
        "organs",
        "heart",
        "artery",
        "vein",
        "nerves",
        "uncut_reference",
        "focus_extra",
    ):
        for entry in selection["included_by_group"][group]:
            index = int(entry["index"])
            name = str(names[index])
            start, stop = (int(value) for value in ranges[index])
            local_faces = _mesh_faces(faces, start=start, stop=stop)
            if len(local_faces) == 0:
                # Keep an explicit None entry for a requested focus mesh with
                # no source faces.  The audit can then report an intentional
                # empty mesh instead of silently dropping the name.
                if focus_names is not None and name in focus_names:
                    paths[group][name] = None
                    if clip_plane is not None:
                        paths["clipped_groups"][group][name] = None
                continue
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            path = _export(geometry_root / f"{safe_name}.obj", vertices[start:stop], local_faces)
            paths[group][name] = path
            if clip_plane is not None and focus_names is not None and name in focus_names:
                origin, normal = clip_plane
                paths["clipped_groups"][group][name] = _slice_export(
                    geometry_root / f"clipped_{safe_name}.obj",
                    vertices[start:stop],
                    local_faces,
                    plane_origin=origin,
                    plane_normal=normal,
                )

    bone_ids = _group_ids(selection, "bones")
    merged_vertices, merged_faces, included_bones = _merge_meshes(vertices, faces, ranges, bone_ids)
    if len(included_bones):
        paths["bones"] = _export(geometry_root / "axial_pelvic_bones.obj", merged_vertices, merged_faces)
    else:
        paths["bones"] = None
    if focus_names is not None:
        focused_bone_ids = [
            index for index in bone_ids if str(names[index]) in focus_names
        ]
        focused_vertices, focused_faces, focused = _merge_meshes(
            vertices, faces, ranges, focused_bone_ids
        )
        paths["focused_bones"] = (
            _export(geometry_root / "focused_bones.obj", focused_vertices, focused_faces)
            if len(focused)
            else None
        )
    else:
        paths["focused_bones"] = None
    paths["clipped_focused_bones"] = None
    if clip_plane is not None and focus_names is not None:
        origin, normal = clip_plane
        clipped_vertices, clipped_faces, clipped_bones = _merge_meshes(
            vertices,
            faces,
            ranges,
            [index for index in bone_ids if str(names[index]) in focus_names],
        )
        if len(clipped_bones):
            paths["clipped_focused_bones"] = _slice_export(
                geometry_root / "clipped_focused_bones.obj",
                clipped_vertices,
                clipped_faces,
                plane_origin=origin,
                plane_normal=normal,
            )
    return paths


def _focus_render_audit(
    selection: Mapping[str, Any],
    paths: Mapping[str, Any],
    focus_names: Sequence[str],
    *,
    clip_plane: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Report the concrete display result for every requested focus mesh.

    Bones are merged into one context OBJ, so their audit entry points to the
    focused merged path.  Other meshes retain one local path per source name.
    A clipped-away mesh is kept as an explicit ``empty_after_clip`` record;
    it must never disappear merely because its name was outside the default
    regional selection.
    """

    entry_by_name: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for group, entries in selection.get("included_by_group", {}).items():
        for entry in entries:
            entry_by_name.setdefault(str(entry["name"]), (str(group), entry))

    audits: list[dict[str, Any]] = []
    clipped_groups = paths.get("clipped_groups", {})
    for requested_name in focus_names:
        name = str(requested_name)
        selected = entry_by_name.get(name)
        if selected is None:
            audits.append(
                {
                    "name": name,
                    "included": False,
                    "rendered": False,
                    "status": "missing_selection",
                    "explicit": True,
                }
            )
            continue
        group, entry = selected
        face_count = int(entry.get("face_count", 0))
        if group == "bones":
            full_path = paths.get("focused_bones")
            display_path = (
                paths.get("clipped_focused_bones") if clip_plane is not None else full_path
            )
        else:
            full_path = (paths.get(group, {}) or {}).get(name)
            display_path = (
                (clipped_groups.get(group, {}) or {}).get(name)
                if clip_plane is not None
                else full_path
            )
        if face_count <= 0:
            status = "empty_no_faces"
        elif clip_plane is not None and display_path is None:
            status = "empty_after_clip"
        elif display_path is None:
            status = "missing_export"
        else:
            status = "rendered"
        audits.append(
            {
                "name": name,
                "index": int(entry["index"]),
                "group": group,
                "tissue": str(entry.get("tissue", "")),
                "face_count": face_count,
                "included": True,
                "rendered": bool(display_path is not None and status == "rendered"),
                "status": status,
                "full_export": str(full_path) if full_path is not None else None,
                "display_export": str(display_path) if display_path is not None else None,
                "clip_applied": bool(clip_plane is not None),
                "explicit": True,
            }
        )
    return audits


def _skin_path(root: Path, skin: np.ndarray, skin_faces: np.ndarray) -> Path:
    path = root / "geometry" / "smplx_skin.obj"
    return _export(path, skin, skin_faces)


def _organ_color(name: str) -> tuple[float, float, float, float]:
    if name == "Heart":
        return (0.92, 0.06, 0.14, 0.96)
    return ORGAN_COLORS.get(name, (0.70, 0.52, 0.20, 0.90))


def _entities_for_layer(
    layer: str,
    *,
    skin_path: Path,
    paths: Mapping[str, Any],
    focus_names: set[str] | None = None,
) -> list[tuple[str, Path, tuple[float, float, float, float]]]:
    entities: list[tuple[str, Path, tuple[float, float, float, float]]] = []
    restrict_to_focus = focus_names is not None and layer in {"context", "viscera"}
    use_clipped_focus = restrict_to_focus and paths.get("clip_plane") is not None

    def add_skin(color: tuple[float, float, float, float]) -> None:
        entities.append(("smplx_skin", skin_path, color))

    def add_group(group: str, color: Any = None) -> None:
        group_paths = paths.get(group, {})
        if use_clipped_focus:
            group_paths = paths.get("clipped_groups", {}).get(group, {})
        for name, path in group_paths.items():
            if restrict_to_focus and str(name) not in focus_names:
                continue
            if path is None:
                continue
            if callable(color):
                chosen = color(str(name))
            elif color is None:
                chosen = _organ_color(str(name))
            else:
                chosen = color
            # Genesis can exhibit triangle ordering artefacts even when the
            # skin is absent if internal meshes themselves use alpha.  All
            # internal review entities are therefore opaque; the skin is a
            # separate opaque layer and can be wiped in the review page.
            chosen = (*tuple(chosen[:3]), 1.0)
            entities.append((f"{group}_{name}", path, chosen))

    if layer == "context":
        # Keep this as a clean internal layer.  Genesis' transparent-mesh
        # depth sorting can hide thin ribs, lungs, or vessels behind the skin;
        # a clean layer makes every selected structure directly auditable.
        if use_clipped_focus:
            bone_path = paths.get("clipped_focused_bones")
        else:
            bone_path = paths.get("focused_bones") if restrict_to_focus else paths.get("bones")
        if bone_path is not None:
            entities.append(("axial_pelvic_bones", bone_path, (0.78, 0.80, 0.84, 1.0)))
        add_group("artery", ARTERY_COLOR)
        add_group("vein", VEIN_COLOR)
        add_group("nerves", NERVE_COLOR)
        add_group("focus_extra", FOCUS_EXTRA_COLOR)
        add_group("organs")
        add_group("heart")
    elif layer == "viscera":
        # Internal structures only: this is the primary vessel/organ review
        # layer and intentionally has no opaque rib or skin occluder.
        add_group("artery", ARTERY_COLOR)
        add_group("vein", VEIN_COLOR)
        add_group("nerves", NERVE_COLOR)
        add_group("focus_extra", FOCUS_EXTRA_COLOR)
        add_group("organs")
        add_group("heart")
    elif layer == "torso":
        # The regional context layer remains clean for the same reason as
        # ``context``; the separate ``skin`` layer supplies the outer boundary.
        if paths.get("bones") is not None:
            entities.append(("axial_pelvic_bones", paths["bones"], (0.78, 0.80, 0.84, 1.0)))
        add_group("artery", ARTERY_COLOR)
        add_group("vein", VEIN_COLOR)
        add_group("nerves", NERVE_COLOR)
        add_group("organs")
        add_group("heart")
    elif layer == "pelvis":
        # Keep bones and internal materials directly visible; no transparent
        # shell is mixed into this diagnostic layer.
        if paths.get("bones") is not None:
            entities.append(("axial_pelvic_bones", paths["bones"], (0.78, 0.80, 0.84, 1.0)))
        add_group("artery", ARTERY_COLOR)
        add_group("vein", VEIN_COLOR)
        add_group("nerves", NERVE_COLOR)
        add_group("organs")
        add_group("heart")
    elif layer in {"skin", "skin_only"}:
        # Opaque shell is a separate boundary reference.  It is never mixed
        # into the clean internal render, avoiding renderer-dependent alpha
        # ordering when judging an organ-to-bone relationship.
        add_skin(SKIN_OPAQUE_COLOR)
    elif layer == "uncut_reference":
        add_group("uncut_reference", lambda name: UNCUT_COLORS.get(name, (0.4, 0.6, 0.9, 0.35)))
    else:
        raise ValueError(f"unknown layer {layer!r}")
    if not entities:
        raise ValueError(f"layer {layer!r} has no selected mesh entities")
    return entities


def _render_variant(
    root: Path,
    *,
    variant: str,
    layer_names: Sequence[str],
    cameras: Mapping[str, Mapping[str, Any]],
    skin_path: Path,
    paths: Mapping[str, Any],
    backend: str,
    focus_names: set[str] | None = None,
) -> dict[str, Any]:
    variant_report: dict[str, Any] = {"layers": {}}
    for layer in layer_names:
        entities = _entities_for_layer(
            layer, skin_path=skin_path, paths=paths, focus_names=focus_names
        )
        layer_root = root / variant / layer
        variant_report["layers"][layer] = _render_layer(
            layer_root,
            entities=entities,
            cameras=dict(cameras),
            backend=backend,
        )
        # The historical helper lays out four columns even when only two to
        # five cameras were requested.  Replace that sheet with a compact one
        # while retaining each authoritative RGB/depth/segmentation file.
        report = variant_report["layers"][layer]
        records = report.get("renders", [])
        compact = _write_contact_sheet(
            [Path(record["rgb"]) for record in records],
            [str(record["camera"]) for record in records],
            root / variant / layer / "contact_sheet.png",
        )
        report["contact_sheet"] = str(compact)
        report["contact_sheet_sha256"] = _sha256(compact)
    return variant_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, help="matching V8 subject/source pack for missing metadata")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--before-key", default="before_vertices", choices=("before_vertices", "source_vertices"))
    parser.add_argument("--layers", nargs="+",
                        choices=("context", "viscera", "torso", "pelvis", "skin", "skin_only", "uncut_reference"),
                        help="layers to render; default is clean context and viscera")
    parser.add_argument("--views", nargs="+", help="camera names; default is five regional Genesis views")
    parser.add_argument(
        "--focus-meshes",
        nargs="+",
        help=(
            "exact source mesh names for a close-up; restricts context/viscera "
            "entities and derives torso camera framing from those meshes"
        ),
    )
    parser.add_argument(
        "--frame-meshes",
        nargs="+",
        help=(
            "optional subset of --focus-meshes used only for torso camera "
            "framing and the clip-plane AABB; every focus mesh is still exported"
        ),
    )
    parser.add_argument(
        "--clip-axis",
        choices=("x", "y", "z"),
        help=(
            "display-only close-up clipping axis; the positive half-space is "
            "kept at the selected focus AABB centre"
        ),
    )
    parser.add_argument(
        "--clip-offset-mm",
        type=float,
        default=None,
        help="offset the display clipping plane along --clip-axis in millimetres",
    )
    parser.add_argument("--discard-render-geometry", action="store_true",
                        help="remove this run's temporary OBJ geometry after RGB/depth/segmentation are retained")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Genesis viscera review: {output}")
    output.mkdir(parents=True)
    requested_layers = list(dict.fromkeys(args.layers or ("context", "viscera")))
    default_views = ("torso_anterior", "torso_posterior", "torso_oblique", "pelvis_ap", "pelvis_oblique")
    requested_views = list(dict.fromkeys(args.views or default_views))
    requested_focus_names = list(dict.fromkeys(args.focus_meshes or []))
    requested_frame_names = list(dict.fromkeys(args.frame_meshes or []))
    invalid_frame_names = sorted(set(requested_frame_names) - set(requested_focus_names))
    if invalid_frame_names:
        raise ValueError(
            "--frame-meshes must be a subset of --focus-meshes: "
            + ", ".join(invalid_frame_names)
        )
    effective_frame_names = requested_frame_names or requested_focus_names
    if args.clip_axis is None and args.clip_offset_mm is not None:
        raise ValueError("--clip-offset-mm requires --clip-axis")
    if args.clip_axis is not None and not requested_focus_names:
        raise ValueError("--clip-axis requires --focus-meshes so the display AABB is defined")
    clip_offset_mm = float(args.clip_offset_mm or 0.0)
    if not np.isfinite(clip_offset_mm):
        raise ValueError("--clip-offset-mm must be finite")
    manifest: dict[str, Any] = {
        "schema_version": 16,
        "renderer": "GenesisPlatformRuntime via render_alignment_truth_genesis_v1._render_layer",
        "artifact_kind": "VisceraReviewGenesisV16",
        "publishable": False,
        "solver_or_fit_run": False,
        "before_key": args.before_key,
        "before_label": args.before_label,
        "layers_requested": requested_layers,
        "views_requested": requested_views,
        "focus_meshes_requested": requested_focus_names,
        "frame_meshes_requested": requested_frame_names,
        "framing_selection": {
            "requested": requested_frame_names,
            "effective": effective_frame_names,
            "used_for": (
                (["torso_* camera framing"] if effective_frame_names else [])
                + (["clip plane AABB"] if args.clip_axis is not None else [])
            ),
            "active": bool(effective_frame_names),
            "all_focus_meshes_still_exported": True,
        },
        "clip_display_policy": {
            "enabled": bool(args.clip_axis),
            "axis": args.clip_axis,
            "offset_mm": clip_offset_mm if args.clip_axis is not None else None,
            "positive_half_space": "axis >= plane_origin[axis]" if args.clip_axis is not None else None,
            "cap": False,
            "cap_status": "uncapped" if args.clip_axis is not None else "not_requested",
            "scope": "focus meshes in context/viscera display exports only",
            "skin_clipped": False,
            "input_geometry_modified": False,
        },
        "source_metadata_policy": "mesh_names/mesh_ranges/mesh_tissues from input NPZ or one authenticated --compiled read",
        "color_policy": {
            "artery": ARTERY_COLOR,
            "vein": VEIN_COLOR,
            "nerve": NERVE_COLOR,
            "organ": "explicit per-mesh palette keyed by source mesh name",
        },
        "alpha_policy": {
            "context": 1.0,
            "viscera": 1.0,
            "torso": 1.0,
            "pelvis": 1.0,
            "uncut_reference": 1.0,
            "skin": 1.0,
            "internal_entities_opaque": True,
            "skin_is_separate_layer": True,
            "reason": "avoid Genesis triangle depth-order artefacts in internal collision review",
        },
        "cells": {},
    }
    for input_path in args.input:
        input_path = input_path.resolve()
        with np.load(input_path, allow_pickle=False) as data:
            required = ("faces", "skin_vertices", "skin_faces", "pose", "smplx_joints", args.before_key, "candidate_vertices")
            missing = [field for field in required if field not in data.files]
            if missing:
                raise ValueError(f"{input_path} missing geometry fields: {missing}")
            faces = np.asarray(data["faces"], dtype=np.int32)
            skin = np.asarray(data["skin_vertices"], dtype=np.float32)
            skin_faces = np.asarray(data["skin_faces"], dtype=np.int32)
            pose = np.asarray(data["pose"], dtype=np.float32)
            joints = np.asarray(data["smplx_joints"], dtype=np.float32)
            before = np.asarray(data[args.before_key], dtype=np.float32)
            candidate = np.asarray(data["candidate_vertices"], dtype=np.float32)
            names, ranges, tissues, metadata_source = _read_metadata(data, compiled=args.compiled)
        # Geometry cells store the evaluated world pose.  Normalize only the
        # root orientation for the review camera, applying exactly the same
        # rigid operation to skin, both variants, and SMPL-X joints.  This
        # keeps a sit/stand or AMASS root turn from making the anatomy appear
        # sideways while preserving every relative internal relationship.
        pose_array = pose.reshape(-1, 3)
        if pose_array.shape[0] != 55:
            raise ValueError(f"pose must contain 55 axis-angle joints in {input_path}")
        if joints.shape != (55, 3):
            raise ValueError(f"smplx_joints must be [55,3] in {input_path}")
        pivot = joints[0].astype(np.float64, copy=True)
        root_inverse = Rotation.from_rotvec(pose_array[0].astype(np.float64)).inv()

        def unrotate(points: np.ndarray) -> np.ndarray:
            values = np.asarray(points, dtype=np.float64)
            return root_inverse.apply(values - pivot) + pivot

        skin = unrotate(skin).astype(np.float32)
        before = unrotate(before).astype(np.float32)
        candidate = unrotate(candidate).astype(np.float32)
        joints = unrotate(joints).astype(np.float32)
        if before.shape != candidate.shape or before.ndim != 2 or before.shape[1] != 3:
            raise ValueError(f"before/candidate vertices must have equal [N,3] shape in {input_path}")
        _validate_metadata(names=names, ranges=ranges, tissues=tissues, vertex_count=len(candidate))
        if faces.size and (faces.min() < 0 or faces.max() >= len(candidate)):
            raise ValueError(f"faces out of bounds in {input_path}")
        selection = _selection_manifest(
            names,
            tissues,
            ranges,
            faces,
            focus_names=requested_focus_names,
        )
        selected_names = {
            str(entry["name"])
            for group in selection["included_by_group"].values()
            for entry in group
        }
        unknown_focus = sorted(set(requested_focus_names) - set(names))
        if unknown_focus:
            raise ValueError(
                f"unknown --focus-meshes {unknown_focus}; source inventory has no exact match"
            )
        excluded_focus = sorted(set(requested_focus_names) - selected_names)
        if excluded_focus:
            raise ValueError(
                "--focus-meshes names are outside the V16 regional selection: "
                + ", ".join(excluded_focus)
            )
        focus_ids = [names.index(name) for name in requested_focus_names] or None
        frame_ids = [names.index(name) for name in effective_frame_names] or None
        clip_plane: tuple[np.ndarray, np.ndarray] | None = None
        clip_manifest: dict[str, Any] = {
            "enabled": False,
            "axis": None,
            "axis_index": None,
            "offset_mm": None,
            "focus_aabb_min_display": None,
            "focus_aabb_max_display": None,
            "focus_aabb_center_display": None,
            "frame_aabb_min_display": None,
            "frame_aabb_max_display": None,
            "frame_aabb_center_display": None,
            "frame_meshes_used_for_aabb": effective_frame_names,
            "plane_origin_display": None,
            "plane_normal_display": None,
            "kept_half_space": None,
            "cap": False,
            "cap_status": "not_requested",
            "scope": "focus meshes in context/viscera display exports only",
            "skin_clipped": False,
            "input_geometry_modified": False,
        }
        if args.clip_axis is not None:
            if not frame_ids:
                raise ValueError("--clip-axis frame selection has no source mesh ids")
            frame_parts = [
                candidate[int(ranges[index, 0]) : int(ranges[index, 1])]
                for index in frame_ids
            ]
            if not frame_parts:
                raise ValueError("--clip-axis frame selection has no vertices")
            frame_points = np.concatenate(frame_parts, axis=0).astype(np.float64, copy=False)
            frame_lower = frame_points.min(axis=0)
            frame_upper = frame_points.max(axis=0)
            frame_center = (frame_lower + frame_upper) * 0.5
            axis_index = {"x": 0, "y": 1, "z": 2}[args.clip_axis]
            plane_origin = frame_center.copy()
            plane_origin[axis_index] += clip_offset_mm / 1000.0
            plane_normal = np.zeros(3, dtype=np.float64)
            plane_normal[axis_index] = 1.0
            clip_plane = (plane_origin, plane_normal)
            clip_manifest = {
                "enabled": True,
                "axis": args.clip_axis,
                "axis_index": axis_index,
                "offset_mm": clip_offset_mm,
                # Keep the historical focus_aabb keys for readers of earlier
                # V16 manifests, while explicitly identifying the effective
                # frame subset that supplied these values.
                "focus_aabb_min_display": frame_lower.astype(float).tolist(),
                "focus_aabb_max_display": frame_upper.astype(float).tolist(),
                "focus_aabb_center_display": frame_center.astype(float).tolist(),
                "frame_aabb_min_display": frame_lower.astype(float).tolist(),
                "frame_aabb_max_display": frame_upper.astype(float).tolist(),
                "frame_aabb_center_display": frame_center.astype(float).tolist(),
                "frame_meshes_used_for_aabb": effective_frame_names,
                "plane_origin_display": plane_origin.astype(float).tolist(),
                "plane_normal_display": plane_normal.astype(float).tolist(),
                "kept_half_space": f"{args.clip_axis} >= plane_origin[{args.clip_axis}]",
                "cap": False,
                "cap_status": "uncapped",
                "scope": "focus meshes in context/viscera display exports only",
                "skin_clipped": False,
                "input_geometry_modified": False,
            }
        cameras = _cameras(skin, candidate, ranges, selection, focus_ids=frame_ids)
        unknown_views = sorted(set(requested_views) - set(cameras))
        if unknown_views:
            raise ValueError(f"unknown views {unknown_views}; available {list(cameras)}")
        cameras = {name: cameras[name] for name in requested_views}
        cell_root = output / input_path.stem
        cell_root.mkdir()
        cell: dict[str, Any] = {
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "source_metadata": metadata_source,
            "root_rotation_removed_for_review": pose_array[0].astype(float).tolist(),
            "focus_meshes": [
                {"name": names[index], "index": int(index)}
                for index in (focus_ids or [])
            ],
            "frame_meshes": [
                {"name": names[index], "index": int(index)}
                for index in (frame_ids or [])
            ],
            "framing_selection": {
                "requested": requested_frame_names,
                "effective": effective_frame_names,
                "used_for": (
                    (["torso_* camera framing"] if effective_frame_names else [])
                    + (["clip plane AABB"] if args.clip_axis is not None else [])
                ),
                "active": bool(effective_frame_names),
                "all_focus_meshes_still_exported": True,
            },
            "focus_changes": {
                "context_and_viscera_content": bool(focus_ids),
                "torso_camera_center_and_range": bool(frame_ids),
                "skin_geometry": False,
                "clip_display_only": bool(clip_plane),
            },
            "clip_plane": clip_manifest,
            "mesh_selection": selection,
            "focus_render_audit": {},
            "camera_manifest": cameras,
            "variants": {},
        }
        variants: list[tuple[str, np.ndarray]] = []
        if not args.candidate_only:
            variants.append((args.before_label, before))
        variants.append(("candidate", candidate))
        # Skin is fixed for the geometry cell and can be exported once.  The
        # selected internal meshes are exported per variant because before and
        # candidate can have different coordinates.
        skin_path = _skin_path(cell_root, skin, skin_faces)
        cell["skin_obj"] = str(skin_path)
        for variant, vertices in variants:
            paths = _mesh_entity_paths(
                cell_root / variant,
                vertices=vertices,
                faces=faces,
                names=names,
                ranges=ranges,
                selection=selection,
                focus_names=set(requested_focus_names) if requested_focus_names else None,
                clip_plane=clip_plane,
            )
            cell["focus_render_audit"][variant] = _focus_render_audit(
                selection,
                paths,
                requested_focus_names,
                clip_plane=clip_plane,
            )
            # _mesh_entity_paths created a variant-local geometry directory;
            # the skin remains shared at cell_root/geometry.
            variant_skin = skin_path
            variant_report = _render_variant(
                cell_root,
                variant=variant,
                layer_names=requested_layers,
                cameras=cameras,
                skin_path=variant_skin,
                paths=paths,
                backend=args.backend,
                focus_names=set(requested_focus_names) if requested_focus_names else None,
            )
            cell["variants"][variant] = variant_report
            print(input_path.stem, variant, "Genesis viscera layers rendered", flush=True)
        if not args.candidate_only and len(variants) == 2:
            for view in requested_views:
                for layer in requested_layers:
                    images: list[Image.Image] = []
                    labels: list[str] = []
                    for variant, _ in variants:
                        image_path = cell_root / variant / layer / "rgb" / f"{view}.png"
                        images.append(Image.open(image_path).convert("RGB"))
                        labels.append(variant)
                    _write_contact_sheet(images, labels, cell_root / "comparison" / layer / f"{view}.png")
        manifest["cells"][input_path.stem] = cell
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if args.discard_render_geometry:
            import shutil
            # Keep the shared skin only while this cell is being rendered; all
            # Genesis layers have already written RGB/depth/segmentation.
            shutil.rmtree(cell_root / "geometry")
            for variant, _ in variants:
                shutil.rmtree(cell_root / variant / "geometry", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
