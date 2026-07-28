"""Compound-anatomy acceptance gates for the V7 operator.

Earlier compound checks either reselected contact probes after posing, treated
an authored skull overhang as a pose regression, or silently passed an oral
cavity that had no tongue mesh at all.  This module freezes every articular
and connection domain once on the immutable rest topology, measures only those
ids after posing, and fails closed when a required structure is missing or
when the source itself already violates a containment threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    JointContactThresholdsV7,
    rigid_edge_metrics_v7,
)


COMPOUND_GATE_SCHEMA_VERSION = 7
SIDES = ("left", "right")
SIDE_SUFFIX = {"left": "L", "right": "R"}

# Verified inventory: 24 ribs, Costal_Cartilage_{L,R}, Sternum, T1..T12.
RIB_MESH_NAMES: tuple[str, ...] = tuple(
    f"Rib_{index}{side}" for side in ("L", "R") for index in range(1, 13)
)
THORACIC_VERTEBRA_NAMES: tuple[str, ...] = tuple(f"T{index}" for index in range(1, 13))
COSTAL_CARTILAGE_NAMES: tuple[str, ...] = ("Costal_Cartilage_L", "Costal_Cartilage_R")
STERNUM_MESH_NAME = "Sternum"
UPPER_SKULL_MESH_NAME = "Upper_Skull"

# Intracranial organ meshes present in the verified 331-mesh anatomy whose
# tissue is ``organ``.  Keeping this explicit prevents a lung or gland from
# being counted as brain containment evidence.
INTRACRANIAL_ORGAN_NAMES: tuple[str, ...] = (
    "Amygdala_L",
    "Amygdala_R",
    "Basal_Ganglia",
    "Cerebellum",
    "Corpus_Callosum_L",
    "Corpus_Callosum_R",
    "Fornix",
    "Frontal_Lobe_L",
    "Frontal_Lobe_R",
    "Hippocampus",
    "Midbrain",
    "Occipital_Lobe_L",
    "Occipital_Lobe_R",
    "Olfactory_Bulb_L",
    "Olfactory_Bulb_R",
    "Parietal_Lobe_L",
    "Parietal_Lobe_R",
    "Pituitary_Gland",
    "Pons",
    "Temporal_Lobe_L",
    "Temporal_Lobe_R",
    "Thalamus_L",
    "Thalamus_R",
    "UNCUT_Cerebrum_L",
    "UNCUT_Cerebrum_R",
    "Ventricles",
)

LEGAL_TONGUE_SEARCH_PATHS: tuple[str, ...] = (
    "outputs/anatomy_retarget/v7_source_bake_001/v71_operator_source_v6.npz",
    "outputs/anatomy_retarget/v7_source_bake_001/rig_inspect.json",
)


@dataclass(frozen=True)
class CompoundGateThresholdsV7:
    """Fail-closed compound limits, in metres and dimensionless ratios."""

    elbow_gap_min_m: float = 0.0
    elbow_gap_max_m: float = 0.003
    elbow_gap_change_m: float = 0.002
    rib_connection_increase_m: float = 0.002
    # Floating and false ribs have no anterior attachment in the source asset;
    # only ends that are already seated at rest carry a connection gate.
    rib_attachment_rest_max_m: float = 0.005
    rib_rigid_q01_ratio: float = 0.99
    rib_rigid_q99_ratio: float = 1.01
    brain_inside_ratio_min: float = 1.0
    brain_max_outside_m: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


def _vertices(value: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"{label} must be a non-empty [N,3] array")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return points


def _faces(value: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(value, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be [F,3]")
    if faces.size and (np.any(faces < 0) or np.any(faces >= int(vertex_count))):
        raise ValueError("faces reference an invalid vertex")
    return faces


def _nearest_distance(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(points) or not len(target):
        return np.full((len(points),), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distance, _indices = cKDTree(target).query(points, k=1)
        return np.asarray(distance, dtype=np.float64)
    except Exception:
        result = np.empty((len(points),), dtype=np.float64)
        batch = max(1, min(2048, int(8_000_000 / max(1, len(target)))))
        for start in range(0, len(points), batch):
            chunk = points[start : start + batch]
            squared = np.sum((chunk[:, None] - target[None, :]) ** 2, axis=2)
            result[start : start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
        return result


def _clearance_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    distances = np.concatenate((_nearest_distance(a, b), _nearest_distance(b, a)))
    return {
        "min_m": float(np.min(distances)),
        "median_m": float(np.median(distances)),
        "q95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(np.max(distances)),
    }


def _distance_summary(points: np.ndarray, target: np.ndarray) -> dict[str, float]:
    distances = _nearest_distance(points, target)
    return {
        "min_m": float(np.min(distances)),
        "median_m": float(np.median(distances)),
        "q95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(np.max(distances)),
    }


def _mesh_lookup(asset: Any) -> dict[str, tuple[int, int, int]]:
    names = list(asset.source_mesh_names)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    if ranges.shape != (len(names), 2):
        raise ValueError("source_vertex_ranges must be [mesh_count,2]")
    tissues = list(asset.source_tissues) if asset.source_tissues is not None else [""] * len(names)
    if len(tissues) != len(names):
        raise ValueError("source_tissues length does not match source_mesh_names")
    return {
        str(name): (index, int(limits[0]), int(limits[1]))
        for index, (name, limits) in enumerate(zip(names, ranges))
    }


def _require_mesh_range(
    lookup: Mapping[str, tuple[int, int, int]],
    name: str,
) -> tuple[int, int]:
    try:
        _index, start, stop = lookup[name]
    except KeyError as exc:
        raise ValueError(f"required mesh {name!r} is absent") from exc
    if stop <= start:
        raise ValueError(f"required mesh {name!r} has an empty vertex range")
    return start, stop


def _mesh_indices(lookup: Mapping[str, tuple[int, int, int]], name: str) -> np.ndarray:
    start, stop = _require_mesh_range(lookup, name)
    return np.arange(start, stop, dtype=np.int64)


def _optional_mesh_indices(
    lookup: Mapping[str, tuple[int, int, int]],
    name: str,
) -> np.ndarray:
    if name not in lookup:
        return np.empty((0,), dtype=np.int64)
    _index, start, stop = lookup[name]
    if stop <= start:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def _union_indices(
    lookup: Mapping[str, tuple[int, int, int]],
    names: Sequence[str],
    *,
    required: bool,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    missing: list[str] = []
    for name in names:
        if name not in lookup:
            missing.append(name)
            continue
        chunks.append(_mesh_indices(lookup, name))
    if required and missing:
        raise ValueError(f"required mesh {missing[0]!r} is absent")
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(chunks))


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


def _mesh_local_surface(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Faces whose three vertices lie in ``[start, stop)``, re-indexed locally."""
    triangles = _faces(faces, len(vertices))
    mask = np.all((triangles >= int(start)) & (triangles < int(stop)), axis=1)
    selected = triangles[mask]
    if not len(selected):
        raise ValueError(
            f"mesh vertex range [{start}, {stop}) contains no complete faces"
        )
    local_vertices = vertices[int(start) : int(stop)]
    local_faces = selected - int(start)
    return local_vertices, local_faces


def _containment_metrics(
    points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
) -> dict[str, float]:
    signed, _closest, _normals = signed_distance(points, surface_vertices, surface_faces)
    outside = signed > 0.0
    return {
        "inside_ratio": float(np.mean(~outside)) if len(signed) else 0.0,
        "max_outside_m": float(max(0.0, float(np.max(signed)))) if len(signed) else 0.0,
        "vertex_count": float(len(signed)),
        "outside_count": float(np.count_nonzero(outside)),
    }


def _elbow_domains_for_side(
    lookup: Mapping[str, tuple[int, int, int]],
    reference: np.ndarray,
    side: str,
) -> dict[str, Any]:
    suffix = SIDE_SUFFIX[side]
    humerus_name = f"Humerus_{suffix}"
    ulna_name = f"Ulna_{suffix}"
    radius_name = f"Radius_{suffix}"
    humerus = _mesh_indices(lookup, humerus_name)
    ulna = _mesh_indices(lookup, ulna_name)
    radius = _mesh_indices(lookup, radius_name)
    forearm = np.unique(np.concatenate((ulna, radius)))
    # Fixed-material-domain rule: select once on rest geometry and reuse ids.
    humerus_contact = _closest_subset(
        humerus,
        reference,
        reference[forearm],
        fraction=0.20,
        minimum=8,
        maximum=512,
    )
    ulna_contact = _closest_subset(
        ulna,
        reference,
        reference[humerus],
        fraction=0.20,
        minimum=8,
        maximum=512,
    )
    radius_contact = _closest_subset(
        radius,
        reference,
        reference[humerus],
        fraction=0.20,
        minimum=8,
        maximum=512,
    )
    if min(len(humerus_contact), len(ulna_contact), len(radius_contact)) < 8:
        return {
            "available": False,
            "reason": (
                f"{side} elbow contact domain needs at least eight vertices "
                f"on Humerus/Ulna/Radius"
            ),
            "pass": False,
        }
    return {
        "available": True,
        "mesh_names": {
            "humerus": humerus_name,
            "ulna": ulna_name,
            "radius": radius_name,
        },
        "humerus_ids": humerus_contact,
        "ulna_ids": ulna_contact,
        "radius_ids": radius_contact,
        "humerus_all": humerus,
        "ulna_all": ulna,
        "radius_all": radius,
    }


def _gap_pair(
    *,
    reference: np.ndarray,
    posed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    limits: CompoundGateThresholdsV7,
) -> dict[str, Any]:
    rest = _clearance_summary(reference[first], reference[second])
    final = _clearance_summary(posed[first], posed[second])
    change = abs(final["min_m"] - rest["min_m"])
    passed = bool(
        limits.elbow_gap_min_m - 1.0e-12 <= final["min_m"] <= limits.elbow_gap_max_m
        and change <= limits.elbow_gap_change_m
    )
    return {
        "reference_clearance": rest,
        "posed_clearance": final,
        "gap_change_m": float(change),
        "pass": passed,
    }


def _evaluate_elbow(
    *,
    lookup: Mapping[str, tuple[int, int, int]],
    reference: np.ndarray,
    posed: np.ndarray,
    faces: np.ndarray,
    limits: CompoundGateThresholdsV7,
) -> dict[str, Any]:
    sides: dict[str, Any] = {}
    failures: list[str] = []
    for side in SIDES:
        domain = _elbow_domains_for_side(lookup, reference, side)
        if not domain.get("available", False):
            sides[side] = {
                "available": False,
                "reason": domain.get("reason", f"{side} elbow unavailable"),
                "pass": False,
            }
            failures.append(f"elbow/{side}")
            continue
        humerus_ulna = _gap_pair(
            reference=reference,
            posed=posed,
            first=domain["humerus_ids"],
            second=domain["ulna_ids"],
            limits=limits,
        )
        humerus_radius = _gap_pair(
            reference=reference,
            posed=posed,
            first=domain["humerus_ids"],
            second=domain["radius_ids"],
            limits=limits,
        )
        rigidity = {
            domain["mesh_names"]["humerus"]: rigid_edge_metrics_v7(
                reference_vertices=reference,
                final_vertices=posed,
                faces=faces,
                indices=domain["humerus_all"],
            ),
            domain["mesh_names"]["ulna"]: rigid_edge_metrics_v7(
                reference_vertices=reference,
                final_vertices=posed,
                faces=faces,
                indices=domain["ulna_all"],
            ),
            domain["mesh_names"]["radius"]: rigid_edge_metrics_v7(
                reference_vertices=reference,
                final_vertices=posed,
                faces=faces,
                indices=domain["radius_all"],
            ),
        }
        side_failures: list[str] = []
        if not humerus_ulna["pass"]:
            side_failures.append("humerus_ulna")
        if not humerus_radius["pass"]:
            side_failures.append("humerus_radius")
        for name, item in rigidity.items():
            if not item.get("available", False) or not item.get("pass", False):
                side_failures.append(f"rigidity/{name}")
        passed = not side_failures
        if not passed:
            failures.append(f"elbow/{side}")
        sides[side] = {
            "available": True,
            "domain": {
                "humerus_ids": np.asarray(domain["humerus_ids"], dtype=np.int64).tolist(),
                "ulna_ids": np.asarray(domain["ulna_ids"], dtype=np.int64).tolist(),
                "radius_ids": np.asarray(domain["radius_ids"], dtype=np.int64).tolist(),
                "humerus_count": int(len(domain["humerus_ids"])),
                "ulna_count": int(len(domain["ulna_ids"])),
                "radius_count": int(len(domain["radius_ids"])),
            },
            "humerus_ulna": humerus_ulna,
            "humerus_radius": humerus_radius,
            "rigidity": rigidity,
            "failures": side_failures,
            "pass": passed,
        }
    available = all(item.get("available", False) for item in sides.values())
    return {
        "left": sides["left"],
        "right": sides["right"],
        "available": available,
        "failures": failures,
        "pass": bool(available and not failures),
    }


def _rib_end_metrics(
    *,
    rib_indices: np.ndarray,
    end_ids: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    posed: np.ndarray,
    limits: CompoundGateThresholdsV7,
) -> dict[str, Any]:
    del rib_indices  # domain membership is carried by end_ids alone
    rest = _distance_summary(reference[end_ids], reference[target])
    final = _distance_summary(posed[end_ids], posed[target])
    increase = float(max(0.0, final["min_m"] - rest["min_m"]))
    attached = bool(rest["min_m"] <= limits.rib_attachment_rest_max_m)
    return {
        "vertex_ids": np.asarray(end_ids, dtype=np.int64).tolist(),
        "vertex_count": int(len(end_ids)),
        "reference_distance": rest,
        "posed_distance": final,
        "increase_m": increase,
        "attached_at_rest": attached,
        "rest_attachment_limit_m": float(limits.rib_attachment_rest_max_m),
        "gated": attached,
        "pass": bool(not attached or increase <= limits.rib_connection_increase_m),
    }


def _evaluate_ribs(
    *,
    lookup: Mapping[str, tuple[int, int, int]],
    reference: np.ndarray,
    posed: np.ndarray,
    faces: np.ndarray,
    limits: CompoundGateThresholdsV7,
    present_rib_names: Sequence[str],
) -> dict[str, Any]:
    present_vertebrae = [name for name in THORACIC_VERTEBRA_NAMES if name in lookup]
    if not present_vertebrae:
        raise ValueError(f"required mesh {THORACIC_VERTEBRA_NAMES[0]!r} is absent")
    vertebra = _union_indices(lookup, present_vertebrae, required=True)
    present_costal = [name for name in COSTAL_CARTILAGE_NAMES if name in lookup]
    if not present_costal:
        raise ValueError(f"required mesh {COSTAL_CARTILAGE_NAMES[0]!r} is absent")
    costal = _union_indices(lookup, present_costal, required=True)
    sternum = _optional_mesh_indices(lookup, STERNUM_MESH_NAME)
    if len(sternum):
        sternal_target = np.unique(np.concatenate((costal, sternum)))
        sternal_target_source = "costal_cartilage+sternum"
    else:
        # Sternum is optional; do not invent a substitute surface.
        sternal_target = costal
        sternal_target_source = "costal_cartilage_only_no_sternum"

    rigid_limits = JointContactThresholdsV7(
        rigid_q01_ratio=limits.rib_rigid_q01_ratio,
        rigid_q99_ratio=limits.rib_rigid_q99_ratio,
    )
    items: dict[str, Any] = {}
    failures: list[str] = []
    for name in present_rib_names:
        rib = _mesh_indices(lookup, name)
        vertebral_end = _closest_subset(
            rib,
            reference,
            reference[vertebra],
            fraction=0.08,
            minimum=8,
            maximum=512,
        )
        sternal_end = _closest_subset(
            rib,
            reference,
            reference[sternal_target],
            fraction=0.08,
            minimum=8,
            maximum=512,
        )
        if min(len(vertebral_end), len(sternal_end)) < 8:
            item = {
                "available": False,
                "reason": f"{name} connection end needs at least eight vertices",
                "pass": False,
            }
            items[name] = item
            failures.append(name)
            continue
        vertebral = _rib_end_metrics(
            rib_indices=rib,
            end_ids=vertebral_end,
            target=vertebra,
            reference=reference,
            posed=posed,
            limits=limits,
        )
        sternal = _rib_end_metrics(
            rib_indices=rib,
            end_ids=sternal_end,
            target=sternal_target,
            reference=reference,
            posed=posed,
            limits=limits,
        )
        rigidity = rigid_edge_metrics_v7(
            reference_vertices=reference,
            final_vertices=posed,
            faces=faces,
            indices=rib,
            thresholds=rigid_limits,
        )
        rib_failures: list[str] = []
        if not vertebral["pass"]:
            rib_failures.append("vertebral_end")
        if not sternal["pass"]:
            rib_failures.append("sternal_end")
        if not rigidity.get("available", False) or not rigidity.get("pass", False):
            rib_failures.append("rigidity")
        passed = not rib_failures
        if not passed:
            failures.append(name)
        items[name] = {
            "available": True,
            "sternal_target_source": sternal_target_source,
            "vertebral_end": vertebral,
            "sternal_end": sternal,
            "rigidity": rigidity,
            "failures": rib_failures,
            "pass": passed,
        }
    available = bool(items) and all(
        item.get("available", False) for item in items.values()
    )
    ungated_sternal = sorted(
        name
        for name, item in items.items()
        if item.get("available", False) and not item["sternal_end"]["attached_at_rest"]
    )
    ungated_vertebral = sorted(
        name
        for name, item in items.items()
        if item.get("available", False) and not item["vertebral_end"]["attached_at_rest"]
    )
    return {
        "items": items,
        "sternal_target_source": sternal_target_source,
        "ungated_sternal_ends": ungated_sternal,
        "ungated_vertebral_ends": ungated_vertebral,
        "available": available,
        "failures": failures,
        "pass": bool(available and not failures),
    }


def _intracranial_organ_names(
    lookup: Mapping[str, tuple[int, int, int]],
    tissues: Sequence[str] | None,
    mesh_names: Sequence[str],
) -> list[str]:
    known = set(INTRACRANIAL_ORGAN_NAMES)
    selected: list[str] = []
    for name in mesh_names:
        if name not in known or name not in lookup:
            continue
        index, _start, _stop = lookup[name]
        tissue = (
            str(tissues[index]).lower()
            if tissues is not None and index < len(tissues)
            else ""
        )
        if tissue and tissue != "organ":
            continue
        selected.append(name)
    return selected


def _evaluate_skull_brain(
    *,
    lookup: Mapping[str, tuple[int, int, int]],
    asset: Any,
    reference: np.ndarray,
    posed: np.ndarray,
    faces: np.ndarray,
    limits: CompoundGateThresholdsV7,
) -> dict[str, Any]:
    start, stop = _require_mesh_range(lookup, UPPER_SKULL_MESH_NAME)
    organ_names = _intracranial_organ_names(
        lookup,
        asset.source_tissues,
        list(asset.source_mesh_names),
    )
    if not organ_names:
        return {
            "available": False,
            "reason": "no intracranial organ meshes are present",
            "pass": False,
        }
    try:
        ref_skull_v, ref_skull_f = _mesh_local_surface(
            vertices=reference, faces=faces, start=start, stop=stop
        )
        posed_skull_v, posed_skull_f = _mesh_local_surface(
            vertices=posed, faces=faces, start=start, stop=stop
        )
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "pass": False}

    structures: dict[str, Any] = {}
    worst_name = ""
    worst_posed_outside = -1.0
    all_ref_signed: list[np.ndarray] = []
    all_posed_signed: list[np.ndarray] = []
    for name in organ_names:
        organ = _mesh_indices(lookup, name)
        ref_signed, _, _ = signed_distance(
            reference[organ], ref_skull_v, ref_skull_f
        )
        posed_signed, _, _ = signed_distance(
            posed[organ], posed_skull_v, posed_skull_f
        )
        all_ref_signed.append(ref_signed)
        all_posed_signed.append(posed_signed)
        ref_outside = ref_signed > 0.0
        posed_outside = posed_signed > 0.0
        posed_max = float(max(0.0, float(np.max(posed_signed))))
        structures[name] = {
            "vertex_count": int(len(organ)),
            "reference_inside_ratio": float(np.mean(~ref_outside)),
            "reference_max_outside_m": float(max(0.0, float(np.max(ref_signed)))),
            "posed_inside_ratio": float(np.mean(~posed_outside)),
            "posed_max_outside_m": posed_max,
        }
        if posed_max >= worst_posed_outside:
            worst_posed_outside = posed_max
            worst_name = name

    reference_signed = np.concatenate(all_ref_signed)
    posed_signed = np.concatenate(all_posed_signed)
    reference_inside_ratio = float(np.mean(reference_signed <= 0.0))
    reference_max_outside_m = float(max(0.0, float(np.max(reference_signed))))
    posed_inside_ratio = float(np.mean(posed_signed <= 0.0))
    posed_max_outside_m = float(max(0.0, float(np.max(posed_signed))))
    added_outside_m = float(max(0.0, posed_max_outside_m - reference_max_outside_m))

    reference_fails = bool(
        reference_inside_ratio < limits.brain_inside_ratio_min
        or reference_max_outside_m > limits.brain_max_outside_m
    )
    posed_ok = bool(
        posed_inside_ratio >= limits.brain_inside_ratio_min
        and posed_max_outside_m <= limits.brain_max_outside_m
    )
    added_ok = bool(added_outside_m <= limits.brain_max_outside_m + 1.0e-9)
    # Source overhang must not be absolved, and must not be blamed on posing.
    # The measured 14.3 mm right-cerebrum protrusion is identical in the
    # pre-beta operator template, so it is an authoring fact: the gate reports
    # it as a publish blocker owned by the source, keeps the retarget's own
    # contribution (added_outside_m) as a separate verdict, and still refuses to
    # report the structure as contained.
    if reference_fails:
        reason = (
            "reference intracranial containment already fails, so absolute "
            "containment is a source-authoring blocker rather than a retarget "
            f"defect: reference_inside_ratio={reference_inside_ratio:.6f}, "
            f"reference_max_outside_m={reference_max_outside_m:.6f}, "
            f"added_outside_m={added_outside_m:.6f}"
        )
        passed = False
    else:
        reason = ""
        passed = posed_ok
        if not passed:
            reason = (
                "posed intracranial containment fails: "
                f"posed_inside_ratio={posed_inside_ratio:.6f}, "
                f"posed_max_outside_m={posed_max_outside_m:.6f}"
            )

    result = {
        "available": True,
        "skull_mesh": UPPER_SKULL_MESH_NAME,
        "structures": structures,
        "worst_structure": worst_name,
        "reference_inside_ratio": reference_inside_ratio,
        "reference_max_outside_m": reference_max_outside_m,
        "posed_inside_ratio": posed_inside_ratio,
        "posed_max_outside_m": posed_max_outside_m,
        "inside_ratio": posed_inside_ratio,
        "max_outside_m": posed_max_outside_m,
        "added_outside_m": added_outside_m,
        "added_outside_pass": added_ok,
        "source_defect": reference_fails,
        "publish_blocker": bool(reference_fails),
        "blocker_kind": "source_authoring" if reference_fails else "",
        "pass": passed,
    }
    if reason:
        result["reason"] = reason
    return result


def _search_tongue_in_npz(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=True) as data:
        if "source_mesh_names" not in data.files:
            return []
        names = [str(value) for value in np.asarray(data["source_mesh_names"]).tolist()]
    return [name for name in names if "tongue" in name.lower()]


def _search_tongue_in_rig_inspect(path: Path) -> list[str]:
    if not path.is_file():
        return []
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    matches: list[str] = []
    meshes = payload.get("meshes")
    if isinstance(meshes, list):
        for item in meshes:
            if isinstance(item, dict):
                name = str(item.get("name", ""))
            else:
                name = str(item)
            if "tongue" in name.lower():
                matches.append(name)
    elif isinstance(meshes, dict):
        for name in meshes:
            if "tongue" in str(name).lower():
                matches.append(str(name))
    return matches


def _evaluate_oral_cavity(
    *,
    lookup: Mapping[str, tuple[int, int, int]],
    asset: Any,
    reference: np.ndarray,
    posed: np.ndarray,
    faces: np.ndarray,
    limits: CompoundGateThresholdsV7,
) -> dict[str, Any]:
    searched = ["asset.source_mesh_names", *LEGAL_TONGUE_SEARCH_PATHS]
    matches: list[dict[str, str]] = []
    asset_matches = [
        name
        for name in asset.source_mesh_names
        if "tongue" in str(name).lower()
    ]
    for name in asset_matches:
        matches.append({"source": "asset.source_mesh_names", "name": str(name)})

    for relative in LEGAL_TONGUE_SEARCH_PATHS:
        path = Path(relative)
        if relative.endswith(".npz"):
            found = _search_tongue_in_npz(path)
        else:
            found = _search_tongue_in_rig_inspect(path)
        for name in found:
            matches.append({"source": relative, "name": name})

    if not matches:
        return {
            "available": True,
            "tongue_present": False,
            "searched": searched,
            "matches": [],
            "pass": False,
            "publish_blocker": True,
            "reason": (
                "source has no tongue mesh; no substitute may be fabricated"
            ),
        }

    tongue_name = str(matches[0]["name"])
    if tongue_name not in lookup:
        return {
            "available": True,
            "tongue_present": True,
            "searched": searched,
            "matches": matches,
            "pass": False,
            "publish_blocker": True,
            "reason": (
                f"tongue name {tongue_name!r} was found in a legal source list "
                "but is absent from the evaluated asset topology"
            ),
        }

    mandible_names = [
        name
        for name in asset.source_mesh_names
        if "mandible" in str(name).lower()
    ]
    maxilla_names = [
        name
        for name in asset.source_mesh_names
        if "maxilla" in str(name).lower()
    ]
    if not mandible_names:
        raise ValueError("required mesh 'Mandible' is absent")
    mandible = _union_indices(lookup, mandible_names, required=True)
    tongue = _mesh_indices(lookup, tongue_name)
    tongue_domain = _closest_subset(
        tongue,
        reference,
        reference[mandible],
        fraction=0.20,
        minimum=8,
        maximum=512,
    )
    if len(tongue_domain) < 8:
        return {
            "available": False,
            "tongue_present": True,
            "searched": searched,
            "matches": matches,
            "pass": False,
            "publish_blocker": False,
            "reason": "oral-cavity tongue domain needs at least eight vertices",
        }

    envelope_names = list(mandible_names) + list(maxilla_names)
    envelope_chunks_ref: list[np.ndarray] = []
    envelope_faces: list[np.ndarray] = []
    vertex_offset = 0
    for name in envelope_names:
        start, stop = _require_mesh_range(lookup, name)
        local_v, local_f = _mesh_local_surface(
            vertices=reference, faces=faces, start=start, stop=stop
        )
        envelope_chunks_ref.append(local_v)
        envelope_faces.append(local_f + vertex_offset)
        vertex_offset += len(local_v)
    envelope_v_ref = np.concatenate(envelope_chunks_ref, axis=0)
    envelope_f = np.concatenate(envelope_faces, axis=0)

    envelope_chunks_posed: list[np.ndarray] = []
    for name in envelope_names:
        start, stop = _require_mesh_range(lookup, name)
        local_v, _local_f = _mesh_local_surface(
            vertices=posed, faces=faces, start=start, stop=stop
        )
        envelope_chunks_posed.append(local_v)
    envelope_v_posed = np.concatenate(envelope_chunks_posed, axis=0)

    reference_metrics = _containment_metrics(
        reference[tongue_domain], envelope_v_ref, envelope_f
    )
    posed_metrics = _containment_metrics(
        posed[tongue_domain], envelope_v_posed, envelope_f
    )
    passed = bool(
        posed_metrics["inside_ratio"] >= limits.brain_inside_ratio_min
        and posed_metrics["max_outside_m"] <= limits.brain_max_outside_m
    )
    return {
        "available": True,
        "tongue_present": True,
        "searched": searched,
        "matches": matches,
        "tongue_mesh": tongue_name,
        "domain": {
            "vertex_ids": np.asarray(tongue_domain, dtype=np.int64).tolist(),
            "vertex_count": int(len(tongue_domain)),
            "envelope_meshes": envelope_names,
        },
        "reference_inside_ratio": reference_metrics["inside_ratio"],
        "reference_max_outside_m": reference_metrics["max_outside_m"],
        "posed_inside_ratio": posed_metrics["inside_ratio"],
        "posed_max_outside_m": posed_metrics["max_outside_m"],
        "pass": passed,
        "publish_blocker": False,
        "reason": (
            ""
            if passed
            else (
                "tongue is outside the mandible/maxilla envelope under brain thresholds"
            )
        ),
    }


def evaluate_compound_gates_v7(
    *,
    asset: Any,
    posed_vertices: np.ndarray,
    domains: Any | None = None,
    reference_vertices: np.ndarray | None = None,
    runtime_coefficients: Mapping[str, Any] | None = None,
    body_surface: Any | None = None,
    thresholds: CompoundGateThresholdsV7 | None = None,
) -> dict[str, Any]:
    """Evaluate elbow, rib, skull/brain, and oral-cavity compound gates.

    ``domains``, ``runtime_coefficients``, and ``body_surface`` are accepted for
    signature symmetry with the vessel gate and are unused.  Inputs are never
    mutated.
    """
    del domains, runtime_coefficients, body_surface
    limits = thresholds or CompoundGateThresholdsV7()
    reference = _vertices(
        asset.vertices_rest if reference_vertices is None else reference_vertices,
        "reference_vertices",
    )
    posed = _vertices(posed_vertices, "posed_vertices")
    if posed.shape != reference.shape:
        raise ValueError("posed_vertices must match reference vertex topology")
    faces = _faces(asset.faces, len(reference))
    lookup = _mesh_lookup(asset)

    present_ribs = [
        name for name in asset.source_mesh_names if str(name) in set(RIB_MESH_NAMES)
    ]
    if not present_ribs:
        raise ValueError(f"required mesh {RIB_MESH_NAMES[0]!r} is absent")

    elbow = _evaluate_elbow(
        lookup=lookup,
        reference=reference,
        posed=posed,
        faces=faces,
        limits=limits,
    )
    ribs = _evaluate_ribs(
        lookup=lookup,
        reference=reference,
        posed=posed,
        faces=faces,
        limits=limits,
        present_rib_names=present_ribs,
    )
    skull_brain = _evaluate_skull_brain(
        lookup=lookup,
        asset=asset,
        reference=reference,
        posed=posed,
        faces=faces,
        limits=limits,
    )
    oral_cavity = _evaluate_oral_cavity(
        lookup=lookup,
        asset=asset,
        reference=reference,
        posed=posed,
        faces=faces,
        limits=limits,
    )

    sections = {
        "elbow": elbow,
        "ribs": ribs,
        "skull_brain": skull_brain,
        "oral_cavity": oral_cavity,
    }
    # Fail-closed: any unavailable section forces overall pass=False.
    failures = [
        name
        for name, section in sections.items()
        if not section.get("available", False) or not section.get("pass", False)
    ]
    available = all(section.get("available", False) for section in sections.values())
    # Blockers owned by the source authoring are listed separately so a reviewer
    # can see that a failing gate is not a retarget regression, without the gate
    # ever reporting them as passed.
    publish_blockers = sorted(
        name for name, section in sections.items() if section.get("publish_blocker")
    )
    return {
        "schema_version": COMPOUND_GATE_SCHEMA_VERSION,
        "available": available,
        "thresholds": limits.to_dict(),
        "publish_blockers": publish_blockers,
        "elbow": {"left": elbow["left"], "right": elbow["right"]},
        "ribs": {
            "items": ribs["items"],
            "failures": ribs["failures"],
            "pass": ribs["pass"],
            "available": ribs["available"],
            "sternal_target_source": ribs["sternal_target_source"],
            "ungated_sternal_ends": ribs["ungated_sternal_ends"],
            "ungated_vertebral_ends": ribs["ungated_vertebral_ends"],
        },
        "skull_brain": skull_brain,
        "oral_cavity": oral_cavity,
        "failures": failures,
        "pass": bool(available and not failures),
    }


__all__ = [
    "COMPOUND_GATE_SCHEMA_VERSION",
    "COSTAL_CARTILAGE_NAMES",
    "INTRACRANIAL_ORGAN_NAMES",
    "RIB_MESH_NAMES",
    "STERNUM_MESH_NAME",
    "THORACIC_VERTEBRA_NAMES",
    "UPPER_SKULL_MESH_NAME",
    "CompoundGateThresholdsV7",
    "evaluate_compound_gates_v7",
]
