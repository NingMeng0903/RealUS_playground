"""Absolute posed poke-through metrics for bone meshes.

``evaluate_posed_body_containment_v10`` only gates a *relative* area-inside
regression against V7, and V7 itself pokes through the skin, so a candidate can
pass the whole ladder while a third of the femur is outside.  The absolute
penetration depth was already computed there as ``max_candidate_outside_m`` and
then dropped on the floor.

This module measures signed, area-weighted penetration per bone-mesh group so
"small poke-through is fine, large posed poke-through is not" becomes a number
a gate can act on.  It changes no geometry and no rest fit.
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .chain_containment_v1 import _signed_distance, _vertex_areas
from .chain_gates_v10 import bone_mesh_group_v10
from .terminal_pose_regression_v6 import _bone_rows


WHOLE_BODY_GROUP = "ALL_BONES"

# Blocking tier.  Pack A (31133af / 142 materialize) is the worst-case linkage
# baseline every later version claims to improve on, so "no group may exceed
# Pack A by more than this" is both meaningful and reachable.  V7 clears it;
# V7 is not literally zero-regression (it is up to +0.51 mm on humerus_R at
# pose_213712), it just stays inside this envelope.
MAX_OUTSIDE_REGRESSION_M = 0.001

# Target tier, report-only.  Nothing meets this yet — Pack A pokes 18.2 mm at
# rest — so gating on it would only produce a gate that can never pass.  It is
# recorded as ``target_met`` to give V12 a scoreboard instead.
TARGET_MAX_OUTSIDE_M = 0.015
TARGET_P95_M = 0.005

# bone_mesh_group_v10's spine pattern requires a trailing underscore
# (``c[1-7]_``, ``t[0-9]+_``, ``l[1-5]_``), so the bare vertebra meshes C3..C7,
# T1..T12, L1..L5 and every tooth fall into "other" — 54 meshes and 28,400 of
# the 95,614 bone vertices.  A poke gate that calls itself per-group cannot
# leave 30% of the skeleton in an unnamed bucket.
_V12_EXTRA_PATTERNS = (
    ("teeth", r"^(canine|central_incisor|lateral_incisor|molar|premolar)"),
    ("cervical", r"^c[1-7]$"),
    ("thoracic", r"^t([1-9]|1[0-2])$"),
    ("lumbar", r"^l[1-5]$"),
)


def bone_mesh_group_v12(name: str) -> str:
    """Group a bone mesh, refusing to leave anything unclassified.

    Delegates to :func:`bone_mesh_group_v10` so the buckets stay comparable to
    the existing V10 gate, then classifies what V10 drops into "other".
    """

    group = bone_mesh_group_v10(name)
    if not group.startswith("other"):
        return group
    lower = str(name).strip().lower()
    for key, pattern in _V12_EXTRA_PATTERNS:
        if re.match(pattern, lower):
            return key
    raise ValueError(f"bone mesh {name!r} has no v12 anatomical group")


def _group_of_vertex(asset: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (bone vertex ids, group index per vertex, group names)."""

    ids: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    names: list[str] = []
    lookup: dict[str, int] = {}
    for name, start, stop in _bone_rows(asset):
        group = bone_mesh_group_v12(name)
        if group not in lookup:
            lookup[group] = len(names)
            names.append(group)
        span = np.arange(int(start), int(stop), dtype=np.int64)
        ids.append(span)
        group_ids.append(np.full(len(span), lookup[group], dtype=np.int64))
    if not ids:
        raise ValueError("asset exposes no bone meshes")
    return np.concatenate(ids), np.concatenate(group_ids), names


def _stats(signed: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    outside = signed > 0.0
    total = float(np.sum(weights))
    if not total > 0.0:
        raise ValueError("bone group has no surface area")
    clamped = np.maximum(signed, 0.0)
    return {
        "vertex_count": int(len(signed)),
        "outside_count": int(np.count_nonzero(outside)),
        "outside_area_fraction": float(np.sum(weights[outside]) / total),
        "max_outside_m": float(max(0.0, float(np.max(signed)))),
        # p95 among the vertices that are actually outside; matches the
        # convention already used by chain_containment_v1._summary.  With a
        # handful of outside vertices this collapses onto the max, so it is
        # reported but not what the target tier judges.
        "outside_p95_m": float(
            np.quantile(signed[outside], 0.95) if np.any(outside) else 0.0
        ),
        # p95 over every vertex in the group; 0.0 unless >5% of them poke.
        "poke_p95_all_m": float(np.quantile(clamped, 0.95)),
        "area_weighted_outside_depth_m": float(np.sum(weights * clamped) / total),
    }


def absolute_poke_metrics(
    vertices: np.ndarray,
    *,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    area_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Signed, area-weighted poke-through per bone-mesh group for one pose.

    Area weights come from ``area_reference`` (default: the materialized rest
    geometry) rather than the posed vertices, so the weighting is identical
    across candidates and cannot drift with a candidate's own deformation.
    ``chain_containment_v1`` weights on rest for the same reason.
    """

    posed = np.asarray(vertices, dtype=np.float64)
    bone_ids, group_ids, group_names = _group_of_vertex(asset)
    faces = np.asarray(asset.faces, dtype=np.int64)
    reference = asset.vertices_rest if area_reference is None else area_reference
    # One exact signed-distance query for every bone vertex, then slice.
    signed = _signed_distance(posed[bone_ids], np.asarray(skin), np.asarray(skin_faces))
    areas = _vertex_areas(np.asarray(reference, dtype=np.float64), faces)[bone_ids]

    groups: dict[str, Any] = {}
    for index, name in enumerate(group_names):
        mask = group_ids == index
        groups[name] = _stats(signed[mask], areas[mask])
    groups[WHOLE_BODY_GROUP] = _stats(signed, areas)
    return groups


def compare_absolute_poke_v12(
    metrics_by_pose: Mapping[str, Mapping[str, Any]],
    *,
    reference: Mapping[str, Mapping[str, Any]],
    max_outside_regression_m: float = MAX_OUTSIDE_REGRESSION_M,
    target_max_outside_m: float = TARGET_MAX_OUTSIDE_M,
    target_p95_m: float = TARGET_P95_M,
    groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the gate to metrics that were already measured.

    ``reference`` maps pose name -> group name -> the stats block produced by
    :func:`absolute_poke_metrics`, normally Pack A (31133af).  It is required:
    without it there is nothing absolute to fail against, which is exactly the
    hole this module exists to close, so a missing reference fails closed.

    ``WHOLE_BODY_GROUP`` is measured and reported but never judged: its max is
    by construction the max of some single group, so an aggregate failure
    always duplicates a group failure that is already listed.
    """

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    target_misses: list[dict[str, Any]] = []
    for pose_name, metrics in metrics_by_pose.items():
        if pose_name not in reference:
            raise KeyError(f"missing reference metrics for pose {pose_name}")
        if groups is not None:
            selected = list(groups)
        else:
            selected = sorted(set(metrics) - {WHOLE_BODY_GROUP})
        pose_failures: list[dict[str, Any]] = []
        pose_target_misses: list[dict[str, Any]] = []
        for group in selected:
            entry = metrics[group]
            reference_entry = reference[pose_name].get(group)
            if reference_entry is None:
                raise KeyError(f"missing reference group {group} for pose {pose_name}")
            regression = entry["max_outside_m"] - float(
                reference_entry["max_outside_m"]
            )
            if regression > max_outside_regression_m:
                pose_failures.append(
                    {
                        "reason": "worse_than_reference",
                        "pose": pose_name,
                        "group": group,
                        "max_outside_m": entry["max_outside_m"],
                        "reference_max_outside_m": float(
                            reference_entry["max_outside_m"]
                        ),
                        "regression_m": float(regression),
                        "limit_m": float(max_outside_regression_m),
                    }
                )
            if (
                entry["max_outside_m"] > target_max_outside_m
                or entry["poke_p95_all_m"] > target_p95_m
            ):
                pose_target_misses.append(
                    {
                        "pose": pose_name,
                        "group": group,
                        "max_outside_m": entry["max_outside_m"],
                        "poke_p95_all_m": entry["poke_p95_all_m"],
                    }
                )
        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "target_met": len(pose_target_misses) == 0,
            "groups": dict(metrics),
            "failures": pose_failures,
            "target_misses": pose_target_misses,
        }
        failures.extend(pose_failures)
        target_misses.extend(pose_target_misses)
    return {
        "schema_version": 12,
        "artifact_kind": "AbsolutePosedPokeV12",
        "passed": len(failures) == 0,
        "target_met": len(target_misses) == 0,
        "publishable": False,
        "metric": "signed_point_to_mesh_distance_area_weighted",
        "gates": {
            "max_outside_regression_m": float(max_outside_regression_m),
            "target_max_outside_m": float(target_max_outside_m),
            "target_p95_m": float(target_p95_m),
            "aggregate_group_judged": False,
        },
        "cells": cells,
        "failures": failures,
        "target_misses": target_misses,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def evaluate_absolute_posed_poke_v12(
    posed_by_pose: Mapping[str, np.ndarray],
    *,
    asset: Any,
    skins: Mapping[str, tuple[np.ndarray, np.ndarray]],
    reference: Mapping[str, Mapping[str, Any]],
    max_outside_regression_m: float = MAX_OUTSIDE_REGRESSION_M,
    target_max_outside_m: float = TARGET_MAX_OUTSIDE_M,
    target_p95_m: float = TARGET_P95_M,
    groups: Sequence[str] | None = None,
    area_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure posed poke-through, then gate it against ``reference``."""

    metrics_by_pose: dict[str, dict[str, Any]] = {}
    for pose_name, vertices in posed_by_pose.items():
        if pose_name not in skins:
            raise KeyError(f"missing skin for pose {pose_name}")
        skin, skin_faces = skins[pose_name]
        metrics_by_pose[pose_name] = absolute_poke_metrics(
            vertices,
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            area_reference=area_reference,
        )
    return compare_absolute_poke_v12(
        metrics_by_pose,
        reference=reference,
        max_outside_regression_m=max_outside_regression_m,
        target_max_outside_m=target_max_outside_m,
        target_p95_m=target_p95_m,
        groups=groups,
    )


__all__ = [
    "MAX_OUTSIDE_REGRESSION_M",
    "TARGET_MAX_OUTSIDE_M",
    "TARGET_P95_M",
    "WHOLE_BODY_GROUP",
    "absolute_poke_metrics",
    "bone_mesh_group_v12",
    "compare_absolute_poke_v12",
    "evaluate_absolute_posed_poke_v12",
]
