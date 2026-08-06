"""V10-aware gate wrappers (joint-anchored FK posing)."""

from __future__ import annotations

import re
import time
from typing import Any, Mapping

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _knee_gap_contact_violation_m
from .dynamic_main_chain_validation_v5 import _area_inside_fraction, _tissue_ranges
from .joint_contact_nonregress_v9 import (
    MAX_CONTACT_VIOLATION_M,
    MAX_FLEX_CONTACT_ABS_M,
    MAX_FLEX_GAP_REGRESSION_M,
    MAX_GAP_REGRESSION_M,
)
from .joint_contact_v7 import FrozenJointMaterialDomainsV7
from .knee_pose_containment_v7 import KNEE_MESH_TOKENS
from .pose_map_v1 import PoseMapV1, pose_whole_chain_vertices
from .pose_map_v10 import pose_whole_chain_vertices_v10
from .smplx_body_surface_v7 import smplx_body_surface_v7
from .terminal_pose_regression_v6 import (
    HAND_FOOT_COLLAPSE_BASELINE_MIN,
    HAND_FOOT_COLLAPSE_CANDIDATE_MAX,
    HAND_FOOT_MEAN_REGRESSION_MAX,
    _bone_rows,
    _is_hand_or_foot,
    _pose_142_vertices,
    area_inside_fraction,
)


MAX_KNEE_OUTSIDE_REGRESSION_M = 0.002
MAX_GROUP_AREA_INSIDE_REGRESSION = 0.02


def _rebased_terminal_baseline_vertices(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    pose_axis_angle: Any,
    candidate_global: np.ndarray,
) -> np.ndarray:
    """142 posed verts, rigidly rebased by each terminal root's V10 correction.

    Report-only: this is a self-referential baseline (same transform as V10
    terminal rebase) and cannot catch absolute hand/foot containment failure.
    """

    from .anatomy_lbs import source_bone_posed_global
    from .pose_map_v10 import FOOT_ROOTS, HAND_ROOTS
    from .whole_chain_rest_fit_v1 import _descendants

    baseline = np.asarray(_pose_142_vertices(value, asset, pose_axis_angle), dtype=np.float64)
    names = [str(n) for n in pose_map.bone_names.tolist()]
    parents = np.asarray(pose_map.bone_parents, dtype=np.int64)
    source_global = np.asarray(
        source_bone_posed_global(asset, pose_axis_angle), dtype=np.float64
    )
    cand_global = np.asarray(candidate_global, dtype=np.float64)
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = [str(t).strip().lower() for t in asset.source_tissues]
    result = baseline.copy()
    for root_name in (*HAND_ROOTS, *FOOT_ROOTS):
        root = names.index(root_name)
        members = _descendants(parents, root)
        rebase = cand_global[root] @ np.linalg.inv(source_global[root])
        R = rebase[:3, :3]
        t = rebase[:3, 3]
        for mesh, (controller, (start, stop), tissue) in enumerate(
            zip(controllers.tolist(), ranges.tolist(), tissues)
        ):
            del mesh
            if tissue != "bone" or int(controller) not in members:
                continue
            pts = result[int(start) : int(stop)]
            result[int(start) : int(stop)] = pts @ R.T + t
    return result


def evaluate_terminal_pose_regression_v10(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
    segment_scales: np.ndarray | None = None,
) -> dict[str, Any]:
    """Hard-gate hand/foot containment vs absolute 142 posed geometry.

    The old rebased-by-wrist baseline is retained as report-only; it is
    mathematically an identity under V10 terminal rebase and cannot fail.
    """

    cells: dict[str, Any] = {}
    hard_failures: list[dict[str, Any]] = []
    for pose_name, pose in poses.items():
        candidate, cand_global = pose_whole_chain_vertices_v10(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose,
            segment_scales=segment_scales,
        )
        baseline_142 = np.asarray(
            _pose_142_vertices(value, asset, pose), dtype=np.float64
        )
        baseline_rebased = _rebased_terminal_baseline_vertices(
            value,
            pose_map,
            asset=asset,
            pose_axis_angle=pose,
            candidate_global=cand_global,
        )
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose
        )
        hand_foot_rows: list[dict[str, Any]] = []
        collapse_failures: list[dict[str, Any]] = []
        for name, start, stop in _bone_rows(asset):
            if not _is_hand_or_foot(name):
                continue
            cand_area, cand_out = area_inside_fraction(
                candidate, asset.faces, skin, skin_faces, start, stop
            )
            base_area, base_out = area_inside_fraction(
                baseline_142, asset.faces, skin, skin_faces, start, stop
            )
            reb_area, reb_out = area_inside_fraction(
                baseline_rebased, asset.faces, skin, skin_faces, start, stop
            )
            geom_vs_142 = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                np.asarray(candidate[start:stop], dtype=np.float64)
                                - baseline_142[start:stop]
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            )
            geom_vs_reb = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                np.asarray(candidate[start:stop], dtype=np.float64)
                                - baseline_rebased[start:stop]
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            )
            entry = {
                "mesh_name": name,
                "candidate_area_inside": cand_area,
                "baseline_142_area_inside": base_area,
                "baseline_rebased_142_area_inside": reb_area,
                "area_inside_delta": cand_area - base_area,
                "area_inside_delta_vs_rebased": cand_area - reb_area,
                "candidate_max_outside_m": cand_out,
                "baseline_142_max_outside_m": base_out,
                "baseline_rebased_142_max_outside_m": reb_out,
                "geometry_rms_vs_142_m": geom_vs_142,
                "geometry_rms_vs_rebased_m": geom_vs_reb,
            }
            hand_foot_rows.append(entry)
            if (
                base_area > HAND_FOOT_COLLAPSE_BASELINE_MIN
                and cand_area < HAND_FOOT_COLLAPSE_CANDIDATE_MAX
            ):
                collapse_failures.append(entry)
        mean_delta = float(
            np.mean([row["area_inside_delta"] for row in hand_foot_rows])
        ) if hand_foot_rows else 0.0
        mean_cand = float(
            np.mean([row["candidate_area_inside"] for row in hand_foot_rows])
        ) if hand_foot_rows else 0.0
        mean_base = float(
            np.mean([row["baseline_142_area_inside"] for row in hand_foot_rows])
        ) if hand_foot_rows else 0.0
        mean_delta_reb = float(
            np.mean([row["area_inside_delta_vs_rebased"] for row in hand_foot_rows])
        ) if hand_foot_rows else 0.0
        mean_geom_142 = float(
            np.mean([row["geometry_rms_vs_142_m"] for row in hand_foot_rows])
        ) if hand_foot_rows else 0.0
        mean_ok = mean_delta >= -HAND_FOOT_MEAN_REGRESSION_MAX
        collapse_ok = len(collapse_failures) == 0
        passed = mean_ok and collapse_ok
        if not passed:
            hard_failures.append(
                {
                    "pose": pose_name,
                    "mean_delta": mean_delta,
                    "mean_candidate": mean_cand,
                    "mean_baseline_142": mean_base,
                    "n_collapse": len(collapse_failures),
                }
            )
        cells[pose_name] = {
            "passed": passed,
            "hand_foot_mean_delta": mean_delta,
            "hand_foot_mean_candidate": mean_cand,
            "hand_foot_mean_baseline_142": mean_base,
            "hand_foot_mean_regression_ok": mean_ok,
            "hand_foot_mean_geometry_rms_vs_142_m": mean_geom_142,
            "hand_foot_mean_delta_vs_rebased": mean_delta_reb,
            "collapse_ok": collapse_ok,
            "n_collapse": len(collapse_failures),
            "collapse_failures": collapse_failures,
            "hand_foot_rows": hand_foot_rows,
            "baseline_policy": "absolute_142",
            "rebased_baseline_role": "report_only",
        }
    return {
        "passed": len(hard_failures) == 0,
        "publishable": False,
        "pose_composition": "joint_anchored_fk_v10",
        "baseline_policy": "absolute_142",
        "gates": {
            "hand_foot_mean_regression_max": float(HAND_FOOT_MEAN_REGRESSION_MAX),
            "hand_foot_collapse_baseline_min": float(HAND_FOOT_COLLAPSE_BASELINE_MIN),
            "hand_foot_collapse_candidate_max": float(HAND_FOOT_COLLAPSE_CANDIDATE_MAX),
        },
        "cells": cells,
        "failures": hard_failures,
    }


def bone_mesh_group_v10(name: str) -> str:
    """Group a bone mesh name into a containment bucket (side-aware)."""

    side = ""
    if name.endswith("_L"):
        side = "_L"
    elif name.endswith("_R"):
        side = "_R"
    lower = name.lower()
    patterns = (
        ("hand", r"metacarp|phalanges_hand|carpal|scaphoid|lunate|triquet|pisiform|hamate|capitate|trapez"),
        ("foot", r"metatars|phalanx_foot|calcaneus|talus|cuneiform|navicular|cuboid|sesamoid"),
        ("forearm", r"radius|ulna"),
        ("humerus", r"humerus"),
        ("scapula", r"scapula|clavicle"),
        ("femur", r"femur"),
        ("patella", r"patella"),
        ("shank", r"tibia|fibula"),
        ("pelvis", r"pelvi|sacrum|ilium|ischium|pubis|coccyx"),
        ("spine", r"vertebra|rib|sternum|c[1-7]_|t[0-9]+_|l[1-5]_"),
        ("skull", r"skull|mandible|maxilla|teeth|hyoid|cranium"),
    )
    for key, pat in patterns:
        if re.search(pat, lower):
            return f"{key}{side}"
    return f"other{side}"


def evaluate_posed_body_containment_v10(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
    baseline_value: ChainRestFitSubjectV1,
    baseline_pose_map: PoseMapV1,
    segment_scales: np.ndarray | None = None,
    baseline_segment_scales: np.ndarray | None = None,
    baseline_composition: str = "right_multiply_bind",
    max_group_area_inside_regression: float = MAX_GROUP_AREA_INSIDE_REGRESSION,
) -> dict[str, Any]:
    """Hard-gate: no bone-mesh group may regress vs baseline by more than the limit.

    ``baseline_composition`` defaults to V7 right-multiply.  Pass
    ``joint_anchored_fk_v10`` for a same-composition rest-fit comparison
    (V11), so FK differences are not mistaken for rest regressions.
    """

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    hard_failures: list[dict[str, Any]] = []
    bone_rows = list(_bone_rows(asset))
    for pose_name, pose in poses.items():
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        candidate, _ = pose_whole_chain_vertices_v10(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose_aa,
            segment_scales=segment_scales,
        )
        if baseline_composition == "joint_anchored_fk_v10":
            baseline, _ = pose_whole_chain_vertices_v10(
                baseline_value,
                baseline_pose_map,
                source_asset=asset,
                pose_axis_angle=pose_aa,
                segment_scales=baseline_segment_scales,
            )
        else:
            baseline, _ = pose_whole_chain_vertices(
                baseline_value,
                baseline_pose_map,
                source_asset=asset,
                pose_axis_angle=pose_aa,
            )
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose_aa
        )
        by_group: dict[str, list[dict[str, Any]]] = {}
        for name, start, stop in bone_rows:
            group = bone_mesh_group_v10(name)
            cand_area, cand_out = area_inside_fraction(
                candidate, asset.faces, skin, skin_faces, start, stop
            )
            base_area, base_out = area_inside_fraction(
                baseline, asset.faces, skin, skin_faces, start, stop
            )
            by_group.setdefault(group, []).append(
                {
                    "mesh_name": name,
                    "candidate_area_inside": float(cand_area),
                    "baseline_area_inside": float(base_area),
                    "area_inside_delta": float(cand_area - base_area),
                    "candidate_max_outside_m": float(cand_out),
                    "baseline_max_outside_m": float(base_out),
                }
            )
        group_rows: dict[str, Any] = {}
        pose_failures: list[dict[str, Any]] = []
        for group, rows in sorted(by_group.items()):
            mean_cand = float(np.mean([r["candidate_area_inside"] for r in rows]))
            mean_base = float(np.mean([r["baseline_area_inside"] for r in rows]))
            delta = mean_cand - mean_base
            max_cand_out = float(max(r["candidate_max_outside_m"] for r in rows))
            max_base_out = float(max(r["baseline_max_outside_m"] for r in rows))
            entry = {
                "n_meshes": len(rows),
                "mean_candidate_area_inside": mean_cand,
                "mean_baseline_area_inside": mean_base,
                "area_inside_delta": delta,
                "max_candidate_outside_m": max_cand_out,
                "max_baseline_outside_m": max_base_out,
                "passed": delta >= -max_group_area_inside_regression,
            }
            group_rows[group] = entry
            if not entry["passed"]:
                pose_failures.append(
                    {
                        "reason": "group_area_inside_regressed",
                        "group": group,
                        "area_inside_delta": delta,
                        "limit": float(-max_group_area_inside_regression),
                        "mean_candidate_area_inside": mean_cand,
                        "mean_baseline_area_inside": mean_base,
                    }
                )
        if pose_failures:
            hard_failures.append({"pose": pose_name, "failures": pose_failures})
        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "groups": group_rows,
            "failures": pose_failures,
        }
    return {
        "passed": len(hard_failures) == 0,
        "publishable": False,
        "pose_composition": "joint_anchored_fk_v10",
        "baseline": "chain_retarget_v7_node2_001",
        "baseline_composition": str(baseline_composition),
        "gates": {
            "max_group_area_inside_regression": float(max_group_area_inside_regression),
        },
        "cells": cells,
        "failures": hard_failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def evaluate_knee_pose_containment_v10(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    smplx_model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
    segment_scales: np.ndarray | None = None,
    baseline_value: ChainRestFitSubjectV1 | None = None,
    baseline_pose_map: PoseMapV1 | None = None,
    baseline_segment_scales: np.ndarray | None = None,
    baseline_composition: str = "right_multiply_bind",
    max_outside_regression_m: float = MAX_KNEE_OUTSIDE_REGRESSION_M,
) -> dict[str, Any]:
    """Posed femur/patella containment; hard-fail outside regression vs baseline."""

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for pose_name, pose in poses.items():
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        candidate, _ = pose_whole_chain_vertices_v10(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose_aa,
            segment_scales=segment_scales,
        )
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=value.betas, pose_axis_angle=pose_aa
        )
        cand_stats = []
        for name, start, stop in _tissue_ranges(asset, {"bone"}):
            if not any(token in name.lower() for token in KNEE_MESH_TOKENS):
                continue
            area, max_out = _area_inside_fraction(
                candidate, asset.faces, skin, skin_faces, start, stop
            )
            cand_stats.append(
                {
                    "mesh_name": name,
                    "area_inside_fraction": float(area),
                    "max_outside_m": float(max_out),
                }
            )
        baseline_stats = None
        if baseline_value is not None and baseline_pose_map is not None:
            if baseline_composition == "joint_anchored_fk_v10":
                base_verts, _ = pose_whole_chain_vertices_v10(
                    baseline_value,
                    baseline_pose_map,
                    source_asset=asset,
                    pose_axis_angle=pose_aa,
                    segment_scales=baseline_segment_scales,
                )
            else:
                base_verts, _ = pose_whole_chain_vertices(
                    baseline_value,
                    baseline_pose_map,
                    source_asset=asset,
                    pose_axis_angle=pose_aa,
                )
            baseline_stats = []
            for name, start, stop in _tissue_ranges(asset, {"bone"}):
                if not any(token in name.lower() for token in KNEE_MESH_TOKENS):
                    continue
                area, max_out = _area_inside_fraction(
                    base_verts, asset.faces, skin, skin_faces, start, stop
                )
                baseline_stats.append(
                    {
                        "mesh_name": name,
                        "area_inside_fraction": float(area),
                        "max_outside_m": float(max_out),
                    }
                )
        compared = []
        pose_failures = []
        base_by_name = {
            row["mesh_name"]: row for row in (baseline_stats or [])
        }
        for row in cand_stats:
            entry = dict(row)
            if row["mesh_name"] in base_by_name:
                base = base_by_name[row["mesh_name"]]
                entry["baseline_max_outside_m"] = base["max_outside_m"]
                entry["outside_delta_m"] = (
                    float(row["max_outside_m"]) - float(base["max_outside_m"])
                )
                entry["baseline_area_inside_fraction"] = base["area_inside_fraction"]
                entry["area_inside_delta"] = (
                    float(row["area_inside_fraction"])
                    - float(base["area_inside_fraction"])
                )
                if entry["outside_delta_m"] > max_outside_regression_m:
                    pose_failures.append(
                        {
                            "reason": "knee_outside_regressed",
                            "mesh_name": row["mesh_name"],
                            "outside_delta_m": entry["outside_delta_m"],
                            "limit_m": float(max_outside_regression_m),
                            "candidate_max_outside_m": entry["max_outside_m"],
                            "baseline_max_outside_m": entry["baseline_max_outside_m"],
                        }
                    )
            compared.append(entry)
        if pose_failures:
            failures.append({"pose": pose_name, "failures": pose_failures})
        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "meshes": compared,
            "failures": pose_failures,
            "candidate_worst_outside_m": float(
                max((row["max_outside_m"] for row in cand_stats), default=0.0)
            ),
            "baseline_worst_outside_m": float(
                max(
                    (row["max_outside_m"] for row in (baseline_stats or [])),
                    default=0.0,
                )
            ),
        }
    return {
        "passed": len(failures) == 0,
        "publishable": False,
        "pose_composition": "joint_anchored_fk_v10",
        "baseline_composition": str(baseline_composition),
        "gates": {
            "max_outside_regression_m": float(max_outside_regression_m),
        },
        "cells": cells,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def evaluate_joint_contact_nonregress_v10(
    value: ChainRestFitSubjectV1,
    *,
    domains: FrozenJointMaterialDomainsV7,
    baseline_vertices: np.ndarray,
    pose_map: PoseMapV1,
    source_asset: Any,
    flex_pose_axis_angle: np.ndarray,
    baseline_flex_vertices: np.ndarray,
    segment_scales: np.ndarray | None = None,
) -> dict[str, Any]:
    """Rest + flex knee gap non-regression using V10 posing for the candidate."""

    started = time.perf_counter()
    candidate = np.asarray(value.vertices_final, dtype=np.float64)
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    sides: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    def _side_check(
        *,
        tag: str,
        cand_verts: np.ndarray,
        base_verts: np.ndarray,
        gap_max_m: float,
        max_viol: float,
        max_reg: float,
    ) -> dict[str, Any]:
        side_failures: list[dict[str, Any]] = []
        rows = {}
        for side in ("left", "right"):
            base_viol, base_gaps = _knee_gap_contact_violation_m(
                domains=domains,
                vertices=base_verts,
                side=side,
                gap_max_m=gap_max_m,
            )
            cand_viol, cand_gaps = _knee_gap_contact_violation_m(
                domains=domains,
                vertices=cand_verts,
                side=side,
                gap_max_m=gap_max_m,
            )
            local: list[dict[str, Any]] = []
            if cand_viol > max_viol:
                local.append(
                    {
                        "reason": f"{tag}_contact_violation_too_high",
                        "candidate_violation_m": float(cand_viol),
                        "max_allowed_m": float(max_viol),
                    }
                )
            if cand_viol > base_viol + max_reg:
                local.append(
                    {
                        "reason": f"{tag}_contact_violation_regressed",
                        "candidate_violation_m": float(cand_viol),
                        "baseline_violation_m": float(base_viol),
                        "max_regression_m": float(max_reg),
                    }
                )
            gap_rows = {}
            for compartment in ("medial", "lateral"):
                base_g = float(base_gaps[compartment])
                cand_g = float(cand_gaps[compartment])
                grow = cand_g - base_g
                gap_rows[compartment] = {
                    "baseline_gap_m": base_g,
                    "candidate_gap_m": cand_g,
                    "gap_delta_m": grow,
                }
                if grow > max_reg:
                    local.append(
                        {
                            "reason": f"{tag}_knee_gap_opened",
                            "compartment": compartment,
                            "baseline_gap_m": base_g,
                            "candidate_gap_m": cand_g,
                            "gap_delta_m": grow,
                            "max_regression_m": float(max_reg),
                        }
                    )
            side_failures.extend({"side": side, **row} for row in local)
            rows[side] = {
                "passed": len(local) == 0,
                "baseline_violation_m": float(base_viol),
                "candidate_violation_m": float(cand_viol),
                "gaps": gap_rows,
                "failures": local,
            }
        return {"sides": rows, "failures": side_failures}

    rest = _side_check(
        tag="rest",
        cand_verts=candidate,
        base_verts=baseline,
        gap_max_m=0.003,
        max_viol=MAX_CONTACT_VIOLATION_M,
        max_reg=MAX_GAP_REGRESSION_M,
    )
    failures.extend(rest["failures"])
    sides["rest"] = rest["sides"]

    cand_flex, _ = pose_whole_chain_vertices_v10(
        value,
        pose_map,
        source_asset=source_asset,
        pose_axis_angle=np.asarray(flex_pose_axis_angle, dtype=np.float32),
        segment_scales=segment_scales,
    )
    flex = _side_check(
        tag="flex",
        cand_verts=np.asarray(cand_flex, dtype=np.float64),
        base_verts=np.asarray(baseline_flex_vertices, dtype=np.float64),
        gap_max_m=0.020,
        max_viol=MAX_FLEX_CONTACT_ABS_M,
        max_reg=MAX_FLEX_GAP_REGRESSION_M,
    )
    failures.extend(flex["failures"])
    sides["flex"] = flex["sides"]
    return {
        "passed": len(failures) == 0,
        "publishable": False,
        "pose_composition": "joint_anchored_fk_v10",
        "sides": sides,
        "failures": failures,
        "flex_evaluated": True,
        "gates": {
            "max_gap_regression_m": float(MAX_GAP_REGRESSION_M),
            "max_contact_violation_m": float(MAX_CONTACT_VIOLATION_M),
            "max_flex_gap_regression_m": float(MAX_FLEX_GAP_REGRESSION_M),
            "max_flex_contact_abs_m": float(MAX_FLEX_CONTACT_ABS_M),
            "baseline": "chain_retarget_v7_node2_001 rest+flex",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = [
    "MAX_GROUP_AREA_INSIDE_REGRESSION",
    "MAX_KNEE_OUTSIDE_REGRESSION_M",
    "bone_mesh_group_v10",
    "evaluate_joint_contact_nonregress_v10",
    "evaluate_knee_pose_containment_v10",
    "evaluate_posed_body_containment_v10",
    "evaluate_terminal_pose_regression_v10",
]
