"""V9 hard gate: rest + flexed knee contact must not regress vs V7."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _knee_gap_contact_violation_m
from .joint_contact_v7 import FrozenJointMaterialDomainsV7
from .pose_map_v1 import PoseMapV1, pose_whole_chain_vertices


MAX_GAP_REGRESSION_M = 0.0015
MAX_CONTACT_VIOLATION_M = 0.003
# Flexed hinge may already be imperfect on V7 (~18mm medial); never allow V9 to
# open it further by more than this budget (the 64mm failure mode).
MAX_FLEX_GAP_REGRESSION_M = 0.005
MAX_FLEX_CONTACT_ABS_M = 0.030


def evaluate_joint_contact_nonregress_v9(
    value: ChainRestFitSubjectV1,
    *,
    domains: FrozenJointMaterialDomainsV7,
    baseline_vertices: np.ndarray,
    pose_map: PoseMapV1 | None = None,
    source_asset: Any | None = None,
    flex_pose_axis_angle: np.ndarray | None = None,
    baseline_flex_vertices: np.ndarray | None = None,
    max_gap_regression_m: float = MAX_GAP_REGRESSION_M,
    max_contact_violation_m: float = MAX_CONTACT_VIOLATION_M,
) -> dict[str, Any]:
    """Compare condyle–plateau gaps vs V7 rest (+ optional flexed pose)."""

    started = time.perf_counter()
    candidate = np.asarray(value.vertices_final, dtype=np.float64)
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate/baseline rest meshes must share topology")
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
        max_viol=max_contact_violation_m,
        max_reg=max_gap_regression_m,
    )
    failures.extend(rest["failures"])
    sides["rest"] = rest["sides"]

    flex_block: dict[str, Any] | None = None
    if (
        pose_map is not None
        and source_asset is not None
        and flex_pose_axis_angle is not None
        and baseline_flex_vertices is not None
    ):
        cand_flex, _ = pose_whole_chain_vertices(
            value,
            pose_map,
            source_asset=source_asset,
            pose_axis_angle=np.asarray(flex_pose_axis_angle, dtype=np.float32),
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
        flex_block = flex

    passed = len(failures) == 0
    return {
        "passed": passed,
        "publishable": False,
        "sides": sides,
        "failures": failures,
        "flex_evaluated": flex_block is not None,
        "gates": {
            "max_gap_regression_m": float(max_gap_regression_m),
            "max_contact_violation_m": float(max_contact_violation_m),
            "max_flex_gap_regression_m": float(MAX_FLEX_GAP_REGRESSION_M),
            "max_flex_contact_abs_m": float(MAX_FLEX_CONTACT_ABS_M),
            "baseline": "chain_retarget_v7_node2_001 rest (+flex when provided)",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = ["evaluate_joint_contact_nonregress_v9"]
