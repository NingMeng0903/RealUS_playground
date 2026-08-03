"""V9 hard gate: rest knee/hip contact must not regress vs V7."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _knee_gap_contact_violation_m
from .joint_contact_v7 import FrozenJointMaterialDomainsV7


# Allow tiny numeric jitter; still reject visible seating regressions.
MAX_GAP_REGRESSION_M = 0.0015
MAX_CONTACT_VIOLATION_M = 0.003


def evaluate_joint_contact_nonregress_v9(
    value: ChainRestFitSubjectV1,
    *,
    domains: FrozenJointMaterialDomainsV7,
    baseline_vertices: np.ndarray,
    max_gap_regression_m: float = MAX_GAP_REGRESSION_M,
    max_contact_violation_m: float = MAX_CONTACT_VIOLATION_M,
) -> dict[str, Any]:
    """Compare condyle–plateau gaps vs a V7 rest mesh; reject seating regressions."""

    started = time.perf_counter()
    candidate = np.asarray(value.vertices_final, dtype=np.float64)
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate/baseline rest meshes must share topology")
    sides: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for side in ("left", "right"):
        base_viol, base_gaps = _knee_gap_contact_violation_m(
            domains=domains, vertices=baseline, side=side
        )
        cand_viol, cand_gaps = _knee_gap_contact_violation_m(
            domains=domains, vertices=candidate, side=side
        )
        side_failures: list[dict[str, Any]] = []
        if cand_viol > max_contact_violation_m:
            side_failures.append(
                {
                    "reason": "contact_violation_too_high",
                    "candidate_violation_m": float(cand_viol),
                    "max_allowed_m": float(max_contact_violation_m),
                }
            )
        if cand_viol > base_viol + max_gap_regression_m:
            side_failures.append(
                {
                    "reason": "contact_violation_regressed",
                    "candidate_violation_m": float(cand_viol),
                    "baseline_violation_m": float(base_viol),
                    "max_regression_m": float(max_gap_regression_m),
                }
            )
        gap_rows = {}
        for compartment in ("medial", "lateral"):
            base_g = float(base_gaps[compartment])
            cand_g = float(cand_gaps[compartment])
            # Larger gap = more open joint; regression if gap grows past budget.
            grow = cand_g - base_g
            gap_rows[compartment] = {
                "baseline_gap_m": base_g,
                "candidate_gap_m": cand_g,
                "gap_delta_m": grow,
            }
            if grow > max_gap_regression_m:
                side_failures.append(
                    {
                        "reason": "knee_gap_opened",
                        "compartment": compartment,
                        "baseline_gap_m": base_g,
                        "candidate_gap_m": cand_g,
                        "gap_delta_m": grow,
                        "max_regression_m": float(max_gap_regression_m),
                    }
                )
        failures.extend({"side": side, **row} for row in side_failures)
        sides[side] = {
            "passed": len(side_failures) == 0,
            "baseline_violation_m": float(base_viol),
            "candidate_violation_m": float(cand_viol),
            "gaps": gap_rows,
            "failures": side_failures,
        }
    passed = len(failures) == 0
    return {
        "passed": passed,
        "publishable": False,
        "sides": sides,
        "failures": failures,
        "gates": {
            "max_gap_regression_m": float(max_gap_regression_m),
            "max_contact_violation_m": float(max_contact_violation_m),
            "baseline": "chain_retarget_v7_node2_001 rest vertices",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = ["evaluate_joint_contact_nonregress_v9"]
