from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.terminal_containment_contract_v2 import (
    terminal_containment_contract_v2,
    terminal_containment_regions_v2,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"


@pytest.fixture(scope="module")
def asset():
    if not OPERATOR.exists():
        pytest.skip("frozen source operator is unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    return materialize_subject(
        operator, betas=np.zeros(10, dtype=np.float64), gender="male"
    ).rigged_asset


def test_regions_partition_core_and_terminal_domains(asset) -> None:
    regions = terminal_containment_regions_v2(asset)
    expected = {
        f"{side}_{region}"
        for side in ("left", "right")
        for region in (
            "lower_core",
            "upper_core",
            "hand",
            "foot_major",
            "toe_phalanges",
        )
    } | {"lower_core", "upper_core", "hand", "foot_major", "toe_phalanges"}
    assert set(regions) == expected
    assert len(regions["left_foot_major"]) == 2712
    assert len(regions["right_foot_major"]) == 2712
    assert len(regions["left_toe_phalanges"]) == 2370
    assert len(regions["right_toe_phalanges"]) == 2370
    for side in ("left", "right"):
        assert not np.intersect1d(
            regions[f"{side}_foot_major"], regions[f"{side}_toe_phalanges"]
        ).size
        assert not np.intersect1d(
            regions[f"{side}_lower_core"], regions[f"{side}_foot_major"]
        ).size
        assert not np.intersect1d(
            regions[f"{side}_upper_core"], regions[f"{side}_hand"]
        ).size


def test_contract_marks_only_toes_as_report_only(asset) -> None:
    first = terminal_containment_contract_v2(asset)
    second = terminal_containment_contract_v2(asset)
    assert first == second
    assert len(first["contract_digest"]) == 64
    assert first["baseline_142_is_report_only"] is False
    assert first["baseline_roles"]["left_lower_core"] == "gate_reference"
    assert first["thresholds"]["left_hand"]["inside_fraction_min"] == 0.98
    assert first["thresholds"]["left_foot_major"]["per_mesh_inside_fraction_min"] == 0.6
    assert first["optimizer_checker_regions_are_identical"] is True
    for label, mode in first["gate_modes"].items():
        expected_report = (
            "toe_phalanges" in label
            or label in {"lower_core", "upper_core", "hand", "foot_major", "toe_phalanges"}
        )
        assert ("report_only" in mode) is expected_report
