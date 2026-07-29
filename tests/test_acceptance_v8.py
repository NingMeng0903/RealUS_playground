from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.acceptance_v8 import (
    FrozenValidationDomainsV8,
    bone_station_profile,
    compare_bone_station_profiles,
    fit_sphere,
    independent_joint_center_gate,
    require_available_gates,
    rigid_compound_gate,
    topology_digest,
)


def _sphere(center: tuple[float, float, float], radius: float, count: int = 80) -> np.ndarray:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    theta = np.pi * (1.0 + np.sqrt(5.0)) * index
    radial = np.sqrt(1.0 - z * z)
    unit = np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z))
    return np.asarray(center) + radius * unit


def test_frozen_domains_reject_overlap_and_topology_change() -> None:
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    digest = topology_digest(8, faces)
    domains = FrozenValidationDomainsV8(
        topology_digest=digest,
        vertex_count=8,
        domains={"fit": np.asarray([0, 1]), "validation": np.asarray([2, 3])},
        fit_validation_pairs=(("fit", "validation"),),
        provenance={"selection": "manual"},
    )
    domains.validate(faces)
    with pytest.raises(ValueError, match="topology"):
        domains.validate(faces[::-1])
    bad = FrozenValidationDomainsV8(
        topology_digest=digest,
        vertex_count=8,
        domains={"fit": np.asarray([0, 1]), "validation": np.asarray([1, 2])},
        fit_validation_pairs=(("fit", "validation"),),
        provenance={},
    )
    with pytest.raises(ValueError, match="overlap"):
        bad.validate()


def test_independent_joint_center_uses_disjoint_validation_points() -> None:
    first_fit = _sphere((0.0, 0.0, 0.0), 0.02)
    second_fit = _sphere((0.0, 0.0, 0.0), 0.02)
    first_validation = _sphere((0.0, 0.0, 0.0), 0.021)
    second_validation = _sphere((0.003, 0.0, 0.0), 0.021)
    vertices = np.vstack(
        (first_fit, second_fit, first_validation, second_validation)
    )
    starts = np.cumsum([0, len(first_fit), len(second_fit), len(first_validation)])
    ranges = {
        "a_fit": np.arange(starts[0], starts[1]),
        "b_fit": np.arange(starts[1], starts[2]),
        "a_validation": np.arange(starts[2], starts[3]),
        "b_validation": np.arange(starts[3], len(vertices)),
    }
    domains = FrozenValidationDomainsV8(
        topology_digest="0" * 64,
        vertex_count=len(vertices),
        domains=ranges,
        fit_validation_pairs=(
            ("a_fit", "a_validation"),
            ("b_fit", "b_validation"),
        ),
        provenance={},
    )
    gate = independent_joint_center_gate(
        vertices,
        domains,
        first_fit="a_fit",
        second_fit="b_fit",
        first_validation="a_validation",
        second_validation="b_validation",
    )
    assert gate["fit_center_error_m"] < 1.0e-9
    assert gate["validation_center_error_m"] == pytest.approx(0.003, abs=1.0e-7)
    assert gate["pass"] is False


def test_rigid_compound_and_radius_profile_gates() -> None:
    rng = np.random.default_rng(4)
    z = rng.uniform(-0.2, 0.2, 1200)
    angle = rng.uniform(0.0, 2.0 * np.pi, len(z))
    radius = 0.015 * (1.0 + 0.03 * z / 0.2)
    bone = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    transform = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    posed = bone @ transform.T + np.asarray([0.2, -0.1, 0.05])
    assert rigid_compound_gate(bone, posed)["pass"] is True

    reference = bone_station_profile(bone)
    candidate = bone_station_profile(bone * np.asarray([0.98, 0.98, 1.0]))
    compared = compare_bone_station_profiles(reference, candidate)
    assert compared["pass"] is True
    thin = bone_station_profile(bone * np.asarray([0.55, 0.55, 1.0]))
    assert compare_bone_station_profiles(reference, thin)["pass"] is False


def test_fail_closed_gate_conjunction() -> None:
    result = require_available_gates(
        {
            "hip": {"available": True, "pass": True},
            "elbow": {"available": False, "pass": True},
        }
    )
    assert result["available"] is False
    assert result["passed"] is False
    assert result["failures"] == ["elbow:unavailable"]
