"""Bounded contracts for the V15 rest translation-map migration."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget import translation_rebuild_v15 as rebuild
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_DIR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION_DIR = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
PACKAGE_DIRS = tuple(
    ROOT / f"outputs/anatomy_retarget/v15_bilateral_collar_{subject}_20260908_001/compiled"
    for subject in ("213328", "213712")
)


@pytest.fixture(scope="module")
def trusted_inputs():
    if not OPERATOR_DIR.is_dir() or not CALIBRATION_DIR.is_dir():
        pytest.skip("authenticated V15 operator/calibration is unavailable")
    operator = load_source_operator(OPERATOR_DIR)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION_DIR, operator=operator, require_complete=True
    )
    return operator, calibration


def _load(subject_index: int):
    if not PACKAGE_DIRS[subject_index].is_dir():
        pytest.skip("authenticated V15 compiled package is unavailable")
    return load_compiled_subject(PACKAGE_DIRS[subject_index])


def test_authenticated_motion_reference_axes_roundtrip(trusted_inputs):
    """The modern branch is a basis change, with no legacy reconstruction."""

    operator, calibration = trusted_inputs
    old = _load(0)
    modern = copy.copy(old)
    modern.provenance = dict(old.provenance)
    modern.provenance["translation_transport"] = "motion_reference_axes"

    expected = (
        np.asarray(old.target_bind)[:, :3, :3]
        @ np.asarray(old.translation_maps)
        @ np.asarray(old.reference_bind)[:, :3, :3].swapaxes(1, 2)
    )
    world, provenance = rebuild.rebuild_rest_world_jacobians_v15(
        modern, operator, calibration
    )

    assert world.shape == (235, 3, 3)
    assert np.isfinite(world).all()
    # ``_world_jacobians`` uses solves/inverses to undo the persisted bind
    # frames.  Validate the contract in the stored basis; this avoids making
    # the test depend on the order of the two equivalent matrix products.
    np.testing.assert_allclose(
        np.asarray(old.target_bind)[:, :3, :3].swapaxes(1, 2)
        @ world
        @ np.asarray(old.reference_bind)[:, :3, :3],
        np.asarray(old.translation_maps),
        atol=1.0e-12,
        rtol=0.0,
    )
    # Keep the explicit WORLD-J formula visible as a second, looser check;
    # inverse/transpose order differs by the persisted bind roundoff.
    np.testing.assert_allclose(world, expected, atol=2.0e-6, rtol=0.0)
    assert provenance["previous_translation_transport"] == "motion_reference_axes"
    assert provenance["world_jacobian_reconstructed"] is True
    assert provenance["runtime_fit"] is False


@pytest.mark.parametrize("subject_index", (0, 1))
def test_legacy_reconstruction_is_authenticated_for_both_subjects(
    trusted_inputs, subject_index
):
    operator, calibration = trusted_inputs
    old = _load(subject_index)
    world, provenance = rebuild.rebuild_rest_world_jacobians_v15(
        old, operator, calibration
    )

    assert world.shape == (235, 3, 3)
    assert np.isfinite(world).all()
    assert np.all(np.linalg.det(world) > 0.0)
    assert provenance["method"] == "reconstruct_saved_arm_rest_field_without_refitting"
    assert old.provenance["method"] == "connected_left_arm_caps_v14"
    assert old.provenance["shape_reference_kind"] == "frozen_operator_template"
    assert provenance["rest_max_abs_error_m"] <= 1.0e-7
    assert provenance["bind_max_abs_error"] <= 1.0e-7
    assert provenance["reference_max_abs_error"] <= 1.0e-7
    assert provenance["world_jacobian_reconstructed"] is True
    assert provenance["optimization_performed"] is False


def test_legacy_unknown_method_fails_closed(trusted_inputs):
    operator, calibration = trusted_inputs
    old = _load(0)
    invalid = copy.copy(old)
    invalid.provenance = dict(old.provenance)
    invalid.provenance["method"] = "unrecorded_arm_solver"
    with pytest.raises(ValueError, match="legacy|method|reconstruct"):
        rebuild.rebuild_rest_world_jacobians_v15(invalid, operator, calibration)


def test_legacy_unknown_shape_reference_fails_closed(trusted_inputs):
    operator, calibration = trusted_inputs
    old = _load(0)
    invalid = copy.copy(old)
    invalid.provenance = dict(old.provenance)
    invalid.provenance["shape_reference_kind"] = "nearest_point_runtime"
    with pytest.raises(ValueError, match="shape|reference|legacy"):
        rebuild.rebuild_rest_world_jacobians_v15(invalid, operator, calibration)


def test_undocumented_transport_fails_closed(trusted_inputs):
    operator, calibration = trusted_inputs
    old = _load(0)
    invalid = copy.copy(old)
    invalid.provenance = dict(old.provenance)
    invalid.provenance["translation_transport"] = "untracked_axes"
    with pytest.raises(ValueError, match="translation|transport|legacy"):
        rebuild.rebuild_rest_world_jacobians_v15(invalid, operator, calibration)
