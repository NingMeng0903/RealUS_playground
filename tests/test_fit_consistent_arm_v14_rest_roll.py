"""Targeted contracts for the opt-in V14 arm rest-roll fit parameter."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.fit_consistent_arm_v14 import (
    ArmMap,
    CAL,
    OP,
    REST_ROLL_LIMIT_DEG,
    _contact_mesh_layout,
    _full_arm_contact_metrics,
    _frozen_full_arm_contact_specs,
    ROOT,
    _initial_parameters_from_report,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    compile_subject,
    original_shape_reference_v14,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


@pytest.fixture(scope="module")
def arm_context():
    capture = ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
    if not OP.is_dir() or not CAL.is_dir() or not capture.is_file():
        pytest.skip("frozen 142 operator, calibration, or 213328 capture is unavailable")
    operator = load_source_operator(OP)
    operator.validate()
    calibration = load_anatomical_calibration_v1(CAL, operator=operator)
    with np.load(capture, allow_pickle=False) as data:
        beta = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)
    pack = materialize_subject(operator, betas=beta, gender="male")
    shape_vertices, shape_bind, _alignment = original_shape_reference_v14(
        operator, pack.rigged_asset
    )
    geometry = replace(
        operator.template_asset,
        vertices_rest=np.asarray(shape_vertices, dtype=np.float64),
        target_rest_global=np.asarray(shape_bind, dtype=np.float64),
    )
    builder = ArmMap(
        geometry,
        calibration,
        motion_reference_bind=pack.rigged_asset.target_bind_global,
        rest_roll=True,
    )
    return {
        "operator": operator,
        "calibration": calibration,
        "beta": beta,
        "pack": pack,
        "geometry": geometry,
        "builder": builder,
    }


def test_legacy_parameter_layout_and_explicit_roll_bounds(arm_context) -> None:
    geometry = arm_context["geometry"]
    calibration = arm_context["calibration"]
    legacy = ArmMap(geometry, calibration)
    assert len(legacy.config(np.zeros(6)).rigid_maps) == 235
    assert len(legacy.config(np.zeros(9)).rigid_maps) == 235

    builder = arm_context["builder"]
    config = builder.config(np.r_[np.zeros(6), [30.0, -15.0]])
    assert config.provenance["rest_roll_enabled"] is True
    assert config.provenance["rest_roll_deg"] == [30.0, -15.0]
    assert config.provenance["rest_roll_unit"] == "degrees"
    assert config.provenance["rest_roll_bounds_deg"] == [
        -REST_ROLL_LIMIT_DEG,
        REST_ROLL_LIMIT_DEG,
    ]
    config_with_contact = builder.config(np.r_[np.zeros(9), [30.0, -15.0]])
    assert config_with_contact.provenance["forearm_cap_offset_mm"] == [0.0, 0.0, 0.0]
    assert config_with_contact.provenance["rest_roll_deg"] == [30.0, -15.0]
    with pytest.raises(ValueError, match="rest-roll|Rest-roll"):
        builder.config(np.r_[np.zeros(6), [REST_ROLL_LIMIT_DEG + 1.0, 0.0]])
    with pytest.raises(ValueError, match="6/9.*two rest-roll"):
        builder.config(np.zeros(6))


@pytest.mark.parametrize("legacy_count", (6, 9))
def test_evaluate_only_migrates_matching_legacy_report_with_zero_roll(
    tmp_path, legacy_count
) -> None:
    report = tmp_path / "legacy_report.json"
    report.write_text(
        '{"best_parameters_mm": ['
        + ", ".join(str(float(i)) for i in range(legacy_count))
        + "]}\n",
        encoding="utf-8",
    )
    loaded = _initial_parameters_from_report(
        report,
        parameter_count=legacy_count + 2,
        legacy_parameter_count=legacy_count,
        rest_roll_enabled=True,
    )
    assert loaded.shape == (legacy_count + 2,)
    np.testing.assert_array_equal(loaded[-2:], [0.0, 0.0])


def test_contact_graph_cannot_silently_switch_to_validation_domains(monkeypatch):
    from projects.genesis_ue_sync.anatomy_retarget.cli import probe_arm_rest_roll_v14 as probe
    from types import SimpleNamespace
    changed = list(probe.CONTACT_SPECS)
    key, domain, target = changed[0]
    changed[0] = (key, domain.replace('.fit', '.validation'), target)
    monkeypatch.setattr(probe, 'CONTACT_SPECS', tuple(changed))
    with pytest.raises(ValueError, match='frozen FIT'):
        _frozen_full_arm_contact_specs()
    with pytest.raises(ValueError, match='frozen FIT'):
        _full_arm_contact_metrics(np.zeros((3, 3)), SimpleNamespace(domains={}), {}, np.arange(3),
                                  specs=[(*changed[0], True)])


def test_nonzero_roll_is_shared_by_geometry_bind_and_full_hand_fk(arm_context) -> None:
    operator = arm_context["operator"]
    beta = arm_context["beta"]
    pack = arm_context["pack"]
    geometry = arm_context["geometry"]
    builder = arm_context["builder"]
    parameters = np.r_[np.zeros(6), [30.0, -15.0]]
    config = builder.config(parameters)
    rest, bind, translation_maps = builder.sample(config, builder.all_ids)
    compiled = compile_subject(
        beta,
        pack,
        replace(config, shape_reference_operator=operator),
    )

    ids = np.asarray(builder.all_ids, dtype=np.int64)
    np.testing.assert_allclose(compiled.target_rest[ids], rest, atol=3e-8, rtol=0.0)
    np.testing.assert_allclose(compiled.target_bind, bind, atol=3e-8, rtol=0.0)
    np.testing.assert_allclose(
        compiled.translation_maps, translation_maps, atol=3e-8, rtol=0.0
    )

    # The forearm roll is carried by the complete Wrist_Rotate_L subtree;
    # child globals are reconstructed from one coherent parent-local bind.
    hand = np.asarray(builder.hand, dtype=np.int64)
    assert np.max(np.abs(compiled.target_bind[hand] - builder.b0[hand])) > 1e-3
    local = compiled.target_local
    reconstructed = np.empty_like(compiled.target_bind)
    for index, parent in enumerate(np.asarray(builder.parents, dtype=np.int64)):
        reconstructed[index] = (
            local[index]
            if parent < 0
            else reconstructed[int(parent)] @ local[index]
        )
    np.testing.assert_allclose(reconstructed, compiled.target_bind, atol=3e-8, rtol=0.0)

    # A non-neutral source pose must still be evaluated through the same full
    # parent-local FK.  This catches a terminal hand world-space override,
    # which would make the rest roll look correct only at T-pose.
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[18] = np.asarray((0.23, -0.17, 0.11), dtype=np.float32)
    _posed_vertices, posed_global = compiled.apply_pose(pose, return_globals=True)
    expected_global = compiled.globals_from_source(compiled.source_globals(pose))
    np.testing.assert_allclose(posed_global, expected_global, atol=3e-8, rtol=0.0)


def test_roll_keeps_cap_shapes_and_source_topology_immutable(arm_context) -> None:
    operator = arm_context["operator"]
    beta = arm_context["beta"]
    pack = arm_context["pack"]
    geometry = arm_context["geometry"]
    calibration = arm_context["calibration"]
    builder = arm_context["builder"]
    config = builder.config(np.r_[np.zeros(6), [60.0, -60.0]])
    compiled = compile_subject(
        beta,
        pack,
        replace(config, shape_reference_operator=operator),
    )

    np.testing.assert_array_equal(
        compiled.source_asset.faces, pack.rigged_asset.faces
    )
    np.testing.assert_array_equal(
        compiled.source_asset.driver_indices, pack.rigged_asset.driver_indices
    )
    np.testing.assert_array_equal(
        compiled.source_asset.driver_weights, pack.rigged_asset.driver_weights
    )

    # These frozen cap domains are all transported by one rigid group map at
    # zero station delta. Pairwise distances catch hidden scale or shearing.
    for domain in (
        "calibration/left/shoulder/humerus.fit",
        "elbow/left/radius.fit",
        "elbow/left/ulna.fit",
    ):
        ids = np.asarray(calibration.domains[domain], dtype=np.int64)
        before = np.asarray(geometry.vertices_rest[ids], dtype=np.float64)
        after = np.asarray(compiled.target_rest[ids], dtype=np.float64)
        before_dist = np.linalg.norm(before[:, None] - before[None, :], axis=-1)
        after_dist = np.linalg.norm(after[:, None] - after[None, :], axis=-1)
        np.testing.assert_allclose(after_dist, before_dist, atol=3e-8, rtol=0.0)


def test_full_arm_graph_uses_frozen_fit_queries_and_strict_targets(arm_context) -> None:
    """The opt-in graph samples every reciprocal arm contact from one field."""

    geometry = arm_context["geometry"]
    calibration = arm_context["calibration"]
    builder = arm_context["builder"]
    specs = _frozen_full_arm_contact_specs()
    layout = _contact_mesh_layout(
        geometry, tuple(dict.fromkeys(row[2] for row in specs))
    )
    used = np.unique(
        np.concatenate(
            [
                np.asarray(builder.all_ids, dtype=np.int64),
                *[
                    np.asarray(calibration.domains[domain], dtype=np.int64)
                    for _key, domain, _target, _gap_enabled in specs
                ],
                *[
                    np.asarray(layout[target]["vertex_ids"], dtype=np.int64)
                    for target in layout
                ],
            ]
        )
    )
    lookup = np.full(len(geometry.vertices_rest), -1, dtype=np.int64)
    lookup[used] = np.arange(len(used), dtype=np.int64)
    config = builder.config(np.r_[np.zeros(6), [30.0, -15.0]])
    rest, _bind, residual = builder.sample(config, used)
    # The motion-reference frame basis is required even for an untouched
    # controller.  This is the runtime/scorer contract for translation maps.
    untouched = next(
        index
        for index in range(len(builder.names))
        if all(index not in group for group in builder.groups)
    )
    expected_residual = (
        builder.b0[untouched, :3, :3].T
        @ builder.motion_reference_bind[untouched, :3, :3]
    )
    np.testing.assert_allclose(residual[untouched], expected_residual, atol=1e-12, rtol=0.0)

    score, metrics = _full_arm_contact_metrics(
        rest, calibration, layout, lookup, specs
    )
    assert np.isfinite(score)
    assert tuple(metrics) == tuple(row[0] for row in specs)
    assert all(metric["calibration_partition"] == "fit" for metric in metrics.values())
    assert all(metric["validation_domains_used"] is False for metric in metrics.values())
    assert all(metric["query_vertices_in_used"] for metric in metrics.values())
    assert all(metric["target_vertices_in_used"] for metric in metrics.values())
    for key in ("wrist_ulna_to_hand", "wrist_hand_to_ulna"):
        assert metrics[key]["gap_penalty_enabled"] is False
        assert metrics[key]["gap_attraction_used"] is False
        assert metrics[key]["penetration_penalty_enabled"] is True
    assert all(
        metric["target_mesh"] in {"Humerus_L", "Radius_L", "Ulna_L", "Scaphoid_L", "Scapula_L"}
        for metric in metrics.values()
    )
