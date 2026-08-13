from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.joint_plausibility_v12 import (
    HINGE_AXIS_REGRESSION_LIMIT_M,
    _min_clearance,
    _point_to_axis_distance,
    compare_joint_plausibility_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.linkage_v12 import (
    evaluate_linkage_v12,
    tube_bone_offset_metrics_v12,
)


def _joint_metrics(
    *,
    hip_error_m: float = 0.003,
    perpendicular_m: float = 0.010,
    mortise_min_m: float = 0.001,
    mortise_max_m: float = 0.010,
) -> dict[str, object]:
    return {
        "hip": {
            "left_hip": {
                "available": True,
                "center_error_m": hip_error_m,
                "head_radius_m": 0.025,
            }
        },
        "hinge_axis": {
            "left_knee": {
                "station_to_axis_perpendicular_m": perpendicular_m,
                "station_along_axis_m": 0.004,
            }
        },
        "ankle_mortise": {
            "left_ankle": {
                "min_m": mortise_min_m,
                "median_m": 0.5 * (mortise_min_m + mortise_max_m),
                "max_m": mortise_max_m,
            }
        },
    }


def test_axis_distance_splits_perpendicular_from_along_axis() -> None:
    origin = np.asarray([0.0, 0.0, 0.0])
    direction = np.asarray([1.0, 0.0, 0.0])
    perpendicular, along = _point_to_axis_distance(
        np.asarray([0.5, 0.03, 0.04]), origin, direction
    )

    assert perpendicular == pytest.approx(0.05)
    assert along == pytest.approx(0.5)


def test_min_clearance_reports_the_nearest_pair() -> None:
    first = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    second = np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, 3.0]])

    summary = _min_clearance(first, second)

    assert summary["min_m"] == pytest.approx(0.2)
    assert summary["max_m"] == pytest.approx(0.8)


def test_hip_absolute_is_reported_not_blocking() -> None:
    """The 2 mm figure fails even on the raw baseline, so it cannot block."""

    reference = {"pose_a": _joint_metrics(hip_error_m=0.003)}
    report = compare_joint_plausibility_v12(
        {"pose_a": _joint_metrics(hip_error_m=0.003)}, reference=reference
    )

    assert report["passed"] is True
    assert report["target_met"] is False
    assert report["target_misses"][0]["metric"] == "hip_head_socket_concentricity"


def test_hip_regression_blocks() -> None:
    reference = {"pose_a": _joint_metrics(hip_error_m=0.003)}
    report = compare_joint_plausibility_v12(
        {"pose_a": _joint_metrics(hip_error_m=0.0045)}, reference=reference
    )

    assert report["passed"] is False
    assert [f["reason"] for f in report["failures"]] == ["hip_seating_regressed"]


@pytest.mark.parametrize(
    ("extra_m", "should_pass"),
    [(0.0025, False), (0.0015, True)],
)
def test_hinge_axis_regression_brackets_its_limit(
    extra_m: float, should_pass: bool
) -> None:
    reference = {"pose_a": _joint_metrics(perpendicular_m=0.010)}
    report = compare_joint_plausibility_v12(
        {"pose_a": _joint_metrics(perpendicular_m=0.010 + extra_m)},
        reference=reference,
    )

    assert (extra_m <= HINGE_AXIS_REGRESSION_LIMIT_M) is should_pass
    assert report["passed"] is should_pass


def test_mortise_collapse_and_liftoff_are_separate_failures() -> None:
    reference = {"pose_a": _joint_metrics(mortise_min_m=0.004, mortise_max_m=0.010)}

    collapsed = compare_joint_plausibility_v12(
        {"pose_a": _joint_metrics(mortise_min_m=0.001, mortise_max_m=0.010)},
        reference=reference,
    )
    assert [f["reason"] for f in collapsed["failures"]] == [
        "ankle_mortise_clearance_collapsed"
    ]

    lifted = compare_joint_plausibility_v12(
        {"pose_a": _joint_metrics(mortise_min_m=0.004, mortise_max_m=0.020)},
        reference=reference,
    )
    assert [f["reason"] for f in lifted["failures"]] == ["ankle_mortise_lifted_off"]


def _linkage_asset():
    """One bone and one vessel mesh sharing a controller, plus a second pair."""

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.1, 0.02, 0.0],
            [0.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.5, 0.02, 0.0],
            [0.6, 0.02, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]], dtype=np.int64
    )
    asset = SimpleNamespace(
        source_mesh_names=np.asarray(["Femur_L", "Artery_L", "Tibia_L", "Vein_L"]),
        source_tissues=np.asarray(["bone", "vessel", "bone", "vessel"]),
        source_vertex_ranges=np.asarray(
            [[0, 2], [2, 4], [4, 6], [6, 8]], dtype=np.int64
        ),
        faces=faces,
        driver_indices=np.asarray([[0]] * 4 + [[1]] * 4, dtype=np.int64),
        driver_weights=np.ones((8, 1), dtype=np.float64),
    )
    return asset, vertices


def _synthetic_counts() -> dict[str, int]:
    """The fixture has 2 tube meshes and 4 tube vertices, not the real 17/55337."""

    return {"expected_tube_mesh_count": 2, "expected_tube_vertex_count": 4}


def test_rigid_motion_conserves_the_tube_to_bone_offset() -> None:
    asset, rest = _linkage_asset()
    posed = rest.copy()
    # Translate each controller group rigidly; tubes ride along with bones.
    posed[0:4] += np.asarray([0.3, 0.0, 0.0])
    posed[4:8] += np.asarray([0.0, 0.4, 0.0])

    metrics = tube_bone_offset_metrics_v12(rest, posed, asset=asset)

    assert metrics["offset_drift_max_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["unpaired_vertex_count"] == 0


def test_a_second_transport_on_the_tubes_shows_up_as_drift() -> None:
    asset, rest = _linkage_asset()
    rigid = rest.copy()
    rigid[0:4] += np.asarray([0.3, 0.0, 0.0])
    rigid[4:8] += np.asarray([0.0, 0.4, 0.0])
    reference = {"pose_a": tube_bone_offset_metrics_v12(rest, rigid, asset=asset)}

    # Move only the vessels, straight away from the bone they ride, which is
    # the exact failure the gate exists to catch.
    tampered = rigid.copy()
    tampered[2:4] += np.asarray([0.0, 0.005, 0.0])

    metrics = {"pose_a": tube_bone_offset_metrics_v12(rest, tampered, asset=asset)}
    report = evaluate_linkage_v12(metrics, reference=reference, **_synthetic_counts())

    assert report["passed"] is False
    assert {f["reason"] for f in report["failures"]} == {"tube_bone_offset_regressed"}


def test_tube_counts_are_absolute_invariants() -> None:
    asset, rest = _linkage_asset()
    metrics = {"pose_a": tube_bone_offset_metrics_v12(rest, rest, asset=asset)}

    report = evaluate_linkage_v12(
        metrics,
        reference=metrics,
        expected_tube_mesh_count=17,
        expected_tube_vertex_count=55337,
    )

    assert report["passed"] is False
    assert {f["reason"] for f in report["failures"]} == {
        "tube_mesh_count_changed",
        "tube_vertex_count_changed",
    }
