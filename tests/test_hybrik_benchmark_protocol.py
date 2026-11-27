from perception.apps.run_hybrik_benchmark import frontend_rejection_reason


def test_backend_replay_preserves_failed_triangulation_even_when_points_were_saved():
    row = {"fit_ok": False, "skip_reason": "bodyhand3d_quality",
           "triangulation_npz_preserved": "rejected/triangulation.npz"}
    assert frontend_rejection_reason(row) == "bodyhand3d_quality"


def test_backend_geometry_failure_can_be_retried_with_valid_observations():
    row = {"fit_ok": False, "skip_reason": "hybrik_geometry",
           "pose2d_quality": {"ok": True}, "bodyhand3d_quality": {"ok": True}}
    assert frontend_rejection_reason(row) is None


def test_explicit_observation_quality_failure_is_preserved_without_skip_reason():
    assert frontend_rejection_reason({"pose2d_quality": {"ok": False}}) == "pose2d_quality"
