"""Regression tests for Stage 2 point collection."""
from __future__ import annotations

import unittest

import numpy as np

from multicam_calib.board.apriltag_board import build_board_geometry
from multicam_calib.calib.plane_fit import fit_plane_svd
from multicam_calib.calib.world_align import _collect_ref_points
from multicam_calib.io.config import BoardConfig, load_board
from multicam_calib.io.results import ExtrinsicsSet, Intrinsics
from multicam_calib.recording.session import RecordingSession, Sample, ViewDetections


def _make_board_geom():
    try:
        return build_board_geometry(load_board())
    except Exception:
        cfg = BoardConfig(rows=2, cols=2, tag_size_m=0.04, pitch_m=0.05)
        return build_board_geometry(cfg)


class TestCollectRefPoints(unittest.TestCase):
    def test_synthetic_floor_points_are_coplanar(self) -> None:
        """Fused board pose + model corners must lie on one plane (not pixel coords)."""
        board_geom = _make_board_geom()
        tag_id = next(iter(board_geom.corners_by_tag))
        corners_px = np.array(
            [[100.0, 200.0], [140.0, 200.0], [140.0, 240.0], [100.0, 240.0]],
            dtype=np.float64,
        )
        views = {"cam1": ViewDetections(alias="cam1", tags={tag_id: corners_px})}

        # Board on horizontal plane z=0.5 m in ref; only geometry matters for collect.
        T_ref_board = np.eye(4, dtype=np.float64)
        T_ref_board[:3, 3] = [0.0, 0.0, 0.5]

        # Monkey-patch estimator to return fixed pose (avoid needing real PnP/intrinsics).
        from multicam_calib.calib import world_align as wa

        def fake_estimate(sample, board_geom, intrinsics, stage1, *, min_tags):
            return T_ref_board.copy(), 1

        orig = wa._estimate_T_ref_board_per_sample
        wa._estimate_T_ref_board_per_sample = fake_estimate
        try:
            session = RecordingSession(
                session_dir=__import__("pathlib").Path("/tmp"),
                stage="test",
                aliases=["cam1"],
                detector=None,  # type: ignore[arg-type]
                recording_cfg=__import__("multicam_calib.io.config", fromlist=["RecordingConfig"]).RecordingConfig(),
                samples=[
                    Sample(index=0, host_timestamp_ns=0, views=views),
                    Sample(index=1, host_timestamp_ns=0, views=views),
                ],
            )
            stage1 = ExtrinsicsSet(reference="cam1", poses={"cam1": np.eye(4)})
            intrinsics = {
                "cam1": Intrinsics(
                    K=np.eye(3),
                    dist=np.zeros(5),
                    image_size=(640, 480),
                    source="test",
                )
            }
            pts, _ = _collect_ref_points(
                session, board_geom, intrinsics, stage1, min_tags=1
            )
        finally:
            wa._estimate_T_ref_board_per_sample = orig

        self.assertGreaterEqual(pts.shape[0], 8)
        res = fit_plane_svd(pts)
        self.assertLess(res.residual_mm, 1.0, msg=f"RMSE {res.residual_mm} mm too large")


if __name__ == "__main__":
    unittest.main()
