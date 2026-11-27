import numpy as np
import pytest
from perception.benchmark.rgb_tracking import RGBPosePrior


class FakeRGBModel:
    def prepare(self, rgb, bbox):
        return rgb, bbox, np.array([50,50]), .001

    def forward_prepared_batch(self, crops, boxes, centers):
        assert len(crops) == 2
        assert crops[0][0,0,0] == 20  # requested camera order, not sorted dict order
        assert crops[1][0,0,0] == 10
        return {'pred_rot_mats': np.tile(np.eye(3), (2,55,1,1))}, .1


def test_rgb_tracking_preserves_camera_order_and_reports_skipped_crops():
    provider = RGBPosePrior.__new__(RGBPosePrior)
    provider.model = FakeRGBModel()
    views = {'cam1': np.full((100,100,3),10,np.uint8), 'cam2': np.full((100,100,3),20,np.uint8)}
    detector = {c: {'selected_person':0,'pose_boxes':[[0,0,99,99]],'bbox_source':'detector'} for c in views}
    hints, meta = provider.infer(views, detector, ['cam2','missing','cam1'])
    assert meta['camera_ids'] == ['cam2','cam1']
    assert meta['failed_views'][0]['camera_id'] == 'missing'
    assert hints['rgb_agreement'][[0,22,23,24]].tolist() == [0,0,0,0]
    assert meta['batch_forward_elapsed_s'] == .1
    assert hints['rgb_rotations'].shape == (55,3,3)
    with pytest.raises(ValueError,match='two valid'):
        provider.infer(views,detector,['cam1'])


def test_rgb_pelvis_fallback_uses_matching_calibration_after_a_skipped_view():
    intrinsic = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    origins = np.array([[-.4, 0., 0.], [.4, 0., 0.], [0., .4, 0.]])
    projections = np.array([intrinsic @ np.c_[np.eye(3), -origin] for origin in origins])
    target = np.array([.1, .1, 3.])
    pixels = projections @ np.r_[target, 1.]
    pixels = pixels[:, :2] / pixels[:, 2:]

    class Model:
        def prepare(self, rgb, bbox):
            return rgb, bbox, np.array([50, 50]), 0.

        def forward_prepared_batch(self, crops, boxes, centers):
            uvd = np.zeros((len(crops), 71, 3))
            for i, crop in enumerate(crops):
                source = int(crop[0, 0, 0])
                uvd[i, 0, :2] = pixels[source] / 99. - .5
            return dict(pred_uvd_jts=uvd,
                        pred_rot_mats=np.tile(np.eye(3), (len(crops), 55, 1, 1))), 0.

    provider = RGBPosePrior.__new__(RGBPosePrior)
    provider.model = Model()
    views = {f'cam{i}': np.full((100, 100, 3), i, np.uint8) for i in range(3)}
    detector = {cid: {'selected_person': 0, 'pose_boxes': [[0, 0, 99, 99]],
                     'bbox_source': 'detector'} for cid in views}
    ids = ['cam2', 'missing', 'cam0', 'cam1']
    hints, meta = provider.infer(views, detector, ids,
                                projection_matrices=projections[[2, 0, 0, 1]])
    np.testing.assert_allclose(hints['rgb_pelvis_world_m'], target, atol=1e-6)
    assert meta['root_triangulation']['used_cameras'] == ['cam2', 'cam0', 'cam1']
    # Two views can supply local pose but cannot claim an observed RGB pelvis.
    hints, meta = provider.infer(views, detector, ['cam1', 'cam0'],
                                projection_matrices=projections[[1, 0]])
    assert 'rgb_pelvis_world_m' not in hints
    assert meta['root_triangulation']['accepted'] is False
