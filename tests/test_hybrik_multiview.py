import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from perception.benchmark.hybrik_multiview import crop_uv_to_pixels, rotation_medoid, triangulate_root


def test_crop_inverse_uses_expanded_bbox_without_half_pixel_or_second_padding():
    actual = crop_uv_to_pixels(np.array([[[-.5,-.5,0],[0,0,0],[.5,.5,0]]]), np.array([[10,20,110,220]]))
    np.testing.assert_allclose(actual, [[[10,20],[60,120],[110,220]]])


def test_root_consensus_rejects_one_bad_view_without_gt_alignment():
    k = np.diag([900., 900., 1.])
    origins = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[.5,.5,0]])
    ps = np.array([k@np.column_stack([np.eye(3), -center]) for center in origins])
    target = np.array([.2,.3,3.])
    q = ps@np.r_[target,1.]
    uv = q[:,:2]/q[:,2:]
    uv[4] += 300
    point, mask, _ = triangulate_root(uv, ps)
    np.testing.assert_allclose(point, target, atol=1e-6)
    assert mask.tolist() == [True,True,True,True,False]
    with pytest.raises(ValueError, match='>=3'):
        triangulate_root(uv[:2], ps[:2])


def test_rotation_consensus_has_proper_so3_mean_and_reports_disagreement():
    matrices = Rotation.from_euler('z',[178,179,-179,0], degrees=True).as_matrix()
    mean, accepted = rotation_medoid(matrices)
    assert accepted and np.linalg.det(mean) == pytest.approx(1)
    assert abs(Rotation.from_matrix(mean).magnitude()-np.pi) < .03
    _, accepted = rotation_medoid(Rotation.from_euler('z',[0,0,180,180],degrees=True).as_matrix())
    assert not accepted
