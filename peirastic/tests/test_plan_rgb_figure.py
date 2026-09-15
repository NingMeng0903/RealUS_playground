"""Offline projection/render for the Orbbec RGB plan figure."""
import numpy as np

from types import SimpleNamespace

from peirastic.DEMO.phathom_scanning.plan_rgb_figure import (
    _crop,
    _square_lr,
    _splat_image,
    camera_elev_azim,
    path_s,
    path_time_s,
    project_points,
    rehome_to_live,
    resample_polyline,
    tool_partial_axes,
)


def test_project_and_splat(tmp_path):
    K = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[2, 3] = 0.4
    pts = np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.04, 0.03, 0.0]])
    uv, ok, xyz = project_points(pts, K, T)
    assert ok.all()
    assert xyz[:, 2].min() > 0.3
    assert np.all(uv[:, 0] > 0) and np.all(uv[:, 1] > 0)
    image = _splat_image(uv, np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.1, 0.2, 1.0]]), (240, 320), radius=2)
    assert image.shape == (240, 320, 3)
    assert image.max() > 0.5
    x0, x1, y0, y1 = _crop(image, (uv,), pad=4)
    assert 0 <= x0 < x1 <= 320
    assert 0 <= y0 < y1 <= 240
    x0b, x1b, y0b, y1b = _crop(image, (uv,), pad=4, extra_right=20, extra_bottom=15)
    assert x1b >= x1 and y1b >= y1
    sx0, sx1, sy0, sy1 = _square_lr((10, 90, 20, 80))
    assert sx1 - sx0 == sy1 - sy0 == 60
    assert sx0 - 10 == 90 - sx1


def test_rehome_centers_on_live_hit():
    poses = np.zeros((5, 6))
    poses[:, 0] = np.linspace(-0.04, 0.04, 5)
    plan = SimpleNamespace(poses=poses, right=np.array([1.0, 0.0, 0.0]))
    hit = SimpleNamespace(
        centroid=np.array([0.3, 0.1, 0.15]),
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.array([[0.25, 0.06, 0.15], [0.35, 0.14, 0.15]]),
    )
    xyz, right, far, normal = rehome_to_live(plan, hit)
    np.testing.assert_allclose(xyz.mean(axis=0), hit.centroid, atol=1e-9)
    np.testing.assert_allclose(right, [1.0, 0.0, 0.0], atol=1e-9)
    assert normal[2] > 0.9
    assert resample_polyline(xyz, 20).shape == (20, 3)


def test_path_time_and_tool_axes():
    xyz = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]])
    t = path_time_s(xyz, speed_m_s=0.01)
    np.testing.assert_allclose(t, [0.0, 5.0, 10.0])
    np.testing.assert_allclose(path_s(xyz), [0.0, 0.5, 1.0])
    poses = np.zeros((3, 6))
    poses[:, 3] = 0.2
    plan = SimpleNamespace(poses=poses)
    x_axis, y_axis, z_axis = tool_partial_axes(plan)
    assert x_axis.shape == y_axis.shape == z_axis.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(x_axis, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(y_axis, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(z_axis, axis=1), 1.0, atol=1e-9)


def test_camera_view_looks_at_center():
    T = np.eye(4)
    T[:3, 3] = [0.4, 0.0, 0.3]
    elev, azim, R = camera_elev_azim(T, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert elev > 20
    assert abs(azim) == 90
    assert R.shape == (3, 3)
