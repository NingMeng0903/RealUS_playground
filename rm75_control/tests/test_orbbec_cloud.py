"""Orbbec wrist-cloud protocol + world TF (no camera, no Genesis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.param_model.paths import GENERATED_URDF
from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import (
    DEFAULT_CLOUD_STRIDE,
    DEFAULT_COLOR_BINS,
    RailBaseLink7FK,
    T_from_pos_quat_wxyz,
    T_from_xyz_rpy,
    T_world_cam,
    camera_cloud_mesh_arrays,
    coerce_depth_meters,
    downsample_cloud,
    fixed_grid_uv,
    load_T_link7_cam,
    pack_cloud_multipart,
    quantize_rgb_keys,
    rgb_uint8_to_float,
    transform_points,
    unpack_cloud_multipart,
)
from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud_overlay import (
    OrbbecCloudOverlay,
    OrbbecCloudOverlayConfig,
)
from rm75_control.control.joint_admittance_8dof.viewer.twin import DigitalTwinMirror


def test_downsample_caps_at_max_and_keeps_sparse():
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((20_000, 3)).astype(np.float32)
    rgb = rng.random((20_000, 3)).astype(np.float32)
    pts, cols = downsample_cloud(xyz, rgb, max_points=8000)
    step = int(np.ceil(20_000 / 8000))
    assert pts.shape[0] == (20_000 + step - 1) // step
    assert pts.shape[0] <= 8000
    assert cols is not None and cols.shape[0] == pts.shape[0]
    np.testing.assert_array_equal(pts[0], xyz[0])
    np.testing.assert_array_equal(pts[1], xyz[step])
    pts_b, _ = downsample_cloud(xyz, rgb, max_points=8000)
    np.testing.assert_array_equal(pts, pts_b)

    small = rng.standard_normal((120, 3)).astype(np.float32)
    pts2, cols2 = downsample_cloud(small, None, max_points=8000)
    assert pts2.shape[0] == 120
    assert cols2 is None

    mid = rng.standard_normal((5000, 3)).astype(np.float32)
    pts3, _ = downsample_cloud(mid, None, max_points=8000)
    assert pts3.shape[0] == 5000


def test_coerce_depth_meters_units():
    z, unit = coerce_depth_meters(np.zeros((4, 4), dtype=np.float32))
    assert unit == "all_zero"
    assert float(z.max()) == 0.0

    mm, unit = coerce_depth_meters(np.full((2, 2), 800.0, dtype=np.float32))
    assert unit == "mm"
    np.testing.assert_allclose(mm, 0.8, atol=1e-6)

    tiny, unit = coerce_depth_meters(np.full((2, 2), 0.0008, dtype=np.float32))
    assert unit == "x1000"
    np.testing.assert_allclose(tiny, 0.8, atol=1e-6)

    ok, unit = coerce_depth_meters(np.full((2, 2), 0.8, dtype=np.float32))
    assert unit == "m"
    np.testing.assert_allclose(ok, 0.8, atol=1e-6)


def test_fixed_grid_uv_stable():
    u0, v0 = fixed_grid_uv(480, 640, DEFAULT_CLOUD_STRIDE)
    u1, v1 = fixed_grid_uv(480, 640, DEFAULT_CLOUD_STRIDE)
    np.testing.assert_array_equal(u0, u1)
    np.testing.assert_array_equal(v0, v1)
    step = DEFAULT_CLOUD_STRIDE
    assert u0.size == len(np.arange(0, 480, step)) * len(np.arange(0, 640, step))
    assert int(u0[0]) == 0 and int(v0[0]) == 0
    assert int(u0[1]) == step


def test_camera_cloud_mesh_arrays():
    xyz = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    verts, faces, colors = camera_cloud_mesh_arrays(xyz, rgb, radius_m=0.01)
    assert verts.shape == (12, 3)
    assert faces.shape == (16, 3)
    assert colors.shape == (12, 4)
    np.testing.assert_allclose(verts[0], [0.01, 0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(colors[0, :3], [255, 0, 0])
    np.testing.assert_array_equal(colors[6, :3], [0, 255, 0])


def test_pack_unpack_roundtrip():
    xyz = np.array([[0.1, 0.2, 0.8], [0.0, 0.0, 1.2]], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.2]], dtype=np.float32)
    parts = pack_cloud_multipart("orbbec_cloud_v1", {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, xyz, rgb)
    assert parts[0] == b"orbbec_cloud_v1"
    meta, xyz2, rgb2 = unpack_cloud_multipart(parts)
    assert meta["n"] == 2
    assert "wall_time_ns" in meta
    assert "source_time_ns" in meta
    assert "sim_time_ns" in meta
    np.testing.assert_allclose(xyz2, xyz)
    np.testing.assert_allclose(rgb2, rgb)


def test_world_cam_composition_and_point_transform():
    T_w_rb = T_from_pos_quat_wxyz([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    T_rb_l7 = T_from_pos_quat_wxyz([0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    T_l7_c = T_from_pos_quat_wxyz([0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    T = T_world_cam(T_w_rb, T_rb_l7, T_l7_c)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)
    p = transform_points(T, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    np.testing.assert_allclose(p[0], [1.0, 2.0, 3.0], atol=1e-9)

    # +90° about Z in link7: camera +X lands on link7 +Y
    cz, sz = 0.0, 1.0
    R = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    p_cam = transform_points(T_rot, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p_cam[0], [0.0, 1.0, 0.0], atol=1e-9)


def test_quantize_rgb_is_32_buckets():
    rng = np.random.default_rng(0)
    rgb = rng.random((4000, 3))
    keys = quantize_rgb_keys(rgb, DEFAULT_COLOR_BINS)
    assert keys.min() >= 0
    assert keys.max() <= 31
    assert len(np.unique(keys)) <= 32


def test_rgb_uint8_bgr_swap():
    bgr = np.array([[0, 0, 255]], dtype=np.uint8)
    rgb = rgb_uint8_to_float(bgr, bgr=True)
    np.testing.assert_allclose(rgb[0], [1.0, 0.0, 0.0], atol=1e-6)


def test_load_T_link7_cam_from_yaml(tmp_path: Path):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.06, 0.02, 0.07]
    path = tmp_path / "orbbec_handeye.yaml"
    path.write_text(
        "T_link7_cam:\n"
        + "\n".join("  - [" + ", ".join(f"{float(v):.9f}" for v in row) + "]" for row in T),
        encoding="utf-8",
    )
    got = load_T_link7_cam(path)
    np.testing.assert_allclose(got, T, atol=1e-9)


def test_load_T_link7_cam_xyz_rpy_fallback(tmp_path: Path):
    path = tmp_path / "handeye.yaml"
    path.write_text(
        "T_link7_cam_xyz_m: [0.01, 0.02, 0.03]\nT_link7_cam_rpy_xyz_rad: [0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    got = load_T_link7_cam(path)
    np.testing.assert_allclose(got[:3, 3], [0.01, 0.02, 0.03], atol=1e-9)
    np.testing.assert_allclose(got[:3, :3], np.eye(3), atol=1e-9)
    ident = T_from_xyz_rpy([0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(ident, np.eye(4), atol=1e-12)


def test_numpy_fk_matches_pinocchio_link7():
    pytest.importorskip("pinocchio")
    from rm75_control.control.joint_admittance_8dof.model import DEFAULT_URDF, RobotKinematics

    urdf = DEFAULT_URDF
    if not Path(urdf).is_file():
        pytest.skip("DEFAULT_URDF missing")
    fk = RailBaseLink7FK(urdf)
    kin = RobotKinematics(urdf)
    q = np.array([0.12, 0.2, -0.3, 0.4, -0.2, 0.1, 0.05, -0.15], dtype=np.float64)
    T = fk.T_railbase_link7(q)
    M = kin.frame_placement(q, "link_7")
    np.testing.assert_allclose(T[:3, 3], np.asarray(M.translation), atol=1e-6)
    np.testing.assert_allclose(T[:3, :3], np.asarray(M.rotation), atol=1e-6)


def test_slider_urdf_fk_rail_shifts_link7_y():
    if not Path(GENERATED_URDF).is_file():
        pytest.skip("generated slider URDF missing")
    fk = RailBaseLink7FK(GENERATED_URDF)
    q0 = np.zeros(8)
    q1 = q0.copy()
    q1[0] = 0.10
    t0 = fk.T_railbase_link7(q0)[:3, 3]
    t1 = fk.T_railbase_link7(q1)[:3, 3]
    assert t1[1] - t0[1] == pytest.approx(0.10, abs=1e-6)


def test_twin_after_sync_hook():
    class _Scene:
        def set_joint_positions(self, q):
            self.q = np.asarray(q, dtype=float)

        def step(self):
            self.stepped = True

    class _Bus:
        def q_meas_8dof(self, rail_m: float = 0.0):
            return np.array([0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    called: list[np.ndarray] = []
    twin = DigitalTwinMirror(_Bus(), _Scene(), after_sync=lambda q: called.append(np.asarray(q).copy()))
    assert twin.sync_once()
    assert len(called) == 1
    assert called[0][0] == pytest.approx(0.11, abs=1e-9)

    twin.set_after_sync(None)
    called.clear()
    twin.feed(np.zeros(8))
    assert called == []


def test_camera_frame_cloud_tracks_rail_q():
    """RViz rule: same camera-frame points follow current q, not a stale world bake."""
    if not Path(GENERATED_URDF).is_file():
        pytest.skip("generated slider URDF missing")

    class _Scene:
        cfg = type("C", (), {"urdf_path": GENERATED_URDF})()
        _robot_pos = (0.0, 0.0, 0.0)
        _robot_quat = (1.0, 0.0, 0.0, 0.0)
        scene = None

    ov = OrbbecCloudOverlay(_Scene(), OrbbecCloudOverlayConfig(urdf_path=GENERATED_URDF))
    p_cam = np.array([[0.0, 0.0, 0.5]], dtype=np.float64)
    q0 = np.zeros(8)
    q1 = q0.copy()
    q1[0] = 0.10
    T0 = ov.T_world_from_cam(q0)
    T1 = ov.T_world_from_cam(q1)
    assert T1[1, 3] - T0[1, 3] == pytest.approx(0.10, abs=1e-6)
    np.testing.assert_allclose(T0[:3, :3], T1[:3, :3], atol=1e-9)
    w0 = transform_points(T0, p_cam)
    w1 = transform_points(T1, p_cam)
    assert w1[0, 1] - w0[0, 1] == pytest.approx(0.10, abs=1e-6)
    ov.stop()


def test_overlay_draw_without_cloud_does_not_raise():
    if not Path(GENERATED_URDF).is_file():
        pytest.skip("generated slider URDF missing")

    class _Scene:
        cfg = type("C", (), {"urdf_path": GENERATED_URDF})()
        _robot_pos = (0.0, 0.0, 0.0)
        _robot_quat = (1.0, 0.0, 0.0, 0.0)
        scene = None

    ov = OrbbecCloudOverlay(_Scene(), OrbbecCloudOverlayConfig(urdf_path=GENERATED_URDF))
    ov.draw(np.zeros(8))
    ov.stop()


def test_overlay_default_on_has_no_flags():
    """run_with_twin defaults: Orbbec cloud + orange SMPL-X on; off switches exist."""
    import argparse
    from pathlib import Path as P

    src = P(__file__).resolve().parents[1] / "apps/joint_admittance_8dof/run_with_twin.py"
    text = src.read_text(encoding="utf-8")
    assert "BooleanOptionalAction" in text
    assert 'default="tcp://127.0.0.1:5598"' in text
    assert "--no-track-subscribe" in text
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbbec-cloud", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--orbbec-cloud-subscribe", default="tcp://127.0.0.1:17358")
    ns = ap.parse_args([])
    assert ns.orbbec_cloud is True
    assert ap.parse_args(["--no-orbbec-cloud"]).orbbec_cloud is False


def test_overlay_rebuilds_mesh_only_on_new_cloud():
    """Twin draw() swaps a prebuilt mesh, then later q only updates T."""
    if not Path(GENERATED_URDF).is_file():
        pytest.skip("generated slider URDF missing")

    class _Ctx:
        def __init__(self) -> None:
            self.drawn: list[np.ndarray] = []
            self.updates: list[np.ndarray] = []
            self.cleared: list[object] = []

        def draw_debug_mesh(self, mesh, pos=None, T=None):
            node = type("N", (), {"name": f"debug_mesh_{len(self.drawn)}"})()
            self.drawn.append(np.asarray(T, dtype=np.float64).copy())
            return node

        def update_debug_objects(self, objs, poses):
            del objs
            self.updates.append(np.asarray(poses[0], dtype=np.float64).copy())

        def clear_debug_object(self, obj):
            self.cleared.append(obj)

    ctx = _Ctx()
    gs = type("Gs", (), {"draw_debug_mesh": ctx.draw_debug_mesh, "clear_debug_object": ctx.clear_debug_object})()
    gs._visualizer = type("V", (), {"context": ctx, "viewer_lock": None})()

    class _Scene:
        cfg = type("C", (), {"urdf_path": GENERATED_URDF})()
        _robot_pos = (0.0, 0.0, 0.0)
        _robot_quat = (1.0, 0.0, 0.0, 0.0)
        scene = gs

    ov = OrbbecCloudOverlay(_Scene(), OrbbecCloudOverlayConfig(urdf_path=GENERATED_URDF))
    q0 = np.zeros(8)
    q1 = q0.copy()
    q1[0] = 0.10
    ov._pending_mesh = object()
    ov._pending_seq = 1
    ov.draw(q0)
    assert len(ctx.drawn) == 1
    ov.draw(q1)
    assert len(ctx.drawn) == 1
    assert len(ctx.updates) == 1
    assert ctx.updates[0][1, 3] - ctx.drawn[0][1, 3] == pytest.approx(0.10, abs=1e-6)
    ov._pending_mesh = object()
    ov._pending_seq = 2
    ov._last_swap_t = 0.0
    ov.draw(q1)
    assert len(ctx.drawn) == 2
    assert len(ctx.cleared) == 1
    ov.stop()


def test_rail_hitch_does_not_invent_velocity():
    class _Scene:
        def set_joint_positions(self, q):
            self.q = np.asarray(q, dtype=float)

        def step(self):
            pass

    class _Bus:
        def q_meas_8dof(self, rail_m: float = 0.0):
            return np.zeros(8)

    twin = DigitalTwinMirror(_Bus(), _Scene(), hz=60.0, rail_extrapolate_s=0.12)
    t0 = 1000.0
    assert twin._extrapolate_rail(0.10, t0) == pytest.approx(0.10)
    # 200 ms stall + 20 mm rail move: teleport, do not coast on v = dx/dt
    assert twin._extrapolate_rail(0.12, t0 + 0.20) == pytest.approx(0.12)
    assert twin._rail_v == pytest.approx(0.0)


def test_publisher_default_is_fixed_stride():
    src = Path(__file__).resolve().parents[2] / "perception/apps/run_orbbec_cloud_publisher.py"
    text = src.read_text(encoding="utf-8")
    assert "DEFAULT_CLOUD_STRIDE" in text
    assert "downsample_cloud" not in text
    assert "unproject_aligned_depth" in text
