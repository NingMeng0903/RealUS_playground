"""Off-screen smoke tests for the paper-style rendering pipeline.

Skipped automatically if PyVista is unavailable. We only verify:
  * colormap has correct output shape and end colours,
  * building the robot MultiBlock yields at least one non-empty PolyData,
  * rendering the sphere-glyph capability plot at low resolution produces a
    non-blank PNG.

Style regressions (SSIM against golden PNGs) are added in PR7.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from rm75_control.tools.reachability.data_model import (  # noqa: E402
    BitmaskLayout,
    CapabilityMap,
    IcosphereToolAxisGrid,
    MapMeta,
    VoxelGrid,
)
from rm75_control.tools.reachability.data_model.capability_map import (  # noqa: E402
    d_value_from_bitmask,
    pack_bits_5dof,
)
from rm75_control.tools.reachability.kinematics.model_locked_rail import (  # noqa: E402
    DEFAULT_URDF,
)
from rm75_control.tools.reachability.viz.colormap import (  # noqa: E402
    make_vahrenkamp_irm_cmap,
    make_zacharias_d_cmap,
    sample_cmap,
)
from rm75_control.tools.reachability.viz.robot_scene import build_robot_pv  # noqa: E402
from rm75_control.tools.reachability.viz.sphere_glyphs import (  # noqa: E402
    render_reachability_index,
    render_slice,
)
from rm75_control.tools.reachability.viz.orientation_glyph import render_direction_spheres  # noqa: E402
from rm75_control.tools.reachability.viz.inversion_scene import (  # noqa: E402
    render_base_candidates,
    render_irm_ground,
)
from rm75_control.tools.reachability.inversion.prefix_solver import PrefixResult  # noqa: E402
from rm75_control.tools.reachability.inversion.trajectory import ScanTrajectory, Waypoint  # noqa: E402


def _fake_map(rng: np.random.Generator) -> CapabilityMap:
    grid = VoxelGrid(origin_m=np.array([-0.6, -0.6, -0.1]), step_m=0.06, shape=(20, 20, 15))
    orient = IcosphereToolAxisGrid.build(subdiv=2)  # 162 dirs
    ijk_all = np.stack(np.meshgrid(np.arange(20), np.arange(20), np.arange(15), indexing="ij"), axis=-1).reshape(-1, 3)
    centers = grid.center_of(ijk_all)
    r = np.linalg.norm(centers, axis=1)
    keep = (r > 0.15) & (r < 0.55)
    ijk = ijk_all[keep].astype(np.int32)
    # graded D(x): highest near a "sweet spot"
    d = 1.0 - np.clip(np.abs(np.linalg.norm(centers[keep], axis=1) - 0.35) / 0.20, 0.0, 1.0)
    n_vox = ijk.shape[0]
    bool_mat = rng.random((n_vox, orient.n)) < d[:, None]
    packed = pack_bits_5dof(bool_mat)
    d_real = d_value_from_bitmask(packed, orient.n)
    layout = BitmaskLayout(n_orient=orient.n, n_roll=0)
    return CapabilityMap(
        grid=grid, orientations=orient, roll=None, layout=layout,
        voxel_ids=ijk, bitmask=packed, d_value=d_real,
        meta=MapMeta(urdf_path=str(DEFAULT_URDF)),
    )


def test_zacharias_cmap_endpoints():
    cmap = make_zacharias_d_cmap()
    rgb = sample_cmap(cmap, n=256)
    assert rgb.shape == (256, 3)
    # red at low D (poor)
    assert rgb[0, 0] > rgb[0, 1] + 40 and rgb[0, 0] > rgb[0, 2] + 40
    # deep blue at high D (good)
    assert rgb[-1, 2] > rgb[-1, 0] + 40 and rgb[-1, 2] > rgb[-1, 1] + 40


def test_vahrenkamp_cmap_shape():
    rgb = sample_cmap(make_vahrenkamp_irm_cmap(), n=64)
    assert rgb.shape == (64, 3)


def test_build_robot_pv():
    if not DEFAULT_URDF.exists():
        pytest.skip("URDF missing")
    scene = build_robot_pv()
    assert scene.n_visuals >= 5  # base + at least a few links
    # each block should be a non-empty polydata
    for i in range(scene.n_visuals):
        pd = scene.mesh_block[i]
        assert pd.n_points > 0
        assert pd.n_cells > 0


def test_render_capability_produces_nonblank_png(tmp_path):
    if not DEFAULT_URDF.exists():
        pytest.skip("URDF missing")
    rng = np.random.default_rng(0)
    cm = _fake_map(rng)
    out = render_reachability_index(
        cm, tmp_path / "cap.png",
        robot_urdf=None,
        d_min=0.02,
        sphere_radius_m=0.014,
        size=(480, 360),
        view="iso",
    )
    assert out.exists() and out.stat().st_size > 5_000
    import imageio.v3 as iio
    img = iio.imread(out)
    assert img.shape[0] > 0 and img.shape[1] > 0
    # picture should not be uniformly white
    assert img.reshape(-1, img.shape[-1]).std(axis=0).sum() > 20


def test_render_slice_produces_png(tmp_path):
    if not DEFAULT_URDF.exists():
        pytest.skip("URDF missing")
    rng = np.random.default_rng(1)
    cm = _fake_map(rng)
    out = render_slice(cm, tmp_path / "slice.png", plane="z=0.30", size=(320, 240))
    assert out.exists() and out.stat().st_size > 3_000


def test_render_direction_spheres_produces_png(tmp_path):
    if not DEFAULT_URDF.exists():
        pytest.skip("URDF missing")
    rng = np.random.default_rng(2)
    cm = _fake_map(rng)
    out = render_direction_spheres(
        cm, tmp_path / "dirs.png", stride=8, d_min=0.05,
        size=(400, 300), max_voxels=12, robot_urdf=None,
    )
    assert out.exists() and out.stat().st_size > 2_000


def test_render_irm_and_placement_png(tmp_path):
    if not DEFAULT_URDF.exists():
        pytest.skip("URDF missing")
    rng = np.random.default_rng(3)
    cm = _fake_map(rng)
    wps = [
        Waypoint(p_world=np.array([0.35, 0.0, 0.35]), tool_axis_world=np.array([0.0, 0.0, -1.0])),
        Waypoint(p_world=np.array([0.40, 0.05, 0.40]), tool_axis_world=np.array([0.0, 0.0, -1.0])),
    ]
    traj = ScanTrajectory(wps)
    irm_out = render_irm_ground(cm, traj, tmp_path / "irm.png", yb_range=(-0.2, 0.2), yb_step=0.1, size=(600, 240))
    assert irm_out.exists() and irm_out.stat().st_size > 1_500
    pref = PrefixResult(
        feasible=True, y_b_best=0.0, last_wp_index=1, arc_len_m=0.1, score=1.0,
        rail_y=0.0, rail_y_series=[0.0, 0.0], relaxed=False, strict_last_wp_index=1,
    )
    pl_out = render_base_candidates(cm, traj, tmp_path / "placement.png", result=pref, yb_range=(-0.2, 0.2), yb_step=0.1, size=(480, 360))
    assert pl_out.exists() and pl_out.stat().st_size > 1_500
