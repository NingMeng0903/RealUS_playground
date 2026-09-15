"""Offline dual-view preview from a saved detect instant."""
import json

import numpy as np

from peirastic.DEMO.phathom_scanning.preview import (
    load_preview_bundle,
    save_plan,
    write_rgb_ply,
)
from peirastic.DEMO.phathom_scanning.render_preview import main as render_main
from peirastic.DEMO.phathom_scanning.s_plan import plan_s_scan
from peirastic.tests.test_phantom_scan_patterns import _top_cloud


def test_write_and_reload_preview_bundle(tmp_path):
    cloud, normal, toward = _top_cloud()
    rgb = np.clip(0.4 + 0.2 * (cloud - cloud.min(axis=0)) / 0.3, 0, 1)
    plan = plan_s_scan(
        cloud, normal, toward,
        scan_length_m=0.14, scan_width_m=0.08, pattern="lissajous",
    )
    hit = type("Hit", (), {})()
    hit.points = cloud
    hit.normal = normal
    hit.centroid = cloud.mean(axis=0)
    hit.table_z = float(cloud[:, 2].min())
    hit.n_points = len(cloud)
    preview = save_plan(
        tmp_path, xyz_cam=cloud, rgb=rgb, q8=np.zeros(8), xyz=cloud,
        hit=hit, plan=plan, probe_width_m=0.05, capture_dir=tmp_path,
    )
    assert preview.is_file()
    assert preview.with_suffix(".pdf").is_file()
    assert (tmp_path / "detect_cloud.ply").is_file()
    assert (tmp_path / "cloud_and_plan.npz").is_file()
    xyz, colors, loaded_hit, loaded_plan, width = load_preview_bundle(tmp_path)
    np.testing.assert_allclose(xyz, cloud)
    np.testing.assert_allclose(loaded_plan.poses, plan.poses)
    assert loaded_plan.pattern == "lissajous"
    assert loaded_plan.normal_offset_deg == 0.0
    assert width == 0.05
    assert loaded_hit.n_points == len(cloud)
    assert render_main([str(tmp_path), "--output", str(tmp_path / "again.png")]) == 0
    assert (tmp_path / "again.png").is_file()


def test_preview_persists_smooth_normal_offset(tmp_path):
    cloud, normal, toward = _top_cloud()
    rgb = np.clip(0.4 + 0.2 * (cloud - cloud.min(axis=0)) / 0.3, 0, 1)
    plan = plan_s_scan(
        cloud, normal, toward,
        scan_length_m=0.14, scan_width_m=0.08, pattern="lissajous",
        normal_offset_deg=20.0,
    )
    hit = type("Hit", (), {})()
    hit.points = cloud
    hit.normal = normal
    hit.centroid = cloud.mean(axis=0)
    hit.table_z = float(cloud[:, 2].min())
    hit.n_points = len(cloud)
    save_plan(
        tmp_path, xyz_cam=cloud, rgb=rgb, q8=np.zeros(8), xyz=cloud,
        hit=hit, plan=plan, probe_width_m=0.05, capture_dir=tmp_path,
    )
    summary = json.loads((tmp_path / "plan.json").read_text())
    assert summary["normal_offset_deg"] == 20.0
    assert "plus_smooth_normal_offset" in summary["orientation_mode"]
    _, _, _, loaded, _ = load_preview_bundle(tmp_path)
    assert loaded.normal_offset_deg == 20.0
    np.testing.assert_allclose(loaded.command_normals, plan.command_normals)
    np.testing.assert_allclose(loaded.poses, plan.poses)


def test_rgb_ply_scales_unit_colors(tmp_path):
    path = write_rgb_ply(
        tmp_path / "cloud.ply",
        np.array([[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    text = path.read_text()
    assert "element vertex 2" in text
    assert "255 0 0" in text
    assert "0 255 0" in text
