"""Calibration and body protocol regressions; no devices or render engine."""
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src", ROOT / "rm75_control"):
    sys.path.insert(0, str(directory))

from scripts.build_realus_scene_from_bundle import (
    DEFAULT_SLIDER_SPEC,
    _camera_scene_entry,
    _load_slider_rail,
    _rail_base_pose,
    build_scene,
)
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import _load_camera_specs
from projects.genesis_ue_sync.tracking.calibration import calibration_from_scene_camera
from projects.genesis_ue_sync.sim_platform.sync.canonical_human_motion import amongus_human_payload_from_smplx_frame
from rm75_control.control.joint_admittance_8dof.viewer.calib_scene import BedSurfaceCalib
from bridge.adapters.ue import ue_camera_payload_from_spec
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.human_refit.placement_resolver import placement_cache_key, preferred_placement_output_path
from projects.genesis_ue_sync.cli.render.unreal import run_scene_init_zmq_ue_bridge as scene_bridge
from projects.genesis_ue_sync.integrations.ue.session import EditorSessionPaths
from projects.genesis_ue_sync.sim_platform.state.scene_init import SceneInitMessageV1


def test_generated_urdf_base_link_matches_controller_world_calibration():
    pose = _rail_base_pose(DEFAULT_SLIDER_SPEC)
    world_root = np.eye(4)
    w, x, y, z = pose["base_quat_wxyz"]
    world_root[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    world_root[:3, 3] = pose["base_pos"]
    urdf = ROOT / "rm75_control/rm75_control/control/joint_admittance_8dof/assets/RM75-6F-8dof.slider.generated.urdf"
    joints = {node.find("child").attrib["link"]: node for node in ET.parse(urdf).getroot().findall("joint")}
    chain = []
    child = "base_link"
    while child in joints:
        node = joints[child]
        origin = node.find("origin")
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        chain.append(transform)
        child = node.find("parent").attrib["link"]
    world_base = world_root.copy()
    for transform in reversed(chain):
        world_base = world_base @ transform
    calibrated = yaml.safe_load((ROOT / "camera_calibration/calibration_results/robot_world.yaml").read_text())
    np.testing.assert_allclose(world_base, calibrated["T_world_baselink_at_rail0"], atol=1e-9)
    assert abs(pose["base_pos"][2] + 0.002) < 1e-9


def test_four_virtual_camera_projections_use_full_k_and_zero_distortion():
    bundle = yaml.safe_load((ROOT / "camera_calibration/calibration_results/genesis_bundle.yaml").read_text())
    for name, source in bundle["cameras"].items():
        entry = _camera_scene_entry(name, source)
        camera = _load_camera_specs([entry])[0]
        calibration = calibration_from_scene_camera(camera)
        np.testing.assert_allclose(calibration.intrinsics, source["intrinsics"])
        np.testing.assert_allclose(calibration.world_from_camera, source["world_from_camera"], atol=1e-9)
        assert not np.any(calibration.distortion)
        ue = ue_camera_payload_from_spec(camera)
        expected_hfov = math.degrees(2 * math.atan(0.5 * camera.res[0] / calibration.intrinsics[0, 0]))
        assert ue["fov"] == pytest.approx(expected_hfov)
        assert ue["fov"] > entry["fov"]
        # Reconstruct UE's native Rotator basis independently, including the
        # signs of Pitch and Roll, and compare all axes with authored WFC.
        ue_basis = (
            Rotation.from_euler("z", ue["yaw"], degrees=True).as_matrix()
            @ Rotation.from_euler("y", -ue["pitch"], degrees=True).as_matrix()
            @ Rotation.from_euler("x", -ue["roll"], degrees=True).as_matrix()
        )
        cv_to_ue_local = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
        expected_basis = np.diag([1, -1, 1]) @ calibration.world_from_camera[:3, :3] @ cv_to_ue_local
        np.testing.assert_allclose(ue_basis, expected_basis, atol=1e-9)


def test_realus_bed_reaches_floor_and_keeps_calibrated_top():
    bundle = yaml.safe_load((ROOT / "camera_calibration/calibration_results/genesis_bundle.yaml").read_text())
    genesis_bed = BedSurfaceCalib.from_bundle(bundle)
    assert genesis_bed is not None
    slab = bundle["bed"]["support_surface"]
    calibrated_top = float(slab["pos"][2]) + 0.5 * float(slab["size"][2])
    expected_quat_xyzw = (
        0.0,
        0.0,
        math.sin(math.radians(genesis_bed.rotation_deg) / 2.0),
        math.cos(math.radians(genesis_bed.rotation_deg) / 2.0),
    )
    for name in ("realus_bed_rail_scene.yaml", "realus_bed_rail_5cam_scene.yaml"):
        scene = load_sync_scene_spec(ROOT / "configs/scenes" / name)
        bed = scene.support_surface
        assert bed.top_z == pytest.approx(calibrated_top)
        assert bed.pos[2] - 0.5 * bed.size[2] == pytest.approx(scene.environment.ground_plane_z_m)
        np.testing.assert_allclose(bed.pos, genesis_bed.center)
        np.testing.assert_allclose(bed.size, genesis_bed.size)
        np.testing.assert_allclose(bed.quat_xyzw, expected_quat_xyzw, atol=1e-12)
        np.testing.assert_allclose(bed.color, genesis_bed.color, atol=1e-12)
    generated = build_scene(
        bundle=bundle,
        slider=_load_slider_rail(DEFAULT_SLIDER_SPEC),
        rail_base=_rail_base_pose(DEFAULT_SLIDER_SPEC),
        body_name="SMPLX-neutral",
        skeletal_mesh_path="/Game/Test.SMPLX-neutral",
    )["support_surface"]
    np.testing.assert_allclose(generated["pos"], genesis_bed.center)
    np.testing.assert_allclose(generated["size"], genesis_bed.size)
    np.testing.assert_allclose(generated["quat_xyzw"], expected_quat_xyzw, atol=1e-12)
    np.testing.assert_allclose(generated["color"], genesis_bed.color, atol=1e-12)
    assert generated["pos"][2] - 0.5 * generated["size"][2] == pytest.approx(0.0)
    assert generated["pos"][2] + 0.5 * generated["size"][2] == pytest.approx(calibrated_top)


def test_capture_component_intrinsics_are_copied_into_pie():
    """UE duplicates reflected non-transient properties when entering PIE.

    These arrays are populated on the editor-side capture component from the
    calibrated scene K.  Marking them Transient makes PIE silently reset them,
    which falls back to the render target centre and loses the authored K.
    """
    header = (
        ROOT
        / "src/projects/genesis_ue_sync/integrations/ue/cpp_plugin/"
        "AmongUsRealtimeCapture/Source/AmongUsRealtimeCapture/Public/"
        "AmongUsTcpCaptureComponent.h"
    ).read_text(encoding="utf-8")
    for name in ("CameraFx", "CameraFy", "CameraCx", "CameraCy"):
        assert f"UPROPERTY()\n\tTArray<float> {name};" in header
        assert f"UPROPERTY(Transient)\n\tTArray<float> {name};" not in header


def test_ue_custom_projection_matrix_maps_cv_pixels_with_offcenter_k():
    """The plugin's row-vector UE matrix must preserve the calibrated CV K."""
    width, height = 1920.0, 1080.0
    fx, fy, cx, cy = 1325.0699, 1328.7095, 964.0668, 532.3394
    matrix = np.asarray(
        [
            [2.0 * fx / width, 0.0, 0.0, 0.0],
            [0.0, 2.0 * fy / height, 0.0, 0.0],
            [2.0 * cx / width - 1.0, 1.0 - 2.0 * cy / height, 0.0, 1.0],
            [0.0, 0.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    points = np.asarray([[0.0, 0.0, 1000.0, 1.0], [100.0, -50.0, 1000.0, 1.0]])
    clip = points @ matrix
    ndc = clip[:, :2] / clip[:, 3:4]
    pixels = np.column_stack(((ndc[:, 0] + 1.0) * width * 0.5, (1.0 - ndc[:, 1]) * height * 0.5))
    expected = np.asarray(
        [[cx, cy], [fx * 100.0 / 1000.0 + cx, -fy * (-50.0) / 1000.0 + cy]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(pixels, expected, atol=1e-9)

    cpp = (
        ROOT
        / "src/projects/genesis_ue_sync/integrations/ue/cpp_plugin/"
        "AmongUsRealtimeCapture/Source/AmongUsRealtimeCapture/Private/"
        "AmongUsTcpCaptureComponent.cpp"
    ).read_text(encoding="utf-8")
    assert "Capture->bUseCustomProjectionMatrix = true;" in cpp
    assert "2.0f * Fx / WidthF" in cpp and "1.0f - 2.0f * Cy / HeightF" in cpp


def test_resume_camera_preview_command_clears_strict_capture_identity():
    loader = (
        ROOT / "src/projects/genesis_ue_sync/cli/render/unreal/ue_common_scene_loader.py"
    ).read_text(encoding="utf-8")
    watcher = (
        ROOT / "src/projects/genesis_ue_sync/cli/render/unreal/ue_editor_session_watch.py"
    ).read_text(encoding="utf-8")
    cpp = (
        ROOT
        / "src/projects/genesis_ue_sync/integrations/ue/cpp_plugin/"
        "AmongUsRealtimeCapture/Source/AmongUsRealtimeCapture/Private/"
        "AmongUsTcpCaptureComponent.cpp"
    ).read_text(encoding="utf-8")
    assert "def resume_camera_preview()" in loader
    assert 'command_type == "resume_camera_preview"' in watcher
    assert "CaptureGroupId.Empty();" in cpp


def test_smplx_hands_and_jaw_survive_without_becoming_legacy_smpl_hands():
    pose = np.zeros((55, 3))
    pose[22] = [0.1, 0.2, 0.3]
    pose[54] = [-0.4, 0.5, 0.6]
    payload = amongus_human_payload_from_smplx_frame(
        frame_index=17, motion_fps=30, root_translation_world_m=[1, 2, 3],
        smplx_pose55=pose, betas=np.arange(10) / 100, gender="female", loop_index=2,
    )
    restored = np.asarray(payload["smplx_pose_axis_angle"]).reshape(55, 3)
    np.testing.assert_allclose(restored[[22, 54]], pose[[22, 54]])
    np.testing.assert_array_equal(np.asarray(payload["smpl_body_pose_axis_angle"])[-6:], 0)
    assert payload["motion_frame_index"] == 17 and payload["loop_index"] == 2
    assert payload["root_translation_world_m"] == [1, 2, 3]


def test_invalid_calibration_and_nonfinite_pose_fail_before_publication(tmp_path):
    with pytest.raises(FileNotFoundError):
        _rail_base_pose(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="finite"):
        amongus_human_payload_from_smplx_frame(
            frame_index=0, motion_fps=30, root_translation_world_m=[0, 0, float("nan")],
            smplx_pose55=np.zeros((55, 3)),
        )


def test_old_motion_asset_cannot_reuse_placement_from_another_bed(tmp_path):
    clip = tmp_path / "shared_amass" / "clip.npz"
    clip.parent.mkdir()
    np.savez(clip, poses=np.zeros((1, 156)))
    scene = load_sync_scene_spec(ROOT / "configs/scenes/realus_bed_rail_scene.yaml")
    key = placement_cache_key(scene, clip)
    modified = replace(scene, support_surface=replace(scene.support_surface, pos=(0, 0, 0.5)))
    assert placement_cache_key(modified, clip) != key
    target = preferred_placement_output_path(scene, clip, repo_root=tmp_path / "realus")
    assert target.is_relative_to(tmp_path / "realus" / "outputs")
    assert not target.is_relative_to(clip.parent)


def test_prepare_tpose_never_launches_animation_bake(monkeypatch, tmp_path):
    def forbidden_bake(**kwargs):
        raise AssertionError("T-pose startup attempted an offline animation bake")
    monkeypatch.setattr(scene_bridge, "_bake_visible_human_fbx", forbidden_bake)
    paths = EditorSessionPaths(tmp_path / "session")
    command, scene_path, _ = scene_bridge._enqueue_apply(
        paths, SceneInitMessageV1(scene_spec={"name": "tpose_test"}),
        scene_apply_mode="prepare", bake_output_root=tmp_path / "bake",
        force_rebuild_bake=False, apply_timeout_s=1,
    )
    assert command.command_type == "prepare_tpose_avatar"
    assert scene_path.is_file()


def test_failed_plugin_build_keeps_installed_binary_and_project(monkeypatch, tmp_path):
    import argparse
    import json
    from projects.genesis_ue_sync.cli.render.unreal import install_amongus_capture_plugin as installer
    project = tmp_path / "project" / "BE_IBL.uproject"
    installed = project.parent / "Plugins/AmongUsRealtimeCapture"
    installed.mkdir(parents=True)
    binary = installed / "old_binary.so"
    binary.write_bytes(b"working-plugin")
    project_text = json.dumps({"Plugins": [{"Name": "AmongUsRealtimeCapture", "Enabled": True}]})
    project.write_text(project_text)
    source = tmp_path / "new_source"
    source.mkdir()
    (source / "AmongUsRealtimeCapture.uplugin").write_text("{}")
    args = argparse.Namespace(uproject=project, plugin_source=source, keep_source_only=False,
                              skip_build=False, unreal_root=None, target_platforms="Linux")
    monkeypatch.setattr(installer, "parse_args", lambda: args)
    monkeypatch.setattr(installer, "_resolve_run_uat", lambda _: Path("unused"))
    def failed_build(*args):
        raise RuntimeError("intentional compiler failure")
    monkeypatch.setattr(installer, "_build_plugin_in_place", failed_build)
    with pytest.raises(RuntimeError, match="compiler failure"):
        installer.main()
    assert binary.read_bytes() == b"working-plugin"
    assert project.read_text() == project_text
