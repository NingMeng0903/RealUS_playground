from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import with_source_driver_coupling
from projects.genesis_ue_sync.anatomy_retarget.cli.run_audit_capture_pose import (
    _region_name_matches,
    _side_matches,
    audit_capture_pose,
)
from projects.genesis_ue_sync.anatomy_retarget.obj_io import write_obj
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset, save_rigged_asset


def _source_asset() -> AnatomyRiggedAsset:
    joints = np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)), dtype=np.float32)
    parents = np.asarray((-1, 0, 1, 2), dtype=np.int32)
    global_bind = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    global_bind[:, :3, 3] = joints
    local_bind = global_bind.copy()
    for index in range(1, 4):
        local_bind[index] = np.linalg.inv(global_bind[index - 1]) @ global_bind[index]
    heads = joints.copy()
    tails = heads + np.asarray((0.0, 0.8, 0.0), dtype=np.float32)
    # A tetrahedron gives the signed-distance audit a closed surface even in a
    # tiny fixture.  The four source meshes deliberately include bone/vessel
    # semantics so subset OBJ and overlay paths are both exercised.
    vertices = np.asarray(
        ((-0.4, -0.4, -0.4), (0.4, -0.4, -0.4), (0.0, 0.4, -0.4), (0.0, 0.0, 0.4)),
        dtype=np.float32,
    )
    base = AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=np.asarray(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=np.int32),
        lbs_weights=None,
        joint_names=["root", "left_elbow", "left_wrist", "left_index1"],
        parents=parents,
        rest_joints=joints,
        inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_mesh_names=["Humerus_L", "Radial_Artery_L", "Humerus_R", "Radial_Artery_R"],
        source_vertex_ranges=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        source_tissues=["bone", "vessel", "bone", "vessel"],
        source_mesh_controller_bones=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_mesh_material_groups=["skeletal", "soft_tissue", "skeletal", "soft_tissue"],
        source_mesh_roles=["authored_mesh"] * 4,
        source_fit_policies=["volume_field"] * 4,
        source_driver_policies=["bind_follow"] * 4,
        source_compound_ids=[""] * 4,
        source_sides=["left", "left", "right", "right"],
        source_landmarks=[("fixture",)] * 4,
        target_landmark_recipes=["fixture"] * 4,
        source_quality_profiles=["default"] * 4,
        driver_indices=np.arange(4, dtype=np.int16)[:, None],
        driver_weights=np.ones((4, 1), dtype=np.float32),
        source_bone_names=["Upper_L", "Forearm_L", "Upper_R", "Forearm_R"],
        source_bone_parents=parents.copy(),
        source_rest_global=global_bind,
        source_rest_local=local_bind,
        source_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_bone_head=heads,
        source_bone_tail=tails,
        source_bone_smplx_a=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 2, 3), dtype=np.int32),
        source_bone_blend=np.zeros(4, dtype=np.float32),
        source_bone_driver_types=["segment_root", "segment_root", "joint_local", "joint_local"],
        source_bone_frame_joints=np.asarray(
            ((0, 1, -1), (1, 2, -1), (2, 2, -1), (3, 3, -1)), dtype=np.int32
        ),
    )
    target = replace(
        base,
        target_rest_global=global_bind.copy(),
        target_rest_local=local_bind.copy(),
        target_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        target_bone_head=heads.copy(),
        target_bone_tail=tails.copy(),
    )
    return with_source_driver_coupling(target)


def _canonical_cache(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path.mkdir()
    write_obj(path / "smpl_canonical_tpose.obj", vertices, faces, comment="fixture canonical body")
    joints = np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)), dtype=np.float32)
    parents = np.asarray((-1, 0, 1, 2), dtype=np.int32)
    bind = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    bind[:, :3, 3] = joints
    np.savez_compressed(
        path / "smpl_canonical_weights.npz",
        lbs_weights=np.eye(4, dtype=np.float32),
        faces=faces,
        rest_joints=joints,
        parents=parents,
        inverse_bind=np.linalg.inv(bind).astype(np.float32),
    )
    (path / "source_manifest.json").write_text(
        json.dumps({"betas": [0.0] * 10}), encoding="utf-8"
    )
    return path


def test_region_matching_does_not_mix_sides_feet_or_heart() -> None:
    assert _side_matches("_1st_Distal_Phalanges_Hand_L", "none", "left")
    assert not _side_matches("_1st_Distal_Phalanges_Hand_R", "none", "left")
    assert _region_name_matches("_1st_Distal_Phalanges_Hand_L", "left_hand")
    assert not _region_name_matches("_1st_Distal_Phalanx_Foot_L", "left_hand")
    assert not _region_name_matches("Heart", "head_neck")
    assert _region_name_matches("Inner_Ear_L", "head_neck")
    assert _region_name_matches("Talus_L", "left_ankle")
    assert _region_name_matches("_1st_Metatarsal_L", "left_foot")


def test_capture_audit_reposes_canonical_body_when_capture_betas_differ(tmp_path: Path) -> None:
    asset = _source_asset()
    asset_path = save_rigged_asset(tmp_path / "anatomy_rigged.npz", asset)
    canonical = _canonical_cache(tmp_path / "canonical", asset.vertices_rest, asset.faces)
    # This direct fit intentionally has a different body location/beta.  The
    # common-body audit must use the re-skinned canonical surface instead.
    motion_path = tmp_path / "smplx_result.npz"
    np.savez_compressed(
        motion_path,
        Rh=np.zeros((1, 3), dtype=np.float32),
        Th=np.zeros((1, 3), dtype=np.float32),
        poses=np.zeros(165, dtype=np.float32),
        shapes=np.asarray([[0.2] + [0.0] * 9], dtype=np.float32),
        vertices=asset.vertices_rest + np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        faces=asset.faces,
    )

    output = tmp_path / "audit"
    report = audit_capture_pose(
        asset_npz=asset_path,
        motion_npz=motion_path,
        output_dir=output,
        canonical_dir=canonical,
        containment_samples_per_mesh=1,
        write_bone_chain_report=False,
    )

    reference = report["reference_surface"]
    assert reference["kind"] == "canonical_reposed_common_body"
    assert reference["beta_mismatch"] is True
    assert np.isclose(reference["capture_vs_canonical_beta_l2"], 0.2)
    assert reference["capture_fit_is_not_used_for_common_body_containment"] is True
    assert report["stage1_runtime_contract"]["passed"] is True
    assert report["containment"]["available"] is True
    assert report["rest_containment"]["available"] is True
    assert report["rest_containment"]["regions"]
    assert (output / "capture_audit.json").is_file()
    assert (output / "objs/anatomy_posed.obj").is_file()
    assert (output / "objs/smplx_capture_fit_posed.obj").is_file()
    assert (output / "objs/smplx_canonical_reposed.obj").is_file()
    assert (output / "objs/source_rig_posed.obj").is_file()
    assert (output / "overlays/posed_bones_vessels_overlay.png").is_file()
    assert (output / "overlays/posed_capture_fit_overlay.png").is_file()
    assert (output / "overlays/rest_bones_vessels_overlay.png").is_file()

    on_disk = json.loads((output / "capture_audit.json").read_text(encoding="utf-8"))
    assert on_disk["reference_surface"]["kind"] == "canonical_reposed_common_body"
    assert on_disk["rest_containment"]["available"] is True
