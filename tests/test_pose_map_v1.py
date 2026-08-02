from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from src.projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    apply_pose_map_global,
    build_pose_map_v1,
    check_pose_map_v1,
    pose_whole_chain_vertices,
)
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from src.projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    build_whole_chain_rest_fit_v1,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_005"
    / "anatomical_calibration_v1"
)
ORACLE = (
    ROOT / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
    / "blender_link_oracle_v7.npz"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURES = {
    label: ROOT / f"smplx_outputs/20260713_{label}/moment_0000/smplx_result.npz"
    for label in ("213328", "213712")
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def matrix():
    paths = (OPERATOR, CALIBRATION, ORACLE, MODEL, *CAPTURES.values())
    if not all(path.exists() for path in paths):
        pytest.skip("frozen PoseMapV1 inputs are unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(MODEL)
    model_sha = _sha(MODEL)
    values = {}
    for label, capture in CAPTURES.items():
        with np.load(capture, allow_pickle=False) as data:
            betas = np.asarray(data["shapes"]).reshape(-1)[:10]
            pose = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=MODEL
            )
        value = build_whole_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas,
            subject_label=label,
            capture_sha256=_sha(capture),
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
        pose_map = build_pose_map_v1(
            value,
            asset=asset,
            calibration=calibration,
            oracle_path=ORACLE,
            source_operator_digest=operator.runtime_digest(validate=False),
        )
        values[label] = (value, asset, pose_map, pose)
    return calibration, values


def test_two_beta_pose_map_is_single_authority_parent_local(matrix) -> None:
    calibration, values = matrix
    for value, asset, pose_map, _pose in values.values():
        report = check_pose_map_v1(pose_map, value, source_asset=asset)
        assert report["passed"] is True, json.dumps(report, indent=2, sort_keys=True)
        np.testing.assert_array_equal(
            pose_map.controller_motion_modes, calibration.controller_motion_modes
        )
        assert len(pose_map.controller_motion_modes) == 235
        assert report["parent_local_mapping_only"] is True
        assert report["pose_time_search"] is False
        assert report["forbidden_global_modes"] == []
        assert max(report["functional_axis_split_error_deg"]) <= 3.0


def test_zero_pose_reconstructs_the_complete_candidate_rest(matrix) -> None:
    _calibration, values = matrix
    zero = np.zeros((55, 3), dtype=np.float32)
    for value, asset, pose_map, _pose in values.values():
        vertices, global_matrices = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=zero
        )
        np.testing.assert_allclose(vertices, value.vertices_final, atol=2.0e-7, rtol=0.0)
        np.testing.assert_allclose(
            global_matrices, pose_map.target_bind_global, atol=2.0e-6, rtol=0.0
        )


def test_recorded_pose_moves_bones_and_tubes_on_target_path(matrix) -> None:
    _calibration, values = matrix
    for value, asset, pose_map, pose in values.values():
        posed, global_matrices = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=pose
        )
        assert np.all(np.isfinite(posed))
        assert np.all(np.isfinite(global_matrices))

        names = list(asset.source_mesh_names or ())
        ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
        pelvis_mesh = names.index("Ilium_L")
        pelvis_start, pelvis_stop = ranges[pelvis_mesh]
        pelvis_delta = np.linalg.norm(
            posed[pelvis_start:pelvis_stop]
            - value.vertices_final[pelvis_start:pelvis_stop],
            axis=1,
        )
        assert float(np.max(pelvis_delta)) > 1.0e-3

        second, _ = pose_whole_chain_vertices(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose,
            include_tube_transport_preview=True,
        )
        tube_ids = np.concatenate(
            [
                np.arange(int(start), int(stop), dtype=np.int64)
                for tissue, (start, stop) in zip(asset.source_tissues, ranges.tolist())
                if str(tissue).strip().lower() in {"vessel", "nerve"}
            ]
        )
        np.testing.assert_array_equal(second[tube_ids], posed[tube_ids])
        assert float(
            np.max(
                np.linalg.norm(
                    posed[tube_ids] - value.vertices_final[tube_ids], axis=1
                )
            )
        ) > 0.0


def test_new_bind_changes_only_the_bind_not_the_142_local_pose_basis(matrix) -> None:
    _calibration, values = matrix
    for _value, asset, pose_map, pose in values.values():
        target_global = apply_pose_map_global(
            pose_map, source_asset=asset, pose_axis_angle=pose
        )
        source_global = apply_pose_map_global(
            type(pose_map)(
                **{
                    **pose_map.__dict__,
                    "target_bind_global": pose_map.source_bind_global,
                    "target_bind_local": pose_map.source_bind_local,
                    "target_inverse_bind": np.linalg.inv(pose_map.source_bind_global),
                    "change_of_bind_basis": np.tile(
                        np.eye(4), (len(pose_map.bone_names), 1, 1)
                    ),
                }
            ),
            source_asset=asset,
            pose_axis_angle=pose,
        )
        target_local = np.empty_like(target_global)
        source_local = np.empty_like(source_global)
        for bone, parent in enumerate(pose_map.bone_parents.tolist()):
            target_local[bone] = (
                target_global[bone]
                if parent < 0
                else np.linalg.inv(target_global[parent]) @ target_global[bone]
            )
            source_local[bone] = (
                source_global[bone]
                if parent < 0
                else np.linalg.inv(source_global[parent]) @ source_global[bone]
            )
        target_basis = np.linalg.inv(pose_map.target_bind_local) @ target_local
        source_basis = np.linalg.inv(pose_map.source_bind_local) @ source_local
        np.testing.assert_allclose(target_basis, source_basis, atol=3.0e-6, rtol=0.0)
