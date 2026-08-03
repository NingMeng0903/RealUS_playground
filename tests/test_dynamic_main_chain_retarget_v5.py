from __future__ import annotations

from pathlib import Path

import numpy as np

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_retarget_v5 import (
    build_dynamic_main_chain_retarget_v5,
    pose_dynamic_main_chain_vertices_v5,
)
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT
    / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_005"
    / "anatomical_calibration_v1"
)
ORACLE = (
    ROOT
    / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
    / "blender_link_oracle_v7.npz"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
NODE2 = ROOT / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node2_004"


def test_v5_single_c_total_and_zero_pose():
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION,
        operator=operator,
        require_complete=False,
        required_scope="full_main_chain",
    )
    _, model_sha = require_frozen_smplx_male_v7(MODEL)
    model = load_smplx_model_v7(MODEL)
    subject, pose_map, asset = build_dynamic_main_chain_retarget_v5(
        operator=operator,
        calibration=calibration,
        whole_chain_subject_dir=NODE2 / "subject_213328",
        smplx_model=model,
        smplx_model_sha256=model_sha,
        oracle_path=ORACLE,
    )
    assert subject.build_report["v4_solver_used"] is False
    assert np.allclose(subject.C_total, subject.C_bone, atol=2e-7)
    zero = np.zeros((55, 3), dtype=np.float32)
    posed, _ = pose_dynamic_main_chain_vertices_v5(
        subject, pose_map, asset=asset, pose_axis_angle=zero
    )
    err = np.max(np.linalg.norm(posed - subject.vertices_final, axis=1))
    assert err < 1e-5
