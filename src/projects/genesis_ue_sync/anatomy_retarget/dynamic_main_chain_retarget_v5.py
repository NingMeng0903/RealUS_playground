"""V5 dynamic main-chain: inherit node2_004 whole-chain bind, single C_total.

Rest/bind authority is the frozen whole-chain subject.  Pose uses the verified
parent-local pose map (source local basis -> target local bind -> FK).  V4 CUDA
multipose root solves are forbidden.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import AnatomicalCalibrationV1
from .chain_rest_fit_v1 import ChainRestFitSubjectV1, _array_digest, _sha256
from .pose_map_v1 import PoseMapV1, build_pose_map_v1, pose_whole_chain_vertices
from .v8_artifacts import SourceOperatorV8, materialize_subject
from .whole_chain_rest_fit_v1 import (
    BASELINE_COMMIT,
    load_whole_chain_rest_fit_v1,
)


DYNAMIC_MAIN_CHAIN_RETARGET_V5_SCHEMA_VERSION = 5
DYNAMIC_MAIN_CHAIN_RETARGET_V5_KIND = "DynamicMainChainSubjectV5"
EXPECTED_POSE_LABELS_V5 = ("tpose", "pose_213328", "pose_213712")


def _c_total_from_binds(B_final: np.ndarray, B_prefit: np.ndarray) -> np.ndarray:
    return np.asarray(B_final, dtype=np.float64) @ np.linalg.inv(
        np.asarray(B_prefit, dtype=np.float64)
    )


@dataclass(frozen=True)
class DynamicMainChainSubjectV5:
    """Shadow subject promoting a whole-chain rest-fit into the V5 contract."""

    whole_chain: ChainRestFitSubjectV1
    C_total: np.ndarray
    source_operator_digest: str
    calibration_digest: str
    smplx_model_sha256: str
    capture_sha256: str
    subject_label: str
    build_report: Mapping[str, Any]

    @property
    def betas(self) -> np.ndarray:
        return self.whole_chain.betas

    @property
    def vertices_final(self) -> np.ndarray:
        return self.whole_chain.vertices_final

    @property
    def B_final(self) -> np.ndarray:
        return self.whole_chain.B_final

    @property
    def B_prefit(self) -> np.ndarray:
        return self.whole_chain.B_prefit

    @property
    def C_bone(self) -> np.ndarray:
        return self.whole_chain.C_bone

    @property
    def target_local_bind(self) -> np.ndarray:
        return self.whole_chain.target_local_bind

    @property
    def inverse_bind(self) -> np.ndarray:
        return self.whole_chain.inverse_bind

    @property
    def bone_parents(self) -> np.ndarray:
        return self.whole_chain.bone_parents

    def validate(self) -> None:
        self.whole_chain.validate()
        total = np.asarray(self.C_total, dtype=np.float64)
        expected = _c_total_from_binds(self.B_final, self.B_prefit)
        if total.shape != (235, 4, 4):
            raise ValueError("V5 C_total must be [235,4,4]")
        if not np.allclose(total, expected, atol=2.0e-7, rtol=0.0):
            raise ValueError("V5 C_total must equal B_final @ inv(B_prefit)")
        if not np.allclose(total, self.C_bone, atol=2.0e-7, rtol=0.0):
            raise ValueError("V5 C_total must equal the single C_bone authority")
        if self.build_report.get("rest_bind_authority") != "whole_chain_node2_004":
            raise ValueError("V5 must declare whole_chain_node2_004 rest/bind authority")
        if self.build_report.get("v4_solver_used") is not False:
            raise ValueError("V5 must not use the quarantined V4 solver")


def build_dynamic_main_chain_retarget_v5(
    *,
    operator: SourceOperatorV8,
    calibration: AnatomicalCalibrationV1,
    whole_chain_subject_dir: Path | str,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    oracle_path: Path | str,
) -> tuple[DynamicMainChainSubjectV5, PoseMapV1, Any]:
    """Promote a frozen whole-chain subject into the V5 single-C_total contract."""

    started = time.perf_counter()
    value = load_whole_chain_rest_fit_v1(
        whole_chain_subject_dir,
        operator=operator,
        calibration=calibration,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
        recheck=False,
    )
    asset = materialize_subject(
        operator, betas=value.betas, gender="male"
    ).rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=Path(oracle_path).resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    c_total = _c_total_from_binds(value.B_final, value.B_prefit)
    report = {
        "rest_bind_authority": "whole_chain_node2_004",
        "pose_authority": "parent_local_pose_map_v1",
        "v4_solver_used": False,
        "baseline_commit": BASELINE_COMMIT,
        "branch_baseline_commit": "31133afba2ced3f4de01df7328d487859c7f9b05",
        "tube_transport_application_count": 1,
        "controller_count": 235,
        "driver_slot_count": 14,
        "build_seconds": float(time.perf_counter() - started),
        "source_whole_chain_dir": str(Path(whole_chain_subject_dir).resolve()),
        "C_total_digest": _array_digest(c_total),
    }
    subject = DynamicMainChainSubjectV5(
        whole_chain=value,
        C_total=c_total,
        source_operator_digest=value.source_operator_digest,
        calibration_digest=value.calibration_digest,
        smplx_model_sha256=value.smplx_model_sha256,
        capture_sha256=value.capture_sha256,
        subject_label=value.subject_label,
        build_report=report,
    )
    subject.validate()
    return subject, pose_map, asset


def pose_dynamic_main_chain_vertices_v5(
    subject: DynamicMainChainSubjectV5,
    pose_map: PoseMapV1,
    *,
    asset: Any,
    pose_axis_angle: Any,
) -> tuple[np.ndarray, np.ndarray]:
    subject.validate()
    return pose_whole_chain_vertices(
        subject.whole_chain,
        pose_map,
        source_asset=asset,
        pose_axis_angle=pose_axis_angle,
    )


def save_dynamic_main_chain_subject_v5(
    path: Path | str,
    subject: DynamicMainChainSubjectV5,
    *,
    checker_report: Mapping[str, Any],
) -> Path:
    subject.validate()
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V5 subject: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        value = subject.whole_chain
        arrays = {
            "betas": np.asarray(value.betas),
            "vertices_prefit": np.asarray(value.vertices_prefit),
            "vertices_final": np.asarray(value.vertices_final),
            "faces": np.asarray(value.faces),
            "bone_parents": np.asarray(value.bone_parents),
            "B_prefit": np.asarray(value.B_prefit),
            "B_final": np.asarray(value.B_final),
            "C_bone": np.asarray(value.C_bone),
            "C_total": np.asarray(subject.C_total),
            "target_local_bind": np.asarray(value.target_local_bind),
            "inverse_bind": np.asarray(value.inverse_bind),
            "prefit_anatomical_frames": np.asarray(value.prefit_anatomical_frames),
            "final_anatomical_frames": np.asarray(value.final_anatomical_frames),
            "smplx_joints_tpose": np.asarray(value.smplx_joints_tpose),
            "station_frame_translation": np.asarray(value.station_frame_translation),
            "centerline_points": np.asarray(value.centerline_points),
            "mesh_policy": np.asarray(value.mesh_policy),
            "moved_vertex_ids": np.asarray(value.moved_vertex_ids),
            "pelvis_cage_vertex_ids": np.asarray(
                value.pelvis_cage_vertex_ids
                if value.pelvis_cage_vertex_ids is not None
                else np.zeros(0, dtype=np.int32)
            ),
            "pelvis_cage_displacements": np.asarray(
                value.pelvis_cage_displacements
                if value.pelvis_cage_displacements is not None
                else np.zeros((0, 3), dtype=np.float64)
            ),
        }
        npz = temporary / "dynamic_main_chain_subject_v5.npz"
        np.savez_compressed(npz, **arrays)
        passed = bool(checker_report.get("passed", False))
        manifest = {
            "schema_version": DYNAMIC_MAIN_CHAIN_RETARGET_V5_SCHEMA_VERSION,
            "artifact_kind": DYNAMIC_MAIN_CHAIN_RETARGET_V5_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "branch_baseline_commit": "31133afba2ced3f4de01df7328d487859c7f9b05",
            "subject_label": subject.subject_label,
            "source_operator_digest": subject.source_operator_digest,
            "calibration_digest": subject.calibration_digest,
            "capture_sha256": subject.capture_sha256,
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "build_report": dict(subject.build_report),
            "checker_report": dict(checker_report),
            "accepted_scope": "full_main_chain_shadow_v5" if passed else "none",
            "decision": (
                "accepted_for_user_genesis_review" if passed else "rejected_for_redesign"
            ),
            "smplx_gender": "male",
            "smplx_model_sha256": subject.smplx_model_sha256,
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
            "v4_solver_used": False,
            "complete": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


__all__ = [
    "DYNAMIC_MAIN_CHAIN_RETARGET_V5_KIND",
    "DYNAMIC_MAIN_CHAIN_RETARGET_V5_SCHEMA_VERSION",
    "EXPECTED_POSE_LABELS_V5",
    "DynamicMainChainSubjectV5",
    "build_dynamic_main_chain_retarget_v5",
    "pose_dynamic_main_chain_vertices_v5",
    "save_dynamic_main_chain_subject_v5",
]
