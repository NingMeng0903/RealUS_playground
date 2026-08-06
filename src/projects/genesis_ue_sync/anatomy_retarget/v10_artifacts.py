"""Load helpers for V10 shadow subjects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .chain_rest_fit_v1 import ChainRestFitSubjectV1


def load_chain_retarget_v10_subject(path: Path | str) -> tuple[ChainRestFitSubjectV1, dict[str, Any]]:
    """Load a subject written by ``run_chain_retarget_v10_shadow``."""

    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    npz_path = root / "whole_chain_rest_fit_subject_v10.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "segment_scales_v10"}
        segment_scales = (
            np.asarray(data["segment_scales_v10"], dtype=np.float64)
            if "segment_scales_v10" in data.files
            else None
        )
    build_report = dict(manifest.get("build_report") or {})
    value = ChainRestFitSubjectV1(
        source_operator_digest=str(
            manifest.get("source_operator_digest")
            or build_report.get("source_operator_digest", "")
        ),
        calibration_digest=str(
            manifest.get("calibration_digest")
            or build_report.get("calibration_digest", "")
        ),
        source_subject_digest=str(
            manifest.get("source_subject_digest")
            or build_report.get("source_subject_digest", "")
        ),
        smplx_model_sha256=str(
            manifest.get("smplx_model_sha256")
            or build_report.get("smplx_model_sha256", "")
        ),
        capture_sha256=str(
            manifest.get("capture_sha256") or build_report.get("capture_sha256", "")
        ),
        subject_label=str(manifest["subject_label"]),
        betas=np.asarray(arrays["betas"], dtype=np.float64),
        vertices_prefit=np.asarray(arrays["vertices_prefit"], dtype=np.float32),
        vertices_final=np.asarray(arrays["vertices_final"], dtype=np.float32),
        faces=np.asarray(arrays["faces"], dtype=np.int32),
        bone_parents=np.asarray(arrays["bone_parents"], dtype=np.int32),
        B_prefit=np.asarray(arrays["B_prefit"], dtype=np.float64),
        B_final=np.asarray(arrays["B_final"], dtype=np.float64),
        C_bone=np.asarray(arrays["C_bone"], dtype=np.float64),
        target_local_bind=np.asarray(arrays["target_local_bind"], dtype=np.float64),
        inverse_bind=np.asarray(arrays["inverse_bind"], dtype=np.float64),
        prefit_anatomical_frames=np.asarray(
            arrays["prefit_anatomical_frames"], dtype=np.float64
        ),
        final_anatomical_frames=np.asarray(
            arrays["final_anatomical_frames"], dtype=np.float64
        ),
        smplx_joints_tpose=np.asarray(arrays["smplx_joints_tpose"], dtype=np.float64),
        station_frame_translation=np.asarray(
            arrays["station_frame_translation"], dtype=np.float64
        ),
        centerline_points=np.asarray(arrays["centerline_points"], dtype=np.float64),
        mesh_policy=np.asarray(arrays["mesh_policy"]),
        moved_vertex_ids=np.asarray(arrays["moved_vertex_ids"], dtype=np.int32),
        build_report=build_report,
        pelvis_cage_vertex_ids=(
            np.asarray(arrays["pelvis_cage_vertex_ids"], dtype=np.int32)
            if "pelvis_cage_vertex_ids" in arrays
            and arrays["pelvis_cage_vertex_ids"].shape != ()
            else None
        ),
        pelvis_cage_displacements=(
            np.asarray(arrays["pelvis_cage_displacements"], dtype=np.float64)
            if "pelvis_cage_displacements" in arrays
            and arrays["pelvis_cage_displacements"].shape != ()
            else None
        ),
    )
    meta = {
        "manifest": manifest,
        "segment_scales": segment_scales,
        "validation_reports": manifest.get("validation_reports"),
    }
    return value, meta


__all__ = ["load_chain_retarget_v10_subject"]
