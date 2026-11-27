"""Keep a selected motion while using a separately specified avatar identity."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence


def with_smplx_body_identity(
    sequence: HumanMotionSequence, *, identity_npz: Path, body_model_npz: Path
) -> HumanMotionSequence:
    """Use avatar shape for rendering, never as an RGB reconstruction input.

    Motion selection, frame rate and translations remain those of the caller.
    Model identity and native55 pose layout are explicit so 16D BEDLAM2 shape
    cannot accidentally be evaluated with a standard 10D male body model.
    """
    from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import pose_to_smplx55_axis_angle

    identity_npz = Path(identity_npz).expanduser().resolve()
    body_model_npz = Path(body_model_npz).expanduser().resolve()
    if not body_model_npz.is_file():
        raise FileNotFoundError(f"SMPL-X avatar body model not found: {body_model_npz}")
    with np.load(identity_npz, allow_pickle=False) as data:
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        gender = str(np.asarray(data["gender"]).reshape(-1)[0]).lower()
    if gender not in {"male", "female", "neutral"} or not np.isfinite(betas).all():
        raise ValueError("Avatar identity requires a valid gender and finite betas")
    if not 1 <= betas.size <= 300:
        raise ValueError("Avatar identity has an unsupported beta dimension")
    poses = np.stack([pose_to_smplx55_axis_angle(row).reshape(165) for row in sequence.poses])
    stat = body_model_npz.stat()
    signature = hashlib.sha256(
        (str(body_model_npz) + f":{stat.st_size}:{stat.st_mtime_ns}:{gender}").encode()
        + betas.tobytes()
    ).hexdigest()[:24]
    return replace(
        sequence,
        model_type="smplx",
        gender=gender,
        betas=betas.copy(),
        poses=poses,
        metadata={
            **sequence.metadata,
            "body_model_path": str(body_model_npz),
            "body_model_variant": "locked_head",
            "pose_layout": "smplx55",
            "avatar_identity_npz": str(identity_npz),
            "avatar_identity_signature": signature,
            "original_motion_model_type": sequence.model_type,
            "original_motion_gender": sequence.gender,
            "shape_usage": "render_source_only; never supplied to reconstruction",
        },
    )
