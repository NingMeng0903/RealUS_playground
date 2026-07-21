"""Load medically constrained TCP poses from the DMP playground.

The DMP replay owns the patient-frame target pose.  Neural IRD training and P1
queries use arm-base coordinates, so a calibrated ``T_arm_base_from_patient``
is intentionally required instead of silently assuming coincident frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class DmpTaskSpec:
    trajectory_npz: Path
    T_arm_base_from_patient: np.ndarray
    pose_key: str = "target"


def load_dmp_task_spec(path: str | Path) -> DmpTaskSpec:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    traj = Path(str(raw.get("trajectory_npz", ""))).expanduser()
    if not traj.is_absolute():
        traj = (path.parent / traj).resolve()
    if not traj.is_file():
        raise FileNotFoundError(f"DMP trajectory not found: {traj}")
    transform = raw.get("T_arm_base_from_patient")
    if transform is None:
        raise ValueError(
            "task manifold requires calibrated T_arm_base_from_patient; "
            "do not assume DMP patient coordinates equal arm-base coordinates"
        )
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_arm_base_from_patient must be a 4x4 homogeneous transform")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_arm_base_from_patient last row must be [0,0,0,1]")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) <= 0.0:
        raise ValueError("T_arm_base_from_patient rotation must be proper orthonormal")
    return DmpTaskSpec(
        trajectory_npz=traj,
        T_arm_base_from_patient=T,
        pose_key=str(raw.get("pose_key", "target")),
    )


def load_task_tcp_poses(spec: DmpTaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return phase and medically legal TCP transforms in arm-base coordinates."""
    data = np.load(spec.trajectory_npz, allow_pickle=False)
    suffix = "target" if spec.pose_key == "target" else str(spec.pose_key)
    p_key, r_key = f"p_{suffix}", f"r_{suffix}"
    if p_key not in data or r_key not in data:
        raise KeyError(f"trajectory requires {p_key!r} and {r_key!r}")
    p = np.asarray(data[p_key], dtype=np.float64).reshape(-1, 3)
    R = np.asarray(data[r_key], dtype=np.float64).reshape(-1, 3, 3)
    if p.shape[0] != R.shape[0]:
        raise ValueError("DMP position and orientation sample counts differ")
    phase = np.asarray(data["s"], dtype=np.float64).reshape(-1) if "s" in data else np.linspace(0.0, 1.0, len(p))
    if len(phase) != len(p):
        raise ValueError("DMP phase and pose sample counts differ")
    T_patient_tcp = np.tile(np.eye(4, dtype=np.float64), (len(p), 1, 1))
    T_patient_tcp[:, :3, :3] = R
    T_patient_tcp[:, :3, 3] = p
    T_base_tcp = spec.T_arm_base_from_patient[None, :, :] @ T_patient_tcp
    return phase, T_base_tcp
