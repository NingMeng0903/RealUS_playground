"""Build a 7-DOF Pinocchio model by freezing ``rail_y`` at 0 in the 8-DOF URDF.

We use :func:`pinocchio.buildReducedModel` rather than maintaining a second
URDF: the 8-DOF asset is the single source of truth (validated by the running
controller) and this file is just a re-parameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

# Same path used by ``rm75_control.control.joint_admittance_8dof.model``.
DEFAULT_URDF = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.urdf"
)

RAIL_JOINT_NAME = "rail_y"
ARM_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
DEFAULT_TCP_FRAME = "tcp"


@dataclass
class LockedRailModel:
    """Reduced 7-DOF Pinocchio model + companion frame ids.

    ``model.nq == 7``; joint order is ``joint_1 .. joint_7`` (rad).
    """

    model: pin.Model
    data: pin.Data
    tcp_frame: str
    tcp_id: int
    q_lower: np.ndarray  # (7,) rad
    q_upper: np.ndarray  # (7,) rad
    v_max: np.ndarray    # (7,) rad/s
    urdf_path: Path
    rail_locked_at_m: float

    def clone(self) -> "LockedRailModel":
        """Return an independent :class:`pin.Data` sharing the model (thread/proc safe copy)."""
        return LockedRailModel(
            model=self.model,
            data=self.model.createData(),
            tcp_frame=self.tcp_frame,
            tcp_id=self.tcp_id,
            q_lower=self.q_lower.copy(),
            q_upper=self.q_upper.copy(),
            v_max=self.v_max.copy(),
            urdf_path=self.urdf_path,
            rail_locked_at_m=self.rail_locked_at_m,
        )


def build_locked_rail_model(
    urdf_path: str | Path | None = None,
    *,
    rail_locked_at_m: float = 0.0,
    tcp_frame: str = DEFAULT_TCP_FRAME,
) -> LockedRailModel:
    """Load the 8-DOF URDF and reduce the ``rail_y`` prismatic joint to a fixed offset."""
    urdf = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    full = pin.buildModelFromUrdf(str(urdf))
    if full.nq != 8:
        raise ValueError(f"expected 8-DOF URDF, got nq={full.nq}")

    q_ref = np.zeros(full.nq, dtype=np.float64)
    rail_qidx = full.joints[full.getJointId(RAIL_JOINT_NAME)].idx_q
    q_ref[rail_qidx] = float(rail_locked_at_m)

    lock_ids = pin.StdVec_Index()
    lock_ids.append(full.getJointId(RAIL_JOINT_NAME))
    reduced = pin.buildReducedModel(full, lock_ids, q_ref)
    if reduced.nq != 7:
        raise RuntimeError(f"reduced model has nq={reduced.nq}, expected 7")

    data = reduced.createData()
    if not reduced.existFrame(tcp_frame):
        raise ValueError(f"frame {tcp_frame!r} missing in reduced model")
    tcp_id = reduced.getFrameId(tcp_frame)

    return LockedRailModel(
        model=reduced,
        data=data,
        tcp_frame=tcp_frame,
        tcp_id=tcp_id,
        q_lower=np.asarray(reduced.lowerPositionLimit, dtype=np.float64).copy(),
        q_upper=np.asarray(reduced.upperPositionLimit, dtype=np.float64).copy(),
        v_max=np.asarray(reduced.velocityLimit, dtype=np.float64).copy(),
        urdf_path=urdf,
        rail_locked_at_m=float(rail_locked_at_m),
    )
