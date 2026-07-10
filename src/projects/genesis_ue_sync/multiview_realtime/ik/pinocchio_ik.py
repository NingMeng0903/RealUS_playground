"""Reusable Pinocchio position IK (damped least squares) over multiple frame targets.

Model-agnostic: feed any Pinocchio model (typically a floating-base humanoid) plus a
set of end-effector frame names, and solve for joint configuration ``q`` whose frame
origins match world-space 3D targets. This is stage-3 of the realtime pipeline,
turning filtered triangulated joints into URDF joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class PinocchioIKConfig:
    max_iters: int = 60
    tol_m: float = 0.01          # stop when RMS target error is below this
    damping: float = 0.1         # Levenberg-Marquardt damping (lambda); JJt += damping**2 * I
    step_scale: float = 1.0      # integration step gain per iteration
    neutral_damping: float = 0.02  # pull unobserved dofs toward neutral each iteration
    clamp_joint_limits: bool = True


def load_model_from_urdf(urdf_path: str | Path, *, floating_base: bool = True) -> pin.Model:
    urdf_path = str(urdf_path)
    if floating_base:
        return pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    return pin.buildModelFromUrdf(urdf_path)


def load_model_from_mjcf(mjcf_path: str | Path) -> pin.Model:
    return pin.buildModelFromMJCF(str(mjcf_path))


class PinocchioIKSolver:
    """Damped least-squares position IK for a fixed set of target frames."""

    def __init__(
        self,
        model: pin.Model,
        target_frames: list[str],
        config: PinocchioIKConfig | None = None,
    ) -> None:
        self.model = model
        self.data = model.createData()
        self.config = config or PinocchioIKConfig()
        self._frame_ids: list[int] = []
        for name in target_frames:
            if not model.existFrame(name):
                raise ValueError(f"Frame '{name}' not in model. Available example frames: "
                                 f"{[f.name for f in list(model.frames)[:12]]}")
            self._frame_ids.append(model.getFrameId(name))
        self.target_frames = list(target_frames)
        self._lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
        self._upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
        self._neutral_q = pin.neutral(model)

    def neutral_q(self) -> np.ndarray:
        return self._neutral_q.copy()

    def solve(
        self,
        targets_world: dict[str, np.ndarray],
        q_init: np.ndarray | None = None,
        weights: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve for ``q`` matching frame origins to world targets.

        ``targets_world`` maps frame name -> (3,) position. Frames absent from the dict
        (or with non-finite targets) are skipped this solve.
        """
        cfg = self.config
        q = (self.neutral_q() if q_init is None else np.asarray(q_init, dtype=np.float64).copy())
        nv = self.model.nv

        active: list[tuple[int, np.ndarray, float]] = []
        for name, fid in zip(self.target_frames, self._frame_ids):
            tgt = targets_world.get(name)
            if tgt is None:
                continue
            tgt = np.asarray(tgt, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(tgt)):
                continue
            w = float((weights or {}).get(name, 1.0))
            active.append((fid, tgt, w))

        diagnostics: dict[str, Any] = {"n_targets": len(active), "iters": 0, "rms_err_m": float("nan")}
        if not active:
            return q, diagnostics

        last_err = float("nan")
        for it in range(int(cfg.max_iters)):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q)

            rows = 3 * len(active)
            e = np.zeros(rows, dtype=np.float64)
            J = np.zeros((rows, nv), dtype=np.float64)
            for i, (fid, tgt, w) in enumerate(active):
                pos = np.asarray(self.data.oMf[fid].translation, dtype=np.float64)
                Jf = pin.getFrameJacobian(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                e[3 * i : 3 * i + 3] = w * (tgt - pos)
                J[3 * i : 3 * i + 3, :] = w * Jf[:3, :]

            last_err = float(np.sqrt(np.mean((e / max(1e-9, 1.0)) ** 2)))
            if last_err < cfg.tol_m:
                diagnostics["iters"] = it
                break

            JJt = J @ J.T + (cfg.damping**2) * np.eye(rows)
            dq = J.T @ np.linalg.solve(JJt, e)
            if cfg.neutral_damping > 0.0:
                dq = dq - float(cfg.neutral_damping) * pin.difference(self.model, q, self._neutral_q)
            q = pin.integrate(self.model, q, cfg.step_scale * dq)
            if cfg.clamp_joint_limits:
                finite = np.isfinite(self._lower) & np.isfinite(self._upper)
                if np.any(finite):
                    q[finite] = np.clip(q[finite], self._lower[finite], self._upper[finite])
            diagnostics["iters"] = it + 1

        diagnostics["rms_err_m"] = last_err
        return q, diagnostics

    def frame_positions(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=np.float64))
        pin.updateFramePlacements(self.model, self.data)
        return {
            name: np.asarray(self.data.oMf[fid].translation, dtype=np.float64).copy()
            for name, fid in zip(self.target_frames, self._frame_ids)
        }


__all__ = [
    "PinocchioIKConfig",
    "PinocchioIKSolver",
    "load_model_from_mjcf",
    "load_model_from_urdf",
]
