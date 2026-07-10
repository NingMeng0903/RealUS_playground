"""Human refit controller — tracking QP archived; passthrough ``q_ref`` until redesign."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from projects.genesis_ue_sync.sim_platform.embodiments.phc_mjcf_retarget import (
    smpl_pose_row_from_phc_bundled_q,
)
from projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic.admittance_wbc import (
    AdmittanceWbcOptions,
)
from projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic.collocation import (
    CollocationOptions,
)


@dataclass(frozen=True)
class ImplicitKinodynamicOptions:
    pin_mjcf_path: Path
    device: str = "cuda"
    dt: float = 1.0 / 30.0
    floating_base_dofs: int = 6
    bed_plane_margin_m: float = 0.005
    collocation: CollocationOptions = field(default_factory=CollocationOptions)
    wbc: AdmittanceWbcOptions = field(default_factory=AdmittanceWbcOptions)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pin_mjcf_path"] = str(self.pin_mjcf_path)
        data["collocation"] = asdict(self.collocation)
        data["wbc"] = asdict(self.wbc)
        return data


@dataclass(frozen=True)
class ImplicitKinodynamicStep:
    frame_index: int
    q: np.ndarray
    diagnostics: dict[str, Any]
    q_phc: np.ndarray | None = None
    smpl_pose_visual: np.ndarray | None = None
    smpl_trans_visual: np.ndarray | None = None
    tau_inject: np.ndarray | None = None


def _clamp_to_limits(
    q: np.ndarray,
    q_lower: np.ndarray | None,
    q_upper: np.ndarray | None,
) -> np.ndarray:
    if q_lower is None or q_upper is None:
        return q
    finite = np.isfinite(q_lower) & np.isfinite(q_upper)
    out = q.copy()
    out[finite] = np.clip(out[finite], q_lower[finite], q_upper[finite])
    return out


class ImplicitKinodynamicRefitController:
    """Passthrough controller: returns ``q_ref``; no WBC/collocation (see ``bak/`` archive)."""

    def __init__(
        self,
        *,
        q_ref: np.ndarray,
        support_plane_z: float,
        options: ImplicitKinodynamicOptions,
        q_lower: np.ndarray | None = None,
        q_upper: np.ndarray | None = None,
        smpl_roi_projector: Any | None = None,
        smpl_pose_ref: np.ndarray | None = None,
        smpl_trans_ref: np.ndarray | None = None,
        vposer: Any | None = None,
        bed_center_xy: np.ndarray | None = None,
        bed_size_xy: np.ndarray | None = None,
        pre_sink_reference: bool = False,
        phc_mjcf_layout_path: Path | str | None = None,
        **_: Any,
    ) -> None:
        del smpl_roi_projector, vposer, bed_center_xy, bed_size_xy, pre_sink_reference
        self.options = options
        self.q_ref = np.asarray(q_ref, dtype=np.float32)
        self.q_ref_gt_obs = self.q_ref.copy()
        self.q_opt = self.q_ref.copy()
        self.support_plane_z = float(support_plane_z)
        self.q_lower = None if q_lower is None else np.asarray(q_lower, dtype=np.float32).reshape(-1)
        self.q_upper = None if q_upper is None else np.asarray(q_upper, dtype=np.float32).reshape(-1)
        self.smpl_pose_ref = None if smpl_pose_ref is None else np.asarray(smpl_pose_ref, dtype=np.float32)
        if smpl_trans_ref is not None:
            self.smpl_trans_ref = np.asarray(smpl_trans_ref, dtype=np.float32).reshape(-1, 3)
        else:
            self.smpl_trans_ref = self.q_ref[:, :3].astype(np.float32).copy()
        self.smpl_pose_visual = None if self.smpl_pose_ref is None else self.smpl_pose_ref.copy()
        self.smpl_trans_visual = self.smpl_trans_ref.copy()
        self.last_diagnostics: dict[str, Any] = {}
        self.phc_mjcf_layout_path = None if phc_mjcf_layout_path is None else Path(phc_mjcf_layout_path)
        self.anchor_fk = None
        self.pin_model = None
        self._support_slot_indices = None
        self._wbc_solver = None
        self._transcriber = None
        print(
            "ImplicitKinodynamicRefitController: physics tracking DISABLED "
            "(q_ref passthrough). Archive: bak/human_dynamics_tracking_archive_20260520/",
            flush=True,
        )

    @property
    def frame_count(self) -> int:
        return int(self.q_ref.shape[0])

    def step(
        self,
        frame_index: int,
        *,
        dynamics_context_qt: dict[str, Any] | None = None,
    ) -> ImplicitKinodynamicStep:
        del dynamics_context_qt
        frame = int(max(0, min(int(frame_index), self.frame_count - 1)))
        q_step = _clamp_to_limits(
            np.asarray(self.q_ref[int(frame)], dtype=np.float32).reshape(-1),
            self.q_lower,
            self.q_upper,
        )
        self.q_opt[frame] = q_step
        smpl_pose_out = self.smpl_pose_row_from_q(q_step, frame=frame)
        smpl_trans_out = q_step[:3].astype(np.float32).copy()
        if self.smpl_pose_visual is not None and smpl_pose_out is not None:
            self.smpl_pose_visual[frame] = smpl_pose_out
        self.smpl_trans_visual[frame] = smpl_trans_out
        diag = {
            "frame_index": int(frame),
            "method": "q_ref_passthrough",
            "refit_mode": "tracking_disabled",
            "wbc": {"skipped": True, "reason": "tracking_archived"},
            "root_z_world_m": float(q_step[2]),
            "support_plane_z_m": float(self.support_plane_z),
            "root_above_support_m": float(q_step[2] - self.support_plane_z),
        }
        self.last_diagnostics = diag
        return ImplicitKinodynamicStep(
            frame_index=int(frame),
            q=q_step,
            diagnostics=diag,
            q_phc=q_step.copy(),
            smpl_pose_visual=smpl_pose_out,
            smpl_trans_visual=smpl_trans_out,
            tau_inject=None,
        )

    def smpl_pose_row_from_q(self, q: np.ndarray, *, frame: int = 0) -> np.ndarray | None:
        if self.smpl_pose_ref is None:
            return None
        fi = int(max(0, min(int(frame), self.frame_count - 1)))
        q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
        if (
            self.phc_mjcf_layout_path is not None
            and self.phc_mjcf_layout_path.is_file()
            and q_arr.shape == self.q_ref[fi].shape
        ):
            return smpl_pose_row_from_phc_bundled_q(
                pose_ref=self.smpl_pose_ref[fi],
                q_ref=self.q_ref[fi],
                q_opt=q_arr,
                layout_path=self.phc_mjcf_layout_path,
            )
        return self.smpl_pose_ref[fi].astype(np.float32).copy()

    def optimized_q(self) -> np.ndarray:
        return self.q_opt.copy()

    def optimized_smpl_visual(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return (
            None if self.smpl_pose_visual is None else self.smpl_pose_visual.copy(),
            self.smpl_trans_visual.copy(),
        )

    def run_offline_collocation(
        self,
        frame_indices: Sequence[int],
        *,
        horizon: int | None = None,
        lbfgs_max_iter: int | None = None,
        body_refinement: bool = True,
        two_stage: bool = True,
    ) -> dict[str, Any]:
        del horizon, lbfgs_max_iter, body_refinement, two_stage, frame_indices
        return {
            "skipped": True,
            "reason": "offline_collocation_archived",
            "archive": "bak/human_dynamics_tracking_archive_20260520/",
        }
