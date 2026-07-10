"""Draw the realtime fitted SMPL mesh (orange human) on a live Genesis runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import human_sequence_from_smpl_pkl
from projects.genesis_ue_sync.sim_platform.human_runtime.gt_smpl_display import GtSmplFrameRenderer
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime


class TrackMeshDrawer:
    """Orange SMPL mesh overlay driven by realtime fitted pose."""

    def __init__(
        self,
        runtime: GenesisPlatformRuntime,
        *,
        smpl_model_dir: str | Path = "dataset/intermediate/humans/body_models/smpl",
        mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 235),
        device: str = "cpu",
        display_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.runtime = runtime
        self._smpl_model_dir = Path(smpl_model_dir)
        self._color = tuple(int(c) for c in mesh_rgba)
        self._device = str(device)
        self._offset = np.asarray(display_offset_m, dtype=np.float32).reshape(3)
        self._renderer: GtSmplFrameRenderer | None = None
        self._renderer_betas: np.ndarray | None = None
        self._mesh_node: Any = None
        self._last_pose_aa: np.ndarray | None = None
        self._last_transl: np.ndarray | None = None

    def _ensure_renderer(self, betas: np.ndarray) -> GtSmplFrameRenderer:
        beta_arr = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
        if (
            self._renderer is None
            or self._renderer_betas is None
            or beta_arr.shape != self._renderer_betas.shape
            or not np.allclose(beta_arr, self._renderer_betas, rtol=0.0, atol=1.0e-5)
        ):
            seq = human_sequence_from_smpl_pkl(self._smpl_model_dir, betas=beta_arr)
            self._renderer = GtSmplFrameRenderer(seq, color=self._color, device=self._device)
            self._renderer_betas = beta_arr.copy()
        return self._renderer

    def draw(self, pose_aa: np.ndarray, betas: np.ndarray, transl: np.ndarray) -> bool:
        pose = np.asarray(pose_aa, dtype=np.float32).reshape(-1)
        transl_w = np.asarray(transl, dtype=np.float32).reshape(3) + self._offset
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(transl_w)):
            return False
        if float(np.max(np.abs(pose))) > 4.0:
            return False
        if (
            self._last_pose_aa is not None
            and self._last_transl is not None
            and self._mesh_node is not None
            and pose.shape == self._last_pose_aa.shape
            and np.allclose(pose, self._last_pose_aa, rtol=0.0, atol=1.0e-5)
            and np.allclose(transl_w, self._last_transl, rtol=0.0, atol=1.0e-5)
        ):
            return True

        renderer = self._ensure_renderer(betas)
        # Bypass GtSmplFrameRenderer frame cache (always uses frame_index=0); go direct SMPL forward.
        vertices, _ = renderer._forward_local_body(pose, transl_m=transl_w)
        span = np.ptp(vertices, axis=0)
        if float(np.max(span)) < 0.45 or float(np.max(span)) > 3.0:
            return False

        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=renderer.faces, process=False)
        mesh.visual.vertex_colors = np.tile(np.asarray(self._color, dtype=np.uint8), (len(mesh.vertices), 1))

        with try_viewer_render_lock(self.runtime, timeout_s=0.15) as acquired:
            if not acquired:
                return False
            ctx = self.runtime.scene._visualizer.context
            if self._mesh_node is not None:
                try:
                    ctx.clear_debug_object(self._mesh_node)
                except Exception:
                    pass
                self._mesh_node = None
            self._mesh_node = ctx.draw_debug_mesh(mesh)

        self._last_pose_aa = pose.copy()
        self._last_transl = transl_w.copy()
        return True

    def clear(self) -> None:
        if self._mesh_node is None:
            return
        with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
            if not acquired:
                return
            ctx = self.runtime.scene._visualizer.context
            try:
                ctx.clear_debug_object(self._mesh_node)
            except Exception:
                pass
        self._mesh_node = None
        self._last_pose_aa = None
        self._last_transl = None
