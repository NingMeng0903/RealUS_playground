"""Genesis debug overlay for vessel planning assets (tube meshes, centerlines, point clouds)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.io import read_centerline_obj, read_obj_mesh
from projects.genesis_ue_sync.multiview_realtime.viz.genesis_viewer_lock import try_viewer_render_lock

logger = logging.getLogger(__name__)

DEFAULT_PLANNING_ROOT = Path("outputs/anatomy_retarget/limb_vessel_planning")

_VESSEL_RGBA = {
    "artery": (220, 40, 40, 210),
    "vein": (40, 90, 220, 210),
    "default": (180, 80, 200, 190),
}
_BONE_RGBA = (240, 220, 80, 220)


def _read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    verts: list[list[float]] = []
    colors: list[list[int]] = []
    in_header = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(parts) >= 6:
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    v = np.asarray(verts, dtype=np.float32)
    c = np.asarray(colors, dtype=np.uint8) if colors else None
    return v, c


def _mesh_rgba_for_name(name: str) -> tuple[int, int, int, int]:
    lower = name.lower()
    if "arter" in lower:
        return _VESSEL_RGBA["artery"]
    if "vein" in lower:
        return _VESSEL_RGBA["vein"]
    return _VESSEL_RGBA["default"]


class PlanningOverlayDrawer:
    """Draw vessel tube meshes, centerline segments, and planning point clouds in Genesis."""

    def __init__(
        self,
        runtime: Any,
        *,
        planning_root: Path | str = DEFAULT_PLANNING_ROOT,
        max_pointcloud_points: int = 2500,
        centerline_radius_m: float = 0.004,
        point_radius_m: float = 0.006,
    ) -> None:
        self.runtime = runtime
        self.planning_root = Path(planning_root)
        self.max_pointcloud_points = max(64, int(max_pointcloud_points))
        self.centerline_radius_m = float(centerline_radius_m)
        self.point_radius_m = float(point_radius_m)
        self._nodes: list[Any] = []
        self._loaded_mtime: float = 0.0

    def reload_if_changed(self, *, force: bool = False) -> bool:
        report = self.planning_root / "planning_report.json"
        mtime = 0.0
        if report.is_file():
            mtime = report.stat().st_mtime
        elif self.planning_root.is_dir():
            try:
                mtime = max(p.stat().st_mtime for p in self.planning_root.rglob("*") if p.is_file())
            except ValueError:
                mtime = 0.0
        if not force and mtime <= self._loaded_mtime + 1e-6:
            return False
        self._loaded_mtime = mtime
        self.redraw()
        return True

    def clear(self) -> None:
        with try_viewer_render_lock(self.runtime, timeout_s=0.05) as acquired:
            if not acquired:
                return
            for node in self._nodes:
                try:
                    self.runtime.scene.clear_debug_object(node)
                except Exception:
                    try:
                        self.runtime.scene._visualizer.context.clear_debug_object(node)
                    except Exception:
                        pass
        self._nodes.clear()

    def redraw(self) -> None:
        self.clear()
        if not self.planning_root.is_dir():
            return
        with try_viewer_render_lock(self.runtime, timeout_s=0.2) as acquired:
            if not acquired:
                return
            ctx = getattr(self.runtime.scene, "_visualizer", None)
            ctx = ctx.context if ctx is not None else self.runtime.scene
            self._draw_vessel_meshes(ctx)
            self._draw_centerlines(ctx)
            self._draw_pointcloud(ctx)
        logger.info("planning overlay redraw root=%s nodes=%d", self.planning_root, len(self._nodes))

    def _draw_vessel_meshes(self, ctx: Any) -> None:
        vessel_dir = self.planning_root / "vessel_segments"
        if not vessel_dir.is_dir():
            return
        import trimesh

        for obj_path in sorted(vessel_dir.glob("*_posed.obj")):
            try:
                verts, faces = read_obj_mesh(obj_path)
            except Exception as exc:
                logger.debug("skip vessel mesh %s: %s", obj_path, exc)
                continue
            if verts.size == 0 or faces.size == 0:
                continue
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            rgba = _mesh_rgba_for_name(obj_path.stem)
            mesh.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
            try:
                node = self.runtime.scene.draw_debug_mesh(mesh)
            except Exception:
                continue
            self._nodes.append(node)

    def _draw_centerlines(self, ctx: Any) -> None:
        cl_dir = self.planning_root / "centerlines"
        if not cl_dir.is_dir():
            return
        posed = cl_dir / "vessel_centerlines_posed.obj"
        if not posed.is_file():
            posed = cl_dir / "vessel_centerlines_rest.obj"
        if not posed.is_file():
            return
        try:
            branches = read_centerline_obj(posed)
        except Exception as exc:
            logger.debug("centerline read failed: %s", exc)
            return
        for label, pts in branches.items():
            arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] < 2:
                continue
            rgba = _mesh_rgba_for_name(str(label))
            color = tuple(float(c) / 255.0 for c in rgba[:3]) + (float(rgba[3]) / 255.0,)
            for i in range(arr.shape[0] - 1):
                try:
                    node = ctx.draw_debug_line(
                        arr[i].tolist(),
                        arr[i + 1].tolist(),
                        radius=self.centerline_radius_m,
                        color=color,
                    )
                    self._nodes.append(node)
                except Exception:
                    pass

    def _draw_pointcloud(self, ctx: Any) -> None:
        ply = self.planning_root / "pointclouds" / "vessel_segments_points.ply"
        if not ply.is_file():
            return
        try:
            pts, colors = _read_colored_ply(ply)
        except Exception as exc:
            logger.debug("ply read failed: %s", exc)
            return
        if pts.shape[0] == 0:
            return
        if pts.shape[0] > self.max_pointcloud_points:
            idx = np.linspace(0, pts.shape[0] - 1, self.max_pointcloud_points, dtype=int)
            pts = pts[idx]
            if colors is not None:
                colors = colors[idx]
        if colors is not None and colors.shape[0] == pts.shape[0]:
            groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
            for p, c in zip(pts, colors, strict=True):
                key = (int(c[0]), int(c[1]), int(c[2]))
                groups.setdefault(key, []).append(p)
            for rgb, batch in groups.items():
                pos = np.asarray(batch, dtype=np.float64)
                color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 0.9)
                try:
                    node = ctx.draw_debug_spheres(pos, radius=self.point_radius_m, color=color)
                    self._nodes.append(node)
                except Exception:
                    pass
        else:
            color = (0.9, 0.85, 0.3, 0.85)
            try:
                node = ctx.draw_debug_spheres(np.asarray(pts, dtype=np.float64), radius=self.point_radius_m, color=color)
                self._nodes.append(node)
            except Exception:
                pass
