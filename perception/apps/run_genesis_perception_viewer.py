#!/usr/bin/env python3
"""Genesis window G — **view** mode: bed + rail robot + SMPL/anatomy (no real robot / no SHM).

For **twin** mode (mirror live robot via SHM), use rm75_control::

  python apps/joint_admittance_8dof/run_with_twin.py --track-subscribe tcp://127.0.0.1:5598

Two modes only:
  view  — this script (static/demo robot pose, ZMQ human overlays)
  twin  — run_with_twin.py (subscribes rm75_state from window A)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PerceptionViewerConfig:
    track_subscribe: str = "tcp://127.0.0.1:5598"
    anatomy_subscribe: str = "tcp://127.0.0.1:5601"
    planning_root: Path = Path("outputs/anatomy_retarget/limb_vessel_planning")
    anatomy_transparent_alpha: float = 0.35
    track_mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 55)
    spawn_robot: bool = True
    backend: str = "cuda"
    reload_planning_s: float = 2.0


class PerceptionViewerOverlay:
    """Subscribe track + anatomy; draw planning assets on a Genesis scene."""

    def __init__(self, scene: object, cfg: PerceptionViewerConfig) -> None:
        self._scene = scene
        self._cfg = cfg
        self._track = None
        self._anatomy_reg = None
        self._anatomy_sub = None
        self._planning = None
        self._latest_pose55 = None
        self._latest_transl = None
        self._anatomy_transparent_applied: set[str] = set()
        self._last_planning_check = 0.0

    def start(self) -> None:
        runtime = getattr(self._scene, "amongus_runtime", None) or _SceneRuntimeAdapter(self._scene)
        try:
            from projects.genesis_ue_sync.multiview_realtime.ingress.track_pose_subscriber import TrackPoseSubscriber

            self._track = TrackPoseSubscriber(
                runtime,
                connect=str(self._cfg.track_subscribe),
                device=str(self._cfg.backend),
                default_betas=__import__("numpy").zeros(10, dtype=__import__("numpy").float32),
                mesh_rgba=self._cfg.track_mesh_rgba,
            )
            self._track.start()
            logging.debug("track subscribe %s", self._cfg.track_subscribe)
        except Exception as exc:
            logging.warning("track overlay unavailable: %s", exc)

        try:
            from projects.genesis_ue_sync.anatomy_retarget.genesis_control import (
                AnatomyAssetRegistry,
                AnatomyAssetSubscriber,
            )

            runtime = getattr(self._scene, "amongus_runtime", None) or _SceneRuntimeAdapter(self._scene)
            alpha = float(self._cfg.anatomy_transparent_alpha)
            self._anatomy_reg = AnatomyAssetRegistry(
                runtime,
                default_color_rgba=(0.2, 0.75, 0.35, alpha),
                default_transparent_alpha=alpha,
            )
            self._anatomy_sub = AnatomyAssetSubscriber(self._anatomy_reg, connect=str(self._cfg.anatomy_subscribe))
            self._anatomy_sub.start()
            logging.debug("anatomy subscribe %s", self._cfg.anatomy_subscribe)
        except Exception as exc:
            logging.warning("anatomy overlay unavailable: %s", exc)

        try:
            from projects.genesis_ue_sync.anatomy_retarget.planning_overlay import PlanningOverlayDrawer

            runtime = getattr(self._scene, "amongus_runtime", None) or _SceneRuntimeAdapter(self._scene)
            self._planning = PlanningOverlayDrawer(runtime, planning_root=self._cfg.planning_root)
            self._planning.reload_if_changed(force=True)
        except Exception as exc:
            logging.warning("planning overlay unavailable: %s", exc)

    def poll_once(self) -> None:
        if self._track is not None:
            try:
                self._track.poll_draw()
            except Exception:
                pass
            try:
                drive = self._track.latest_anatomy_drive()
            except Exception:
                drive = None
            if drive is not None:
                self._latest_pose55, self._latest_transl = drive

        if self._anatomy_reg is not None and self._latest_pose55 is not None:
            try:
                from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import anatomy_transl_from_track_drive

                for model_id in self._anatomy_reg.model_ids:
                    drawer = self._anatomy_reg._drawers.get(model_id)
                    if drawer is not None and not getattr(drawer, "_realus_opaque_layer", False):
                        drawer.set_render_mode("opaque")
                        drawer._realus_opaque_layer = True
                transl = anatomy_transl_from_track_drive(
                    self._latest_pose55,
                    self._latest_transl,
                    self._anatomy_reg.canonical_pelvis(),
                )
                self._anatomy_reg.draw_all(self._latest_pose55, transl=transl)
            except Exception:
                pass

        now = time.monotonic()
        if self._planning is not None and (now - self._last_planning_check) >= float(self._cfg.reload_planning_s):
            self._last_planning_check = now
            try:
                self._planning.reload_if_changed()
            except Exception:
                pass


class _SceneRuntimeAdapter:
    def __init__(self, scene: object) -> None:
        self.scene = getattr(scene, "scene", scene)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        choices=("view",),
        default="view",
        help="view=static robot + ZMQ human (default). Twin: run_with_twin.py",
    )
    ap.add_argument("--track-subscribe", type=str, default="tcp://127.0.0.1:5598")
    ap.add_argument("--anatomy-subscribe", type=str, default="tcp://127.0.0.1:5601")
    ap.add_argument("--planning-root", type=Path, default=Path("outputs/anatomy_retarget/limb_vessel_planning"))
    ap.add_argument("--anatomy-alpha", type=float, default=0.35)
    ap.add_argument(
        "--track-mesh-alpha",
        type=int,
        default=55,
        metavar="0-255",
        help="Orange SMPL-X skin opacity (default 55)",
    )
    ap.add_argument(
        "--no-robot",
        action="store_true",
        help="Hide RM75 + rail (default: show static arm at demo pose)",
    )
    ap.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--reload-planning-s", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true", help="Log ZMQ overlay subscribe details")
    args = ap.parse_args()

    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    rm75 = repo / "rm75_control"
    if rm75.is_dir():
        sys.path.insert(0, str(rm75))

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    for _name in (
        "OpenGL",
        "OpenGL.acceleratesupport",
        "OpenGL.arrays",
        "projects.genesis_ue_sync.multiview_realtime.ingress.track_pose_subscriber",
        "projects.genesis_ue_sync.anatomy_retarget.genesis_control",
        "projects.genesis_ue_sync.anatomy_retarget.planning_overlay",
        "genesis",
    ):
        logging.getLogger(_name).setLevel(logging.DEBUG if args.verbose else logging.ERROR)

    cfg = PerceptionViewerConfig(
        track_subscribe=str(args.track_subscribe),
        anatomy_subscribe=str(args.anatomy_subscribe),
        planning_root=args.planning_root,
        anatomy_transparent_alpha=float(args.anatomy_alpha),
        track_mesh_rgba=(250, 122, 31, max(0, min(255, int(args.track_mesh_alpha)))),
        spawn_robot=not bool(args.no_robot),
        backend=str(args.backend),
        reload_planning_s=float(args.reload_planning_s),
    )

    if cfg.backend == "cuda":
        from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import ensure_cuda_driver_for_taichi

        ensure_cuda_driver_for_taichi(require_gpu=True)

    from rm75_control.control.joint_admittance_8dof.viewer.scene import RailGenesisConfig, RailGenesisScene

    scene = RailGenesisScene(
        RailGenesisConfig(
            backend=cfg.backend,
            show_viewer=True,
            spawn_robot=cfg.spawn_robot,
            load_calib_scene=True,
        )
    )
    try:
        scene.build()
    except ImportError as exc:
        logging.error("Genesis unavailable: %s", exc)
        return 1

    overlay = PerceptionViewerOverlay(scene, cfg)
    overlay.start()
    try:
        while True:
            scene.step()
            overlay.poll_once()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
