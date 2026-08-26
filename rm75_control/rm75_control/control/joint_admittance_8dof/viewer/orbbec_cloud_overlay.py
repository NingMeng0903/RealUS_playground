"""Optional Orbbec colored-cloud overlay for the 8-DOF twin (default off).

CPU mesh is built on the ZMQ thread. ``draw_debug_mesh`` must run on the
twin thread (Genesis / OpenGL). ``after_sync`` writes ``T_world_cam`` every
frame and only swaps a prebuilt mesh when one is waiting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import (
    DEFAULT_ORBBEC_CLOUD_BIND,
    DEFAULT_ORBBEC_CLOUD_TOPIC,
    DEFAULT_SPHERE_RADIUS_M,
    RailBaseLink7FK,
    T_from_pos_quat_wxyz,
    T_world_cam,
    camera_cloud_mesh_arrays,
    load_T_link7_cam,
    unpack_cloud_multipart,
)

_MIN_SWAP_S = 0.05


@dataclass
class OrbbecCloudOverlayConfig:
    subscribe: str = DEFAULT_ORBBEC_CLOUD_BIND
    topic: str = DEFAULT_ORBBEC_CLOUD_TOPIC
    handeye_yaml: Path | None = None
    urdf_path: Path | None = None
    sphere_radius_m: float = DEFAULT_SPHERE_RADIUS_M
    draw_hz: float = 0.0  # 0 = every twin frame


class OrbbecCloudOverlay:
    """Subscribe camera-frame cloud; keep one debug mesh and move it with TF."""

    def __init__(self, scene: Any, config: OrbbecCloudOverlayConfig | None = None) -> None:
        self._scene = scene
        self._cfg = config or OrbbecCloudOverlayConfig()
        self._T_link7_cam = load_T_link7_cam(self._cfg.handeye_yaml)
        urdf = self._cfg.urdf_path
        if urdf is None:
            urdf = getattr(getattr(scene, "cfg", None), "urdf_path", None)
        self._fk: RailBaseLink7FK | None = None
        if urdf is not None and Path(urdf).is_file():
            try:
                self._fk = RailBaseLink7FK(urdf)
            except Exception:
                self._fk = None
        self._stop = threading.Event()
        self._recv_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending_mesh: Any = None
        self._pending_seq = -1
        self._pending_n = 0
        self._cloud_seq = 0
        self._drawn_seq = -1
        self._last_q: np.ndarray | None = None
        self._last_T: np.ndarray | None = None
        self._node: Any = None
        self._pose_ok = False
        self._last_draw_t = 0.0
        self._last_swap_t = 0.0
        self._slow_n = 0
        self._recv_log_n = 0
        self._swap_fail_n = 0

    def start(self) -> None:
        if self._recv_thread is not None and self._recv_thread.is_alive():
            return
        self._stop.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, name="orbbec-cloud-sub", daemon=True)
        self._recv_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.5)
            self._recv_thread = None
        node = None
        with self._lock:
            node = self._node
            self._node = None
            self._pending_mesh = None
        if node is not None:
            self._clear_one(node)

    def T_world_from_cam(self, q8: np.ndarray) -> np.ndarray:
        if self._fk is None:
            raise RuntimeError("no rail_base→link_7 FK")
        T_rb_l7 = self._fk.T_railbase_link7(q8)
        pos = getattr(self._scene, "_robot_pos", (0.0, 0.0, 0.0))
        quat = getattr(self._scene, "_robot_quat", (1.0, 0.0, 0.0, 0.0))
        return T_world_cam(T_from_pos_quat_wxyz(pos, quat), T_rb_l7, self._T_link7_cam)

    def draw(self, q8: np.ndarray) -> None:
        """Twin thread: swap a prebuilt mesh if ready, then write T."""
        if self._fk is None:
            return
        hz = float(self._cfg.draw_hz)
        now = time.monotonic()
        if hz > 0.0 and now - self._last_draw_t < 1.0 / hz:
            return
        q = np.asarray(q8, dtype=np.float64).reshape(-1)
        t0 = time.monotonic()
        try:
            T = np.asarray(self.T_world_from_cam(q), dtype=np.float64).reshape(4, 4)
        except Exception:
            return
        with self._lock:
            self._last_T = T
            has_pending = self._pending_mesh is not None
            node = self._node
            q_same = (
                self._last_q is not None
                and self._last_q.shape == q.shape
                and np.allclose(q, self._last_q, atol=1e-6)
            )
            retry = not self._pose_ok
        swapped = False
        if has_pending and (node is None or now - self._last_swap_t >= _MIN_SWAP_S):
            swapped = self.upload_pending(T)
            if swapped:
                self._last_swap_t = now
                with self._lock:
                    node = self._node
                    self._last_q = q.copy()
                    self._pose_ok = True
        if node is None:
            with self._lock:
                self._last_q = q.copy()
            return
        if swapped or (q_same and not retry):
            self._last_draw_t = now
            return
        ok = self._update_pose(node, T, blocking=False)
        with self._lock:
            self._last_q = q.copy()
            if ok:
                self._pose_ok = True
        self._last_draw_t = now
        dt = time.monotonic() - t0
        if dt > 0.012:
            self._slow_n += 1
            if self._slow_n <= 3 or self._slow_n % 30 == 0:
                print(
                    f"rm75 twin: orbbec overlay {dt * 1000:.1f} ms "
                    f"(swap+TF hitch #{self._slow_n})",
                    flush=True,
                )

    def upload_pending(self, T: np.ndarray | None = None) -> bool:
        """Attach a prebuilt mesh. Must run on the twin / Genesis thread."""
        with self._lock:
            mesh = self._pending_mesh
            seq = self._pending_seq
            n = int(self._pending_n)
            if T is None:
                T = self._last_T
        if mesh is None or T is None:
            return False
        if not self._swap_mesh(mesh, np.asarray(T, dtype=np.float64).reshape(4, 4)):
            return False
        with self._lock:
            if self._pending_seq == seq:
                self._pending_mesh = None
            self._drawn_seq = seq
            self._pose_ok = False
        if self._drawn_seq == seq and n and (seq <= 2 or seq % 30 == 0):
            print(f"rm75 twin: orbbec cloud mesh n={n} seq={seq}", flush=True)
        return True

    def _recv_loop(self) -> None:
        try:
            import zmq
        except ImportError:
            print("rm75 twin: orbbec cloud recv disabled (no zmq)", flush=True)
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 4)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.setsockopt(zmq.SUBSCRIBE, self._cfg.topic.encode("utf-8"))
        try:
            sock.connect(str(self._cfg.subscribe))
        except Exception as exc:
            print(f"rm75 twin: orbbec cloud subscribe failed ({exc})", flush=True)
            sock.close(0)
            return
        print(
            f"rm75 twin: orbbec cloud SUB connected {self._cfg.subscribe} topic={self._cfg.topic}",
            flush=True,
        )
        last_wait_log = time.monotonic()
        while not self._stop.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                if self._recv_log_n == 0 and time.monotonic() - last_wait_log >= 3.0:
                    print(
                        "rm75 twin: orbbec cloud waiting for ZMQ "
                        f"({self._cfg.subscribe}) — publisher must print published n=",
                        flush=True,
                    )
                    last_wait_log = time.monotonic()
                continue
            except Exception:
                if self._stop.is_set():
                    break
                time.sleep(0.05)
                continue
            try:
                _meta, xyz, rgb = unpack_cloud_multipart(parts)
            except Exception:
                continue
            if rgb is None or rgb.shape[0] != xyz.shape[0]:
                rgb = np.full((xyz.shape[0], 3), 0.75, dtype=np.float32)
            mesh = self._make_mesh(xyz, rgb)
            if mesh is None:
                continue
            with self._lock:
                self._cloud_seq += 1
                self._pending_mesh = mesh
                self._pending_seq = self._cloud_seq
                self._pending_n = int(xyz.shape[0])
                seq = self._cloud_seq
                n = int(xyz.shape[0])
            self._recv_log_n += 1
            if self._recv_log_n <= 2 or self._recv_log_n % 30 == 0:
                print(f"rm75 twin: orbbec cloud recv n={n} seq={seq}", flush=True)
        try:
            sock.close(0)
        except Exception:
            pass

    def _gs_scene(self) -> Any | None:
        return getattr(self._scene, "scene", None)

    def _rasterizer_ctx(self) -> Any | None:
        gs_scene = self._gs_scene()
        if gs_scene is None:
            return None
        viz = getattr(gs_scene, "_visualizer", None)
        return getattr(viz, "context", None) if viz is not None else None

    def _render_lock(self) -> Any | None:
        gs_scene = self._gs_scene()
        viz = getattr(gs_scene, "_visualizer", None) if gs_scene is not None else None
        wrapper = getattr(viz, "viewer_lock", None) if viz is not None else None
        inner = getattr(wrapper, "_pyrender_viewer", None)
        return getattr(inner, "render_lock", None)

    def _with_lock(self, fn, *, blocking: bool = True):
        rl = self._render_lock()
        if rl is not None:
            acquired = rl.acquire(blocking=blocking)
            if not acquired:
                return False
            try:
                fn()
                return True
            finally:
                rl.release()
        gs_scene = self._gs_scene()
        viz = getattr(gs_scene, "_visualizer", None) if gs_scene is not None else None
        lock = getattr(viz, "viewer_lock", None)
        if lock is None:
            fn()
            return True
        with lock:
            fn()
        return True

    def _make_mesh(self, xyz: np.ndarray, rgb: np.ndarray) -> Any | None:
        try:
            import trimesh
        except ImportError:
            print("rm75 twin: orbbec cloud mesh skipped (no trimesh)", flush=True)
            return None
        verts, faces, colors = camera_cloud_mesh_arrays(
            xyz, rgb, radius_m=float(self._cfg.sphere_radius_m)
        )
        if verts.shape[0] == 0:
            return None
        visual = trimesh.visual.ColorVisuals()
        visual._data["vertex_colors"] = colors
        return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)

    def _swap_mesh(self, mesh: Any, T: np.ndarray) -> bool:
        gs_scene = self._gs_scene()
        ctx = self._rasterizer_ctx()
        new_node = None
        try:
            if gs_scene is not None and hasattr(gs_scene, "draw_debug_mesh"):
                new_node = gs_scene.draw_debug_mesh(mesh, T=T)
            elif ctx is not None and hasattr(ctx, "draw_debug_mesh"):
                holder: list[Any] = []

                def _draw() -> None:
                    holder.append(ctx.draw_debug_mesh(mesh, T=T))

                if not self._with_lock(_draw, blocking=True):
                    return False
                new_node = holder[0] if holder else None
        except Exception as exc:
            self._swap_fail_n += 1
            if self._swap_fail_n <= 3 or self._swap_fail_n % 30 == 0:
                print(f"rm75 twin: orbbec cloud mesh swap failed ({exc})", flush=True)
            return False
        if new_node is None:
            self._swap_fail_n += 1
            if self._swap_fail_n <= 3:
                print("rm75 twin: orbbec cloud mesh swap returned None", flush=True)
            return False
        with self._lock:
            old = self._node
            self._node = new_node
        if old is not None:
            self._clear_one(old)
        return True

    def _update_pose(self, node: Any, T: np.ndarray, *, blocking: bool) -> bool:
        ctx = self._rasterizer_ctx()
        host = ctx if ctx is not None and hasattr(ctx, "update_debug_objects") else self._gs_scene()
        if host is None or not hasattr(host, "update_debug_objects"):
            return False
        try:
            return bool(
                self._with_lock(lambda: host.update_debug_objects([node], [T]), blocking=blocking)
            )
        except Exception:
            return False

    def _clear_one(self, node: Any) -> None:
        gs_scene = self._gs_scene()
        ctx = self._rasterizer_ctx()
        for host in (gs_scene, ctx):
            if host is None:
                continue
            clearer = getattr(host, "clear_debug_object", None)
            if clearer is None:
                continue
            try:
                clearer(node)
                return
            except Exception:
                continue
