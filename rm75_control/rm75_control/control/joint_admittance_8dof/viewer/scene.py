"""Genesis scene bootstrap for RM75-6F on Y-axis rail (self-contained, no Among_US)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.generator import (
    compute_layout,
    generate_urdf,
    load_spec,
)
from rm75_control.control.joint_admittance_8dof.param_model.paths import (
    ASSETS_DIR,
    DEFAULT_SPEC_YAML,
    DEFAULT_URDF,
    GENERATED_URDF,
)
from rm75_control.control.joint_admittance_8dof.param_model.placement import (
    entity_pose_from_calib,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import prepare_genesis_urdf
from rm75_control.control.joint_admittance_8dof.viewer.calib_scene import (
    add_calibration_scene,
    default_calib_bundle_path,
    load_calibration_scene,
)
from rm75_control.control.joint_admittance_8dof.viewer.tensor_utils import to_numpy

DEFAULT_Q = np.zeros(8, dtype=np.float64)

# link_7 DAE: light-gray body (mat_0) + black stripe (mat_2). Prefer mesh materials.
RAIL_BOX_HEIGHT_M = 0.08
DEFAULT_ROBOT_POS = (0.0, 0.0, RAIL_BOX_HEIGHT_M)
DEFAULT_RAIL_Y_LIMIT_M = 0.80


@dataclass
class RailGenesisConfig:
    backend: str = "cuda"
    show_viewer: bool = True
    dt: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_pos: tuple[float, float, float] = DEFAULT_ROBOT_POS
    urdf_path: Path = DEFAULT_URDF
    init_q: np.ndarray | None = None
    kinematic: bool = True
    spec_yaml: Path | None = DEFAULT_SPEC_YAML
    calib_bundle: Path | None = None
    load_calib_scene: bool = True
    spawn_robot: bool = True


class RailGenesisScene:
    """Kinematic Genesis viewer: policy joint angles drive pose (no free dynamics)."""

    def __init__(self, cfg: RailGenesisConfig | None = None) -> None:
        self.cfg = cfg or RailGenesisConfig()
        self._gs = None
        self.scene = None
        self.robot = None
        self._q_cmd = DEFAULT_Q.copy()
        self._rail_y_lower = 0.0
        self._rail_y_upper = DEFAULT_RAIL_Y_LIMIT_M
        self._robot_pos = tuple(float(v) for v in self.cfg.robot_pos)
        self._robot_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._world_base_pos = np.array(self._robot_pos, dtype=np.float64)
        self._calib_spec = None
        if self.cfg.load_calib_scene:
            bundle_path = self.cfg.calib_bundle
            if bundle_path is None:
                bundle_path = default_calib_bundle_path()
            self._calib_spec = load_calibration_scene(bundle_path)
        self._resolve_spec()

    def _resolve_spec(self) -> None:
        if self.cfg.spec_yaml is None:
            return
        spec = load_spec(self.cfg.spec_yaml)
        layout = compute_layout(spec)
        generate_urdf(spec, GENERATED_URDF)
        self.cfg.urdf_path = GENERATED_URDF
        calib = resolve_world_calib(spec, layout)
        entity = entity_pose_from_calib(calib)
        self._robot_pos = entity["pos"]
        self._robot_quat = entity["quat_wxyz"]
        self._world_base_pos = np.asarray(calib["base_pos_m"], dtype=np.float64)
        self._rail_y_lower = 0.0
        self._rail_y_upper = float(layout["travel"])

    def build(self) -> None:
        import os

        # Avoid intermittent AssertionError(fast_checksum) when reopening the
        # viewer after the controller session restarted.
        os.environ.setdefault("GS_ENABLE_FASTCACHE", "0")
        import genesis as gs

        self._gs = gs
        backend = gs.cuda if self.cfg.backend == "cuda" else gs.cpu
        if not bool(getattr(gs, "_initialized", False)):
            gs.init(backend=backend, precision="32", logging_level="warning")

        look_z = float(self._world_base_pos[2]) + 0.32
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.dt),
            rigid_options=gs.options.RigidOptions(
                gravity=self.cfg.gravity,
                enable_collision=False,
                enable_self_collision=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.8, -1.2, look_z + 0.65),
                camera_lookat=tuple(float(x) for x in self._world_base_pos),
                camera_fov=45,
                refresh_rate=60,
            ),
            show_viewer=self.cfg.show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0)),
            name="ground",
        )
        if self._calib_spec is not None:
            add_calibration_scene(self.scene, gs, self._calib_spec)
        if self.cfg.spawn_robot:
            urdf = Path(self.cfg.urdf_path)
            if not urdf.is_absolute():
                urdf = ASSETS_DIR / urdf
            if not urdf.exists():
                raise FileNotFoundError(f"Genesis URDF not found: {urdf}")

            urdf = prepare_genesis_urdf(urdf)
            morph_kwargs: dict = {
                "file": str(urdf),
                "pos": self._robot_pos,
                "quat": self._robot_quat,
                "fixed": True,
                "merge_fixed_links": False,
                "decimate": False,
                # False: keep Collada/DAE per-submesh colors (probe + link_7 stripes).
                "prioritize_urdf_material": False,
            }
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(**morph_kwargs),
                material=gs.materials.Rigid(),
                surface=gs.surfaces.Default(),
                name="rm75_rail",
            )
        self.scene.build()
        if self.robot is not None:
            if hasattr(self.robot, "material") and hasattr(self.robot.material, "gravity_compensation"):
                self.robot.material.gravity_compensation = 1.0
            self._apply_pd_gains()
            q0 = DEFAULT_Q if self.cfg.init_q is None else np.asarray(self.cfg.init_q, dtype=float)
            self.set_joint_positions(q0)

    def _apply_pd_gains(self) -> None:
        n = int(to_numpy(self.robot.get_dofs_position()).reshape(-1).size)
        if n != 8:
            raise RuntimeError(f"expected 8 DOFs from Genesis URDF, got {n}")
        effort = np.array([500.0, 60.0, 60.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float32)
        kp = np.array([800.0, 3400.0, 3400.0, 2600.0, 2600.0, 1100.0, 850.0, 850.0], dtype=np.float32)
        kv = np.array([80.0, 380.0, 380.0, 290.0, 290.0, 125.0, 95.0, 95.0], dtype=np.float32)
        if self.cfg.kinematic:
            kp *= 4.0
            kv *= 2.0
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)
        self.robot.set_dofs_force_range(-effort, effort)

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q_cmd = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        if self.robot is None:
            return
        qf = self._q_cmd.astype(np.float32)
        self.robot.set_dofs_position(self._q_cmd)
        self.robot.control_dofs_position(qf)

    def step(self) -> None:
        if self.robot is not None:
            self.robot.control_dofs_position(self._q_cmd.astype(np.float32))
        self.scene.step()

    def joint_positions(self) -> np.ndarray:
        if self.robot is None:
            return self._q_cmd.copy()
        return to_numpy(self.robot.get_dofs_position()).reshape(-1).astype(float)

    def set_rail_y(self, y_m: float) -> None:
        q = self._q_cmd.copy()
        q[0] = float(np.clip(y_m, self._rail_y_lower, self._rail_y_upper))
        self.set_joint_positions(q)

    def rail_y(self) -> float:
        return float(self._q_cmd[0])
