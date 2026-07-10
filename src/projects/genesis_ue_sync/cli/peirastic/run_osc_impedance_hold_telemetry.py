"""Hold current TCP pose with OSC impedance; prints JSON telemetry lines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common.project import project_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genesis OSC impedance hold telemetry (no ZMQ).")
    p.add_argument("--scene-spec", type=Path, default=None)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--state-hz", type=float, default=100.0, help="GenesisPlatformRuntime dt = 1/state_hz.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--target-delta-m", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--target-rotvec-rad", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument(
        "--osc-yaml",
        type=Path,
        default=None,
        help="Override configs/controllers/franka_panda_osc_impedance_default.yaml",
    )
    p.add_argument("--print-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    from projects.genesis_ue_sync.sim_platform.control.controllers.base import CartesianControlTarget
    from projects.genesis_ue_sync.sim_platform.control.controllers.common import apply_pose_delta_wxyz
    from projects.genesis_ue_sync.sim_platform.control.controllers.osc_impedance import (
        OSCImpedanceController,
        OSCImpedanceControllerConfig,
        load_osc_impedance_yaml,
    )
    from projects.genesis_ue_sync.sim_platform.embodiments import build_panda_ultrasound_preset
    from projects.genesis_ue_sync.sim_platform.scenes import default_sync_scene_spec_path, load_sync_scene_spec
    from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
        BoxEntityConfig,
        GenesisPlatformRuntime,
        GenesisRuntimeConfig,
    )

    args = parse_args()
    root = project_paths(__file__).root
    scene_path = Path(args.scene_spec) if args.scene_spec else default_sync_scene_spec_path()
    scene_spec = load_sync_scene_spec(scene_path)
    dt_sim = 1.0 / max(float(args.state_hz), 1e-6)

    rt = GenesisPlatformRuntime(GenesisRuntimeConfig(backend=args.backend, show_viewer=False, dt=float(dt_sim)))
    rt.initialize()
    rt.add_ground_plane(color=scene_spec.environment.ground_plane_color)
    if scene_spec.support_surface is not None and scene_spec.support_surface.spawn_in_genesis:
        rt.add_box(
            BoxEntityConfig(
                name=scene_spec.support_surface.name,
                pos=scene_spec.support_surface.pos,
                size=scene_spec.support_surface.size,
                quat_xyzw=scene_spec.support_surface.quat_xyzw,
                color=scene_spec.support_surface.color,
            ),
        )

    robot_urdf = Path(scene_spec.robot.resolved_urdf_path)
    emb = build_panda_ultrasound_preset(urdf_path=robot_urdf, camera_names=())
    robot_name = scene_spec.robot.name
    rt.add_articulated_entity(
        emb,
        name=robot_name,
        pos=scene_spec.robot.base_pos,
        quat_xyzw=scene_spec.robot.base_quat_xyzw,
    )
    rt.set_robot_gravity_compensation(robot_name, 1.0)
    rt.build()
    rt.apply_franka_like_arm_pd_gains(robot_name)
    rt.reset()

    motion = rt.get_motion_interface(robot_name)
    jp = np.asarray(scene_spec.robot.joint_positions, dtype=np.float32).reshape(-1)
    motion.set_joint_positions(jp)
    motion.control_joint_positions(jp)

    yaml_path = Path(args.osc_yaml) if args.osc_yaml else root / "configs/controllers/franka_panda_osc_impedance_default.yaml"
    osc_data = load_osc_impedance_yaml(yaml_path)
    osc_cfg = OSCImpedanceControllerConfig.from_yaml_dict(osc_data, float(dt_sim))
    tcp_ov = scene_spec.robot.tcp_control
    if tcp_ov is not None:
        if tcp_ov.link_name:
            osc_cfg.tcp_link_name = tcp_ov.link_name
        if tcp_ov.local_point_m is not None:
            osc_cfg.tcp_local_point_m = np.asarray(tcp_ov.local_point_m, dtype=np.float32).reshape(3)

    osc = OSCImpedanceController(motion, osc_cfg)
    hold = np.asarray(osc.current_pose(), dtype=np.float32).reshape(7)
    target_pose = apply_pose_delta_wxyz(
        hold,
        np.concatenate(
            [
                np.asarray(args.target_delta_m, dtype=np.float32).reshape(3),
                np.asarray(args.target_rotvec_rad, dtype=np.float32).reshape(3),
            ],
            dtype=np.float32,
        ),
    )
    zero_twist = np.zeros(6, dtype=np.float32)
    q_home = jp.copy()

    max_pose_err = 0.0
    max_tau_inf = 0.0

    pe = args.print_every
    for step in range(int(max(args.steps, 1))):
        res = osc.step(
            CartesianControlTarget(
                pose=target_pose,
                twist=zero_twist,
                nullspace_target=q_home,
                metadata={"dt": float(dt_sim)},
            ),
        )
        rt.step()

        pose_err_inf = float(np.max(np.abs(res.pose_error)))
        tau_inf = float(np.max(np.abs(res.command)))
        max_pose_err = max(max_pose_err, pose_err_inf)
        max_tau_inf = max(max_tau_inf, tau_inf)

        if pe > 0 and step % pe == 0:
            payload = {
                "step": int(step),
                "pose_err_max_abs": pose_err_inf,
                "tau_max_abs_Nm": tau_inf,
                "gain_ramp_meta": float(res.metadata.get("gain_ramp", 1.0)),
            }
            print(json.dumps(payload), flush=True)

    print(
        json.dumps(
            {
                "summary_max_pose_err": max_pose_err,
                "summary_max_tau_inf": max_tau_inf,
                "hold_pose_tcp_wxyz": hold.tolist(),
                "target_pose_tcp_wxyz": target_pose.tolist(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
