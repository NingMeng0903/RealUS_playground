from __future__ import annotations

"""
Smoke test: subprocess Genesis server + ``FrankaInterface`` client.

Requires ``pip install -e ref_code_library/PEIRASTIC_control`` and a working Genesis install.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--server-startup-s", type=float, default=5.0)
    return parser.parse_args()


def repo_root() -> Path:
    from common.project import project_paths

    return project_paths(__file__).root


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    root = repo_root()

    import numpy as np
    from projects.genesis_ue_sync.integrations.peirastic.pose_codec import o_t_ee_flat_from_homogeneous, tcp_pose_to_homogeneous

    T = np.eye(4)
    flat = o_t_ee_flat_from_homogeneous(T)
    R = np.array(flat).reshape(4, 4).transpose()
    assert np.allclose(R, T)

    iface_yaml = root / "ref_code_library" / "PEIRASTIC_control" / "config" / "local-host.yml"
    if not iface_yaml.is_file():
        logging.error("Missing %s", iface_yaml)
        return 2

    py = sys.executable
    srv_script = root / "src" / "projects" / "genesis_ue_sync" / "cli" / "peirastic" / "run_genesis_franka_sim_server.py"
    env = os.environ.copy()
    pp = f"{root / 'src'}:{root / 'ref_code_library' / 'PEIRASTIC_control'}"
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    proc = subprocess.Popen(
        [py, str(srv_script), "--backend", args.backend, "--interface-yaml", str(iface_yaml)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(float(args.server_startup_s))

    if proc.poll() is not None:
        err = (proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "") or ""
        if "Genesis is required" in err or "No module named 'genesis'" in err:
            logging.warning("Genesis not installed; skipping ZMQ integration portion of smoke.")
            logging.warning("%s", err.strip()[-1500:])
            return 0
        logging.error("Server exited early: %s", err[-2000:])
        return 6

    try:
        from peirastic.franka_interface import FrankaInterface
        from peirastic.utils.config_utils import get_default_controller_config
    except ImportError as exc:
        logging.error("Install PEIRASTIC_control (pip install -e ref_code_library/PEIRASTIC_control): %s", exc)
        proc.terminate()
        proc.wait(timeout=15)
        return 3

    robot = FrankaInterface(str(iface_yaml), has_gripper=False, use_visualizer=False)
    try:
        if not robot.wait_for_state(timeout=15.0):
            logging.error("No robot state received.")
            return 4
        q0 = robot.last_q
        if q0 is None:
            logging.error("last_q is None.")
            return 5
        logging.info("Joint state ok shape=%s", getattr(q0, "shape", None))

        cfg_j = get_default_controller_config("JOINT_POSITION")
        robot.move_joints(
            [0.1, 0.35, 0.0, -1.2, 0.0, 1.1, 0.0],
            controller_cfg=cfg_j,
            blocking=True,
            timeout=8.0,
            position_tolerance=0.08,
        )

        cfg_osc = get_default_controller_config("OSC_POSE")
        robot.control("OSC_POSE", [0.0, 0.0, -0.005, 0.0, 0.0, 0.0], controller_cfg=cfg_osc)
        time.sleep(0.2)

        cfg_imp = get_default_controller_config("JOINT_IMPEDANCE")
        robot.control(
            "JOINT_IMPEDANCE",
            [0.0, 0.35, 0.0, -1.2, 0.0, 1.1, 0.0],
            controller_cfg=cfg_imp,
        )
        time.sleep(0.2)
    finally:
        robot.close()

    proc.terminate()
    try:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if err.strip():
            logging.debug("server stderr tail: %s", err[-2000:])
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    logging.info("Smoke finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
