#!/usr/bin/env python3
"""Genesis viewer demo for RM75-6F on parametric slider/rail.

Run::

  source env.sh
  python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.paths import DEFAULT_SPEC_YAML
from rm75_control.control.joint_admittance_8dof.viewer.scene import (
    DEFAULT_Q,
    RailGenesisConfig,
    RailGenesisScene,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show-viewer", action="store_true")
    p.add_argument("--seconds", type=float, default=0.0, help="Auto-exit after N s (0 = run until Ctrl+C)")
    p.add_argument("--rail-y", type=float, default=0.0, help="Initial rail_y position (m)")
    p.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_YAML,
        help="Slider/rail YAML spec (default: config/slider_rail.yaml)",
    )
    p.add_argument("--legacy-urdf", action="store_true", help="Use legacy blue-box URDF (no parametric spec)")
    p.add_argument("--no-calib-scene", action="store_true", help="Skip auto-loading camera_calibration bundle")
    p.add_argument("--gravity", action="store_true", help="Enable gravity (default off)")
    return p.parse_args()


def _require_cuda_gpu() -> None:
    from rm75_control.control.joint_admittance_8dof.viewer.cuda_env import (
        ensure_cuda_driver_for_taichi,
    )

    ensure_cuda_driver_for_taichi(require_gpu=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for viewer.demo (default). "
            "Check: nvidia-smi, and install CUDA PyTorch via viewer/install_torch.sh"
        )
    print(f"backend: cuda ({torch.cuda.get_device_name(0)})", flush=True)


def main() -> int:
    args = parse_args()
    try:
        _require_cuda_gpu()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    q = DEFAULT_Q.copy()
    q[0] = float(args.rail_y)
    cfg = RailGenesisConfig(
        backend="cuda",
        show_viewer=args.show_viewer,
        init_q=q,
        gravity=(0.0, 0.0, -9.81) if args.gravity else (0.0, 0.0, 0.0),
        spec_yaml=None if args.legacy_urdf else args.spec,
        load_calib_scene=not args.no_calib_scene,
    )
    scene = RailGenesisScene(cfg)
    try:
        scene.build()
    except ImportError as exc:
        print(
            "Genesis import failed (usually missing PyTorch or genesis-world):\n"
            f"  {exc}\n\n"
            "Install (from repo root, after source env.sh):\n"
            "  bash rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh\n"
            "  pip install -r rm75_control/control/joint_admittance_8dof/viewer/requirements.txt",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(
        f"RM75-6F-8dof: rail_y={scene.rail_y():+.3f} m "
        f"(travel +-{scene._rail_y_limit:.3f} m, spec={args.spec if not args.legacy_urdf else 'legacy'})",
        flush=True,
    )
    if scene._calib_spec is not None:
        bed = scene._calib_spec.bed
        bed_txt = (
            f"size={bed.size[0]:.2f}x{bed.size[1]:.2f}m rot={bed.rotation_deg:.1f}deg"
            if bed is not None
            else "no bed"
        )
        print(
            f"calib scene: {len(scene._calib_spec.camera_ids)} cameras, {bed_txt} "
            f"from {scene._calib_spec.bundle_path}",
            flush=True,
        )
    elif not args.no_calib_scene:
        print("calib scene: bundle not found (see camera_calibration/calibration_results/)", flush=True)
    t0 = time.monotonic()
    try:
        while True:
            scene.step()
            if args.seconds > 0 and (time.monotonic() - t0) >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
