"""Optional beta-conditioned MJCF via PHC upstream ``SMPL_Robot`` (SMPLSim)."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence, resolve_body_model_dir


def _gender_to_smpl_sim(g: str) -> list[int]:
    s = str(g).strip().lower()
    if s in ("male", "m"):
        return [1]
    if s in ("female", "f"):
        return [2]
    return [0]


def _ensure_smplsim_on_path(smplsim_root: Path) -> None:
    root = str(Path(smplsim_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def try_write_smpl_robot_mjcf(
    sequence: HumanMotionSequence,
    dest_xml: Path,
    *,
    smpl_model_dir: Path | None = None,
    smplsim_root: Path | None = None,
) -> bool:
    """Build ``dest_xml`` with ``smpl_sim.smpllib.smpl_local_robot.SMPL_Robot`` for current ``betas``.

    Uses capsule/hinge MJCF (``mesh=False``), ``upright_start=True`` to stay consistent with
    ``AMONGUS_PHC_UPRIGHT_FIX`` retargeting.

    Returns
    -------
    True if the file was written; False if SMPLSim import/run failed (caller may fall back).
    """

    if str(sequence.model_type).lower() != "smpl":
        warnings.warn("SMPL_Robot MJCF is only implemented for model_type=smpl; skipping.")
        return False

    paths_default = Path(__file__).resolve().parents[5] / "ref_code_library" / "SMPLSim"
    sim_root = Path(smplsim_root or os.environ.get("AMONGUS_SMPLSIM_ROOT", str(paths_default))).expanduser().resolve()
    if not (sim_root / "smpl_sim").is_dir():
        warnings.warn(f"SMPLSim not found at {sim_root}; set AMONGUS_SMPLSIM_ROOT. Skipping SMPL_Robot MJCF.")
        return False

    model_dir = Path(smpl_model_dir or resolve_body_model_dir("smpl")).expanduser().resolve()
    if not model_dir.is_dir():
        warnings.warn(f"SMPL model dir missing: {model_dir}. Skipping SMPL_Robot MJCF.")
        return False

    _ensure_smplsim_on_path(sim_root)

    try:
        from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
    except ImportError as e:
        warnings.warn(f"Could not import smpl_sim ({e}). Install SMPLSim deps (torch, lxml, numpy-stl, mujoco).")
        return False

    try:
        import torch
    except ImportError:
        warnings.warn("torch required for SMPL_Robot MJCF. Skipping.")
        return False

    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "replace_feet": True,
        "remove_toe": False,
        "freeze_hand": False,
        "big_ankle": True,
        "box_body": False,
        "real_weight": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "rel_joint_lm": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }

    try:
        robot = SMPL_Robot(robot_cfg, data_dir=str(model_dir))
        beta = np.asarray(sequence.betas, dtype=np.float32).reshape(-1)[:10]
        if beta.size < 10:
            beta = np.pad(beta, (0, 10 - int(beta.size)))
        betas_t = torch.from_numpy(beta[None, :].copy()).float()
        g = _gender_to_smpl_sim(sequence.gender)
        robot.load_from_skeleton(betas=betas_t, gender=g, objs_info=None)
        dest_xml = Path(dest_xml)
        dest_xml.parent.mkdir(parents=True, exist_ok=True)
        robot.write_xml(str(dest_xml))
    except Exception as e:
        warnings.warn(f"SMPL_Robot.write_xml failed: {e}")
        return False

    return dest_xml.is_file()
