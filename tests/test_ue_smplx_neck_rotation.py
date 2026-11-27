"""A neck turn must not rotate either collar, arm, or hand chain."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_neck_turn_is_independent_of_both_arm_chains(monkeypatch):
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace(Rotator=lambda *a: a))
    loader = importlib.import_module(
        "projects.genesis_ue_sync.cli.render.unreal.ue_common_scene_loader"
    )
    matrices = []

    def capture(matrix):
        matrices.append(np.array(matrix, copy=True))
        return matrix

    monkeypatch.setattr(loader, "ue_world_rotation_from_genesis", capture)
    monkeypatch.setattr(loader, "ue_rotator_deg_from_matrix", lambda m: (0, 0, 0))
    pose = np.zeros((55, 3))
    pose[12, 2] = np.deg2rad(35.892)
    pose[0] = [0.4, -0.2, 0.7]  # Legacy root replaces this duplicate root.
    root = Rotation.from_euler("xyz", [0.1, 0.2, -0.3]).as_matrix()
    loader._smplx55_component_space_rotators(pose, smpl_root_rot_genesis=root)
    np.testing.assert_allclose(matrices[12], root @ Rotation.from_rotvec(pose[12]).as_matrix())
    np.testing.assert_allclose(matrices[15], matrices[12])  # Head follows neck.
    for joint in (13, 14, 16, 17, 18, 19, 20, 21, *range(25, 55)):
        np.testing.assert_allclose(matrices[joint], root, atol=1e-12)
