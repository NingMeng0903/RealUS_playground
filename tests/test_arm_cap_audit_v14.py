"""The cap gate must distinguish legal rigid motion from pose deformation."""
import numpy as np
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.cli.audit_consistent_arm_v14 import _cap_shape_metrics


def test_cap_gate_allows_large_rigid_motion_but_rejects_local_shape_change():
    points = np.random.default_rng(104).normal(size=(64, 3)) * .02
    rotation = Rotation.from_euler('xyz', [.7, -.4, 1.2]).as_matrix()
    posed = points @ rotation.T + [.3, -.6, 1.1]
    assert _cap_shape_metrics(points, posed, apply_gate=True)['shape_preservation_passed']
    posed[0] += [.0008, 0, 0]
    measured = _cap_shape_metrics(points, posed, apply_gate=True)
    assert measured['rigid_procrustes_max_m'] > .0001
    assert not measured['shape_preservation_passed']
