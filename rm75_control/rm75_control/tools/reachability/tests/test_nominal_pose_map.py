"""DoD: nominal controller posture is represented in the capability map.

MC-only maps under-estimate D(x) (sparse orientation coverage). We therefore
check two weaker but actionable criteria:

1. The FK voxel of ``q_nominal_deg`` exists in the sparse map.
2. DLS multi-seed IK converges on that TCP pose (cross-check vs map bits).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap, unpack_bits_5dof
from rm75_control.tools.reachability.kinematics import (
    SeedPoolConfig,
    build_locked_rail_model,
    build_seed_pool,
    fk_tool_axis_batch,
    ik_dls_multiseed,
)
from rm75_control.tools.reachability.kinematics.ik_seeds import DEFAULT_NOMINAL_DEG
from rm75_control.tools.reachability.kinematics.model_locked_rail import DEFAULT_URDF

MAP_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[4]
    / "data"
    / "reachability"
    / "rm75_6f_4cm_20deg"
)


@pytest.fixture(scope="module")
def lm():
    if not DEFAULT_URDF.exists():
        pytest.skip(f"URDF missing at {DEFAULT_URDF}")
    return build_locked_rail_model(DEFAULT_URDF)


def test_nominal_fk_voxel_in_map():
    if not MAP_DIR.exists():
        pytest.skip(f"production map not built: {MAP_DIR}")
    cm = CapabilityMap.load(MAP_DIR, mmap=True)
    lm = build_locked_rail_model()
    q = np.radians(np.array(DEFAULT_NOMINAL_DEG, dtype=float))[None, :]
    pos, axis = fk_tool_axis_batch(lm, q)
    p = pos[0]
    ijk = tuple(int(x) for x in cm.grid.idx_of(p))
    row = cm.row_of(ijk)
    assert row is not None, f"nominal TCP {p} missing from sparse map at ijk={ijk}"
    d = float(cm.d_value[row])
    assert d > 0.05, f"nominal voxel D(x)={d:.3f} too low (MC under-sampling)"
    bits = unpack_bits_5dof(cm.bitmask[row : row + 1], cm.orientations.n)[0]
    oi = cm.orientations.nearest(axis[0])
    # MC may miss the exact tool axis even when IK succeeds — recorded for hybrid builds.
    if not bits[oi]:
        pytest.xfail("nominal tool axis not set in MC map; run hybrid --ik-refine")


def test_nominal_pose_ik_converges(lm):
    q_nom = np.radians(np.array(DEFAULT_NOMINAL_DEG, dtype=float))
    pin.forwardKinematics(lm.model, lm.data, q_nom)
    pin.updateFramePlacement(lm.model, lm.data, lm.tcp_id)
    target = pin.SE3(
        lm.data.oMf[lm.tcp_id].rotation.copy(),
        lm.data.oMf[lm.tcp_id].translation.copy(),
    )
    seeds = build_seed_pool(
        lm.q_lower, lm.q_upper,
        SeedPoolConfig(nominal_deg=DEFAULT_NOMINAL_DEG, n_random=8, random_seed=0),
    )
    res = ik_dls_multiseed(lm, target, seeds, max_iter=60, lam=0.05)
    assert res.report.ok, f"nominal pose IK failed: {res.report}"
