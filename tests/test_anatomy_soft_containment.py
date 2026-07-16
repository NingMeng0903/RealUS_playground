from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import _incremental_harmonic_field


def test_incremental_harmonic_field_reaches_boundary_without_flips() -> None:
    nodes = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.2, 0.2)),
        dtype=np.float64,
    )
    elements = np.asarray(((0, 1, 2, 4), (0, 1, 4, 3), (0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32)
    boundary = np.asarray((0, 1, 2, 3), dtype=np.int32)
    values = np.asarray(((0, 0, 0), (0.05, 0, 0), (0, 0.03, 0), (0, 0, -0.02)), dtype=np.float64)
    field, report = _incremental_harmonic_field(nodes, elements, boundary, values)
    np.testing.assert_allclose(field[boundary], values, atol=1.0e-7)
    assert report["inverted_tetrahedra"] == 0
    assert report["minimum_jacobian_ratio"] > 0.0


def test_incremental_harmonic_field_rejects_unreachable_inverted_target() -> None:
    nodes = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    elements = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    boundary = np.arange(4, dtype=np.int32)
    values = np.zeros((4, 3), dtype=np.float64)
    values[3, 2] = -2.0
    with pytest.raises(RuntimeError, match="cannot avoid tetrahedron inversion"):
        _incremental_harmonic_field(nodes, elements, boundary, values)
