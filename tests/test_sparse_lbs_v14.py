import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import _weighted_rest_correction
from projects.genesis_ue_sync.anatomy_retarget.sparse_lbs_v14 import SparseLBSV14


@pytest.mark.parametrize('dense', [False, True])
def test_sparse_lbs_preserves_original_fourteen_slot_reduction_exactly(dense):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(211, 3))
    indices = rng.integers(0, 17, (211, 14))  # Repeated controllers are legal.
    weights = rng.uniform(size=indices.shape)
    if not dense:
        weights[rng.uniform(size=indices.shape) < .8] = 0
    weights[:, 6] += 1
    weights /= weights.sum(1)[:, None]
    transforms = rng.normal(size=(17, 4, 4))
    evaluator = SparseLBSV14(points, indices, weights)
    expected = _weighted_rest_correction(points, indices, weights, transforms)
    np.testing.assert_array_equal(evaluator(transforms), expected)
    # The layout must not alias mutable caller arrays.
    points[:] = 0
    weights[:] = 0
    indices[:] = 0
    np.testing.assert_array_equal(evaluator(transforms), expected)
    transforms[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        evaluator(transforms)
