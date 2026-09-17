import numpy as np

from peirastic.contact_qp.mean_confidence_map import contact_us_mask, mean_confidence_map


def test_contact_mask_keeps_force_above_threshold():
    force_t = np.array([0.0, 1.0, 2.0, 3.0])
    force_fz = np.array([0.1, 0.4, 2.0, 4.0])
    us_t = np.array([0.0, 0.5, 1.5, 2.5, 3.5])
    keep = contact_us_mask(us_t, force_t, force_fz, fz_min=0.8)
    np.testing.assert_array_equal(keep, [False, False, True, True, False])


def test_mean_map_is_pixelwise_average():
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.0, 1.0], [0.5, 1.0]])
    np.testing.assert_allclose(mean_confidence_map([a, b]), [[0.5, 0.5], [0.5, 0.75]])
