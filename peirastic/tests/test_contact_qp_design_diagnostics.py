import numpy as np
from peirastic.apps.contact_qp_design_diagnostics import rocking_interval, summary


def test_two_endpoint_force_priority_needs_normal_headroom():
    # At fixed vn, same-sign endpoint priority forbids any change in rocking.
    assert np.allclose(rocking_interval((0,0),(-1,1),force_sign=1),[0,0])
    # .5 mm/s retract headroom allows +/- .02 rad/s at a=25 mm.
    assert np.allclose(rocking_interval((-.0005,0),(-1,1),force_sign=1),[-.02,.02])
    assert np.allclose(rocking_interval((0,.0005),(-1,1),force_sign=-1),[-.02,.02])


def test_aperture_and_mechanical_limits_are_intersected():
    assert np.allclose(rocking_interval((0,0),(-1,1),aperture=.00075),[-.03,.03])
    assert np.allclose(rocking_interval((0,0),(-.01,.02),aperture=.00075),[-.01,.02])
    assert rocking_interval((.1,.2),(-1,1),force_sign=1,aperture=.00075) is None


def test_signed_error_is_not_replaced_by_absolute_bias():
    assert summary([-.1,-.2])['mean'] < 0
    assert summary([.1,.2])['mean'] > 0
    assert summary([])=={'count':0}
