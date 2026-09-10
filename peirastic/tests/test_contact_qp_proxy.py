"""The production signature-based runner must reach the child through ProxyOuter."""
import inspect

import numpy as np

from peirastic.realman8dof.session import ProxyOuter


def test_proxy_explicitly_advertises_and_forwards_execution_observations():
    captured = {}

    class Child:
        def sample(self, t, pose, force, *, slack_norm=None, measured_twist_base=None,
                   measurement_time_s=None, measurement_id=None,
                   feedback_velocity_valid=None, measured_twist_valid=None,
                   measured_twist_fresh=None, measured_twist_metadata=None):
            captured.update(slack=slack_norm, velocity=measured_twist_base,
                            time=measurement_time_s, seq=measurement_id,
                            valid=feedback_velocity_valid, full_valid=measured_twist_valid,
                            fresh=measured_twist_fresh, metadata=measured_twist_metadata)
            return np.zeros(6)

    proxy = ProxyOuter(Child())
    names = inspect.signature(proxy.sample).parameters
    available = dict(slack_norm=.3, measured_twist_base=np.arange(6.),
                     measurement_time_s=5., measurement_id=4, feedback_velocity_valid=True,
                     measured_twist_valid=False, measured_twist_fresh=False,
                     measured_twist_metadata={"reason": "rail_stale", "port_verified": False})
    kwargs = {k: v for k, v in available.items() if k in names}
    proxy.sample(0., np.zeros(6), np.zeros(6), **kwargs)
    assert captured["slack"] == .3 and captured["seq"] == 4
    assert captured["time"] == 5. and captured["valid"]
    assert captured["full_valid"] is False and captured["fresh"] is False
    assert captured["metadata"] == available["measured_twist_metadata"]
    np.testing.assert_array_equal(captured["velocity"], np.arange(6.))
