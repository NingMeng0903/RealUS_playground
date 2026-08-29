"""Certificate 1 filter: 20 Hz first-order, not 10 Hz second-order."""

from __future__ import annotations

from rm75_control.control.admittance_common.controller import AdmittanceConfig
from rm75_control.control.admittance_common.observer import ForceObserverConfig


def test_observer_default_is_20hz_first_order() -> None:
    cfg = ForceObserverConfig()
    assert cfg.causal_fc_hz == 20.0
    assert cfg.causal_order == 1


def test_dimeas_detector_cutoff_is_below_limit_cycle() -> None:
    assert AdmittanceConfig().var_damping_omega_c_hz == 1.2
    assert AdmittanceConfig().var_damping_omega_c_hz < 1.5
