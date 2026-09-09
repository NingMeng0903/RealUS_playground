"""Retired activation requests must fail before a controller can run."""

import pytest

from peirastic.realman8dof.force.config import build_force_controller
from rm75_control.control.admittance_common.controller import AdmittanceConfig


RETIRED = (
    "force_dob",
    "energy_tank",
    "cdyob",
    "force_corridor",
    "surface_force_modulation",
)


@pytest.mark.parametrize("name", RETIRED)
@pytest.mark.parametrize("section", (None, "controller", "hybrid_motion"))
def test_retired_activation_is_rejected(name, section):
    block = {name: {"enabled": True}}
    raw = block if section is None else {section: block}
    with pytest.raises(ValueError, match=f"{name} has been removed"):
        AdmittanceConfig.from_dict(raw)


@pytest.mark.parametrize("name", RETIRED)
@pytest.mark.parametrize("mode", ("off", False, None))
def test_old_disabled_settings_do_not_restore_retired_state(name, mode):
    cfg = AdmittanceConfig.from_dict(
        {"hybrid_motion": {name: {"enabled": False, "mode": mode}}}
    )
    assert not hasattr(cfg, name)


@pytest.mark.parametrize("mode", ("shadow", "active"))
def test_cdyob_mode_activation_is_rejected(mode):
    with pytest.raises(ValueError, match="cdyob has been removed"):
        AdmittanceConfig.from_dict({"hybrid_motion": {"cdyob": {"mode": mode}}})


@pytest.mark.parametrize("name", RETIRED)
@pytest.mark.parametrize("nested", (False, True))
def test_mode_payload_cannot_reactivate_retired_mechanism(name, nested):
    block = {name: {"enabled": True}}
    payload = {"hybrid_motion": block} if nested else block
    with pytest.raises(ValueError, match=f"{name} has been removed"):
        build_force_controller(0.005, payload=payload)
