"""Polyline + contact_qp wrap and FiniteInterval duration_s callable."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.api.payloads import HfpcPayload
from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.contact_qp.runtime_config import load_study_config
from peirastic.core.ipc import PAYLOAD_MAX
from peirastic.core.modes import Mode, ModeRequest
from peirastic.DEMO.phathom_scanning.comparison_run import (
    DEFAULT_CONTACT_QP,
    comparison_hfpc_options,
    write_hfpc_polyline,
)
from peirastic.realman8dof.force.torque_tilt import LegacyForceWithTilt
from peirastic.realman8dof.modes.contact_qp import wrap_study_phase
from peirastic.realman8dof.modes.track import _hybrid_force_law
from peirastic.realman8dof.session import _polyline_ref
from rm75_control.control.joint_admittance_8dof.loop import Phase
from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference


class DummyOuter:
    def sample(self, t_s, current_pose, f_ext, **kwargs):
        del t_s, current_pose, f_ext, kwargs
        return np.zeros(6)


def test_finite_interval_accepts_polyline_duration_method():
    points = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    reference = WorldPolylineReference(points, rpy=np.zeros((2, 3)), speed_m_s=0.02, soft_start=False)
    assert callable(reference.duration_s)
    adapter = FiniteIntervalReference(reference, dt_s=0.005)
    assert adapter.duration_s == pytest.approx(reference.duration_s())
    assert adapter.duration_s == pytest.approx(5.0)


def test_wrap_study_phase_accepts_polyline(tmp_path):
    phase = wrap_study_phase(
        Phase(outer=DummyOuter(), label="compare"),
        {
            "reference": "polyline",
            "contact_qp": {"mode": "baseline", "log_path": str(tmp_path / "contact_qp.jsonl")},
        },
        SimpleNamespace(inner=SimpleNamespace(kin=None)),
    )
    assert hasattr(phase.outer, "sample")
    phase.on_exit()


def test_wrap_study_phase_still_rejects_hold():
    with pytest.raises(ValueError, match="icra_path or polyline"):
        wrap_study_phase(
            Phase(outer=DummyOuter(), label="bad"),
            {"reference": "hold", "contact_qp": {"mode": "baseline"}},
            SimpleNamespace(),
        )


def test_embedded_lissajous_ultrapoc_exceeds_mailbox():
    poses = np.linspace(0.2, 0.4, 115 * 6).reshape(115, 6).tolist()
    config = load_study_config(DEFAULT_CONTACT_QP)
    config["log_path"] = "/tmp/contact_qp.jsonl"
    payload = HfpcPayload(
        reference="polyline",
        poses=poses,
        law="contact_qp",
        contact_qp=config,
        speed_m_s=0.01,
        force=4.0,
        label="phathom_s_scan",
    ).to_json()
    blob = json.dumps(ModeRequest(Mode.TRACK_HYBRID, payload).to_json(), separators=(",", ":"))
    assert len(blob.encode("utf-8")) > PAYLOAD_MAX


def test_comparison_hfpc_uses_paths_and_fits_mailbox(tmp_path):
    poses = np.linspace(0.2, 0.4, 115 * 6).reshape(115, 6)
    options = comparison_hfpc_options(
        "ultrapoc", force=4.0, run_dir=tmp_path, config_path=DEFAULT_CONTACT_QP,
    )
    payload = HfpcPayload(
        reference="polyline",
        plan_path=str(write_hfpc_polyline(tmp_path, poses)),
        law=options["law"],
        contact_qp=options["contact_qp"],
        speed_m_s=0.01,
        force=4.0,
        label="phathom_s_scan",
        extra=options["extra"],
    ).to_json()
    blob = json.dumps(ModeRequest(Mode.TRACK_HYBRID, payload).to_json(), separators=(",", ":"))
    assert "poses" not in payload
    assert payload["plan_path"].endswith("hfpc_polyline.json")
    assert isinstance(payload["contact_qp"], str)
    assert len(blob.encode("utf-8")) < PAYLOAD_MAX
    ref = _polyline_ref(payload, "xyz")
    assert ref.points.shape == (115, 3)


def test_contact_qp_keeps_tilt_when_demo_mask_claims_omega_y():
    law, f_des, tilt_cfg = _hybrid_force_law(
        0.005,
        {
            "law": "contact_qp",
            "contact_qp": {"mode": "active"},
            "force_axes": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "desired_z": 4.0,
        },
        control_frame="tool",
    )
    assert tilt_cfg.enabled
    assert isinstance(law, LegacyForceWithTilt)
    assert f_des[2] == pytest.approx(4.0)


def test_demo_z_only_mask_still_disables_tilt_without_contact_qp():
    law, _f_des, tilt_cfg = _hybrid_force_law(
        0.005,
        {"force_axes": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "desired_z": 4.0},
        control_frame="tool",
    )
    assert not tilt_cfg.enabled
    assert not isinstance(law, LegacyForceWithTilt)
