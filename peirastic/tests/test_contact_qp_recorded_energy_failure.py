"""Numerical trigger from 2026-09-11 uncalibrated/003, control sample 1487.

The last fully logged raw wrench is used; the failed row's raw wrench was not
logged. This reproduces the numerical failure, not a closed-loop replay.
"""
import numpy as np
import pytest

from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.runtime_config import load_study_config, calibrated_geometry
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal
from peirastic.contact_qp.types import ContactStatus


@pytest.mark.parametrize('available', [.543, .0001, 0.])
def test_recorded_failure_keeps_energy_row_and_solves_or_refuses(available):
    config = load_study_config('peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml')
    law, _ = build_contact_nominal()
    settings = dict(config['qp'])
    vel = np.asarray(law.controller.cfg.max_velocity).copy()
    acc = np.asarray(law.controller.cfg.max_acceleration).copy()
    vel[2] = min(vel[2], law.controller.cfg.max_vz_tool_m_s)
    vel[4] = min(vel[4], law.tilt.cfg.vmax_rad_s)
    acc[4] = min(acc[4], law.tilt.cfg.a_max)
    settings.update(c_min=.8, quality_policy_version=config['feature']['quality_policy_version'],
                    max_velocity=vel, max_acceleration=acc, angle_limit_rad=law.tilt.cfg.theta_max_rad)
    geometry, _ = calibrated_geometry(config)
    nominal = np.array([.0014366754000970715, .004939262814845449, -.000307383861478358,
                        .00005040561896274133, 0., .0006118759464102322])
    previous = np.array([.00035372356323227035, .0012004395816588766, -.0002881421534556955,
                         .00002961360274241947, 4.954938245177022e-9, .00016270956518239026])
    path = nominal.copy(); path[[2, 4]] = 0
    wrench = -np.array([.6219733239625533, 1.8536964730267598, 4.111339843014162,
                        .104189445381939, -.013351890153415777, .06898028399491514])
    energy = PortEnergyConstraint(wrench, available, .05)
    solver = ContactQp(QpConfig(**settings))
    result = solver.solve(QpInput(geometry, nominal, path, 4.136201260066533,
        .004983089999768708, 4850.778024143, previous_twist=previous, energy=energy,
        acceleration_dt_s=.004983089999768708, measured_angle=.0004029468203759989))
    if available == .543:
        assert result.qp_twist is not None, result.diagnostics
        assert result.diagnostics['iterations'] < solver.config.max_iterations
    if result.qp_twist is not None:
        assert energy.admissible(result.qp_twist)
        assert result.energy_certificate is energy
    else:
        assert result.status in (ContactStatus.TASK_INFEASIBLE, ContactStatus.SOLVER_FAILED)
