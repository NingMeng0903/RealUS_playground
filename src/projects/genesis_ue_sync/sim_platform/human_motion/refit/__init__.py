"""Implicit Kinodynamic Whole-Body Refit interfaces."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AdmittanceWbcOptions": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic.admittance_wbc",
        "AdmittanceWbcOptions",
    ),
    "CollocationOptions": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic.collocation",
        "CollocationOptions",
    ),
    "ImplicitKinodynamicOptions": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic_refit",
        "ImplicitKinodynamicOptions",
    ),
    "ImplicitKinodynamicRefitController": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic_refit",
        "ImplicitKinodynamicRefitController",
    ),
    "ImplicitKinodynamicStep": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.implicit_kinodynamic_refit",
        "ImplicitKinodynamicStep",
    ),
    "SmplRoiProjector": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.smpl_roi",
        "SmplRoiProjector",
    ),
    "SmplRoiSpec": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.smpl_roi",
        "SmplRoiSpec",
    ),
    "VPoserAdapter": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.vposer_adapter",
        "VPoserAdapter",
    ),
    "abdomen_vertex_indices_for_sequence": (
        "projects.genesis_ue_sync.sim_platform.human_motion.refit.smpl_roi",
        "abdomen_vertex_indices_for_sequence",
    ),
}

__all__ = [
    "AdmittanceWbcOptions",
    "CollocationOptions",
    "ImplicitKinodynamicOptions",
    "ImplicitKinodynamicRefitController",
    "ImplicitKinodynamicStep",
    "SmplRoiProjector",
    "SmplRoiSpec",
    "VPoserAdapter",
    "abdomen_vertex_indices_for_sequence",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
