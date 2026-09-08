"""Genesis viewer for the parametric 8-DOF slider/rail model."""

from importlib import import_module

__all__ = [
    "DEFAULT_Q",
    "DEFAULT_RAIL_Y_LIMIT_M",
    "DEFAULT_ROBOT_POS",
    "DEFAULT_SPEC_YAML",
    "DigitalTwinMirror",
    "RailGenesisConfig",
    "RailGenesisScene",
]


def __getattr__(name: str):
    # Module entrypoints need to set CPU/thread budgets before importing
    # NumPy or Genesis. Importing a cloud-protocol helper needs neither scene.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "DEFAULT_SPEC_YAML":
        module = "rm75_control.control.joint_admittance_8dof.param_model.paths"
    elif name == "DigitalTwinMirror":
        module = __name__ + ".twin"
    else:
        module = __name__ + ".scene"
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
