"""Deprecated import path — use ``param_model`` and ``viewer`` instead.

Layout::

    joint_admittance_8dof/
      config/slider_rail.yaml   # geometry + world_calib
      param_model/              # parametric URDF generation
      viewer/                   # Genesis scene + digital twin
      genesis/                  # thin re-exports (this package)
"""

from rm75_control.control.joint_admittance_8dof.param_model import (
    ASSETS_DIR,
    DEFAULT_SPEC_YAML,
    DEFAULT_URDF,
    GENERATED_URDF,
    compute_layout,
    generate_urdf,
    load_spec,
    prepare_genesis_urdf,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.viewer import (
    DigitalTwinMirror,
    RailGenesisConfig,
    RailGenesisScene,
)

__all__ = [
    "ASSETS_DIR",
    "DEFAULT_SPEC_YAML",
    "DEFAULT_URDF",
    "DigitalTwinMirror",
    "GENERATED_URDF",
    "RailGenesisConfig",
    "RailGenesisScene",
    "compute_layout",
    "generate_urdf",
    "load_spec",
    "prepare_genesis_urdf",
    "resolve_world_calib",
]
