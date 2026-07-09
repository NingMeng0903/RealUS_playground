"""Parametric slider/rail URDF generation and world calibration."""

from rm75_control.control.joint_admittance_8dof.param_model.generator import (
    DEFAULT_SPEC,
    SliderRailSpecError,
    build_urdf_string,
    compute_layout,
    generate_urdf,
    load_spec,
)
from rm75_control.control.joint_admittance_8dof.param_model.paths import (
    ASSETS_DIR,
    DEFAULT_SPEC_YAML,
    DEFAULT_URDF,
    GENERATED_URDF,
)
from rm75_control.control.joint_admittance_8dof.param_model.placement import (
    base_offset_in_rail_base,
    entity_pose_from_calib,
    resolve_world_calib,
)
from rm75_control.control.joint_admittance_8dof.param_model.urdf_prepare import (
    package_assets_dir,
    prepare_genesis_urdf,
)

__all__ = [
    "ASSETS_DIR",
    "DEFAULT_SPEC",
    "DEFAULT_SPEC_YAML",
    "DEFAULT_URDF",
    "GENERATED_URDF",
    "SliderRailSpecError",
    "base_offset_in_rail_base",
    "build_urdf_string",
    "compute_layout",
    "entity_pose_from_calib",
    "generate_urdf",
    "load_spec",
    "package_assets_dir",
    "prepare_genesis_urdf",
    "resolve_world_calib",
]
