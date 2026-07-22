"""Parametric slider/rail URDF generator for the RM75 8-DOF base.

Reads a YAML (or dict) spec and emits a Genesis-loadable URDF whose kinematic
tree is::

    rail_base (root, fixed in world)
     |-- frame_link   (fixed)     -> frame
     |-- rail_link    (fixed)     -> rail assembly (deck + tracks + end plates)
     |     |-- slider_link (prismatic rail_y, axis Y) -> slider; driven DOF #0
     |           |-- base_link (fixed arm_mount) -> RM75 arm links (verbatim)

Design notes
------------
- Every physical part is one URDF link; the rail assembly carries several
  colored ``<box>`` visuals (URDF allows multiple ``<visual>`` per link).
- The slider is a visual on ``slider_link`` so it translates with ``rail_y``.
- The arm block (``base_link`` .. ``tcp``) is copied verbatim from
  ``RM75-6F-8dof.genesis.urdf`` so FK / WBC joint origins are byte-identical;
  only the ``arm_mount`` origin (base coordinate on the slider top) is set here.
- Model Z origin is ``rail_base`` (frame bottom).  ``rail_link`` frame sits at the
  rail module floor (top of frame / rail bottom).  ``slider_link`` frame sits at
  the slider top center; ``arm_mount`` is identity offset on that plane.
- Rail long axis = Y (prismatic travel). "Outer side" = +X: the arm is offset
  toward +X and flush with the rail +X face; the frame protrudes toward -X.

The generator is pure text (no numpy / Genesis import) so its output can be
parsed and asserted in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Arm block copied verbatim from RM75-6F-8dof.genesis.urdf (base_link .. tcp).
# joint_N / link_N origins and inertials MUST match the WBC URDF (verified FK).
# Only the parent `arm_mount` joint (generated below) positions this block.
_ARM_BLOCK = """  <link name="base_link">
    <inertial>
      <origin xyz="0.00049987 5.2709E-05 0.060019" rpy="0 0 0" />
      <mass value="1.862" />
      <inertia ixx="0.0017232" ixy="-3.1058E-06" ixz="-3.7924E-05"
               iyy="0.0017051" iyz="1.3691E-06" izz="0.00090158" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/base_link.dae" />
      </geometry>
    </visual>
  </link>
  <link name="link_1">
    <inertial>
      <origin xyz="0.000241 -0.013273 -0.00995" rpy="0 0 0" />
      <mass value="1.574" />
      <inertia ixx="0.002487573" ixy="0.000009663" ixz="-0.000007909"
               iyy="0.002321038" iyz="0.000179393" izz="0.001450554" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_1.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_1" type="revolute">
    <origin xyz="0 0 0.2405" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="link_1" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="60" velocity="3.14" />
  </joint>
  <link name="link_2">
    <inertial>
      <origin xyz="-0.000357 -0.106789 0.005329" rpy="0 0 0" />
      <mass value="1.217" />
      <inertia ixx="0.003494121" ixy="0.000002921" ixz="-0.000005613"
               iyy="0.000892721" iyz="-0.000583884" izz="0.003444080" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_2.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_2" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_1" />
    <child link="link_2" />
    <axis xyz="0 0 1" />
    <limit lower="-2.2689" upper="2.2689" effort="60" velocity="3.14" />
  </joint>
  <link name="link_3">
    <inertial>
      <origin xyz="0.000003 -0.01398 -0.011324" rpy="0 0 0" />
      <mass value="1.11" />
      <inertia ixx="0.001836663" ixy="0.000002259" ixz="-0.000004216"
               iyy="0.001498875" iyz="0.000037167" izz="0.001062545" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_3.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_3" type="revolute">
    <origin xyz="0 -0.256 0" rpy="1.5708 0 0" />
    <parent link="link_2" />
    <child link="link_3" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="30" velocity="3.14" />
  </joint>
  <link name="link_4">
    <inertial>
      <origin xyz="-0.000005 -0.084658 0.004747" rpy="0 0 0" />
      <mass value="0.685" />
      <inertia ixx="0.001282444" ixy="-0.000000551" ixz="-0.000000630"
               iyy="0.000373013" iyz="-0.000232084" izz="0.001256177" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_4.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_4" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_3" />
    <child link="link_4" />
    <axis xyz="0 0 1" />
    <limit lower="-2.356" upper="2.356" effort="30" velocity="3.14" />
  </joint>
  <link name="link_5">
    <inertial>
      <origin xyz="0.000078 -0.012937 -0.008781" rpy="0 0 0" />
      <mass value="0.619" />
      <inertia ixx="0.000627336" ixy="0.000001636" ixz="-0.000001345"
               iyy="0.000542455" iyz="0.000034970" izz="0.000370291" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_5.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_5" type="revolute">
    <origin xyz="0 -0.21 0" rpy="1.5708 0 0" />
    <parent link="link_4" />
    <child link="link_5" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="10" velocity="3.14" />
  </joint>
  <link name="link_6">
    <inertial>
      <origin xyz="-0.000014 -0.078524 0.002819" rpy="0 0 0" />
      <mass value="0.602" />
      <inertia ixx="0.000780774" ixy="-0.000000121" ixz="-0.000000469"
               iyy="0.000289973" iyz="-0.000120513" izz="0.000763955" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_6.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_6" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_5" />
    <child link="link_6" />
    <axis xyz="0 0 1" />
    <limit lower="-2.234" upper="2.234" effort="10" velocity="3.14" />
  </joint>
  <link name="link_7">
    <inertial>
      <origin xyz="0.001094 -0.000077 -0.010119" rpy="0 0 0" />
      <mass value="0.144" />
      <inertia ixx="0.000044123" ixy="-0.000000064" ixz="0.0000003"
               iyy="0.000035078" iyz="-0.000000029" izz="0.000065445" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/link_7.dae" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_7" type="revolute">
    <origin xyz="0 -0.1612 0" rpy="1.5708 0 0" />
    <parent link="link_6" />
    <child link="link_7" />
    <axis xyz="0 0 1" />
    <limit lower="-6.28" upper="6.28" effort="10" velocity="3.14" />
  </joint>
  <link name="tcp" />
  <joint name="link_7_to_tcp" type="fixed">
    <origin xyz="0 0 0.220" rpy="0 0 0" />
    <parent link="link_7" />
    <child link="tcp" />
  </joint>
"""

from rm75_control.control.joint_admittance_8dof.param_model.paths import DEFAULT_SPEC_YAML

DEFAULT_SPEC: dict[str, Any] = {
    "arm": "rm75",
    "rail": {
        "effective_travel_mm": 360.0,
        "end_overhead_mm": 0.0,
        "width_mm": 150.0,
        "base_plate_thickness_mm": 12.0,
        "track_height_mm": 18.0,
        "track_width_mm": 22.0,
        "track_gap_mm": 66.0,
        "side_plate_thickness_mm": 14.0,
        "side_plate_height_mm": 80.0,
    },
    "slider": {
        "width_mm": 160.0,
        "length_mm": 170.0,
        "top_to_rail_bottom_mm": 66.0,
    },
    "frame": {
        "height_mm": 40.0,
        "width_mm": 220.0,
    },
    "arm_mount": {
        "offset_x_mm": 40.0,
        "offset_y_mm": 0.0,
    },
    "world_calib": {
        "base_pos_m": [0.0, 0.0, 0.266],
        "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
    "colors": {
        "frame": [0.75, 0.75, 0.78, 1.0],
        "rail_metal": [0.60, 0.62, 0.65, 1.0],
        "dark": [0.18, 0.18, 0.20, 1.0],
    },
}


class SliderRailSpecError(ValueError):
    """Raised when a slider/rail spec is malformed."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_spec(spec: dict | str | Path) -> dict[str, Any]:
    """Return a full spec dict (defaults merged) from a dict, YAML path, or str."""
    if isinstance(spec, (str, Path)):
        import yaml

        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}\n"
                f"  slider_rail.yaml: {DEFAULT_SPEC_YAML}\n"
                "  Omit --spec for the default, or pass:\n"
                "    --spec rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml"
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif isinstance(spec, dict):
        raw = spec
    else:  # pragma: no cover - defensive
        raise SliderRailSpecError(f"unsupported spec type: {type(spec)!r}")
    # Allow either the bare body or a {"slider_rail": {...}} wrapper.
    body = raw.get("slider_rail", raw) if isinstance(raw, dict) else {}
    if not isinstance(body, dict):
        raise SliderRailSpecError("slider_rail spec must be a mapping")
    return _deep_merge(DEFAULT_SPEC, body)


def _rgba(color: Any) -> str:
    vals = [float(x) for x in color]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        raise SliderRailSpecError(f"color must have 3 or 4 components, got {color!r}")
    return " ".join(f"{v:.6f}" for v in vals)


def _link_inertial(
    *,
    mass: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    indent: str = "    ",
) -> str:
    ox, oy, oz = origin
    return (
        f"{indent}<inertial>\n"
        f'{indent}  <origin xyz="{ox:.6f} {oy:.6f} {oz:.6f}" rpy="0 0 0" />\n'
        f'{indent}  <mass value="{mass:.3f}" />\n'
        f'{indent}  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />\n'
        f"{indent}</inertial>\n"
    )


def _box_visual(
    *,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    mat_name: str,
    rgba: str,
    indent: str = "    ",
) -> str:
    sx, sy, sz = size
    cx, cy, cz = center
    return (
        f'{indent}<visual>\n'
        f'{indent}  <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0" />\n'
        f'{indent}  <geometry>\n'
        f'{indent}    <box size="{sx:.6f} {sy:.6f} {sz:.6f}" />\n'
        f'{indent}  </geometry>\n'
        f'{indent}  <material name="{mat_name}">\n'
        f'{indent}    <color rgba="{rgba}" />\n'
        f'{indent}  </material>\n'
        f'{indent}</visual>\n'
    )


def compute_layout(spec: dict[str, Any]) -> dict[str, float]:
    """Resolve derived geometry (meters). Model Z origin = frame bottom.

    Rail length is sized so the slider is **flush** with each end-plate *inner*
    face at the travel limits (``rail_y=0`` and ``rail_y=travel``)::

        rail_len_y = travel + slider_length + 2 * side_plate_thickness
                     + end_overhead   # optional extra total gap (0 = exact flush)

    Joint origin is at the slider-center pose for ``rail_y = 0``.
    """
    full = load_spec(spec) if "rail" not in spec else spec
    rail = full["rail"]
    slider = full["slider"]
    frame = full["frame"]

    m = 1e-3  # mm -> m

    travel = float(rail["effective_travel_mm"]) * m
    # Optional *extra* total length beyond the flush fit (split equally both ends).
    end_extra = float(rail.get("end_overhead_mm", 0.0)) * m
    rail_w = float(rail["width_mm"]) * m
    base_t = float(rail["base_plate_thickness_mm"]) * m
    track_h = float(rail["track_height_mm"]) * m
    track_w = float(rail["track_width_mm"]) * m
    track_gap = float(rail["track_gap_mm"]) * m
    side_t = float(rail["side_plate_thickness_mm"]) * m
    side_h = float(rail["side_plate_height_mm"]) * m

    frame_h = float(frame["height_mm"]) * m
    frame_w = float(frame["width_mm"]) * m

    slider_w = float(slider["width_mm"]) * m
    slider_l = float(slider["length_mm"]) * m
    top_to_rail_bottom = float(slider["top_to_rail_bottom_mm"]) * m

    # Flush fit: slider face against end-plate inner face at both travel limits.
    rail_len_y = travel + slider_l + 2.0 * side_t + end_extra
    clearance_each = 0.5 * end_extra
    # Slider-center Y in rail_link when rail_y = 0.
    rail_y_origin_y = -0.5 * rail_len_y + side_t + clearance_each + 0.5 * slider_l

    rail_bottom_z = frame_h
    base_plate_top_z = rail_bottom_z + base_t
    track_top_z = base_plate_top_z + track_h
    slider_top_z = rail_bottom_z + top_to_rail_bottom
    slider_h = slider_top_z - track_top_z
    if slider_h <= 0.0:
        raise SliderRailSpecError(
            "slider.top_to_rail_bottom_mm must exceed base_plate + track height "
            f"({(base_t + track_h) * 1e3:.1f} mm); got {top_to_rail_bottom * 1e3:.1f} mm"
        )

    return {
        "m": m,
        "travel": travel,
        "rail_len_y": rail_len_y,
        "rail_y_origin_y": rail_y_origin_y,
        "end_extra": end_extra,
        "rail_w": rail_w,
        "base_t": base_t,
        "track_h": track_h,
        "track_w": track_w,
        "track_gap": track_gap,
        "side_t": side_t,
        "side_h": side_h,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "slider_w": slider_w,
        "slider_l": slider_l,
        "slider_h": slider_h,
        "rail_bottom_z": rail_bottom_z,
        "base_plate_top_z": base_plate_top_z,
        "track_top_z": track_top_z,
        "slider_top_z": slider_top_z,
        "top_to_rail_bottom": top_to_rail_bottom,
        "rail_plus_x_face": rail_w / 2.0,
    }


def build_urdf_string(spec: dict | str | Path) -> str:
    full = load_spec(spec)
    lay = compute_layout(full)
    colors = full["colors"]
    arm_mount = full["arm_mount"]
    m = lay["m"]

    frame_rgba = _rgba(colors["frame"])
    metal_rgba = _rgba(colors["rail_metal"])
    dark_rgba = _rgba(colors["dark"])

    # Frame: +X face flush with rail +X face -> center shifts toward -X.
    frame_cx = lay["rail_plus_x_face"] - lay["frame_w"] / 2.0
    frame_cz = lay["frame_h"] / 2.0
    frame_visual = _box_visual(
        size=(lay["frame_w"], lay["rail_len_y"], lay["frame_h"]),
        center=(frame_cx, 0.0, frame_cz),
        mat_name="frame_mat",
        rgba=frame_rgba,
    )

    # Deck + tracks fit between end plates (no overlap with black end caps).
    # rail_link frame origin = rail module floor (top of frame); visuals are local Z.
    deck_len_y = lay["rail_len_y"] - 2.0 * lay["side_t"]
    base_plate = _box_visual(
        size=(lay["rail_w"], deck_len_y, lay["base_t"]),
        center=(0.0, 0.0, lay["base_t"] / 2.0),
        mat_name="rail_metal_mat",
        rgba=metal_rgba,
    )
    track_cz = lay["base_t"] + lay["track_h"] / 2.0
    track_x = lay["track_gap"] / 2.0
    track_len_y = deck_len_y
    track_l = _box_visual(
        size=(lay["track_w"], track_len_y, lay["track_h"]),
        center=(-track_x, 0.0, track_cz),
        mat_name="rail_metal_mat",
        rgba=metal_rgba,
    )
    track_r = _box_visual(
        size=(lay["track_w"], track_len_y, lay["track_h"]),
        center=(track_x, 0.0, track_cz),
        mat_name="rail_metal_mat",
        rgba=metal_rgba,
    )
    # Black END plates at the two Y-axis ends (local Z from rail bottom).
    side_cz = lay["side_h"] / 2.0
    side_y = lay["rail_len_y"] / 2.0 - lay["side_t"] / 2.0
    side_l = _box_visual(
        size=(lay["rail_w"], lay["side_t"], lay["side_h"]),
        center=(0.0, -side_y, side_cz),
        mat_name="rail_dark_mat",
        rgba=dark_rgba,
    )
    side_r = _box_visual(
        size=(lay["rail_w"], lay["side_t"], lay["side_h"]),
        center=(0.0, side_y, side_cz),
        mat_name="rail_dark_mat",
        rgba=dark_rgba,
    )

    # Slider: link frame at top-center; box hangs below the mounting plane.
    slider_visual = _box_visual(
        size=(lay["slider_w"], lay["slider_l"], lay["slider_h"]),
        center=(0.0, 0.0, -lay["slider_h"] / 2.0),
        mat_name="slider_mat",
        rgba=dark_rgba,
    )

    rail_y_lower = 0.0
    rail_y_upper = lay["travel"]
    rail_y_origin_y = lay["rail_y_origin_y"]
    arm_x = float(arm_mount["offset_x_mm"]) * m
    arm_y = float(arm_mount["offset_y_mm"]) * m
    rail_mount_z = lay["rail_bottom_z"]
    slider_joint_z = lay["top_to_rail_bottom"]

    header = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- GENERATED by slider_rail_gen.py - do not edit by hand.\n"
        "     Parametric slider/rail + RM75 8-DOF arm. Rail travel = Y, Z up.\n"
        "     rail_y = 0 at -Y end, rail_y = travel at +Y end (0..travel_m).\n"
        "     Model Z origin = rail_base (frame bottom).\n"
        "     rail_link frame = rail module floor; slider_link frame = slider top center.\n"
        "     Arm block (base_link..tcp) is verbatim from RM75-6F-8dof.genesis.urdf. -->\n"
        '<robot name="RM75-6F-8dof-slider">\n'
    )

    rail_base = (
        '  <link name="rail_base">\n'
        "    <inertial>\n"
        '      <origin xyz="0 0 0" rpy="0 0 0" />\n'
        '      <mass value="5.0" />\n'
        '      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05" />\n'
        "    </inertial>\n"
        "  </link>\n"
    )

    frame_link = (
        '  <link name="frame_link">\n'
        f"{_link_inertial(mass=8.0)}"
        f"{frame_visual}"
        "  </link>\n"
        '  <joint name="frame_mount" type="fixed">\n'
        '    <origin xyz="0 0 0" rpy="0 0 0" />\n'
        '    <parent link="rail_base" />\n'
        '    <child link="frame_link" />\n'
        "  </joint>\n"
    )

    rail_link = (
        '  <link name="rail_link">\n'
        f"{_link_inertial(mass=6.0, origin=(0.0, 0.0, track_cz))}"
        f"{base_plate}{track_l}{track_r}{side_l}{side_r}"
        "  </link>\n"
        '  <joint name="rail_mount" type="fixed">\n'
        f'    <origin xyz="0 0 {rail_mount_z:.6f}" rpy="0 0 0" />\n'
        '    <parent link="rail_base" />\n'
        '    <child link="rail_link" />\n'
        "  </joint>\n"
    )

    slider_link = (
        '  <link name="slider_link">\n'
        f"{_link_inertial(mass=2.0, origin=(0.0, 0.0, -lay['slider_h'] / 2.0))}"
        f"{slider_visual}"
        "  </link>\n"
        '  <joint name="rail_y" type="prismatic">\n'
        f'    <origin xyz="0 {rail_y_origin_y:.6f} {slider_joint_z:.6f}" rpy="0 0 0" />\n'
        '    <parent link="rail_link" />\n'
        '    <child link="slider_link" />\n'
        '    <axis xyz="0 1 0" />\n'
        f'    <limit lower="{rail_y_lower:.6f}" upper="{rail_y_upper:.6f}" '
        'velocity="0.20" effort="500" />\n'
        "  </joint>\n"
    )

    arm_mount_joint = (
        '  <joint name="arm_mount" type="fixed">\n'
        f'    <origin xyz="{arm_x:.6f} {arm_y:.6f} 0.000000" rpy="0 0 0" />\n'
        '    <parent link="slider_link" />\n'
        '    <child link="base_link" />\n'
        "  </joint>\n"
    )

    return (
        header
        + rail_base
        + frame_link
        + rail_link
        + slider_link
        + arm_mount_joint
        + _ARM_BLOCK
        + "</robot>\n"
    )


def generate_urdf(spec: dict | str | Path, out_path: str | Path) -> Path:
    """Write the slider/rail URDF to ``out_path`` and return it."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_urdf_string(spec), encoding="utf-8")
    return out


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate slider/rail URDF from a YAML spec.")
    p.add_argument("--spec", type=Path, default=None, help="YAML spec (default: built-in defaults)")
    p.add_argument("--out", type=Path, required=True, help="Output URDF path")
    args = p.parse_args(argv)
    spec: dict | Path = args.spec if args.spec is not None else DEFAULT_SPEC_YAML
    out = generate_urdf(spec, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
