"""Build a PyVista MultiBlock of the RM75 robot at a given joint config.

Walks the URDF ``VISUAL`` geometry model with Pinocchio, computes each link's
world-frame placement at ``q_full``, loads meshes via trimesh (Collada / STL /
OBJ all covered) and stacks them into a PyVista ``MultiBlock``.

Collada arm meshes are multi-material Scenes. Loading with ``force="mesh"``
merges them into a single grey surface — we instead expand each Scene geometry
and keep its PBR ``baseColorFactor`` / ``main_color``.

Paper mount-compare figures skip the rail extrusion and probe/TCP head meshes,
and draw a cylinder at the true ``tcp`` frame instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin
import pyvista as pv

from rm75_control.tools.reachability.viz.colormap import ZACHARIAS_ROBOT_GRAY


_DEFAULT_VISUAL_URDF = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.genesis.urdf"
)

# Substrings matched against Pinocchio visual object names (case-insensitive).
DEFAULT_SKIP_VISUAL_SUBSTRINGS: tuple[str, ...] = (
    "rail_visual",
    "link_8",
    "probe",
)


@dataclass
class RobotPvScene:
    urdf_path: Path
    mesh_block: pv.MultiBlock
    n_visuals: int
    link_colors: list[str] = field(default_factory=list)
    tcp_pose_world: pin.SE3 | None = None


def _rgba_to_hex(rgba) -> str | None:
    if rgba is None:
        return None
    rgb = np.asarray(rgba, dtype=np.float64).reshape(-1)[:3]
    if rgb.max() <= 1.0 + 1.0e-6:
        rgb = np.clip(rgb * 255.0, 0, 255)
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _color_from_material(mat) -> str | None:
    if mat is None:
        return None
    for attr in ("baseColorFactor", "main_color"):
        hex_c = _rgba_to_hex(getattr(mat, attr, None))
        if hex_c is not None:
            return hex_c
    return None


def _color_from_trimesh(tm) -> str | None:
    vis = getattr(tm, "visual", None)
    mat = getattr(vis, "material", None) if vis is not None else None
    return _color_from_material(mat)


def _pv_from_trimesh(tm) -> pv.PolyData | None:
    if tm is None or not hasattr(tm, "vertices"):
        return None
    return pv.wrap(tm)


def _meshes_from_path(mesh_path: str) -> list[tuple[pv.PolyData, str | None]]:
    """Load a path as one or more (mesh, hex colour) pairs.

    Collada files arrive as ``trimesh.Scene`` graphs. Submesh vertices are in
    local geometry space; the shared scene-graph transform (often a 90° axis
    flip from Collada Y-up) **must** be applied or every link looks exploded
    relative to the URDF joint frames.
    """
    import trimesh

    loaded = trimesh.load(mesh_path, force=None)
    out: list[tuple[pv.PolyData, str | None]] = []
    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            T, geom_name = loaded.graph.get(node_name)
            geom = loaded.geometry[geom_name]
            if geom is None or not hasattr(geom, "vertices"):
                continue
            # Copy before transform so cached scene geometries stay pristine.
            geom = geom.copy()
            geom.apply_transform(np.asarray(T, dtype=np.float64))
            pd = _pv_from_trimesh(geom)
            if pd is None:
                continue
            out.append((pd, _color_from_trimesh(geom)))
        return out
    pd = _pv_from_trimesh(loaded)
    if pd is None:
        return []
    return [(pd, _color_from_trimesh(loaded))]


def _mesh_from_geom_object(go: pin.GeometryObject) -> list[tuple[pv.PolyData, str | None]]:
    """Turn a Pinocchio VISUAL GeometryObject into one or more PolyData + colours."""
    mesh_path = str(go.meshPath)
    if mesh_path and mesh_path != "BOX" and Path(mesh_path).exists():
        parts = _meshes_from_path(mesh_path)
        s = np.asarray(getattr(go, "meshScale", np.ones(3)), dtype=float).reshape(3)
        if not np.allclose(s, 1.0):
            scaled = []
            for pd, col in parts:
                pd2 = pd.copy()
                pd2.points = pd2.points * s[None, :]
                scaled.append((pd2, col))
            return scaled
        return parts

    geom = getattr(go, "geometry", None)
    if geom is None:
        return []
    node_type = getattr(geom, "getNodeType", lambda: None)()
    name = str(node_type) if node_type is not None else type(geom).__name__

    if "BOX" in name.upper() or type(geom).__name__.startswith("Box"):
        halfside = np.asarray(geom.halfSide, dtype=float)
        return [(
            pv.Box(bounds=(-halfside[0], halfside[0], -halfside[1], halfside[1], -halfside[2], halfside[2])),
            "#888888",
        )]
    if "CYL" in name.upper() or type(geom).__name__.startswith("Cyl"):
        radius = float(getattr(geom, "radius", 0.02))
        length = float(getattr(geom, "halfLength", 0.05)) * 2.0
        return [(pv.Cylinder(radius=radius, height=length, direction=(0.0, 0.0, 1.0)), "#888888")]
    if "SPH" in name.upper() or type(geom).__name__.startswith("Sph"):
        radius = float(getattr(geom, "radius", 0.02))
        return [(pv.Sphere(radius=radius), "#888888")]
    return []


def _apply_se3(pd: pv.PolyData, M: pin.SE3) -> pv.PolyData:
    T = np.eye(4)
    T[:3, :3] = np.asarray(M.rotation)
    T[:3, 3] = np.asarray(M.translation)
    out = pd.copy()
    out.transform(T, inplace=True)
    return out


def _should_skip(name: str, skip_substrings: tuple[str, ...] | None) -> bool:
    if not skip_substrings:
        return False
    low = name.lower()
    return any(s.lower() in low for s in skip_substrings)


def build_robot_pv(
    urdf_path: str | Path | None = None,
    q_full: np.ndarray | None = None,
    *,
    base_pose_world: pin.SE3 | None = None,
    color: str | None = None,
    skip_visual_substrings: tuple[str, ...] | None = DEFAULT_SKIP_VISUAL_SUBSTRINGS,
) -> RobotPvScene:
    """Return a MultiBlock of link visuals at joint config ``q_full``.

    By default skips rail extrusion visuals and probe/TCP-head meshes
    (``rail_visual``, ``link_8``, ``probe``). Pass
    ``skip_visual_substrings=()`` to keep everything.
    """
    urdf = Path(urdf_path) if urdf_path is not None else _DEFAULT_VISUAL_URDF
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    _ = color
    model = pin.buildModelFromUrdf(str(urdf))
    if q_full is None:
        q_full = np.zeros(model.nq, dtype=np.float64)
    elif q_full.size != model.nq:
        raise ValueError(f"q_full must have {model.nq} entries, got {q_full.size}")

    package_dirs = [str(urdf.parent)]
    gm = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.VISUAL, package_dirs=package_dirs)
    gdata = gm.createData()
    data = model.createData()
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, gm, gdata)

    block = pv.MultiBlock()
    colors: list[str] = []
    for i, go in enumerate(gm.geometryObjects):
        if _should_skip(str(go.name), skip_visual_substrings):
            continue
        parts = _mesh_from_geom_object(go)
        if not parts:
            continue
        M_world = gdata.oMg[i]
        for j, (pd, link_color) in enumerate(parts):
            pd_world = _apply_se3(pd, M_world)
            if base_pose_world is not None:
                pd_world = _apply_se3(pd_world, base_pose_world)
            block.append(pd_world, name=f"{go.name}_{j}")
            colors.append(link_color or ZACHARIAS_ROBOT_GRAY)

    tcp_pose = None
    if model.existFrame("tcp"):
        tcp_pose = data.oMf[model.getFrameId("tcp")].copy()
        if base_pose_world is not None:
            tcp_pose = base_pose_world * tcp_pose

    return RobotPvScene(
        urdf_path=urdf,
        mesh_block=block,
        n_visuals=len(block),
        link_colors=colors,
        tcp_pose_world=tcp_pose,
    )


def estimate_tcp_shaft_height_m(urdf_path: str | Path, *, fallback: float = 0.10) -> float:
    """Flange→TCP length used as the green marker shaft (tip on TCP)."""
    try:
        model = pin.buildModelFromUrdf(str(urdf_path))
        data = model.createData()
        q = pin.neutral(model)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        if not model.existFrame("tcp"):
            return float(fallback)
        t_tcp = np.asarray(data.oMf[model.getFrameId("tcp")].translation, dtype=np.float64)
        # Prefer link_7 frame if present; else joint_7 placement.
        if model.existFrame("link_7"):
            t_flange = np.asarray(data.oMf[model.getFrameId("link_7")].translation, dtype=np.float64)
        elif model.existJointName("joint_7"):
            t_flange = np.asarray(data.oMi[model.getJointId("joint_7")].translation, dtype=np.float64)
        else:
            return float(fallback)
        dist = float(np.linalg.norm(t_tcp - t_flange))
        if not np.isfinite(dist) or dist < 1e-3:
            return float(fallback)
        return float(np.clip(dist * 0.95, 0.06, 0.22))
    except Exception:
        return float(fallback)


def tcp_cylinder_marker(
    tcp_pose_world: pin.SE3,
    *,
    radius_m: float = 0.012,
    height_m: float = 0.100,
    color: str = "#1b5e20",
) -> tuple[pv.PolyData, str]:
    """Cylinder along tool −Z with the **distal tip** on the TCP frame origin.

    Tool +Z points out of the tip; the shaft therefore occupies
    ``[tcp − height·ẑ, tcp]`` so the outermost end sits on TCP (not the
    flange-facing end).
    """
    R = np.asarray(tcp_pose_world.rotation, dtype=np.float64)
    p = np.asarray(tcp_pose_world.translation, dtype=np.float64).reshape(3)
    z_hat = R @ np.array([0.0, 0.0, 1.0])
    height = float(height_m)
    # Center is halfway back along −Z from TCP → distal tip lands on ``p``.
    center = p - 0.5 * height * z_hat
    cyl = pv.Cylinder(
        center=tuple(center),
        direction=tuple(z_hat),
        radius=float(radius_m),
        height=height,
        resolution=48,
    )
    return cyl, color


def add_tcp_marker_to_plotter(
    pl: pv.Plotter,
    scene: RobotPvScene,
    *,
    radius_m: float = 0.012,
    height_m: float | None = None,
    color: str = "#1b5e20",
) -> None:
    if scene.tcp_pose_world is None:
        return
    h = float(height_m) if height_m is not None else estimate_tcp_shaft_height_m(scene.urdf_path)
    cyl, col = tcp_cylinder_marker(
        scene.tcp_pose_world, radius_m=radius_m, height_m=h, color=color
    )
    pl.add_mesh(
        cyl,
        color=col,
        opacity=1.0,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.7,
        name="tcp_cylinder",
    )


def add_rest_pose_annotation(
    pl: pv.Plotter,
    *,
    ground_radius_m: float = 1.2,
    ground_color: str = "#eeeeee",
    axes_length_m: float = 0.15,
) -> None:
    """Add a light ground disk + world-frame axes at (0,0,0), matching paper figs."""
    disk = pv.Disc(center=(0, 0, 0), inner=0.0, outer=ground_radius_m, normal=(0, 0, 1), r_res=64, c_res=1)
    pl.add_mesh(disk, color=ground_color, opacity=0.4, name="ground_disk", show_edges=False)
    for axis, color in zip(
        (np.array([axes_length_m, 0, 0]), np.array([0, axes_length_m, 0]), np.array([0, 0, axes_length_m])),
        ("red", "green", "blue"),
    ):
        arr = pv.Arrow(
            start=(0, 0, 0), direction=tuple(axis),
            tip_length=0.2, tip_radius=0.02, shaft_radius=0.008, scale=1.0,
        )
        pl.add_mesh(arr, color=color, name=f"axis_{color}")


def add_robot_to_plotter(
    pl: pv.Plotter,
    scene: RobotPvScene,
    *,
    color: str = ZACHARIAS_ROBOT_GRAY,
    opacity: float = 1.0,
    show_edges: bool = False,
    use_dae_colors: bool = True,
    on_top: bool = False,
    show_tcp_marker: bool = True,
) -> None:
    """Add robot meshes; default preserves per-submesh DAE colours + TCP cylinder."""
    for i, name in enumerate(scene.mesh_block.keys()):
        mesh_color = scene.link_colors[i] if use_dae_colors and i < len(scene.link_colors) else color
        mesh = scene.mesh_block[i].copy()
        mesh.point_data.clear()
        mesh.cell_data.clear()
        actor = pl.add_mesh(
            mesh,
            color=mesh_color,
            opacity=float(opacity),
            show_edges=show_edges,
            smooth_shading=True,
            ambient=0.28,
            diffuse=0.72,
            specular=0.05,
            name=f"robot_{name}",
        )
        if on_top:
            try:
                vtk_prop = actor.GetProperty()
                if hasattr(vtk_prop, "SetPolygonOffsetFactor"):
                    vtk_prop.SetPolygonOffsetFactor(-4.0)
                    vtk_prop.SetPolygonOffsetUnits(-4.0)
            except Exception:
                pass
    if show_tcp_marker:
        add_tcp_marker_to_plotter(pl, scene)
