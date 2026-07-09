"""Build a PyVista MultiBlock of the RM75 robot at a given joint config.

Walks the URDF ``VISUAL`` geometry model with Pinocchio, computes each link's
world-frame placement at ``q_full``, loads meshes via trimesh (Collada / STL /
OBJ all covered) and stacks them into a PyVista ``MultiBlock`` so the caller
can drop it into any plotter with a single ``add_mesh`` call.

By default each link keeps the **DAE material colour** (``main_color`` from
the Collada PBR material). Pass ``use_dae_colors=False`` for the uniform grey
fallback used in some schematic views.
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


@dataclass
class RobotPvScene:
    urdf_path: Path
    mesh_block: pv.MultiBlock
    n_visuals: int
    link_colors: list[str] = field(default_factory=list)


def _color_from_trimesh(tm) -> str | None:
    vis = getattr(tm, "visual", None)
    mat = getattr(vis, "material", None) if vis is not None else None
    if mat is None:
        return None
    mc = getattr(mat, "main_color", None)
    if mc is None:
        return None
    rgb = np.asarray(mc, dtype=np.uint8).reshape(-1)[:3]
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _mesh_from_geom_object(go: pin.GeometryObject) -> tuple[pv.PolyData | None, str | None]:
    """Turn a Pinocchio VISUAL GeometryObject into a PolyData + optional DAE colour."""
    mesh_path = str(go.meshPath)
    link_color: str | None = None
    if mesh_path and mesh_path != "BOX" and Path(mesh_path).exists():
        import trimesh

        tm = trimesh.load(mesh_path, force="mesh")
        if tm is None or not hasattr(tm, "vertices"):
            return None, None
        link_color = _color_from_trimesh(tm)
        pd = pv.wrap(tm)
        s = np.asarray(getattr(go, "meshScale", np.ones(3)), dtype=float).reshape(3)
        if not np.allclose(s, 1.0):
            pd.points = pd.points * s[None, :]
        return pd, link_color

    geom = getattr(go, "geometry", None)
    if geom is None:
        return None, None
    node_type = getattr(geom, "getNodeType", lambda: None)()
    name = str(node_type) if node_type is not None else type(geom).__name__

    if "BOX" in name.upper() or type(geom).__name__.startswith("Box"):
        halfside = np.asarray(geom.halfSide, dtype=float)
        return pv.Box(bounds=(-halfside[0], halfside[0], -halfside[1], halfside[1], -halfside[2], halfside[2])), "#888888"
    if "CYL" in name.upper() or type(geom).__name__.startswith("Cyl"):
        radius = float(getattr(geom, "radius", 0.02))
        length = float(getattr(geom, "halfLength", 0.05)) * 2.0
        return pv.Cylinder(radius=radius, height=length, direction=(0.0, 0.0, 1.0)), "#888888"
    if "SPH" in name.upper() or type(geom).__name__.startswith("Sph"):
        radius = float(getattr(geom, "radius", 0.02))
        return pv.Sphere(radius=radius), "#888888"
    return None, None


def _apply_se3(pd: pv.PolyData, M: pin.SE3) -> pv.PolyData:
    T = np.eye(4)
    T[:3, :3] = np.asarray(M.rotation)
    T[:3, 3] = np.asarray(M.translation)
    out = pd.copy()
    out.transform(T, inplace=True)
    return out


def build_robot_pv(
    urdf_path: str | Path | None = None,
    q_full: np.ndarray | None = None,
    *,
    base_pose_world: pin.SE3 | None = None,
    color: str | None = None,
) -> RobotPvScene:
    """Return a MultiBlock of link visuals at joint config ``q_full``."""
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
    pin.updateGeometryPlacements(model, data, gm, gdata)

    block = pv.MultiBlock()
    colors: list[str] = []
    for i, go in enumerate(gm.geometryObjects):
        pd, link_color = _mesh_from_geom_object(go)
        if pd is None:
            continue
        M_world = gdata.oMg[i]
        pd_world = _apply_se3(pd, M_world)
        if base_pose_world is not None:
            pd_world = _apply_se3(pd_world, base_pose_world)
        block.append(pd_world, name=go.name)
        colors.append(link_color or ZACHARIAS_ROBOT_GRAY)

    return RobotPvScene(urdf_path=urdf, mesh_block=block, n_visuals=len(block), link_colors=colors)


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
) -> None:
    """Add robot meshes; default preserves per-link DAE ``main_color``."""
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
