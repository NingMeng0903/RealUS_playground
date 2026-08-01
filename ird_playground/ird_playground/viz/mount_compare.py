"""Three TCP-mount paper figures: reachability map + global IRD each (6 PNGs).

Mounts are typically:

* ``probe45`` — current physical probe TCP
* ``tcp220`` — vertical stock tool at 220 mm along link_7 +Z
* ``horizontal`` — horizontal ultrasound probe URDF

Shared ``clim`` / ``bar_max`` keep colours comparable across all six figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ird_playground.viz.viz_style import (
    PROBE_COMPARE_BAR_MAX,
    PROBE_COMPARE_CLIM,
    PROBE_COMPARE_D_MIN,
    PROBE_COMPARE_N_LEVELS,
    MOUNT_COMPARE_BOUNDS,
    MOUNT_COMPARE_FOCUS,
    MOUNT_COMPARE_IRD_BOUNDS,
    MOUNT_COMPARE_IRD_FOCUS,
    MOUNT_COMPARE_OBLIQUE_SPAN,
    MOUNT_COMPARE_PARALLEL_SCALE,
)


@dataclass(frozen=True)
class MountSpec:
    id: str
    title: str
    map_dir: Path
    robot_urdf: Path


@dataclass(frozen=True)
class MountCompareStyle:
    clim: tuple[float, float] = PROBE_COMPARE_CLIM
    bar_max: float = PROBE_COMPARE_BAR_MAX
    n_color_levels: int = PROBE_COMPARE_N_LEVELS
    d_min: float = PROBE_COMPARE_D_MIN
    figsize: tuple[int, int] = (3200, 1100)


@dataclass(frozen=True)
class MountCompareConfig:
    mounts: tuple[MountSpec, ...]
    style: MountCompareStyle
    out_dir: Path
    repo_root: Path


def _resolve(path: str | Path, *, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def _pick_map_dir(raw: dict[str, Any], *, root: Path) -> Path:
    primary = _resolve(str(raw["map_dir"]), root=root)
    if primary.is_dir() and (primary / "manifest.yaml").is_file():
        return primary
    fb = raw.get("map_dir_fallback")
    if fb:
        fallback = _resolve(str(fb), root=root)
        if fallback.is_dir() and (fallback / "manifest.yaml").is_file():
            return fallback
    return primary


def load_mount_compare_config(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> MountCompareConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if repo_root is not None:
        root = Path(repo_root).resolve()
    else:
        # ird_playground/configs/*.yaml → RealUS_playground
        root = cfg_path.parents[2]
        if not (root / "rm75_control").is_dir():
            root = cfg_path.parents[1]

    style_raw = dict(raw.get("style") or {})
    clim = style_raw.get("clim", list(PROBE_COMPARE_CLIM))
    figsize = style_raw.get("figsize", [3200, 1100])
    style = MountCompareStyle(
        clim=(float(clim[0]), float(clim[1])),
        bar_max=float(style_raw.get("bar_max", PROBE_COMPARE_BAR_MAX)),
        n_color_levels=int(style_raw.get("n_color_levels", PROBE_COMPARE_N_LEVELS)),
        d_min=float(style_raw.get("d_min", PROBE_COMPARE_D_MIN)),
        figsize=(int(figsize[0]), int(figsize[1])),
    )

    mounts: list[MountSpec] = []
    for row in raw.get("mounts") or []:
        mounts.append(
            MountSpec(
                id=str(row["id"]),
                title=str(row.get("title") or row["id"]),
                map_dir=_pick_map_dir(row, root=root),
                robot_urdf=_resolve(str(row["robot_urdf"]), root=root),
            )
        )
    if len(mounts) < 1:
        raise ValueError("mount_compare config needs at least one mount")

    io = dict(raw.get("io") or {})
    out_dir = _resolve(
        str(io.get("out_dir", "ird_playground/data/reports/mount_compare")),
        root=root,
    )
    return MountCompareConfig(
        mounts=tuple(mounts), style=style, out_dir=out_dir, repo_root=root,
    )


def _load_capability_map(map_dir: Path):
    from ird_playground.viz.rm75_ns import ensure_rm75_namespace

    ensure_rm75_namespace()
    from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap

    if not (map_dir / "manifest.yaml").is_file():
        raise FileNotFoundError(
            f"capability map missing at {map_dir} "
            f"(build with rm75_control reachability configs first)"
        )
    return CapabilityMap.load(map_dir, mmap=True)


def _abs_mesh_urdf(src: Path, dst: Path) -> Path:
    """Rewrite relative mesh filenames to absolute paths for off-tree PyVista loads."""
    import re

    text = src.read_text(encoding="utf-8")
    mesh_root = src.parent

    def _abs(m: re.Match[str]) -> str:
        rel = m.group(1)
        if Path(rel).is_absolute():
            return m.group(0)
        return f'filename="{(mesh_root / rel).resolve()}"'

    text = re.sub(r'filename="([^"]+)"', _abs, text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return dst


def prepare_robot_urdf(robot_urdf: Path, *, cache_dir: Path) -> Path:
    """Return a PyVista-safe URDF path (absolute mesh filenames)."""
    if not robot_urdf.is_file():
        raise FileNotFoundError(robot_urdf)
    cached = cache_dir / f"{robot_urdf.stem}.absmeshes.urdf"
    return _abs_mesh_urdf(robot_urdf, cached)


def _base_link_y(robot_urdf: Path) -> float:
    """``base_link`` Y at neutral (rail locked) in the URDF root frame."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(robot_urdf))
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    if model.existFrame("base_link"):
        return float(data.oMf[model.getFrameId("base_link")].translation[1])
    return float(data.oMi[1].translation[1])


def base_link_y_centering(robot_urdf: Path):
    """Display offset that puts ``base_link`` on the Y=0 rail plane.

    Stock RM75 URDFs place the locked rail shoulder at ``y=-0.4``; the horizontal
    probe patch uses ``y=0``. Without this shift, ``y>=0`` hemisphere cuts leave
    the arm on the rim of the cloud.

    Returns ``(display_offset, shoulder_y_in_map_frame)``.
    """
    import numpy as np

    y = _base_link_y(robot_urdf)
    offset = np.array([0.0, -y, 0.0], dtype=np.float64)
    return offset, y


def _map_build_urdf(map_dir: Path, *, repo_root: Path) -> Path | None:
    """URDF used to build the capability map (manifest), if resolvable."""
    import yaml

    man_path = map_dir / "manifest.yaml"
    if not man_path.is_file():
        return None
    man = yaml.safe_load(man_path.read_text(encoding="utf-8")) or {}
    rel = man.get("urdf_path")
    if not rel:
        return None
    p = Path(str(rel))
    if p.is_file():
        return p.resolve()
    for root in (repo_root, repo_root / "rm75_control"):
        cand = (root / p).resolve()
        if cand.is_file():
            return cand
    return None


def cut_plane_for_mount(
    map_dir: Path,
    viz_urdf: Path,
    *,
    repo_root: Path,
):
    """Y-cut through the map-frame shoulder; viz arm parked on the same plane.

    Cut Y must come from the **map build** URDF (manifest), not the genesis viz
    URDF — horizontal kinematics use ``rail_y`` origin at 0 while an outdated
    genesis copy still has ``-0.4``, which shoved the cut to the cloud rim.
    """
    import numpy as np
    import pinocchio as pin

    map_urdf = _map_build_urdf(map_dir, repo_root=repo_root)
    cut_y = _base_link_y(map_urdf) if map_urdf is not None else _base_link_y(viz_urdf)
    viz_y = _base_link_y(viz_urdf)
    display_offset = np.array([0.0, -cut_y, 0.0], dtype=np.float64)
    base_pose_world = pin.SE3(np.eye(3), np.array([0.0, -viz_y, 0.0], dtype=np.float64))
    return display_offset, cut_y, base_pose_world


def render_mount_pair(
    mount: MountSpec,
    *,
    out_dir: Path,
    style: MountCompareStyle,
    robot_urdf: Path,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Write ``{id}_reachability.png`` and ``{id}_ird.png``."""
    from ird_playground.viz.rm75_ns import ensure_rm75_namespace

    ensure_rm75_namespace()
    from rm75_control.tools.reachability.viz.sphere_glyphs import render_reachability_index

    from ird_playground.viz.global_ird import render_global_ird_from_capability

    cm = _load_capability_map(mount.map_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rm_path = out_dir / f"{mount.id}_reachability.png"
    ird_path = out_dir / f"{mount.id}_ird.png"

    # Classic Zacharias Y-hemisphere after shoulder centering; locked camera so
    # every mount puts the arm at the same size and place. Left = 45° onto the
    # cut (opening toward the right); right = orthographic onto the cut face.
    root = repo_root or Path(__file__).resolve().parents[3]
    display_offset, cut_y, base_pose_world = cut_plane_for_mount(
        mount.map_dir, robot_urdf, repo_root=root,
    )
    render_reachability_index(
        cm,
        rm_path,
        robot_urdf=robot_urdf,
        d_min=style.d_min,
        clim=style.clim,
        clim_auto=False,
        size=style.figsize,
        n_color_levels=style.n_color_levels,
        bar_max=style.bar_max,
        view="cross",
        fixed_camera=True,
        plane=f"y={cut_y}",
        display_offset=display_offset,
        base_pose_world=base_pose_world,
        camera_bounds=MOUNT_COMPARE_BOUNDS,
        camera_focus=MOUNT_COMPARE_FOCUS,
        parallel_scale=MOUNT_COMPARE_PARALLEL_SCALE,
        oblique_span=MOUNT_COMPARE_OBLIQUE_SPAN,
    )
    render_global_ird_from_capability(
        cm,
        ird_path,
        robot_urdf=robot_urdf,
        title=f"Global IRD — {mount.title}",
        d_min=style.d_min,
        clim=style.clim,
        clim_auto=False,
        size=style.figsize,
        n_color_levels=style.n_color_levels,
        bar_max=style.bar_max,
        camera_bounds=MOUNT_COMPARE_IRD_BOUNDS,
        camera_focus=MOUNT_COMPARE_IRD_FOCUS,
        parallel_scale=MOUNT_COMPARE_PARALLEL_SCALE,
        oblique_span=MOUNT_COMPARE_OBLIQUE_SPAN,
    )
    return {"reachability": rm_path, "ird": ird_path}


def render_mount_compare(
    config: MountCompareConfig,
    *,
    skip_missing: bool = False,
) -> dict[str, dict[str, str]]:
    """Render all configured mounts; return id → {reachability, ird} paths."""
    cache = config.out_dir / "_urdf_cache"
    report: dict[str, dict[str, str]] = {}
    for mount in config.mounts:
        if not (mount.map_dir / "manifest.yaml").is_file():
            if skip_missing:
                report[mount.id] = {
                    "status": "skipped_missing_map",
                    "map_dir": str(mount.map_dir),
                }
                continue
            raise FileNotFoundError(
                f"[{mount.id}] capability map not found: {mount.map_dir}\n"
                f"Build it first (see rm75_control/configs/reachability/)."
            )
        robot = prepare_robot_urdf(mount.robot_urdf, cache_dir=cache)
        paths = render_mount_pair(
            mount,
            out_dir=config.out_dir,
            style=config.style,
            robot_urdf=robot,
            repo_root=config.repo_root,
        )
        report[mount.id] = {
            "status": "ok",
            "title": mount.title,
            "map_dir": str(mount.map_dir),
            "reachability": str(paths["reachability"]),
            "ird": str(paths["ird"]),
        }
    return report


__all__ = [
    "MountCompareConfig",
    "MountCompareStyle",
    "MountSpec",
    "load_mount_compare_config",
    "prepare_robot_urdf",
    "render_mount_compare",
    "render_mount_pair",
]
