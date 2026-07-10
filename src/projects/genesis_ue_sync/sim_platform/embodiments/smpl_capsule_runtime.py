from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.embodiments.loaders.urdf_loader import (
    URDFToolFrames,
    parse_root_link_name,
)
from projects.genesis_ue_sync.sim_platform.embodiments.mjcf_loader import build_embodiment_from_mjcf
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import EmbodimentProfile
from projects.genesis_ue_sync.sim_platform.embodiments.phc_bundled_mjcf_proxy import sync_phc_bundled_proxy_to_cache
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import ProxyGeometry

DEFAULT_SMPL_PROXY_VISUAL_RGBA = (0.98, 0.48, 0.12, 0.52)


@dataclass(frozen=True)
class SmplCapsuleRuntimeAsset:
    urdf_path: Path
    root_link_name: str
    proxy_geometry: ProxyGeometry
    mjcf_path: Path | None = None
    mjcf_dof_layout_path: Path | None = None


def prepare_phc_bundled_smpl_proxy_asset(
    sequence: HumanMotionSequence,
    *,
    cache_dir: Path,
    phc_root: Path | None = None,
    force_rewrite: bool = False,
) -> SmplCapsuleRuntimeAsset:
    """PHC bundled MJCF proxy (``phc/data/assets/mjcf``): no SMPLSim, fixed gender templates.

    Drive Genesis with ``build_smpl_capsule_embodiment(..., genesis_proxy=\"mjcf\")``.
    """

    from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies

    deps = {d.name: d for d in human_motion_dependencies()}
    root = Path(phc_root).expanduser().resolve() if phc_root is not None else deps["PHC"].resolved_path()
    mjcf_path, layout_path, urdf_placeholder, proxy_geometry = sync_phc_bundled_proxy_to_cache(
        sequence,
        cache_dir=Path(cache_dir),
        phc_root=root,
        force_rewrite=bool(force_rewrite),
    )
    return SmplCapsuleRuntimeAsset(
        urdf_path=urdf_placeholder,
        root_link_name=parse_root_link_name(urdf_placeholder),
        proxy_geometry=proxy_geometry,
        mjcf_path=mjcf_path,
        mjcf_dof_layout_path=layout_path,
    )


def prepare_smpl_capsule_runtime_asset(
    sequence: HumanMotionSequence,
    *,
    cache_dir: Path,
    device: str | None = "cpu",
    visual_rgba: tuple[float, float, float, float] = DEFAULT_SMPL_PROXY_VISUAL_RGBA,
    force_rewrite: bool = False,
    genesis_proxy: Literal["urdf", "mjcf"] = "mjcf",
) -> SmplCapsuleRuntimeAsset:
    del device, visual_rgba, genesis_proxy
    return prepare_phc_bundled_smpl_proxy_asset(sequence, cache_dir=Path(cache_dir), force_rewrite=bool(force_rewrite))


def build_smpl_capsule_embodiment(
    *,
    name: str,
    asset: SmplCapsuleRuntimeAsset,
    fixed_base: bool = False,
    genesis_proxy: Literal["urdf", "mjcf"] = "mjcf",
) -> EmbodimentProfile:
    if genesis_proxy != "mjcf":
        raise RuntimeError(
            "Hand-written SMPL capsule URDF was removed; use genesis_proxy='mjcf' (PHC bundled MJCF)."
        )
    root = asset.root_link_name
    tool = URDFToolFrames(
        base_frame=root,
        eef_link=root,
        tool_frame=root,
        tcp_frame=root,
    )
    if asset.mjcf_path is None or asset.mjcf_dof_layout_path is None:
        raise RuntimeError("PHC MJCF asset is missing mjcf_path or mjcf_dof_layout_path.")
    return build_embodiment_from_mjcf(
        name=name,
        mjcf_path=asset.mjcf_path,
        layout_path=asset.mjcf_dof_layout_path,
        tool_frames=tool,
        fixed_base=fixed_base,
    )
