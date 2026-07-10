from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import (
    prepare_phc_bundled_smpl_proxy_asset,
    prepare_smpl_capsule_runtime_asset,
)
from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies


@dataclass(frozen=True)
class ProxyAssetSummary:
    provider: str
    urdf_path: str
    root_link_name: str
    body_count: int
    total_mass_kg: float
    shape_key: str = ""
    mjcf_path: str = ""
    mjcf_dof_layout_path: str = ""
    diagnostics: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["diagnostics"] = dict(self.diagnostics or {})
        return data


class HumanProxyProvider(Protocol):
    def prepare(self, sequence: HumanMotionSequence) -> ProxyAssetSummary:
        ...


@dataclass(frozen=True)
class LocalCapsuleProxyProvider:
    """Use the existing project SMPL capsule URDF generator."""

    cache_dir: Path | None = None
    device: str = "cpu"
    force_rewrite: bool = False

    def prepare(self, sequence: HumanMotionSequence) -> ProxyAssetSummary:
        root = project_paths(__file__).outputs_root
        cache = self.cache_dir or (root / "genesis_capsule_urdf_cache")
        asset = prepare_smpl_capsule_runtime_asset(
            sequence,
            cache_dir=cache,
            device=self.device,
            force_rewrite=self.force_rewrite,
        )
        total_mass = sum(float(body.mass_kg) for body in asset.proxy_geometry.bodies)
        return ProxyAssetSummary(
            provider="local_capsule",
            urdf_path=str(asset.urdf_path),
            root_link_name=str(asset.root_link_name),
            body_count=len(asset.proxy_geometry.bodies),
            total_mass_kg=float(total_mass),
            shape_key=str(asset.proxy_geometry.shape_key),
            mjcf_path=str(asset.mjcf_path) if asset.mjcf_path else "",
            mjcf_dof_layout_path=str(asset.mjcf_dof_layout_path) if asset.mjcf_dof_layout_path else "",
            diagnostics={
                "model_type": asset.proxy_geometry.model_type,
                "gender": asset.proxy_geometry.gender,
                "hip_width_m": asset.proxy_geometry.hip_width_m,
                "shoulder_width_m": asset.proxy_geometry.shoulder_width_m,
                "torso_height_m": asset.proxy_geometry.torso_height_m,
                "primitive_types": sorted({body.primitive_type for body in asset.proxy_geometry.bodies}),
                "provider_note": "SMPL shape-derived collision primitives with volume-based sizing and density compensation.",
            },
        )


@dataclass(frozen=True)
class PhcProxyProvider:
    """PHC bundled MJCF humanoid (``phc/data/assets/mjcf``) as Genesis proxy — no SMPLSim import."""

    phc_root: Path | None = None
    cache_dir: Path | None = None
    force_rewrite: bool = False

    def availability(self) -> dict[str, Any]:
        deps = {dep.name: dep for dep in human_motion_dependencies()}
        phc = self.phc_root or deps["PHC"].resolved_path()
        mjcf_dir = phc / "phc" / "data" / "assets" / "mjcf"
        return {
            "phc_root": str(phc),
            "phc_exists": bool(phc.exists()),
            "phc_mjcf_dir": str(mjcf_dir),
            "phc_mjcf_dir_exists": bool(mjcf_dir.is_dir()),
            "ready": bool(phc.exists() and mjcf_dir.is_dir()),
        }

    def prepare(self, sequence: HumanMotionSequence) -> ProxyAssetSummary:
        status = self.availability()
        if not status["ready"]:
            hint = (
                "Fix: clone PHC next to your other reference repos, e.g.\n"
                f"  git clone https://github.com/ZhengyiLuo/PHC.git {Path(status['phc_root'])}\n"
                "Or: export AMONGUS_PHC_ROOT=/path/to/PHC\n"
                "Then check: ls \"$AMONGUS_PHC_ROOT/phc/data/assets/mjcf/*.xml\""
            )
            raise FileNotFoundError(f"PHC bundled MJCF proxy is not ready: {status}\n{hint}")
        root = project_paths(__file__).outputs_root
        cache = self.cache_dir or (root / "phc_bundled_mjcf_proxy_cache")
        asset = prepare_phc_bundled_smpl_proxy_asset(
            sequence,
            cache_dir=cache,
            phc_root=self.phc_root,
            force_rewrite=self.force_rewrite,
        )
        total_mass = sum(float(body.mass_kg) for body in asset.proxy_geometry.bodies)
        return ProxyAssetSummary(
            provider="phc_bundled_mjcf",
            urdf_path=str(asset.urdf_path),
            root_link_name=str(asset.root_link_name),
            body_count=len(asset.proxy_geometry.bodies),
            total_mass_kg=float(total_mass),
            shape_key=str(asset.proxy_geometry.shape_key),
            mjcf_path=str(asset.mjcf_path) if asset.mjcf_path else "",
            mjcf_dof_layout_path=str(asset.mjcf_dof_layout_path) if asset.mjcf_dof_layout_path else "",
            diagnostics={
                "proxy_note": "PHC pre-built MuJoCo SMPL asset (gender template). Beta-conditioned mesh requires upstream SMPL_Robot tooling.",
                "genesis_proxy": "mjcf",
                "template_source_dir": status["phc_mjcf_dir"],
            },
        )
