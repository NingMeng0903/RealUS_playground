from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _candidate_roots(anchor: Path) -> list[Path]:
    if anchor.is_file():
        anchor = anchor.parent
    return [anchor, *anchor.parents]


def _looks_like_project_root(path: Path) -> bool:
    src_root = path / "src"
    return (
        src_root.is_dir()
        and (src_root / "projects" / "genesis_ue_sync").is_dir()
        and (src_root / "bridge").is_dir()
        and (path / "MD").is_dir()
    )


def discover_project_root(anchor: str | Path | None = None) -> Path:
    for key in ("REALUS_PROJECT_ROOT", "AMONGUS_PROJECT_ROOT"):
        env_root = os.environ.get(key, "").strip()
        if env_root:
            return Path(env_root).expanduser().resolve()
    start = Path(anchor).expanduser().resolve() if anchor is not None else Path(__file__).resolve()
    for candidate in _candidate_roots(start):
        if _looks_like_project_root(candidate):
            return candidate
    raise RuntimeError(f"Cannot discover project root from anchor: {start}")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def src_root(self) -> Path:
        return self.root / "src"

    @property
    def md_root(self) -> Path:
        return self.root / "MD"

    @property
    def configs_root(self) -> Path:
        return self.root / "configs"

    @property
    def scene_configs_root(self) -> Path:
        return self.configs_root / "scenes"

    @property
    def dataset_root(self) -> Path:
        return self.root / "dataset"

    @property
    def outputs_root(self) -> Path:
        return self.root / "outputs"

    @property
    def tmp_root(self) -> Path:
        return self.root / "tmp"

    @property
    def assets_root(self) -> Path:
        return self.root / "assets"

    @property
    def reference_root(self) -> Path:
        return self.root / "ref_code_library"

    @property
    def default_scene_spec_path(self) -> Path:
        return self.scene_configs_root / "amass_lie_sync_scene.yaml"

    @property
    def bedlam_unreal_root(self) -> Path:
        env = os.environ.get("REALUS_BEDLAM_UNREAL_ROOT", "").strip() or os.environ.get(
            "AMONGUS_BEDLAM_UNREAL_ROOT", ""
        ).strip()
        if env:
            return Path(env).expanduser().resolve()
        local = self.assets_root / "humans" / "bedlam2" / "unreal"
        if local.is_dir():
            return local
        # Shared install under Among_US (do not duplicate the UE project tree).
        shared = Path("/media/camp/EXT_DRIVE/Among_US/assets/humans/bedlam2/unreal")
        return shared if shared.is_dir() else local

    @property
    def bedlam_unreal_project_root(self) -> Path:
        return self.bedlam_unreal_root / "projects" / "BE_IBL"

    @property
    def bedlam_unreal_project_file(self) -> Path:
        return self.bedlam_unreal_project_root / "BE_IBL.uproject"

    @property
    def bedlam_engine_python_root(self) -> Path:
        return self.bedlam_unreal_root / "engine_content" / "PS" / "Bedlam" / "Core" / "Python"

    @property
    def bedlam_retarget_root(self) -> Path:
        env = os.environ.get("REALUS_BEDLAM_RETARGET_ROOT", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        local = self.reference_root / "bedlam2_retargeting"
        if (local / "retargeting" / "retargeting.uproject").is_file() or (local / "processing").is_dir():
            shared = Path("/media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam2_retargeting")
            # Prefer Among_US full retargeting.uproject when local copy is processing-only.
            if not (local / "retargeting" / "retargeting.uproject").is_file() and (
                shared / "retargeting" / "retargeting.uproject"
            ).is_file():
                return shared
            return local
        shared = Path("/media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam2_retargeting")
        return shared if shared.is_dir() else local

    @property
    def bedlam_retarget_project_file(self) -> Path:
        return self.bedlam_retarget_root / "retargeting" / "retargeting.uproject"

    @property
    def bedlam_retarget_python_root(self) -> Path:
        return self.bedlam_retarget_root / "retargeting" / "Content" / "Python"

    @property
    def smplx_blender_addon_root(self) -> Path:
        return self.reference_root / "smplx_blender_addon"

    @property
    def smplx_blender_required_blend(self) -> Path:
        return self.smplx_blender_addon_root / "data" / "smplx_model_lh_20230302.blend"

    @property
    def ue_generated_cache_root(self) -> Path:
        return self.outputs_root / "ue_cache"

    def resolve_from_root(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.root / candidate).resolve()


@lru_cache(maxsize=None)
def project_paths(anchor: str | Path | None = None) -> ProjectPaths:
    return ProjectPaths(root=discover_project_root(anchor))
