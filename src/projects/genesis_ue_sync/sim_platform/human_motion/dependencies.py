from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.layout import HumanDatasetLayout


def _vposer_weights_default(reference_root: Path, models_root: Path) -> Path:
    """Prefer bundled ref_code_library/V02_05 when checkpoints are present."""
    ref_vposer = reference_root / "V02_05"
    snaps = ref_vposer / "snapshots"
    if snaps.is_dir() and (any(snaps.glob("*.ckpt")) or any(snaps.glob("*.pt"))):
        return ref_vposer
    return models_root / "vposer"


def _phc_mjcf_bundle_ready(phc_root: Path) -> bool:
    lib = phc_root / "phc" / "data" / "assets" / "mjcf"
    if not lib.is_dir():
        return False
    for name in ("smpl_0_humanoid.xml", "smpl_humanoid.xml", "smpl_1_humanoid.xml"):
        if (lib / name).is_file():
            return True
    return False


@dataclass(frozen=True)
class ExternalDependency:
    name: str
    kind: str
    source: str
    default_path: Path
    required_for_mvp: bool
    env_var: str = ""

    def resolved_path(self) -> Path:
        default = self.default_path.expanduser().resolve()
        if not self.env_var:
            return default
        raw = os.environ.get(self.env_var, "").strip()
        if not raw:
            return default
        env_path = Path(raw).expanduser().resolve()
        if self.name == "PHC":
            env_ok = _phc_mjcf_bundle_ready(env_path)
            if env_ok:
                return env_path
            if _phc_mjcf_bundle_ready(default):
                warnings.warn(
                    f"{self.env_var}={raw!r} is missing phc/data/assets/mjcf templates; "
                    f"using repo default {default}. Unset the variable or point it to a real PHC clone.",
                    UserWarning,
                    stacklevel=2,
                )
                return default
        return env_path

    def status(self) -> dict[str, Any]:
        path = self.resolved_path()
        return {
            **asdict(self),
            "default_path": str(self.default_path),
            "resolved_path": str(path),
            "exists": bool(path.exists()),
        }


def human_motion_dependencies() -> tuple[ExternalDependency, ...]:
    paths = project_paths(__file__)
    layout = HumanDatasetLayout.default()
    models_root = layout.models_root
    return (
        ExternalDependency(
            name="Qwen2.5-7B-Instruct",
            kind="huggingface_model",
            source="Qwen/Qwen2.5-7B-Instruct",
            default_path=models_root / "llm" / "Qwen2.5-7B-Instruct",
            required_for_mvp=False,
            env_var="AMONGUS_LLM_MODEL_DIR",
        ),
        ExternalDependency(
            name="MotionDiffuse",
            kind="git_repo",
            source="https://github.com/MotrixLab/MotionDiffuse.git",
            default_path=paths.reference_root / "MotionDiffuse",
            required_for_mvp=False,
            env_var="AMONGUS_MOTIONDIFFUSE_ROOT",
        ),
        ExternalDependency(
            name="MotionDiffuse-checkpoints",
            kind="model_weights",
            source="MotionDiffuse release assets or upstream instructions",
            default_path=models_root / "motion" / "MotionDiffuse",
            required_for_mvp=False,
            env_var="AMONGUS_MOTIONDIFFUSE_MODEL_DIR",
        ),
        ExternalDependency(
            name="PHC",
            kind="git_repo",
            source="https://github.com/ZhengyiLuo/PHC.git",
            default_path=paths.reference_root / "PHC",
            required_for_mvp=False,
            env_var="AMONGUS_PHC_ROOT",
        ),
        ExternalDependency(
            name="SMPLSim",
            kind="git_repo",
            source="https://github.com/ZhengyiLuo/SMPLSim.git",
            default_path=paths.reference_root / "SMPLSim",
            required_for_mvp=False,
            env_var="AMONGUS_SMPLSIM_ROOT",
        ),
        ExternalDependency(
            name="human_body_prior",
            kind="git_repo",
            source="https://github.com/nghorbani/human_body_prior.git",
            default_path=paths.reference_root / "human_body_prior",
            required_for_mvp=False,
            env_var="AMONGUS_HUMAN_BODY_PRIOR_ROOT",
        ),
        ExternalDependency(
            name="VPoser-weights",
            kind="model_weights",
            source="human_body_prior VPoser model release",
            default_path=_vposer_weights_default(paths.reference_root, models_root),
            required_for_mvp=False,
            env_var="AMONGUS_VPOSER_MODEL_DIR",
        ),
        ExternalDependency(
            name="SMPL-body-models",
            kind="licensed_body_models",
            source="SMPL/SMPL-X official downloads",
            default_path=layout.body_models_root,
            required_for_mvp=True,
            env_var="AMONGUS_SMPL_MODEL_DIR",
        ),
    )


def dependency_report() -> dict[str, Any]:
    deps = human_motion_dependencies()
    return {
        "dependencies": [dep.status() for dep in deps],
        "missing_required": [dep.name for dep in deps if dep.required_for_mvp and not dep.resolved_path().exists()],
        "missing_optional": [dep.name for dep in deps if not dep.required_for_mvp and not dep.resolved_path().exists()],
    }
