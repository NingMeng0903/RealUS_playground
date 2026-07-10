from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.project import project_paths


@dataclass(frozen=True)
class HumanDatasetLayout:
    root: Path

    @classmethod
    def default(cls) -> "HumanDatasetLayout":
        return cls(root=project_paths(__file__).dataset_root)

    @property
    def raw_root(self) -> Path:
        return self.root / "raw" / "humans"

    @property
    def processed_root(self) -> Path:
        return self.root / "processed" / "humans"

    @property
    def intermediate_root(self) -> Path:
        return self.root / "intermediate" / "humans"

    @property
    def demo_root(self) -> Path:
        return self.root / "demo_video" / "humans"

    @property
    def amass_raw_root(self) -> Path:
        return self.raw_root / "amass"

    @property
    def amass_hf_root(self) -> Path:
        return self.raw_root / "amass_hf"

    @property
    def babel_raw_root(self) -> Path:
        return self.raw_root / "babel"

    @property
    def babel_release_root(self) -> Path:
        return self.babel_raw_root / "babel_v1.0_release"

    @property
    def bedlam_raw_root(self) -> Path:
        return self.raw_root / "bedlam"

    @property
    def bedlam_hf_root(self) -> Path:
        return self.raw_root / "bedlam_hf"

    @property
    def unified_sequences_root(self) -> Path:
        return self.processed_root / "unified_sequences"

    @property
    def mesh_sequences_root(self) -> Path:
        return self.processed_root / "mesh_sequences"

    @property
    def support_reference_batches_root(self) -> Path:
        return self.processed_root / "support_reference_batches"

    @property
    def generated_sequences_root(self) -> Path:
        return self.processed_root / "generated_sequences"

    @property
    def refit_sequences_root(self) -> Path:
        return self.processed_root / "refit_sequences"

    @property
    def motion_manifests_root(self) -> Path:
        return self.processed_root / "motion_manifests"

    @property
    def diagnostics_root(self) -> Path:
        return self.processed_root / "diagnostics"

    @property
    def body_models_root(self) -> Path:
        return self.intermediate_root / "body_models"

    @property
    def models_root(self) -> Path:
        return self.intermediate_root / "models"

    @property
    def support_semantics_models_root(self) -> Path:
        return self.models_root / "support_semantics"

    def ensure(self) -> None:
        for path in (
            self.raw_root,
            self.processed_root,
            self.intermediate_root,
            self.demo_root,
            self.amass_raw_root,
            self.amass_hf_root,
            self.babel_raw_root,
            self.bedlam_raw_root,
            self.bedlam_hf_root,
            self.unified_sequences_root,
            self.mesh_sequences_root,
            self.support_reference_batches_root,
            self.generated_sequences_root,
            self.refit_sequences_root,
            self.motion_manifests_root,
            self.diagnostics_root,
            self.body_models_root,
            self.models_root,
            self.support_semantics_models_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
