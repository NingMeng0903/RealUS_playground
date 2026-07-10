from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.human_motion.contracts import ActionBlock, GeneratedMotionMetadata
from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies


def _metadata(metadata: GeneratedMotionMetadata) -> dict[str, Any]:
    return {"human_motion": {"generated": metadata.to_json_dict()}}


@dataclass(frozen=True)
class PlaceholderMotionGenerator:
    """Small deterministic generator used for plumbing tests before MotionDiffuse is installed."""

    fps: float = 30.0
    gender: str = "neutral"
    model_type: str = "smpl"

    def generate(
        self,
        *,
        prompt: str,
        action_blocks: tuple[ActionBlock, ...],
        betas: np.ndarray | None = None,
        seed: int = 0,
    ) -> HumanMotionSequence:
        total_s = sum(float(block.duration_s) for block in action_blocks) if action_blocks else 2.0
        frame_count = max(int(round(total_s * self.fps)), 2)
        poses = np.zeros((frame_count, 72), dtype=np.float32)
        trans = np.zeros((frame_count, 3), dtype=np.float32)
        trans[:, 2] = 0.02
        if frame_count > 1:
            phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
            trans[:, 0] = 0.08 * np.sin(np.pi * phase)
            poses[:, 2] = 0.15 * np.sin(2.0 * np.pi * phase)
        metadata = GeneratedMotionMetadata(
            source="placeholder",
            prompt=prompt,
            action_blocks=tuple(action_blocks),
            model_name="placeholder_zero_smpl_v1",
            fps=float(self.fps),
            seed=int(seed),
        )
        return HumanMotionSequence(
            source_dataset="generated",
            sequence_name="placeholder_human_motion",
            source_path="human_motion.placeholder",
            model_type=self.model_type,
            fps=float(self.fps),
            gender=self.gender,
            betas=np.zeros(10, dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32).reshape(-1)[:10],
            poses=poses,
            trans=trans,
            metadata=_metadata(metadata),
        )


@dataclass(frozen=True)
class MotionDiffuseAdapter:
    """Adapter boundary for upstream MotionDiffuse without importing it at module import time."""

    repo_root: Path | None = None
    checkpoint_dir: Path | None = None

    def availability(self) -> dict[str, Any]:
        deps = {dep.name: dep for dep in human_motion_dependencies()}
        repo = self.repo_root or deps["MotionDiffuse"].resolved_path()
        ckpt = self.checkpoint_dir or deps["MotionDiffuse-checkpoints"].resolved_path()
        return {
            "repo_root": str(repo),
            "repo_exists": bool(repo.exists()),
            "checkpoint_dir": str(ckpt),
            "checkpoint_exists": bool(ckpt.exists()),
            "ready": bool(repo.exists() and ckpt.exists()),
        }

    def generate(self, *args: Any, **kwargs: Any) -> HumanMotionSequence:
        status = self.availability()
        if not status["ready"]:
            raise FileNotFoundError(
                "MotionDiffuse is not ready. Install the repo/checkpoints or use PlaceholderMotionGenerator. "
                f"Status: {status}"
            )
        raise NotImplementedError("MotionDiffuse invocation is isolated behind this adapter and must be wired after dependency inspection.")
