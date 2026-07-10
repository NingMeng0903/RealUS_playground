from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CONTACT_KEYS: tuple[str, ...] = (
    "back_contact",
    "chest_contact",
    "left_side_contact",
    "right_side_contact",
    "pelvis_contact",
    "left_elbow_push",
    "right_elbow_push",
    "left_palm_support",
    "right_palm_support",
    "left_knee_contact",
    "right_knee_contact",
)


@dataclass(frozen=True)
class ContactMask:
    """Heuristic semantic contact hints used to weight refit losses."""

    values: dict[str, float] = field(default_factory=dict)

    def normalized(self) -> dict[str, float]:
        out = {key: 0.0 for key in CONTACT_KEYS}
        for key, value in self.values.items():
            if key not in out:
                continue
            out[key] = float(max(0.0, min(1.0, value)))
        return out

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ContactMask":
        if not payload:
            return cls()
        return cls(values={str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float, bool))})


@dataclass(frozen=True)
class ActionBlock:
    action: str
    duration_s: float
    start_time_s: float = 0.0
    target_pose: str = ""
    facing: str = ""
    bed_region: str = "center"
    contact_mask: ContactMask = field(default_factory=ContactMask)
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["contact_mask"] = self.contact_mask.normalized()
        return data

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ActionBlock":
        return cls(
            action=str(payload.get("action") or payload.get("name") or "unknown"),
            duration_s=float(payload.get("duration_s", 1.0)),
            start_time_s=float(payload.get("start_time_s", 0.0)),
            target_pose=str(payload.get("target_pose", "")),
            facing=str(payload.get("facing", "")),
            bed_region=str(payload.get("bed_region", "center")),
            contact_mask=ContactMask.from_mapping(payload.get("contact_mask") or {}),
            notes=str(payload.get("notes", "")),
        )


def merged_contact_mask_from_action_blocks(action_blocks: tuple[ActionBlock, ...]) -> dict[str, float]:
    """Merge semantic contact hints across blocks (per-key max) for manifests and diagnostics."""

    out: dict[str, float] = {key: 0.0 for key in CONTACT_KEYS}
    for block in action_blocks:
        mask = block.contact_mask.normalized()
        for key, value in mask.items():
            if key in out:
                out[key] = max(out[key], float(value))
    return out


@dataclass(frozen=True)
class GeneratedMotionMetadata:
    source: str
    prompt: str
    action_blocks: tuple[ActionBlock, ...]
    model_name: str = ""
    fps: float = 30.0
    seed: int = 0
    adapter_version: str = "human_motion_generation_v1"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "prompt": self.prompt,
            "action_blocks": [block.to_json_dict() for block in self.action_blocks],
            "model_name": self.model_name,
            "fps": float(self.fps),
            "seed": int(self.seed),
            "adapter_version": self.adapter_version,
        }


@dataclass(frozen=True)
class PhysicalRefitDiagnostics:
    """Offline refit summary stored in ``MotionManifest.refit`` and under ``metrics['physical_refit']``.

    Typical ``metrics`` entries: ``support_plane_z_m``, ``loss_weights`` (see HamiltonianLossWeights),
    ``contact_mask_normalized`` (CONTACT_KEYS), explanatory ``note``. The same contact mask and
    loss weights are also exposed as top-level optional fields for stable JSON consumers.
    """

    method: str
    input_npz_path: str
    output_npz_path: str
    frame_count: int
    max_bed_penetration_m: float = 0.0
    mean_root_shift_m: float = 0.0
    max_joint_speed_rad_s: float = 0.0
    stages: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    # Semantic contact hints (normalized CONTACT_KEYS) used to scale damping / penetration vs tracking; see loss_terms.
    contact_mask_normalized: dict[str, float] | None = None
    loss_weights: dict[str, float] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class MotionManifest:
    sequence_npz_path: str
    prompt: str = ""
    action_blocks: tuple[ActionBlock, ...] = ()
    generated: GeneratedMotionMetadata | None = None
    refit: PhysicalRefitDiagnostics | None = None
    tags: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sequence_npz_path": self.sequence_npz_path,
            "prompt": self.prompt,
            "action_blocks": [block.to_json_dict() for block in self.action_blocks],
            "generated": self.generated.to_json_dict() if self.generated is not None else None,
            "refit": self.refit.to_json_dict() if self.refit is not None else None,
            "tags": list(self.tags),
            "metrics": dict(self.metrics),
        }

    def save(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_json_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return output_path
