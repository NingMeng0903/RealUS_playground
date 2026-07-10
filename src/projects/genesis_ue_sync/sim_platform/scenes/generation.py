from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import project_paths

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class WeightedOptionSpec:
    value: Any
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariationAxisSpec:
    name: str
    sampler: str = "choice"
    options: list[WeightedOptionSpec] = field(default_factory=list)
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionSourceSelectorSpec:
    source_type: str = "prepared_sequence"
    path_glob: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGenerationJobSpec:
    name: str
    base_scene_path: str
    output_root: str
    sample_count: int
    seed: int = 0
    motion_sources: list[MotionSourceSelectorSpec] = field(default_factory=list)
    variation_axes: list[VariationAxisSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_base_scene_path(self) -> Path:
        return project_paths(__file__).resolve_from_root(self.base_scene_path)

    @property
    def resolved_output_root(self) -> Path:
        return project_paths(__file__).resolve_from_root(self.output_root)


def _load_payload(path: Path) -> dict[str, Any]:
    raw_text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level mapping in generation config: {path}")
    return payload


def load_scene_generation_job(path: str | Path) -> SceneGenerationJobSpec:
    payload = _load_payload(Path(path))
    motion_sources = [
        MotionSourceSelectorSpec(
            source_type=str(item.get("source_type", "prepared_sequence")),
            path_glob=str(item.get("path_glob", "")),
            tags=[str(tag) for tag in item.get("tags", [])],
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("motion_sources", [])
    ]
    variation_axes = [
        VariationAxisSpec(
            name=str(item["name"]),
            sampler=str(item.get("sampler", "choice")),
            options=[
                WeightedOptionSpec(
                    value=option.get("value"),
                    weight=float(option.get("weight", 1.0)),
                    metadata=dict(option.get("metadata", {})),
                )
                for option in item.get("options", [])
            ],
            enabled=bool(item.get("enabled", True)),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("variation_axes", [])
    ]
    return SceneGenerationJobSpec(
        name=str(payload["name"]),
        base_scene_path=str(payload["base_scene_path"]),
        output_root=str(payload["output_root"]),
        sample_count=int(payload["sample_count"]),
        seed=int(payload.get("seed", 0)),
        motion_sources=motion_sources,
        variation_axes=variation_axes,
        metadata=dict(payload.get("metadata", {})),
    )
