"""BEDLAM / Meshcapade avatar catalog helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class BedlamAvatarRecord:
    avatar_id: str
    body_name: str
    gender: str
    skeletal_mesh_path: str
    texture_body: str | None = None
    texture_clothing: str | None = None
    texture_clothing_overlay: str | None = None
    smplx_beta_prior_key: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BedlamAvatarIndex:
    records: tuple[BedlamAvatarRecord, ...]
    beta_priors: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def by_id(self) -> dict[str, BedlamAvatarRecord]:
        return {rec.avatar_id: rec for rec in self.records}

    def beta_prior_values(self, key: str | None) -> list[float] | None:
        if not key:
            return None
        entry = self.beta_priors.get(str(key))
        if not isinstance(entry, dict):
            return None
        vals = entry.get("values")
        if not isinstance(vals, list):
            return None
        return [float(v) for v in vals]


def load_bedlam_avatar_index(path: Path) -> BedlamAvatarIndex:
    path = Path(path).expanduser().resolve()
    raw_text = path.read_text(encoding="utf-8")
    if yaml is not None and path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in avatar index: {path}")
    entries = payload.get("avatars") or payload.get("records") or []
    pri_raw = payload.get("beta_priors") or {}
    pri_copy: dict[str, Any] = {}
    for key, val in pri_raw.items():
        pri_copy[str(key)] = dict(val) if isinstance(val, dict) else val
    records: list[BedlamAvatarRecord] = []
    for item in entries:
        records.append(
            BedlamAvatarRecord(
                avatar_id=str(item["avatar_id"]),
                body_name=str(item["body_name"]),
                gender=str(item.get("gender", "neutral")),
                skeletal_mesh_path=str(item["skeletal_mesh_path"]),
                texture_body=item.get("texture_body"),
                texture_clothing=item.get("texture_clothing"),
                texture_clothing_overlay=item.get("texture_clothing_overlay"),
                smplx_beta_prior_key=item.get("smplx_beta_prior_key"),
                notes=str(item.get("notes", "")),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return BedlamAvatarIndex(records=tuple(records), beta_priors=pri_copy, source_path=path)


def default_avatar_index_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "bedlam" / "avatar_index.yaml"
