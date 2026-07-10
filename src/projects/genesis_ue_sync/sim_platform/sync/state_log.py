"""Append-only JSONL log for canonical scene states (offline UE replay, dataset provenance)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from projects.genesis_ue_sync.sim_platform.state.canonical import CanonicalSceneStateV1, canonical_scene_state_to_dict


def append_canonical_state_jsonl(path: Path | str, state: CanonicalSceneStateV1 | dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_scene_state_to_dict(state) if isinstance(state, CanonicalSceneStateV1) else dict(state)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def iter_canonical_state_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
