from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from common.project import ProjectPaths, project_paths


def editor_python_search_paths(paths: ProjectPaths | None = None) -> list[Path]:
    paths = paths or project_paths()
    candidates = [paths.bedlam_engine_python_root]
    env_extra = os.environ.get("AMONGUS_UE_PYTHON_PATHS", "").strip()
    if env_extra:
        candidates.extend(Path(item).expanduser() for item in env_extra.split(os.pathsep) if item.strip())
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def ensure_editor_python_paths(paths: ProjectPaths | None = None) -> None:
    for candidate in reversed(editor_python_search_paths(paths)):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def import_editor_python_module(module_name: str, paths: ProjectPaths | None = None):
    ensure_editor_python_paths(paths)
    return importlib.import_module(module_name)
