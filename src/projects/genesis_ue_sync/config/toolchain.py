from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from common.project import ProjectPaths, project_paths


def discover_python_executable() -> str:
    return sys.executable


def discover_python_command(*, preferred_conda_env: str | None = "genesis") -> list[str]:
    conda_bin = shutil.which("conda")
    if conda_bin is not None and preferred_conda_env:
        return [conda_bin, "run", "-n", preferred_conda_env, "python"]
    return [discover_python_executable()]


def _resolve_executable(
    *,
    env_var: str,
    candidate_paths: list[Path],
    candidate_names: list[str],
    description: str,
) -> str:
    env_path = os.environ.get(env_var, "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    for candidate in candidate_paths:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    for candidate_name in candidate_names:
        resolved = shutil.which(candidate_name)
        if resolved:
            return resolved
    raise RuntimeError(f"Cannot find {description}. Set {env_var} or install it in PATH.")


def _blender_candidates_on_drive(drive_root: Path) -> list[Path]:
    out: list[Path] = []
    for rel in (Path("blender") / "blender", Path("Blender") / "blender"):
        p = drive_root / rel
        if p.is_file() and os.access(p, os.X_OK):
            out.append(p)
    software = drive_root / "software"
    if software.is_dir():
        try:
            for child in sorted(software.iterdir()):
                if not child.is_dir() or "blender" not in child.name.lower():
                    continue
                exe = child / "blender"
                if exe.is_file() and os.access(exe, os.X_OK):
                    out.append(exe)
        except OSError:
            pass
    return out


def discover_blender_executable(paths: ProjectPaths | None = None) -> str:
    paths = paths or project_paths()
    home = Path.home()
    repo_parent = paths.root.resolve().parent
    blender_dir = os.environ.get("AMONGUS_BLENDER_DIR", "").strip()
    dir_candidates: list[Path] = []
    if blender_dir:
        p = Path(blender_dir).expanduser().resolve()
        if p.is_file() and os.access(p, os.X_OK):
            dir_candidates.append(p)
        else:
            exe = p / "blender"
            if exe.is_file() and os.access(exe, os.X_OK):
                dir_candidates.append(exe)
    drive_candidates = _blender_candidates_on_drive(repo_parent)
    return _resolve_executable(
        env_var="AMONGUS_BLENDER",
        candidate_paths=dir_candidates
        + drive_candidates
        + [
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            home / "software" / "blender-4.5.8-linux-x64" / "blender",
            home / "software" / "blender-5.0.1-linux-x64" / "blender",
            paths.root / "third_party" / "blender" / "blender",
            home / "Applications" / "blender" / "blender",
        ],
        candidate_names=["blender"],
        description=(
            "Blender executable (set AMONGUS_BLENDER to the binary, "
            "or AMONGUS_BLENDER_DIR to the extracted folder containing `blender`)"
        ),
    )


def discover_unreal_editor_executable(paths: ProjectPaths | None = None) -> str:
    _ = paths or project_paths()
    return _resolve_executable(
        env_var="UNREAL_EDITOR_CMD",
        candidate_paths=[
            Path("/media/camp/EXT_DRIVE/ue/UnrealEngine-5.3.2") / "Engine" / "Binaries" / "Linux" / "UnrealEditor-Cmd",
            Path("/media/camp/EXT_DRIVE/ue/UnrealEngine-5.3.2") / "Engine" / "Binaries" / "Linux" / "UnrealEditor",
            Path("/media/camp/EXT_DRIVE/ue/UnrealEngine") / "Engine" / "Binaries" / "Linux" / "UnrealEditor-Cmd",
            Path("/media/camp/EXT_DRIVE/ue/UnrealEngine") / "Engine" / "Binaries" / "Linux" / "UnrealEditor",
            Path.home() / "software" / "UnrealEngine" / "Engine" / "Binaries" / "Linux" / "UnrealEditor-Cmd",
            Path.home() / "software" / "UnrealEngine" / "Engine" / "Binaries" / "Linux" / "UnrealEditor",
        ],
        candidate_names=["UnrealEditor-Cmd", "UnrealEditor"],
        description="Unreal Editor executable",
    )
