from __future__ import annotations

import os
import sys
from pathlib import Path

import unreal

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.integrations.ue import EditorSessionPaths, EditorSessionStatus


def _current_level_path() -> str:
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
        if world is None:
            return ""
        return str(world.get_path_name())
    except Exception:
        return ""


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: ue_editor_session_ready.py <session_dir>")
    session_dir = Path(sys.argv[1]).expanduser().resolve()
    status = EditorSessionStatus(
        state="ready",
        project_path=str(project_paths(__file__).bedlam_unreal_project_file),
        ready=True,
        level_path=_current_level_path(),
        detail="manual_ready_signal",
        process_pid=os.getpid(),
    )
    status.save(EditorSessionPaths(session_dir))
    unreal.log(f"UE session ready written: {session_dir}")


if __name__ == "__main__":
    main()
