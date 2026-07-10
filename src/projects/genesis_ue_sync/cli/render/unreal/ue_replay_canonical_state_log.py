from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _ensure_src_on_path() -> None:
    script = Path(__file__).resolve()
    for parent in script.parents:
        if parent.name == "src" and (parent / "common" / "project.py").is_file():
            sp = str(parent)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return
    raise RuntimeError("Cannot locate src/ containing common/project.py")


def main() -> None:
    _ensure_src_on_path()
    from projects.genesis_ue_sync.integrations.ue import (
        EditorSessionPaths,
        enqueue_apply_canonical_scene_tick,
        wait_for_command_result,
    )
    from projects.genesis_ue_sync.sim_platform.sync import iter_canonical_state_jsonl

    parser = argparse.ArgumentParser(description="Replay canonical JSONL rows into UE editor session commands.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    paths = EditorSessionPaths(Path(args.session_dir).expanduser().resolve())
    rows = list(iter_canonical_state_jsonl(Path(args.jsonl).expanduser().resolve()))
    if args.max_rows > 0:
        rows = rows[: int(args.max_rows)]
    for idx, row in enumerate(rows):
        cmd = enqueue_apply_canonical_scene_tick(paths, row)
        result = wait_for_command_result(paths, cmd.request_id, timeout_s=float(args.timeout_s))
        if not result.success:
            raise RuntimeError(f"Replay failed at row {idx}: {result.detail}")
        if args.sleep_s > 0:
            time.sleep(float(args.sleep_s))


if __name__ == "__main__":
    main()
