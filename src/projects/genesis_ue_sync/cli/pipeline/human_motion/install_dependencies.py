from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print or execute dependency install commands for the human motion pipeline.")
    p.add_argument("--execute", action="store_true", help="Run clone/download commands. Default only prints the plan.")
    p.add_argument("--include-models", action="store_true", help="Include Hugging Face model downloads when --execute is set.")
    return p.parse_args()


def _commands() -> list[dict[str, str]]:
    cmds: list[dict[str, str]] = []
    for dep in human_motion_dependencies():
        target = dep.resolved_path()
        if dep.kind == "git_repo":
            cmds.append({"name": dep.name, "kind": dep.kind, "command": f"git clone {dep.source} {target}"})
        elif dep.kind == "huggingface_model":
            cmds.append({"name": dep.name, "kind": dep.kind, "command": f"hf download {dep.source} --local-dir {target}"})
    return cmds


def main() -> None:
    args = parse_args()
    commands = _commands()
    if not args.include_models:
        commands = [cmd for cmd in commands if cmd["kind"] != "huggingface_model"]
    if not args.execute:
        print(json.dumps({"commands": commands, "execute": False}, indent=2, ensure_ascii=True))
        return
    results = []
    for item in commands:
        target = Path(item["command"].split()[-1])
        if target.exists():
            results.append({**item, "skipped": True, "reason": "target exists"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(item["command"], shell=True, check=False)
        results.append({**item, "returncode": int(completed.returncode)})
    print(json.dumps({"commands": results, "execute": True}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
