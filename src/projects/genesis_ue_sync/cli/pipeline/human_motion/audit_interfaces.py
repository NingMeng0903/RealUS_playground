from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.human_motion.interfaces import current_motion_interface_audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write the current human motion interface audit.")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = current_motion_interface_audit().to_json_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
