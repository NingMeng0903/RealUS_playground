from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.sim_platform.human_motion.planning import QwenActionParser, RuleBasedActionParser


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse a bed-motion prompt into action blocks.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--default-duration-s", type=float, default=2.0)
    p.add_argument("--parser", choices=("rule", "qwen"), default="rule")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.parser == "qwen":
        parser = QwenActionParser(fallback=RuleBasedActionParser(default_duration_s=float(args.default_duration_s)))
    else:
        parser = RuleBasedActionParser(default_duration_s=float(args.default_duration_s))
    payload = {"prompt": args.prompt, "action_blocks": [block.to_json_dict() for block in parser.parse(args.prompt)]}
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
