from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize generated human motion manifests.")
    p.add_argument("--manifest-root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object in manifest: {path}")
    return payload


def main() -> None:
    args = parse_args()
    paths = sorted({*args.manifest_root.rglob("motion_manifest.json"), *args.manifest_root.rglob("*manifest*.json")})
    action_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    frame_counts: list[int] = []
    failures = 0
    for path in paths:
        payload = _load(path)
        for block in payload.get("action_blocks") or []:
            if isinstance(block, dict):
                action_counts[str(block.get("action", "unknown"))] += 1
        for tag in payload.get("tags") or []:
            tag_counts[str(tag)] += 1
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        if "frame_count" in metrics:
            frame_counts.append(int(metrics["frame_count"]))
        refit = payload.get("refit") if isinstance(payload.get("refit"), dict) else {}
        failures += int(bool(refit.get("warnings")))
    summary = {
        "manifest_root": str(args.manifest_root),
        "manifest_count": len(paths),
        "action_counts": dict(action_counts),
        "tag_counts": dict(tag_counts),
        "min_frame_count": min(frame_counts) if frame_counts else 0,
        "max_frame_count": max(frame_counts) if frame_counts else 0,
        "warning_count": int(failures),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
