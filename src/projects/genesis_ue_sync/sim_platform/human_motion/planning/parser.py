from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from projects.genesis_ue_sync.sim_platform.human_motion.contracts import ActionBlock, ContactMask


_ACTION_ALIASES: tuple[tuple[str, str], ...] = (
    ("躺下", "lie_down"),
    ("仰卧", "supine"),
    ("正面躺", "supine"),
    ("翻身", "roll_over"),
    ("侧身", "side_lying"),
    ("趴下", "prone"),
    ("俯卧", "prone"),
    ("爬起", "get_up"),
    ("撑床", "push_on_bed"),
    ("四肢着地", "all_fours"),
    ("爬行", "crawl"),
)


def _contact_for_action(action: str) -> ContactMask:
    if action in {"supine", "lie_down"}:
        return ContactMask({"back_contact": 1.0, "pelvis_contact": 0.8})
    if action == "prone":
        return ContactMask({"chest_contact": 1.0, "pelvis_contact": 0.7})
    if action == "side_lying":
        return ContactMask({"left_side_contact": 0.6, "right_side_contact": 0.6, "pelvis_contact": 0.7})
    if action in {"roll_over", "get_up", "push_on_bed"}:
        return ContactMask({"left_elbow_push": 0.8, "right_elbow_push": 0.8, "left_palm_support": 0.6, "right_palm_support": 0.6})
    if action in {"all_fours", "crawl"}:
        return ContactMask({"left_palm_support": 1.0, "right_palm_support": 1.0, "left_knee_contact": 1.0, "right_knee_contact": 1.0})
    return ContactMask()


@dataclass(frozen=True)
class RuleBasedActionParser:
    """Deterministic fallback parser for prompts before a local LLM is wired in."""

    default_duration_s: float = 2.0

    def parse(self, prompt: str) -> tuple[ActionBlock, ...]:
        text = str(prompt).strip()
        if not text:
            return ()
        parsed_json = self._parse_json_blocks(text)
        if parsed_json:
            return parsed_json

        parts = [p.strip() for p in re.split(r"->|→|，|,|然后|再|and then", text) if p.strip()]
        if not parts:
            parts = [text]
        blocks: list[ActionBlock] = []
        t = 0.0
        for raw in parts:
            action = self._classify_action(raw)
            duration = self._duration_from_text(raw) or self.default_duration_s
            blocks.append(
                ActionBlock(
                    action=action,
                    duration_s=duration,
                    start_time_s=t,
                    target_pose=action,
                    bed_region="center",
                    contact_mask=_contact_for_action(action),
                    notes=raw,
                )
            )
            t += duration
        return tuple(blocks)

    def _parse_json_blocks(self, text: str) -> tuple[ActionBlock, ...]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ()
        blocks_payload: Any
        if isinstance(payload, dict):
            blocks_payload = payload.get("action_blocks") or payload.get("actions") or []
        else:
            blocks_payload = payload
        if not isinstance(blocks_payload, list):
            return ()
        blocks = [ActionBlock.from_mapping(item) for item in blocks_payload if isinstance(item, dict)]
        return tuple(blocks)

    def _classify_action(self, text: str) -> str:
        for needle, action in _ACTION_ALIASES:
            if needle in text:
                return action
        return "generic_bed_motion"

    def _duration_from_text(self, text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|秒|sec|seconds)", text, flags=re.IGNORECASE)
        if not match:
            return None
        return max(float(match.group(1)), 0.1)
