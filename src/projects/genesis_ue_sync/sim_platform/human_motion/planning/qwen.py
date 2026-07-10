from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.sim_platform.human_motion.contracts import ActionBlock
from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies
from projects.genesis_ue_sync.sim_platform.human_motion.planning.parser import RuleBasedActionParser


ACTION_BLOCK_SYSTEM_PROMPT = """You convert bed human-motion requests into JSON.
Return only JSON with an action_blocks array. Each item must include:
action, duration_s, start_time_s, target_pose, facing, bed_region, contact_mask, notes.
contact_mask values must be numbers in [0, 1].
"""


@dataclass(frozen=True)
class QwenActionParser:
    model_dir: Path | None = None
    fallback: RuleBasedActionParser = RuleBasedActionParser()

    def availability(self) -> dict[str, Any]:
        deps = {dep.name: dep for dep in human_motion_dependencies()}
        model_dir = self.model_dir or deps["Qwen2.5-7B-Instruct"].resolved_path()
        return {"model_dir": str(model_dir), "model_exists": bool(model_dir.exists()), "ready": bool(model_dir.exists())}

    def parse(self, prompt: str) -> tuple[ActionBlock, ...]:
        status = self.availability()
        if not status["ready"]:
            return self.fallback.parse(prompt)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            return self.fallback.parse(prompt)

        model_dir = Path(status["model_dir"])
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        messages = [
            {"role": "system", "content": ACTION_BLOCK_SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        response_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return self.fallback.parse(prompt)
        raw_blocks = payload.get("action_blocks") if isinstance(payload, dict) else payload
        if not isinstance(raw_blocks, list):
            return self.fallback.parse(prompt)
        blocks = [ActionBlock.from_mapping(item) for item in raw_blocks if isinstance(item, dict)]
        return tuple(blocks) if blocks else self.fallback.parse(prompt)
