"""Backend-independent inference loop with real PIL image observations."""

from dataclasses import dataclass, field
import json
import time
from typing import Protocol

from PIL import Image

from .crop import CropSession
from .policies import POLICIES, fixed_regions, random_regions
from .protocol import parse_action, system_prompt
from .vision import ImageBudget, display_image


@dataclass
class Generation:
    text: str
    generated_tokens: int | None = None
    visual_tokens: int | None = None
    prompt_tokens: int | None = None
    truncated: bool | None = None
    diagnostics: dict = field(default_factory=dict)


class Backend(Protocol):
    def generate(self, messages: list[dict]) -> Generation: ...


def image_count(messages):
    return sum(block.get("type") == "image" for message in messages
               if isinstance(message["content"], list) for block in message["content"])


def image_pixels(messages):
    return sum(block["image"].width * block["image"].height for message in messages
               if isinstance(message["content"], list) for block in message["content"] if block.get("type") == "image")


def tool_feedback(result, remaining):
    record = {**result.event, "remaining_calls": remaining}
    text = "Tool result: " + json.dumps(record, ensure_ascii=False, allow_nan=False)
    text += (" Give the final <answer> now; no tool calls remain." if remaining == 0
             else " Continue with one crop request or a final <answer>.")
    return text


def run_episode(backend: Backend, observation: dict, original: Image.Image, policy="O4", seed=42,
                whole_budget=ImageBudget(), high_budget=ImageBudget(max_pixels=2007040),
                crop_budget=ImageBudget(max_pixels=501760), diagnostic_regions=None):
    """Return raw trajectory and unscored prediction; labels are never accepted.

    O2/O3 expose two predetermined crops before generation. They are forced
    observation controls, not learned tool use. O4 has <=2 attempts and a third
    generation opportunity to answer. D0 is diagnostic only and must be gated
    to validation data by the caller.
    """
    allowed = {"id", "image", "question", "width", "height"}
    if set(observation) != allowed:
        raise ValueError("Pass observation_record(), never the labeled manifest row, to inference")
    if policy not in POLICIES:
        raise ValueError(f"Unknown observation policy: {policy}")
    if policy == "D0" and (not isinstance(diagnostic_regions, list) or not 1 <= len(diagnostic_regions) <= 2):
        raise ValueError("D0 requires one or two explicitly supplied diagnostic regions")
    start = time.perf_counter()
    initial = display_image(original, high_budget if policy == "O1" else whole_budget)
    session = CropSession(original, initial.size, max_calls=2 if policy in ("O2", "O3", "O4", "D0") else 0)
    messages = [
        {"role": "system", "content": system_prompt(initial.size, 2 if policy == "O4" else 0)},
        {"role": "user", "content": [{"type": "image", "image": initial}, {"type": "text", "text": observation["question"]}]},
    ]
    regions = fixed_regions(initial.size) if policy == "O2" else random_regions(initial.size, observation["id"], seed) if policy == "O3" else diagnostic_regions if policy == "D0" else []
    for bbox in regions:
        result = session.execute(bbox)
        result.event["origin"] = "diagnostic" if policy == "D0" else "forced_observation"
        if not result.event["success"]:
            raise ValueError(f"Invalid predeclared observation: {result.event['error']}")
        crop = display_image(result.image, crop_budget)
        result.event["display_size"] = list(crop.size)
        messages[1]["content"].extend([
            {"type": "text", "text": "Additional observation, original-reference bbox: " + json.dumps(bbox)},
            {"type": "image", "image": crop},
        ])
    turns, final, termination = [], "", "no_final_answer"
    for turn in range(3 if policy == "O4" else 1):
        pixels = image_pixels(messages)
        result = backend.generate(messages)
        turns.append({"turn": turn + 1, "text": result.text, "generated_tokens": result.generated_tokens,
                      "visual_tokens": result.visual_tokens, "prompt_tokens": result.prompt_tokens,
                      "image_pixels": pixels, "input_images": image_count(messages),
                      "truncated": result.truncated, "diagnostics": result.diagnostics})
        messages.append({"role": "assistant", "content": result.text})
        action = parse_action(result.text)
        final = result.text
        if result.truncated:
            termination = "generation_length_limit"
            break
        if action.kind == "answer":
            termination = "final_answer"
            break
        if policy != "O4":
            if action.kind == "tool":
                denied = session.execute(action.bbox)
                denied.event["origin"] = "model"
                denied.event["error"] = "tools_disallowed_by_policy"
            termination = "invalid_or_disallowed_action"
            break
        tool = session.execute(action.bbox if action.kind == "tool" else None)
        tool.event["origin"] = "model" if action.kind == "tool" else "invalid_action"
        if action.kind == "invalid":
            tool.event["parse_error"] = action.error
        if turn == 2:
            termination = "final_answer_opportunity_exhausted"
            break
        crop = None
        if tool.image is not None:
            crop = display_image(tool.image, crop_budget)
            tool.event["display_size"] = list(crop.size)
        content = [{"type": "text", "text": tool_feedback(tool, session.remaining)}]
        if crop is not None:
            content.append({"type": "image", "image": crop})
        messages.append({"role": "user", "content": content})
    def total(field):
        return sum(turn[field] for turn in turns) if all(turn[field] is not None for turn in turns) else None
    truncation = any(turn["truncated"] for turn in turns) if all(turn["truncated"] is not None for turn in turns) else None
    prediction = {
        "id": observation["id"], "completion": final, "policy": policy, "seed": seed,
        "tool_events": session.events, "truncated": truncation, "termination": termination,
        "cost": {"generated_tokens": total("generated_tokens"), "visual_tokens": total("visual_tokens"),
                 "image_pixels": total("image_pixels"), "latency_seconds": time.perf_counter() - start},
    }
    return {"prediction": prediction, "messages": messages, "turns": turns, "reference_size": list(initial.size)}
