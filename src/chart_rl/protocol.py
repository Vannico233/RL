"""Strict, shared action grammar for evaluation and training rewards."""

from dataclasses import dataclass
import json
import re
from typing import Any


TOOL_NAME = "image_zoom_in_tool"
_THOUGHT = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
_ANSWER = re.compile(r"\A<answer>([^<>]+)</answer>\Z", re.DOTALL)
_TOOL = re.compile(r"\A<tool_call>(.*?)</tool_call>\Z", re.DOTALL)


@dataclass(frozen=True)
class Action:
    kind: str
    answer: str | None = None
    bbox: Any = None
    error: str | None = None


def _reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_action(text: str) -> Action:
    """One optional think block followed by exactly one answer or tool call.

    Reject ambiguous multiple answers, trailing prose, and mixed actions. No
    numeric extraction fallback is applied. Bounding-box geometry is validated
    by the tool, so malformed geometry still consumes an attempted call.
    """
    if not isinstance(text, str):
        return Action("invalid", error="non_text_completion")
    text = _THOUGHT.sub("", text, count=1).strip()
    answer = _ANSWER.fullmatch(text)
    if answer:
        value = answer.group(1).strip()
        return Action("answer", answer=value) if value else Action("invalid", error="empty_answer")
    tool = _TOOL.fullmatch(text)
    if not tool:
        return Action("invalid", error="invalid_action_format")
    try:
        value = json.loads(tool.group(1), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return Action("invalid", error="invalid_tool_json")
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return Action("invalid", error="invalid_tool_schema")
    if value["name"] != TOOL_NAME:
        return Action("invalid", error="unknown_tool")
    args = value["arguments"]
    if not isinstance(args, dict) or set(args) != {"bbox_2d"}:
        return Action("invalid", error="invalid_tool_arguments")
    return Action("tool", bbox=args["bbox_2d"])


def tool_action(bbox) -> str:
    payload = {"name": TOOL_NAME, "arguments": {"bbox_2d": list(bbox)}}
    return "<tool_call>" + json.dumps(payload, allow_nan=False, separators=(",", ":")) + "</tool_call>"


def system_prompt(reference_size: tuple[int, int], max_calls: int = 2) -> str:
    width, height = reference_size
    return (
        "Answer the chart question using the image. You may answer directly or inspect a crop. "
        f"The first image is the coordinate reference: width={width}, height={height}. "
        "Every bbox uses pixels of that FIRST image, including on later calls; never use crop-local coordinates. "
        f"At most {max_calls} tool attempts are allowed, including invalid attempts. "
        "Each turn must contain exactly one action, optionally preceded by one <think>...</think> block. "
        'To crop, output <tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[x1,y1,x2,y2]}}</tool_call>. '
        "Coordinates must satisfy 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height. "
        "To finish, output <answer>short answer</answer>. Preserve the units requested by the question. "
        "Do not put any prose after the action. Cropping itself earns no reward."
    )
