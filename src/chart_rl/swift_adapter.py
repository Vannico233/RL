"""CPU-testable hooks for ms-swift v4.6.0's MultiTurnScheduler contract.

The external plugin injects the real framework base. Unit tests may inject a
minimal base, which checks our hooks but does not prove GPU/framework behavior.
"""

import base64
from io import BytesIO
import json
import inspect
import math
import os
from pathlib import Path
import time
import uuid

from PIL import Image

from .crop import CropSession
from .data import sha256_file
from .episode import tool_feedback
from .protocol import parse_action
from .reward import score_completion
from .vision import ImageBudget, display_image


SWIFT_COMMIT = "ad818a2cdfafaf4305ac40c4acd892a211978bc2"


def load_received_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return image.convert("RGB").copy()
    if not isinstance(value, str) or value.startswith(("http://", "https://")):
        raise ValueError("Expected an in-memory, base64, or local frozen image")
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Expected a base64 image data URI")
    else:
        encoded = value
        if len(value) < 4096:
            try:
                path = Path(value)
                if path.is_file():
                    with Image.open(path) as image:
                        return image.convert("RGB").copy()
            except OSError:
                pass
    return load_received_image(base64.b64decode(encoded, validate=True))


def reward_batch(completions, solution):
    if len(completions) != len(solution):
        raise ValueError("Completion/solution lengths differ; refusing silent zip truncation")
    return [score_completion(text, target).reward for text, target in zip(completions, solution)]


def build_scheduler_class(base):
    class ChartCropScheduler(base):
        def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
            if max_turns not in (None, 3):
                raise ValueError("Use max_turns=3: two tool attempts plus one final-answer opportunity")
            super().__init__(infer_engine, 3, *args, **kwargs)

        async def on_trajectory_start(self, requests):
            for request in requests:
                keys = ("chart_id", "chart_original_image", "chart_image_sha256", "chart_reference_size",
                        "chart_crop_max_pixels", "chart_min_pixels")
                metadata = {key: request.data_dict[key] for key in keys}
                original_path = Path(metadata["chart_original_image"])
                if sha256_file(original_path) != metadata["chart_image_sha256"]:
                    raise ValueError("Original image changed after manifest/export freeze")
                if len(request.images) != 1:
                    raise ValueError("Each training trajectory must start with exactly one whole image")
                first = load_received_image(request.images[0])
                if list(first.size) != metadata["chart_reference_size"]:
                    raise ValueError("Displayed first-image size differs from exported coordinate reference")
                with Image.open(original_path) as original:
                    session = CropSession(original, first.size)
                request._chart_session = session
                request._chart_metadata = metadata
                request._chart_turns = []
                request._chart_started = time.perf_counter()
                request._chart_crop_budget = ImageBudget(max_pixels=metadata["chart_crop_max_pixels"], min_pixels=metadata["chart_min_pixels"])
                request._chart_loaded_adapters = None
                engine = getattr(getattr(self, "infer_engine", None), "engine", None)
                if engine is not None and hasattr(engine, "list_loras"):
                    loaded = engine.list_loras()
                    if inspect.isawaitable(loaded):
                        loaded = await loaded
                    request._chart_loaded_adapters = sorted(int(value) for value in loaded)
                # Do not pass labels or arbitrary dataset metadata into inference.
                # The trainer retains its original sample.extra for reward scoring.
                request.data_dict = dict(metadata)

        def check_finished(self, infer_request, response_choice, current_turn):
            return (super().check_finished(infer_request, response_choice, current_turn)
                    or parse_action(response_choice.message.content).kind == "answer")

        async def on_turn_end(self, infer_request, response_choice, current_turn):
            session = infer_request._chart_session
            action = parse_action(response_choice.message.content)
            token_ids = response_choice.token_ids
            logprob_data = getattr(response_choice, "logprobs", None)
            logprobs = logprob_data.get("content") if isinstance(logprob_data, dict) else None
            sizes = [list(load_received_image(value).size) for value in infer_request.images]
            infer_request._chart_turns.append({
                "turn": current_turn, "completion": response_choice.message.content,
                "action_kind": action.kind, "finish_reason": response_choice.finish_reason,
                "generated_tokens": len(token_ids) if token_ids is not None else None,
                "logprob_count": len(logprobs) if logprobs is not None else None,
                "logprobs_align_with_sampled_ids": (len(logprobs) == len(token_ids)) if logprobs is not None and token_ids is not None else None,
                "logprobs_finite": all(math.isfinite(item["logprob"]) for item in logprobs) if logprobs is not None else None,
                "display_image_sizes": sizes, "display_image_pixels": sum(w * h for w, h in sizes),
                "elapsed_seconds": time.perf_counter() - infer_request._chart_started,
            })
            if current_turn >= 3 and action.kind != "answer" and response_choice.finish_reason != "length":
                event = session.execute(action.bbox if action.kind == "tool" else None).event
                event["origin"] = "model" if action.kind == "tool" else "invalid_action"
            info = {
                "images": list(infer_request.images),
                "chart_id": infer_request._chart_metadata["chart_id"],
                "chart_tool_events": list(session.events), "chart_turns": list(infer_request._chart_turns),
                "chart_loaded_adapter_ids": infer_request._chart_loaded_adapters,
                "chart_trajectory_time": time.time(),
            }
            if self.check_finished(infer_request, response_choice, current_turn):
                trace_dir = os.environ.get("CHART_RL_TRACE_DIR")
                if trace_dir:
                    path = Path(trace_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    trace = {key: value for key, value in info.items() if key != "images"}
                    trace["reference_size"] = list(session.reference_size)
                    trace["original_size"] = list(session.original.size)
                    trace["framework_commit"] = SWIFT_COMMIT
                    (path / f"{uuid.uuid4().hex}.json").write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            return {"rollout_infos": info}

        def step(self, infer_request, response_choice, current_turn):
            session = infer_request._chart_session
            action = parse_action(response_choice.message.content)
            result = session.execute(action.bbox if action.kind == "tool" else None)
            result.event["origin"] = "model" if action.kind == "tool" else "invalid_action"
            if action.kind == "invalid":
                result.event["parse_error"] = action.error
            crop = None
            if result.image is not None:
                crop = display_image(result.image, infer_request._chart_crop_budget)
                result.event["display_size"] = list(crop.size)
            feedback = tool_feedback(result, session.remaining)
            if crop is not None:
                feedback += "\n<image>"
                infer_request.images.append(crop)
            infer_request.messages.append({"role": "user", "content": feedback})
            # The framework retains sampled assistant IDs/log-probs and masks
            # user observations. Never synthesize or rewrite sampled responses.
            return {"infer_request": infer_request, "rollout_infos": {
                "images": list(infer_request.images), "chart_tool_events": list(session.events),
            }}

    return ChartCropScheduler
