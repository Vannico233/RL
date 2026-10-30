"""Read-only training instrumentation for the pinned ms-swift release.

Hooks wrap framework calls and return their results unchanged. Runtime evidence
is written under the run directory; importing this file does not load torch.
"""

from collections import defaultdict
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import time


def write_event(path, kind, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": kind, "time": time.time(), **payload}, ensure_ascii=False, allow_nan=False) + "\n")


def record_rewards(rewards, kwargs):
    directory = os.environ.get("CHART_RL_AUDIT_DIR")
    if not directory:
        return
    ids = kwargs.get("chart_id", [])
    if len(ids) != len(rewards):
        raise ValueError("Missing row identity in training reward audit")
    grouped = defaultdict(list)
    # Chart identity includes the image. Generic question text alone is not
    # sufficient to identify a multimodal GRPO group.
    for prompt, reward in zip(ids, rewards):
        grouped[str(prompt)].append(reward)
    state = kwargs.get("trainer_state")
    write_event(Path(directory) / "rewards.jsonl", "reward_batch", step=getattr(state, "global_step", None),
                chart_ids=ids, groups=[{"chart_id": key, "rewards": values, "has_variance": len(set(values)) > 1}
                                      for key, values in grouped.items()])


def _as_list(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def _expected_ids(sample):
    ids = sample.response_token_ids
    if not ids:
        return None
    if isinstance(ids[0], int):
        ids = [ids]
    masks = sample.response_loss_mask
    if masks and isinstance(masks[0], int):
        masks = [masks]
    if masks and (len(masks) != len(ids) or any(len(a) != len(b) for a, b in zip(masks, ids))):
        raise ValueError("Framework response token IDs and loss masks have inconsistent shapes")
    return [token for turn, tokens in enumerate(ids) for index, token in enumerate(tokens)
            if not masks or masks[turn][index] == 1]


def build_callback_class(base):
    class ChartAuditCallback(base):
        def __init__(self, args, trainer):
            super().__init__(args, trainer)
            self.event_path = Path(args.output_dir) / "audit" / "training.jsonl"
            self._initial = None
            self._last = None
            self._original_sync = None
            self._original_encode = None

        def _fingerprint(self):
            digest = hashlib.sha256()
            for name, param in self.trainer.model.named_parameters():
                if param.requires_grad:
                    digest.update(name.encode("utf-8"))
                    digest.update(param.detach().float().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        def on_train_begin(self, args, state, control, **kwargs):
            from swift.rlhf_trainers import grpo_trainer as module
            parameters = [(name, param) for name, param in self.trainer.model.named_parameters() if param.requires_grad]
            if not parameters or any("lora_" not in name for name, _ in parameters):
                raise ValueError("Expected non-empty LoRA-only trainable parameters")
            if any(any(part in name.lower() for part in ("visual", "vision", "merger", "multi_modal_projector")) for name, _ in parameters):
                raise ValueError("Vision/connector parameters were unexpectedly made trainable")
            self._initial = self._last = self._fingerprint()
            write_event(self.event_path, "train_begin", step=state.global_step, fingerprint=self._initial,
                        trainable_parameters=sum(param.numel() for _, param in parameters),
                        trainable_names=[name for name, _ in parameters])
            self._original_sync = self.trainer._move_model_to_vllm
            original_sync = self._original_sync

            @wraps(original_sync)
            def audited_sync(*call_args, **call_kwargs):
                fingerprint = self._fingerprint()
                result = original_sync(*call_args, **call_kwargs)
                write_event(self.event_path, "weight_sync_ack", step=self.trainer.state.global_step,
                            fingerprint=fingerprint, rollout_enable_lora=bool(self.trainer.rollout_enable_lora),
                            note="Framework sync and cache-reset calls returned successfully; not a remote tensor checksum.")
                return result

            self.trainer._move_model_to_vllm = audited_sync
            self._original_encode = module.encode_sample
            original_encode = self._original_encode

            @wraps(original_encode)
            def audited_encode(sample, template, **encode_kwargs):
                encoded = original_encode(sample, template, **encode_kwargs)
                labels = encoded.get("labels")
                expected = _expected_ids(sample)
                if labels is not None:
                    labels = _as_list(labels)
                    actual = [token for token in labels if token != -100]
                    grid = encoded.get("image_grid_thw")
                    grid = _as_list(grid) if grid is not None else None
                    info = sample.rollout_infos or {}
                    turns = info.get("chart_turns", [])
                    sizes = turns[-1]["display_image_sizes"] if turns else None
                    expected_grid = [[1, h // 14, w // 14] for w, h in sizes] if sizes else None
                    logprobs = sample.rollout_logprobs
                    logprob_count = sum(len(turn) for turn in logprobs) if logprobs else None
                    visual_tokens = (sum(sum(w * h // (28 * 28) for w, h in turn["display_image_sizes"])
                                         for turn in turns) if turns else
                                     sum(t * h * w // 4 for t, h, w in grid) if grid else None)
                    generated_tokens = (sum(turn["generated_tokens"] for turn in turns)
                                        if turns and all(turn["generated_tokens"] is not None for turn in turns)
                                        else len(expected) if expected is not None else None)
                    write_event(self.event_path, "encoded_trajectory", step=self.trainer.state.global_step,
                                request_id=sample.request_id, mode="train" if self.trainer.model.training else "eval",
                                chart_id=sample.extra.get("chart_id"), input_tokens=len(labels),
                                generated_tokens=generated_tokens, visual_tokens=visual_tokens,
                                truncated=sample.is_truncated,
                                loss_tokens=len(actual), loss_tokens_match_rollout=(actual == expected) if expected is not None else None,
                                image_grid_thw=grid, expected_image_grid_thw=expected_grid,
                                image_grid_matches=(grid == expected_grid) if expected_grid is not None else None,
                                sampled_turns=len(turns), masked_token_count=sum(token == -100 for token in labels))
                    write_event(self.event_path, "logprob_alignment", step=self.trainer.state.global_step,
                                chart_id=sample.extra.get("chart_id"), rollout_logprob_count=logprob_count,
                                sampled_loss_token_count=len(expected) if expected is not None else None,
                                matches=(logprob_count == len(expected)) if logprob_count is not None and expected is not None else None)
                return encoded

            module.encode_sample = audited_encode

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            import torch
            terms = [param.grad.detach().float().square().sum() for param in self.trainer.model.parameters()
                     if param.requires_grad and param.grad is not None]
            norm = float(torch.stack(terms).sum().sqrt().item()) if terms else 0.0
            finite = math.isfinite(norm)
            write_event(self.event_path, "gradient", step=state.global_step, gradient_norm=norm if finite else None,
                        nonzero_gradient=norm > 0 and finite, finite=finite)
            if not finite:
                raise RuntimeError("Non-finite gradient; inspect the training audit before continuing")

        def on_step_end(self, args, state, control, **kwargs):
            import torch
            fingerprint = self._fingerprint()
            write_event(self.event_path, "optimizer_step", step=state.global_step, fingerprint=fingerprint,
                        changed_since_previous=(fingerprint != self._last), changed_since_initial=(fingerprint != self._initial),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            self._last = fingerprint

        def on_train_end(self, args, state, control, **kwargs):
            from swift.rlhf_trainers import grpo_trainer as module
            if self._original_sync is not None:
                self.trainer._move_model_to_vllm = self._original_sync
            if self._original_encode is not None:
                module.encode_sample = self._original_encode
            write_event(self.event_path, "train_end", step=state.global_step, fingerprint=self._last)

    return ChartAuditCallback
