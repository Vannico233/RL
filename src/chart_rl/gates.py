"""G1 evidence checks; never infer success from a checkpoint's existence alone."""

import json
from pathlib import Path

from .data import read_jsonl


def audit_training_run(directory):
    root = Path(directory)
    training_path, reward_path = root / "audit/training.jsonl", root / "audit/rewards.jsonl"
    events = read_jsonl(training_path) if training_path.exists() else []
    rewards = read_jsonl(reward_path) if reward_path.exists() else []
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "rollouts").glob("*.json"))]
    plan = json.loads((root / "launch-plan.json").read_text(encoding="utf-8")) if (root / "launch-plan.json").exists() else {}
    encoded = [event for event in events if event["kind"] == "encoded_trajectory"]
    alignments = [event for event in events if event["kind"] == "logprob_alignment"]
    changed = [event for event in events if event["kind"] == "optimizer_step" and event["changed_since_initial"]]
    changed_hashes = {event["fingerprint"] for event in changed}
    synced = [event for event in events if event["kind"] == "weight_sync_ack" and event["fingerprint"] in changed_hashes]
    after_sync = [trace for trace in traces if trace.get("chart_loaded_adapter_ids") and
                  any(trace.get("chart_trajectory_time", 0) > event["time"] for event in synced)]
    cropped = [trace for trace in traces if any(event["success"] for event in trace.get("chart_tool_events", []))]
    checks = {
        "initialization_and_code_recorded": bool(plan.get("model_identity", {}).get("sha256")) and bool(plan.get("code_identity", {}).get("sha256")),
        "training_finished": any(event["kind"] == "train_end" for event in events),
        "reward_variation_observed": any(group["has_variance"] for batch in rewards for group in batch["groups"]),
        "finite_nonzero_gradient_observed": any(event["kind"] == "gradient" and event["finite"] and event["nonzero_gradient"] for event in events),
        "no_nonfinite_gradient": not any(event["kind"] == "gradient" and not event["finite"] for event in events),
        "trainable_weights_changed": bool(changed),
        "updated_weights_sync_acknowledged": bool(synced),
        "loss_tokens_match_sampled_responses": bool(encoded) and all(event["loss_tokens_match_rollout"] is True for event in encoded),
        "logprob_counts_match_sampled_loss_tokens": bool(alignments) and all(event["matches"] is True for event in alignments),
    }
    if plan.get("setting") == "B2":
        checks.update({
            "real_crop_then_later_generation": any(len(trace.get("chart_turns", [])) >= 2 for trace in cropped),
            "no_more_than_two_budgeted_attempts": bool(traces) and all(sum(event["consumed_budget"] for event in trace.get("chart_tool_events", [])) <= 2 for trace in traces),
            "training_image_grids_match_observations": bool(encoded) and all(event["image_grid_matches"] is True for event in encoded),
            "generation_after_sync_has_registered_adapter": bool(after_sync),
            "rollout_logprobs_finite_and_aligned": bool(traces) and all(
                turn.get("logprobs_align_with_sampled_ids") is True and turn.get("logprobs_finite") is True
                for trace in traces for turn in trace.get("chart_turns", [])),
        })
    else:
        checks["B2_tool_smoke_setting"] = False
    report = {
        "status": "pass" if checks and all(checks.values()) else "needs_attention",
        "checks": checks, "model": plan.get("model"), "setting": plan.get("setting"),
        "model_sha256": plan.get("model_identity", {}).get("sha256"),
        "code_sha256": plan.get("code_identity", {}).get("sha256"),
        "run_directory": str(root.resolve()), "evidence_counts": {"training_events": len(events), "reward_batches": len(rewards), "rollout_traces": len(traces)},
        "limitations": ["A sync acknowledgement and later registered-adapter inference do not constitute a remote tensor checksum.",
                        "This gate verifies implementation evidence; it does not establish tool usefulness or benchmark improvement.",
                        "Constant-reward groups are legitimate; find a real training microbatch with reward variation before asserting nonzero GRPO learning."],
    }
    (root / "g1-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
