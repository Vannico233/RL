"""Summarize recorded training evidence without counting re-encodes twice."""

import json
from pathlib import Path

from .data import read_jsonl


def summarize_training(directory):
    root = Path(directory)
    events = read_jsonl(root / "audit/training.jsonl")
    rewards = read_jsonl(root / "audit/rewards.jsonl")
    plan = json.loads((root / "launch-plan.json").read_text(encoding="utf-8"))
    process = json.loads((root / "process-status.json").read_text(encoding="utf-8"))
    trajectories = {}
    missing_identity = 0
    for event in events:
        if event["kind"] != "encoded_trajectory":
            continue
        if not event.get("request_id"):
            missing_identity += 1
            continue
        # A rollout may be encoded again for loss calculation. Count its actual
        # generation once; mode separates periodic validation from training.
        trajectories[(event["mode"], event["request_id"])] = event
    groups = {}
    for mode in ("train", "eval"):
        rows = [row for (row_mode, _), row in trajectories.items() if row_mode == mode]
        costs = {}
        for key in ("generated_tokens", "visual_tokens"):
            present = [row[key] for row in rows if row.get(key) is not None]
            costs[key] = {"sum": sum(present) if present else None, "observed": len(present),
                          "coverage": len(present) / len(rows) if rows else None}
        groups[mode] = {"trajectories": len(rows), "unique_questions": len({row["chart_id"] for row in rows}),
                        "truncation_rate": sum(bool(row["truncated"]) for row in rows) / len(rows) if rows else None,
                        "costs": costs}
    steps = [row for row in events if row["kind"] == "optimizer_step"]
    gaps = [b["time"] - a["time"] for a, b in zip(steps, steps[1:]) if b["step"] == a["step"] + 1]
    return {
        "setting": plan["setting"], "smoke": plan["smoke"], "process_status": process["status"],
        "latest_invocation_wall_seconds": process.get("wall_seconds"), "optimizer_events": len(steps),
        "mean_inter_step_wall_seconds": sum(gaps) / len(gaps) if gaps else None,
        "peak_trainer_allocated_bytes": max((row["peak_allocated_bytes"] for row in steps), default=None),
        "peak_trainer_reserved_bytes": max((row["peak_reserved_bytes"] for row in steps), default=None),
        "modes": groups, "encoding_events_without_request_id": missing_identity,
        "reward_group_batches": sum(len(row["groups"]) for row in rewards),
        "constant_reward_group_batches": sum(not group["has_variance"] for row in rewards for group in row["groups"]),
        "limitations": ["Counts cover unique trajectories that reached the encoder; failures before encoding are not measured.",
                        "Visual tokens sum each generation input, including repeated history images; they are not FLOPs.",
                        "Wall time includes loading, synchronization, checkpointing and validation; step gaps may include resumed downtime.",
                        "CUDA peaks describe the trainer process only, not the separate vLLM process.",
                        "Reward group statistics include periodic validation and any repeated reward evaluations."],
    }
