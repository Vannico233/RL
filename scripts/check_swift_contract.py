#!/usr/bin/env python3
"""Inspect a pinned upstream checkout without importing its GPU dependencies.

This is a source-contract check, not a runtime integration test.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys


def inspect_contract(source):
    source = Path(source).resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=True).stdout.strip()
    checks = {"commit_matches": commit == "ad818a2cdfafaf4305ac40c4acd892a211978bc2"}
    module = ast.parse((source / "swift/rollout/multi_turn.py").read_text(encoding="utf-8"))
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    for name, methods in {"RolloutScheduler": ("on_trajectory_start", "on_turn_end"),
                          "MultiTurnScheduler": ("step", "check_finished", "run")}.items():
        found = {node.name for node in classes[name].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        checks[name + "_hooks"] = set(methods) <= found
    data = (source / "swift/rl_core/data.py").read_text(encoding="utf-8")
    checks["dynamic_image_override_contract"] = "rollout_output.rollout_infos[key]" in data and "_MULTIMODAL_KEYS" in data
    trainer = (source / "swift/rlhf_trainers/grpo_trainer.py").read_text(encoding="utf-8")
    checks["per_sample_encoder_hook"] = "encode_sample(s, template)" in trainer
    registry = ast.parse((source / "swift/callbacks/mapping.py").read_text(encoding="utf-8"))
    checks["callback_registry"] = any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "callbacks_map" for target in node.targets) for node in registry.body)
    return {"status": "pass" if all(checks.values()) else "fail", "commit": commit, "checks": checks,
            "scope": "Static source interface inspection only; does not validate GPU imports, generation, masks, or updates at runtime."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect_contract(args.source)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["status"] == "pass" else 1)
