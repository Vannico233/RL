#!/usr/bin/env python3
"""Run the predefined pilot or final comparison matrix sequentially."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from chart_rl.data import sha256_file
from chart_rl.provenance import code_identity, model_identity
from collect_results import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--prepared", default="data_manifests/v1")
    parser.add_argument("--initial-model")
    parser.add_argument("--b1-adapter")
    parser.add_argument("--b2-adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prepared, output = Path(args.prepared).resolve(), Path(args.output).resolve()
    metadata = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    model = str(Path(args.initial_model or metadata["model_path"]).resolve())
    if args.phase == "pilot":
        settings = [("B0", policy, None) for policy in ("O0", "O1", "O2", "O3", "O4")]
        manifest = prepared / "pilot.jsonl"
    else:
        if not args.b1_adapter or not args.b2_adapter:
            raise ValueError("Final matrix requires both B1 and B2 adapters selected using validation results")
        settings = [("B0", "O0", None), ("B0", "O4", None), ("B1", "O0", str(Path(args.b1_adapter).resolve()))]
        settings += [("B2", policy, str(Path(args.b2_adapter).resolve())) for policy in ("O0", "O1", "O2", "O3", "O4")]
        manifest = prepared / "test.jsonl"
    commands = []
    for setting, policy, adapter in settings:
        command = [sys.executable, "-m", "chart_rl.cli", "evaluate", "--manifest", str(manifest),
                   "--data-root", metadata["dataset_root"], "--model", model, "--policy", policy,
                   "--seed", str(args.seed), "--device", args.device, "--output", str(output / f"{setting}-{policy}")]
        if adapter:
            command += ["--adapter", adapter]
        commands.append(command)
    plan = {"phase": args.phase, "model": model, "manifest_sha256": sha256_file(manifest), "commands": commands}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    plan["model_identity"] = model_identity(model)["sha256"]
    plan["adapter_identities"] = {adapter: model_identity(adapter)["sha256"] for _, _, adapter in settings if adapter}
    plan["code_identity"] = code_identity(project)["sha256"]
    if args.resume:
        prior = json.loads((output / "matrix-plan.json").read_text(encoding="utf-8"))
        if prior != plan:
            raise ValueError("Matrix configuration or source/model contents changed; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "matrix-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed = []
    for (setting, policy, _), command in zip(settings, commands):
        directory = output / f"{setting}-{policy}"
        if directory.exists():
            run_path = directory / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
            if (args.resume and run.get("status") == "complete" and run.get("manifest_sha256") == plan["manifest_sha256"]
                    and (directory / "summary.json").exists()):
                print(f"Keeping verified completed run {setting}-{policy}", flush=True)
                completed.append(directory)
                continue
            raise ValueError(f"Incomplete output exists: {directory}. Keep its logs and move it aside before resuming the matrix.")
        print(f"Evaluating {setting}-{policy}", flush=True)
        subprocess.run(command, cwd=project, check=True)
        completed.append(directory)
    result_path = output / "results.csv"
    if not result_path.exists():
        collect(completed, result_path)
    print(f"Completed {len(completed)} evaluation conditions: {result_path}")


if __name__ == "__main__":
    main()
