#!/usr/bin/env python3
"""Own the rollout process lifecycle and run B1/B2 on separate local GPUs."""

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from chart_rl.data import read_jsonl, sha256_file, write_jsonl
from chart_rl.gates import audit_training_run
from chart_rl.launch import launch_plan
from chart_rl.provenance import code_identity, model_identity, package_versions


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    # Each child below is created in its own session. Only that owned process
    # group is signalled, including its vLLM workers.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_ready(process, port, timeout=900):
    opener = build_opener(ProxyHandler({}))
    deadline, last_notice = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Rollout server exited; inspect rollout.log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health/", timeout=3) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (URLError, OSError, ValueError):
            pass
        if time.monotonic() - last_notice > 15:
            print("Waiting for the rollout server to load the model; details are in rollout.log", flush=True)
            last_notice = time.monotonic()
        time.sleep(1)
    raise TimeoutError("Rollout server did not become ready within the startup timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default="data_manifests/v1")
    parser.add_argument("--setting", choices=("B1", "B2"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-model", help="Optional merged common SFT initialization for both B1 and B2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainer-gpu", type=int, default=0)
    parser.add_argument("--rollout-gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gate-report", help="Passing G1 report for this initialization, required for a full run")
    parser.add_argument("--resume-from", help="An existing checkpoint inside the output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prepared, output = Path(args.prepared).resolve(), Path(args.output).resolve()
    metadata = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    model = Path(args.initial_model or metadata["model_path"]).resolve()
    train_data = prepared / "exports/train" / f"{args.setting}.jsonl"
    val_data = prepared / "exports/val" / f"{args.setting}.jsonl"
    for path in (model, train_data, val_data):
        if not path.exists():
            raise FileNotFoundError(path)
    max_steps = args.max_steps if args.max_steps is not None else (8 if args.smoke else 128)
    resume = Path(args.resume_from).resolve() if args.resume_from else None
    if resume and (not resume.is_relative_to(output) or not (resume / "trainer_state.json").is_file()):
        raise ValueError("Resume checkpoint must be inside this run and contain trainer_state.json")
    plan = launch_plan(project, model, train_data, val_data, output, args.setting, max_steps,
                       args.seed, args.trainer_gpu, args.rollout_gpu, args.port, args.smoke, resume)
    plan["prepared_directory"] = str(prepared)
    plan["train_source_sha256"] = sha256_file(train_data)
    plan["val_source_sha256"] = sha256_file(val_data)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual training is supported on Linux; use --dry-run to inspect commands elsewhere")
    print("Recording initialization and source checksums before launching GPU processes...", flush=True)
    plan["model_identity"] = model_identity(model)
    plan["code_identity"] = code_identity(project)
    plan["environment"] = package_versions()
    if not args.smoke:
        if not args.gate_report:
            raise ValueError("Complete a B2 --smoke run and provide its passing --gate-report before long training")
        gate = json.loads(Path(args.gate_report).read_text(encoding="utf-8"))
        if (gate.get("status") != "pass" or gate.get("model_sha256") != plan["model_identity"]["sha256"]
                or gate.get("code_sha256") != plan["code_identity"]["sha256"]):
            raise ValueError("G1 evidence must pass and use the identical initialization")
    if resume:
        prior = json.loads((output / "launch-plan.json").read_text(encoding="utf-8"))
        for key in ("setting", "model", "seed", "train_source_sha256", "val_source_sha256", "smoke"):
            if prior.get(key) != plan[key]:
                raise ValueError("Resume configuration changed: " + key)
        for key in ("model_identity", "code_identity"):
            if prior[key]["sha256"] != plan[key]["sha256"]:
                raise ValueError("Resume content changed: " + key)
    elif output.exists():
        raise FileExistsError("Output exists; choose a new directory or explicitly resume its checkpoint")
    else:
        output.mkdir(parents=True)
    subprocess.run([sys.executable, str(project / "scripts/preflight.py"), "--output", str(output / "preflight.json")], check=True)
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    if args.smoke and args.setting == "B2":
        rows = read_jsonl(train_data)[:64]
        for row in rows:
            width, height = row["chart_reference_size"]
            row["messages"][-1]["content"] += (
                f"\nTechnical diagnostic only: on your first turn request a crop with bbox_2d [0,0,{width / 2},{height / 2}]. "
                "Then inspect the returned image and answer the original question."
            )
        train_data = output / "smoke-input.jsonl"
        write_jsonl(train_data, rows)
        index = plan["training_command"].index("--dataset") + 1
        plan["training_command"][index] = str(train_data)
        plan["smoke_note"] = "Forces an initial tool request for technical diagnosis; excluded from all scientific comparisons."
    plan_name = f"resume-plan-{time.time_ns()}.json" if resume else "launch-plan.json"
    (output / plan_name).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    server, training = None, None
    started = time.time()
    status = {"status": "starting", "started_at": started}
    status_path = output / "process-status.json"
    try:
        with (output / "rollout.log").open("a", encoding="utf-8") as server_log, (output / "training.log").open("a", encoding="utf-8") as train_log:
            server = subprocess.Popen(plan["rollout_command"], cwd=project, env={**os.environ, **plan["rollout_env"]},
                                      stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(rollout_pid=server.pid)
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            wait_ready(server, args.port)
            training = subprocess.Popen(plan["training_command"], cwd=project, env={**os.environ, **plan["training_env"]},
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            status.update(status="training", training_pid=training.pid)
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            for line in training.stdout:
                print(line, end="", flush=True)
                train_log.write(line)
                train_log.flush()
            code = training.wait()
            if code:
                raise RuntimeError(f"Training exited with code {code}; inspect training.log")
            status.update(status="training_finished", finished_at=time.time(), wall_seconds=time.time() - started)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error), finished_at=time.time())
        raise
    finally:
        stop_owned_process(training)
        stop_owned_process(server)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.smoke:
        report = audit_training_run(output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "pass" else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
