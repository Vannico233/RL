#!/usr/bin/env python3
"""Collect complete evaluation runs; does not mix or silently pair datasets."""

import argparse
import csv
import json
from pathlib import Path


def collect(directories, output):
    rows = []
    for directory in directories:
        directory = Path(directory)
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run["status"] != "complete":
            raise ValueError(f"Refusing incomplete run: {directory}")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        metric = summary["overall"]
        row = {"run": str(directory), "model": run["provenance"].get("model_path"),
               "adapter": run["provenance"].get("adapter_path"), "policy": run["policy"], "seed": run["seed"],
               "manifest_sha256": run["manifest_sha256"], "count": metric["count"], "protocol": summary["protocol"],
               "accuracy": metric["accuracy"], "strict_accuracy": metric["strict_accuracy"], "parse_rate": metric["parse_rate"],
               "human_accuracy": summary["by_source"].get("human", {}).get("accuracy"),
               "augmented_accuracy": summary["by_source"].get("augmented", {}).get("accuracy"),
               "tool_attempts": metric["tools"]["mean_attempts"], "tool_metric_coverage": metric["tools"]["coverage"],
               "invalid_request_rate": metric["tools"]["invalid_request_rate"],
               "truncation_rate": metric["truncation"]["rate"]}
        for field, values in metric["cost"].items():
            row[field] = values["mean"]
            row[field + "_coverage"] = values["coverage"]
        rows.append(row)
    if not rows:
        raise ValueError("No evaluation directories supplied")
    with Path(output).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Collected {collect(args.runs, args.output)} complete runs")
