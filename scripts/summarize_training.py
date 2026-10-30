#!/usr/bin/env python3
"""Export actual training cost, coverage and reward diagnostics."""

import argparse
import json
from pathlib import Path

from chart_rl.training_summary import summarize_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output or str(Path(args.run) / "training-summary.json"))
    report = summarize_training(args.run)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
