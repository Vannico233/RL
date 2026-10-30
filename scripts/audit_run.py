#!/usr/bin/env python3
import argparse
import json
import sys

from chart_rl.gates import audit_training_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check G1 evidence from an actual B2 smoke run")
    parser.add_argument("run_directory")
    args = parser.parse_args()
    report = audit_training_run(args.run_directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["status"] == "pass" else 2)
