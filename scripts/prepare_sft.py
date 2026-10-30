#!/usr/bin/env python3
import argparse
import json

from chart_rl.sft import prepare_reviewed_sft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build optional warmup examples from reviewed training regions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(prepare_reviewed_sft(args.manifest, args.data_root, args.reviews, args.output, args.limit), ensure_ascii=False, indent=2))
