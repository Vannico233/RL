#!/usr/bin/env python3
"""Run on the server; the default download includes model weights and data."""

import argparse
import json
from pathlib import Path

from chart_rl.assets import download_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/project.json"))
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_assets(args.config, args.output_root, args.model_only, args.data_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
