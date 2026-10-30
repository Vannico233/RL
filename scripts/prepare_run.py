#!/usr/bin/env python3
"""Freeze a question pool and export matched B1/B2 train/validation datasets."""

import argparse
import json
from pathlib import Path

from chart_rl.data import prepare_chartqa, sha256_file
from chart_rl.swift_data import export_swift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="data/assets.json")
    parser.add_argument("--output", default="data_manifests/v1")
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assets_path = Path(args.assets).resolve()
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    root = Path(assets["dataset_root"])
    output = Path(args.output).resolve()
    audit = prepare_chartqa(root, output, args.train_limit, args.val_limit, args.test_limit, seed=args.seed)
    for split in ("train", "val"):
        export_swift(output / f"{split}.jsonl", root, output / "exports" / split)
    prepared = {"asset_record": str(assets_path), "asset_record_sha256": sha256_file(assets_path),
                "dataset_root": str(root), "model_path": assets.get("model_path"),
                "selected_counts": audit["selected_counts"], "seed": args.seed}
    (output / "prepared.json").write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(prepared, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
