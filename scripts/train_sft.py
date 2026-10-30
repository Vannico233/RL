#!/usr/bin/env python3
"""Optional LoRA protocol warmup. Merge its selected adapter before RL controls."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    command = ["swift", "sft", "--model", str(Path(args.model).resolve()), "--model_type", "qwen2_5_vl",
               "--dataset", str(Path(args.dataset).resolve()), "--split_dataset_ratio", "0", "--eval_strategy", "no",
               "--tuner_type", "lora", "--target_modules", "all-linear", "--freeze_vit", "true", "--freeze_aligner", "true",
               "--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0", "--torch_dtype", "bfloat16",
               "--attn_impl", "sdpa", "--gradient_checkpointing", "true", "--loss_scale", "default",
               "--max_length", "8192", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "4",
               "--learning_rate", "5e-5", "--num_train_epochs", "1", "--save_strategy", "epoch",
               "--logging_steps", "1", "--report_to", "tensorboard", "--seed", str(args.seed),
               "--output_dir", str(output), "--add_version", "false"]
    plan = {"command": command, "gpu": args.gpu, "seed": args.seed}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if platform.system() != "Linux":
        raise RuntimeError("Run actual SFT in the validated Linux environment")
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "NPROC_PER_NODE": "1",
                   "MAX_PIXELS": "1003520", "MIN_PIXELS": "3136", "USE_HF": "1"}
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
