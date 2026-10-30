"""Construct the pinned two-GPU training plan without importing GPU packages."""

from pathlib import Path


def launch_plan(project, model, train_data, eval_data, output, setting="B2", max_steps=128,
                seed=42, trainer_gpu=0, rollout_gpu=1, port=8001, smoke=False, resume=None):
    if setting not in ("B1", "B2"):
        raise ValueError("Training setting must be B1 or B2")
    if trainer_gpu == rollout_gpu or min(trainer_gpu, rollout_gpu) < 0:
        raise ValueError("Training and rollout must use two distinct GPU indices")
    if max_steps < 1 or not 1024 <= port <= 65535:
        raise ValueError("Invalid training budget or server port")
    project, output = Path(project).resolve(), Path(output).resolve()
    plugin = str(project / "plugins" / "chartqa_plugin.py")
    common = ["--model", str(model), "--model_type", "qwen2_5_vl", "--external_plugins", plugin,
              "--torch_dtype", "bfloat16"]
    rollout = ["swift", "rollout", *common,
               "--host", "127.0.0.1", "--port", str(port),
               "--vllm_use_async_engine", "true", "--vllm_tensor_parallel_size", "1",
               "--vllm_data_parallel_size", "1", "--vllm_gpu_memory_utilization", "0.8",
               "--vllm_max_model_len", "8192", "--vllm_max_num_seqs", "8",
               "--vllm_limit_mm_per_prompt", '{"image":3}',
               "--vllm_enable_lora", "true", "--vllm_max_lora_rank", "16",
               "--vllm_enable_prefix_caching", "false", "--vllm_enforce_eager", "true"]
    training = ["swift", "rlhf", "--rlhf_type", "grpo", *common,
                "--dataset", str(train_data), "--split_dataset_ratio", "0",
                "--use_vllm", "true", "--vllm_mode", "server",
                "--async_generate", "false",
                "--vllm_server_host", "127.0.0.1", "--vllm_server_port", str(port),
                "--vllm_server_timeout", "600", "--vllm_server_pass_dataset", "true",
                "--reward_funcs", "chartqa_answer", "--callbacks", "chart_audit",
                "--tuner_type", "lora", "--target_modules", "all-linear",
                "--freeze_vit", "true", "--freeze_aligner", "true", "--freeze_llm", "false",
                "--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0",
                "--attn_impl", "sdpa", "--gradient_checkpointing", "true", "--padding_free", "false",
                "--loss_scale", "default", "--max_length", "8192", "--max_completion_length", "256",
                "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "4" if smoke else "8",
                "--num_generations", "4", "--learning_rate", "1e-5", "--beta", "0",
                "--temperature", "0.9", "--top_p", "0.95", "--max_steps", str(max_steps),
                "--warmup_ratio", "0" if smoke else "0.03", "--max_grad_norm", "1",
                "--logging_steps", "1", "--save_steps", "1" if smoke else "64", "--save_total_limit", "3",
                "--dataloader_num_workers", "2", "--dataset_num_proc", "1",
                "--seed", str(seed), "--data_seed", str(seed), "--report_to", "tensorboard",
                "--log_completions", "true", "--output_dir", str(output), "--add_version", "false"]
    if smoke:
        training += ["--eval_strategy", "no"]
    else:
        training += ["--val_dataset", str(eval_data), "--eval_strategy", "steps", "--eval_steps", "64",
                     "--per_device_eval_batch_size", "1", "--num_generations_eval", "1"]
    if setting == "B2":
        rollout += ["--multi_turn_scheduler", "chartqa_crop", "--max_turns", "3"]
        training += ["--multi_turn_scheduler", "chartqa_crop", "--max_turns", "3"]
    if resume:
        training += ["--resume_from_checkpoint", str(resume)]
    shared_env = {"MAX_PIXELS": "1003520", "MIN_PIXELS": "3136", "USE_HF": "1", "TOKENIZERS_PARALLELISM": "false"}
    return {
        "setting": setting, "model": str(model), "output": str(output), "smoke": smoke,
        "seed": seed, "max_steps": max_steps, "rollout_command": rollout, "training_command": training,
        "rollout_env": {**shared_env, "CUDA_VISIBLE_DEVICES": str(rollout_gpu),
                        "CHART_RL_TRACE_DIR": str(output / "rollouts")},
        "training_env": {**shared_env, "CUDA_VISIBLE_DEVICES": str(trainer_gpu), "NPROC_PER_NODE": "1",
                         "CHART_RL_AUDIT_DIR": str(output / "audit")},
    }
