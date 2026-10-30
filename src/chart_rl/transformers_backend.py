"""Real Qwen2.5-VL inference, imported only by the GPU evaluation command."""

from .episode import Generation


class TransformersBackend:
    def __init__(self, model_path, adapter_path=None, device="cuda:0", max_new_tokens=256, seed=42,
                 max_pixels=2007040, min_pixels=3136, temperature=0.0):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available() or not device.startswith("cuda"):
            raise RuntimeError("Real-model evaluation requires a CUDA GPU; CPU tests use an explicitly scripted test backend")
        if type(max_new_tokens) is not int or max_new_tokens < 1 or temperature < 0:
            raise ValueError("Invalid generation budget or temperature")
        self.torch, self.device = torch, device
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels,
                                                      local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()

    def generate(self, messages):
        torch = self.torch
        images = [block["image"] for message in messages if isinstance(message["content"], list)
                  for block in message["content"] if block.get("type") == "image"]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        grid = inputs["image_grid_thw"].tolist()
        processor = self.processor.image_processor
        patch_size, merge_size = processor.patch_size, processor.merge_size
        expected = [[1, image.height // patch_size, image.width // patch_size] for image in images]
        if grid != expected:
            raise ValueError(f"Processor resized an image unexpectedly: grid={grid}, expected={expected}")
        visual_tokens = sum(t * h * w // (merge_size ** 2) for t, h, w in grid)
        image_token_id = self.model.config.image_token_id
        actual_image_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
        if actual_image_tokens != visual_tokens:
            raise ValueError("Image placeholders and processor visual-token counts do not align")
        inputs = inputs.to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        options = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            options["temperature"] = self.temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **options)
        ids = generated[0, prompt_tokens:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        truncated = len(ids) >= self.max_new_tokens and (not ids or ids[-1] not in eos_ids)
        completion = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return Generation(
            completion, generated_tokens=len(ids), visual_tokens=visual_tokens, prompt_tokens=prompt_tokens,
            truncated=truncated,
            diagnostics={"image_grid_thw": grid, "image_token_count": actual_image_tokens,
                         "cost_semantics": "visual input tokens per generation call, including repeated history images"},
        )
