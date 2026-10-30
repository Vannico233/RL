import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from chart_rl.data import read_jsonl, sha256_file, write_jsonl
from chart_rl.gates import audit_training_run
from chart_rl.launch import launch_plan
from chart_rl.sft import prepare_reviewed_sft


class LaunchAndGateTests(unittest.TestCase):
    def test_b1_and_b2_share_initialization_and_have_separate_gpu_processes(self):
        common = dict(project=".", model="model", train_data="train.jsonl", eval_data="val.jsonl", output="run")
        b1, b2 = launch_plan(**common, setting="B1"), launch_plan(**common, setting="B2")
        self.assertEqual(b1["model"], b2["model"])
        self.assertNotEqual(b2["training_env"]["CUDA_VISIBLE_DEVICES"], b2["rollout_env"]["CUDA_VISIBLE_DEVICES"])
        self.assertNotIn("--multi_turn_scheduler", b1["rollout_command"])
        self.assertIn("--multi_turn_scheduler", b2["rollout_command"])
        self.assertIn("--multi_turn_scheduler", b2["training_command"])
        position = b2["training_command"].index("--loss_scale")
        self.assertEqual(b2["training_command"][position + 1], "default")
        command = b2["training_command"]
        eval_batch = int(command[command.index("--per_device_eval_batch_size") + 1])
        eval_generations = int(command[command.index("--num_generations_eval") + 1])
        self.assertEqual(eval_batch % eval_generations, 0)
        with self.assertRaises(ValueError):
            launch_plan(**common, trainer_gpu=0, rollout_gpu=0)

    def test_checkpoint_and_finished_event_are_not_sufficient_for_g1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-1").mkdir()
            (root / "audit").mkdir()
            write_jsonl(root / "audit/training.jsonl", [{"kind": "train_end", "step": 1}])
            (root / "launch-plan.json").write_text(json.dumps({"setting": "B2", "model": "model"}))
            report = audit_training_run(root)
            self.assertEqual(report["status"], "needs_attention")
            self.assertTrue(report["checks"]["training_finished"])
            self.assertFalse(report["checks"]["trainable_weights_changed"])
            self.assertFalse(report["checks"]["generation_after_sync_has_registered_adapter"])


class SFTTests(unittest.TestCase):
    def test_only_reviewed_training_regions_become_tool_demonstrations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (280, 280), (1, 2, 3)).save(image)
            row = {"id": "one", "image": image.name, "image_sha256": sha256_file(image), "question": "Question?", "answer": "4", "split": "train"}
            manifest, reviews = root / "train.jsonl", root / "reviews.jsonl"
            write_jsonl(manifest, [row])
            review = {"id": "one", "reference_size": [280, 280], "regions": [[0, 0, 140, 140]],
                      "reviewed": True, "review_note": "Synthetic test fixture, not a real annotation"}
            write_jsonl(reviews, [review])
            output = root / "sft"
            audit = prepare_reviewed_sft(manifest, root, reviews, output)
            example = read_jsonl(output / "sft.jsonl")[0]
            self.assertEqual(audit["count"], 1)
            self.assertEqual(len(example["images"]), 2)
            self.assertEqual(example["messages"][-1]["content"], "<answer>4</answer>")
            self.assertNotIn(review["review_note"], str(example["messages"]))
            review["reviewed"] = False
            write_jsonl(reviews, [review])
            with self.assertRaises(ValueError):
                prepare_reviewed_sft(manifest, root, reviews, root / "unreviewed")
            row["split"] = "test"
            write_jsonl(manifest, [row])
            with self.assertRaises(ValueError):
                prepare_reviewed_sft(manifest, root, reviews, root / "test-leak")


if __name__ == "__main__":
    unittest.main()
