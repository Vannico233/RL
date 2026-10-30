import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from chart_rl.data import sha256_file, write_jsonl, read_jsonl
from chart_rl.protocol import tool_action
from chart_rl.swift_adapter import build_scheduler_class, reward_batch
from chart_rl.swift_data import export_swift


class ContractBase:
    """Minimal contract double. This is not an imported ms-swift scheduler."""
    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        self.max_turns = max_turns

    def check_finished(self, request, choice, turn):
        return choice.finish_reason == "length" or turn >= self.max_turns


def response(text):
    return SimpleNamespace(message=SimpleNamespace(content=text), token_ids=[3, 7, 9], finish_reason="stop")


class SwiftAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "original.png"
        Image.new("RGB", (560, 560), (15, 25, 35)).save(self.path)
        self.request = SimpleNamespace(images=[Image.new("RGB", (280, 280), (15, 25, 35))],
            messages=[{"role": "user", "content": "<image>\nQuestion"}],
            data_dict={"chart_id": "x", "chart_original_image": str(self.path),
                       "chart_image_sha256": sha256_file(self.path), "chart_reference_size": [280, 280],
                       "chart_crop_max_pixels": 501760, "chart_min_pixels": 3136,
                       "solution": "PRIVATE_TARGET", "untrusted_extra": "PRIVATE_METADATA"})
        self.scheduler = build_scheduler_class(ContractBase)()

    def tearDown(self):
        self.temp.cleanup()

    def test_hooks_return_dynamic_images_and_preserve_sampled_response(self):
        original_extra = self.request.data_dict
        asyncio.run(self.scheduler.on_trajectory_start([self.request]))
        self.assertNotIn("solution", self.request.data_dict)
        self.assertEqual(original_extra["solution"], "PRIVATE_TARGET")
        for turn in (1, 2):
            choice = response(tool_action([0, 0, 140, 140]))
            self.request.messages.append({"role": "assistant", "content": choice.message.content})
            sampled = deepcopy(self.request.messages[-1])
            asyncio.run(self.scheduler.on_turn_end(self.request, choice, turn))
            self.assertFalse(self.scheduler.check_finished(self.request, choice, turn))
            result = self.scheduler.step(self.request, choice, turn)
            self.assertEqual(self.request.messages[-2], sampled)
            self.assertEqual(len(result["rollout_infos"]["images"]), turn + 1)
            self.assertEqual(sum(m["content"].count("<image>") for m in self.request.messages), turn + 1)
            self.assertNotIn("response_token_ids", result)
            self.assertNotIn("response_loss_mask", result)
        final = response("<answer>4</answer>")
        self.request.messages.append({"role": "assistant", "content": final.message.content})
        metadata = asyncio.run(self.scheduler.on_turn_end(self.request, final, 3))["rollout_infos"]
        self.assertTrue(self.scheduler.check_finished(self.request, final, 3))
        self.assertEqual(metadata["chart_tool_events"][0]["executed_bbox_original"], [0, 0, 280, 280])
        self.assertEqual(len(metadata["chart_turns"]), 3)
        self.assertEqual(len(metadata["images"]), 3)

    def test_image_tampering_or_coordinate_mismatch_fails(self):
        self.request.data_dict["chart_reference_size"] = [56, 56]
        with self.assertRaises(ValueError):
            asyncio.run(self.scheduler.on_trajectory_start([self.request]))
        self.request.data_dict["chart_reference_size"] = [280, 280]
        Image.new("RGB", (560, 560), "red").save(self.path)
        with self.assertRaises(ValueError):
            asyncio.run(self.scheduler.on_trajectory_start([self.request]))

    def test_reward_batch_lengths_and_no_tool_bonus(self):
        self.assertEqual(reward_batch([tool_action([1, 2, 3, 4]), "<answer>4</answer>"], ["4", "4"]), [0, 1])
        with self.assertRaises(ValueError):
            reward_batch(["<answer>4</answer>"], [])

    def test_export_paired_inputs_preserves_labels_only_as_metadata(self):
        manifest = self.root / "train.jsonl"
        write_jsonl(manifest, [{"id": "x", "image": self.path.name, "image_sha256": sha256_file(self.path),
                              "width": 560, "height": 560, "question": "Q", "answer": "PRIVATE_TARGET", "split": "train"}])
        output = self.root / "export"
        export_swift(manifest, self.root, output)
        left, right = read_jsonl(output / "B1.jsonl")[0], read_jsonl(output / "B2.jsonl")[0]
        self.assertEqual(left["images"], right["images"])
        self.assertEqual(left["solution"], right["solution"])
        self.assertNotIn("PRIVATE_TARGET", str(right["messages"]))
        self.assertEqual(right["chart_reference_size"], [560, 560])


if __name__ == "__main__":
    unittest.main()
