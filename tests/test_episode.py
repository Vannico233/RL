from copy import deepcopy
import unittest

from PIL import Image

from chart_rl.episode import Generation, image_count, run_episode
from chart_rl.policies import random_regions
from chart_rl.protocol import tool_action
from chart_rl.vision import ImageBudget, display_image


class ScriptedBackend:
    """Test-only deterministic backend; does not represent model inference."""
    def __init__(self, completions):
        self.completions = iter(completions)
        self.histories = []

    def generate(self, messages):
        self.histories.append(deepcopy(messages))
        return Generation(next(self.completions), generated_tokens=8, visual_tokens=None, truncated=False)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (280, 280), (120, 22, 255))
        self.observation = {"id": "test", "image": "local.png", "question": "What is the value?", "width": 280, "height": 280}

    def test_two_real_images_returned_then_final_answer_opportunity(self):
        backend = ScriptedBackend([tool_action([0, 0, 140, 140]), tool_action([140, 140, 280, 280]), "<answer>4</answer>"])
        trajectory = run_episode(backend, self.observation, self.image)
        self.assertEqual([image_count(history) for history in backend.histories], [1, 2, 3])
        self.assertEqual(trajectory["prediction"]["termination"], "final_answer")
        self.assertEqual(len(trajectory["prediction"]["tool_events"]), 2)
        self.assertIn("no tool calls remain", backend.histories[2][-1]["content"][0]["text"])
        self.assertIsNone(trajectory["prediction"]["cost"]["visual_tokens"])
        self.assertEqual(trajectory["prediction"]["cost"]["generated_tokens"], 24)

    def test_invalid_actions_use_budget_but_still_allow_answer(self):
        backend = ScriptedBackend(["not JSON", tool_action([-1, 0, 30, 30]), "<answer>4</answer>"])
        trajectory = run_episode(backend, self.observation, self.image)
        self.assertEqual([image_count(history) for history in backend.histories], [1, 1, 1])
        self.assertEqual(sum(event["consumed_budget"] for event in trajectory["prediction"]["tool_events"]), 2)
        self.assertEqual(trajectory["prediction"]["termination"], "final_answer")

    def test_third_tool_request_never_executes(self):
        backend = ScriptedBackend([tool_action([0, 0, 140, 140])] * 3)
        trajectory = run_episode(backend, self.observation, self.image)
        self.assertEqual(sum(event["success"] for event in trajectory["prediction"]["tool_events"]), 2)
        self.assertEqual(trajectory["prediction"]["tool_events"][-1]["error"], "budget_exhausted")

    def test_forced_controls_have_real_images_and_declared_origins(self):
        for policy in ("O2", "O3"):
            backend = ScriptedBackend(["<answer>4</answer>"])
            trajectory = run_episode(backend, self.observation, self.image, policy=policy)
            self.assertEqual(image_count(backend.histories[0]), 3)
            self.assertTrue(all(e["origin"] == "forced_observation" for e in trajectory["prediction"]["tool_events"]))

    def test_labels_rejected_and_no_tool_direct_answer_allowed(self):
        backend = ScriptedBackend(["<answer>4</answer>"])
        with self.assertRaises(ValueError):
            run_episode(backend, {**self.observation, "answer": "4"}, self.image)
        trajectory = run_episode(backend, self.observation, self.image)
        self.assertEqual(trajectory["prediction"]["tool_events"], [])

    def test_disallowed_tool_request_is_recorded_without_execution(self):
        backend = ScriptedBackend([tool_action([0, 0, 140, 140])])
        trajectory = run_episode(backend, self.observation, self.image, policy="O0")
        event = trajectory["prediction"]["tool_events"][0]
        self.assertFalse(event["success"])
        self.assertFalse(event["consumed_budget"])
        self.assertEqual(event["error"], "tools_disallowed_by_policy")

    def test_random_regions_deterministic_and_area_bounded(self):
        regions = random_regions((280, 560), "x", 42)
        self.assertEqual(regions, random_regions((280, 560), "x", 42))
        for x1, y1, x2, y2 in regions:
            area = (x2 - x1) * (y2 - y1) / (280 * 560)
            self.assertGreaterEqual(area, 0.2)
            self.assertLessEqual(area, 0.5)

    def test_display_budget_and_patch_alignment(self):
        image = Image.new("RGB", (1800, 1200))
        budget = ImageBudget(max_pixels=1003520)
        display = display_image(image, budget)
        self.assertEqual(display.width % 28, 0)
        self.assertEqual(display.height % 28, 0)
        self.assertLessEqual(display.width * display.height, budget.max_pixels)
        self.assertEqual(display_image(display, budget).size, display.size)


if __name__ == "__main__":
    unittest.main()
