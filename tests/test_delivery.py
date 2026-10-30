"""CPU checks for reproducibility and server-facing delivery commands."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from chart_rl.data import sha256_file, write_jsonl
from chart_rl.episode import Generation
from chart_rl.provenance import model_identity
from chart_rl.run_evaluation import evaluate_run
from chart_rl.training_summary import summarize_training

PROJECT = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def test_weight_content_changes_identity_but_secret_environment_file_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(b"synthetic fixture A")
            first = model_identity(root)
            (root / ".env").write_text("TEST_ONLY=not-a-real-secret")
            self.assertEqual(first, model_identity(root))
            weights.write_bytes(b"synthetic fixture B")
            self.assertNotEqual(first["sha256"], model_identity(root)["sha256"])

    def test_training_summary_deduplicates_rollouts_and_separates_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audit").mkdir()
            row = {"kind": "encoded_trajectory", "request_id": "r1", "mode": "train", "chart_id": "q1",
                   "generated_tokens": 9, "visual_tokens": 200, "truncated": False}
            rows = [row, dict(row), {**row, "request_id": "r2", "mode": "eval", "generated_tokens": None}]
            write_jsonl(root / "audit/training.jsonl", rows)
            write_jsonl(root / "audit/rewards.jsonl", [{"groups": [{"has_variance": False}]}])
            (root / "launch-plan.json").write_text(json.dumps({"setting": "B2", "smoke": True}))
            (root / "process-status.json").write_text(json.dumps({"status": "failed"}))
            result = summarize_training(root)
            self.assertEqual(result["process_status"], "failed")
            self.assertEqual(result["modes"]["train"]["trajectories"], 1)
            self.assertEqual(result["modes"]["train"]["costs"]["generated_tokens"]["sum"], 9)
            self.assertIsNone(result["modes"]["eval"]["costs"]["generated_tokens"]["sum"])
            self.assertEqual(result["modes"]["eval"]["costs"]["generated_tokens"]["coverage"], 0)

    def test_offline_case_viewer_keeps_real_images_and_escapes_model_text(self):
        class ScriptedBackend:
            count = 0

            def generate(self, messages):
                self.count += 1
                text = ('<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,140,140]}}</tool_call>'
                        if self.count == 1 else '<answer>4</answer>')
                return Generation(text)

        spec = importlib.util.spec_from_file_location("render_cases_for_test", PROJECT / "scripts/render_cases.py")
        viewer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viewer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "chart.png"
            Image.new("RGB", (280, 280), "blue").save(image)
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, [{"id": "one", "image": image.name, "image_sha256": sha256_file(image),
                                   "width": 280, "height": 280, "question": "<script>example</script>",
                                   "answer": "4", "source": "human", "split": "val", "group_id": "chart"}])
            run = root / "run"
            evaluate_run(ScriptedBackend(), manifest, root, run, case_limit=2)
            output = run / "cases/index.html"
            self.assertEqual(viewer.render(run, output), 1)
            html = output.read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;example&lt;/script&gt;", html)
            self.assertIn('<rect x="0" y="0" width="140" height="140"', html)
            self.assertTrue(next((run / "cases").glob("*/original.png")).is_file())
            self.assertIn("image_1.png", html)

    def test_final_matrix_dry_run_builds_all_eight_conditions_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared.json").write_text(json.dumps({"model_path": str(root / "model"), "dataset_root": str(root)}))
            write_jsonl(root / "test.jsonl", [{"id": "synthetic"}])
            command = [sys.executable, str(PROJECT / "scripts/evaluate_matrix.py"), "--phase", "final",
                       "--prepared", str(root), "--b1-adapter", str(root / "b1"), "--b2-adapter", str(root / "b2"),
                       "--output", str(root / "output"), "--dry-run"]
            result = subprocess.run(command, check=True, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": str(PROJECT / "src")}, timeout=30)
            plan = json.loads(result.stdout)
            self.assertEqual(len(plan["commands"]), 8)
            self.assertFalse((root / "output").exists())
            self.assertEqual(sum("--adapter" in cmd for cmd in plan["commands"]), 6)


if __name__ == "__main__":
    unittest.main()
