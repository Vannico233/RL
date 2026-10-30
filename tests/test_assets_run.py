import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from chart_rl.assets import extract_dataset
from chart_rl.data import sha256_file, write_jsonl
from chart_rl.episode import Generation
from chart_rl.run_evaluation import evaluate_run


class Backend:
    def generate(self, messages):
        return Generation("<answer>100</answer>", generated_tokens=8, visual_tokens=100, truncated=False)


class AssetsAndRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_archive_path_escape_rejected_before_extracting(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../outside.txt", "bad")
        destination = self.root / "output"
        with self.assertRaises(ValueError):
            extract_dataset(archive, destination)
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / "outside.txt").exists())

    def test_extract_expected_official_layout(self):
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for split in ("train", "val", "test"):
                for source in ("human", "augmented"):
                    handle.writestr(f"ChartQA Dataset/{split}/{split}_{source}.json", "[]")
                handle.writestr(f"ChartQA Dataset/{split}/png/", "")
        directory = extract_dataset(archive, self.root / "extracted")
        self.assertEqual(directory.name, "ChartQA Dataset")

    def _manifest(self):
        image = self.root / "chart.png"
        Image.new("RGB", (280, 280), (1, 2, 3)).save(image)
        row = {"id": "sample", "image": image.name, "question": "Question", "width": 280, "height": 280,
               "image_sha256": sha256_file(image), "group_id": "one-image", "answer": "100", "source": "human", "split": "test"}
        path = self.root / "test.jsonl"
        write_jsonl(path, [row])
        return path

    def test_complete_run_exports_real_case_images_and_summary(self):
        output = self.root / "run"
        summary = evaluate_run(Backend(), self._manifest(), self.root, output, case_limit=2)
        self.assertEqual(summary["overall"]["accuracy"], 1)
        self.assertEqual(json.loads((output / "run.json").read_text())["status"], "complete")
        case = next((output / "cases").glob("*/trajectory.json"))
        trace = json.loads(case.read_text())
        self.assertEqual(trace["score"]["reward"], 1)
        self.assertTrue((case.parent / "image_0.png").is_file())

    def test_infrastructure_failure_never_produces_complete_summary(self):
        class FailureBackend:
            def generate(self, messages):
                raise RuntimeError("simulated GPU failure")
        output = self.root / "failed"
        with self.assertRaises(RuntimeError):
            evaluate_run(FailureBackend(), self._manifest(), self.root, output)
        self.assertEqual(json.loads((output / "run.json").read_text())["status"], "failed")
        self.assertFalse((output / "summary.json").exists())

    def test_human_region_diagnostic_disallowed_on_test_split(self):
        with self.assertRaises(ValueError):
            evaluate_run(Backend(), self._manifest(), self.root, self.root / "d0", policy="D0", diagnostic_file="regions.jsonl")


if __name__ == "__main__":
    unittest.main()
