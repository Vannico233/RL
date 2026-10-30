import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from chart_rl.cli import main
from chart_rl.data import load_official, observation_record, prepare_chartqa, read_jsonl, write_jsonl
from chart_rl.evaluation import paired_cluster_comparison, score_predictions, summarize


def make_dataset(root):
    for split_index, split in enumerate(("train", "val", "test")):
        folder = root / split
        (folder / "png").mkdir(parents=True)
        for source_index, source in enumerate(("human", "augmented")):
            name = f"{split}_{source}.png"
            Image.new("RGB", (12, 10), (split_index * 60, source_index * 90, 25)).save(folder / "png" / name)
            rows = [{"imgname": name, "query": "What is the value?", "label": "100"}]
            (folder / f"{split}_{source}.json").write_text(json.dumps(rows), encoding="utf-8")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "dataset"
        make_dataset(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_dedupe_pixel_alias_preserves_test_and_audits(self):
        with Image.open(self.root / "test/png/test_human.png") as image:
            image.save(self.root / "train/png/renamed.png", compress_level=0)
        path = self.root / "train/train_human.json"
        rows = json.loads(path.read_text())
        rows.append({"imgname": "renamed.png", "query": "Another question?", "label": "3"})
        path.write_text(json.dumps(rows))
        output = Path(self.temp.name) / "manifest"
        audit = prepare_chartqa(self.root, output, pilot_limit=1)
        self.assertEqual(audit["input_counts"]["train"], 3)
        self.assertEqual(audit["selected_counts"]["train"], 2)
        self.assertIn("pixel_sha256", audit["removed"][0]["matched_by"])
        train = read_jsonl(output / "train.jsonl")
        test = read_jsonl(output / "test.jsonl")
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in test})
        self.assertEqual(set(observation_record(train[0])), {"id", "image", "question", "width", "height"})

    def test_manifests_reproducible_and_pilot_is_validation_subset(self):
        first, second = Path(self.temp.name) / "a", Path(self.temp.name) / "b"
        one = prepare_chartqa(self.root, first, pilot_limit=1, seed=7)
        two = prepare_chartqa(self.root, second, pilot_limit=1, seed=7)
        self.assertEqual(one, two)
        self.assertLessEqual({r["id"] for r in read_jsonl(first / "pilot.jsonl")},
                             {r["id"] for r in read_jsonl(first / "val.jsonl")})
        with self.assertRaises(FileExistsError):
            prepare_chartqa(self.root, first)

    def test_image_path_escape_rejected(self):
        path = self.root / "train/train_human.json"
        path.write_text(json.dumps([{"imgname": "../../other.png", "query": "Q", "label": "A"}]))
        with self.assertRaises(ValueError):
            load_official(self.root)

    def test_cli_prepare_score_end_to_end(self):
        manifests = Path(self.temp.name) / "manifests"
        with redirect_stdout(StringIO()):
            main(["prepare", "--data-root", str(self.root), "--output", str(manifests)])
        records = read_jsonl(manifests / "test.jsonl")
        predictions = Path(self.temp.name) / "predictions.jsonl"
        write_jsonl(predictions, [{"id": r["id"], "completion": "<answer>100</answer>",
                                  "tool_events": [], "cost": {"generated_tokens": 8}} for r in records])
        report = Path(self.temp.name) / "report"
        with redirect_stdout(StringIO()):
            main(["score", "--manifest", str(manifests / "test.jsonl"), "--predictions", str(predictions), "--output", str(report)])
        result = json.loads((report / "summary.json").read_text())
        self.assertEqual(result["overall"]["accuracy"], 1)
        self.assertEqual(result["overall"]["cost"]["visual_tokens"]["coverage"], 0)
        self.assertIsNone(result["overall"]["cost"]["visual_tokens"]["mean"])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [{"id": str(i), "group_id": str(i // 2), "source": "human" if i % 2 else "augmented",
                          "split": "test", "answer": "100"} for i in range(4)]

    def test_missing_or_duplicate_predictions_fail(self):
        predictions = [{"id": r["id"], "completion": "<answer>100</answer>"} for r in self.manifest]
        with self.assertRaises(ValueError):
            score_predictions(self.manifest, predictions[:-1])
        with self.assertRaises(ValueError):
            score_predictions(self.manifest, predictions + predictions[:1])

    def test_failures_remain_in_all_denominators(self):
        predictions = [{"id": r["id"], "completion": "<answer>100</answer>" if i == 0 else "",
                        "cost": {"generated_tokens": 10}, "tool_events": []} for i, r in enumerate(self.manifest)]
        result = summarize(score_predictions(self.manifest, predictions))
        self.assertEqual(result["overall"]["accuracy"], 0.25)
        self.assertEqual(result["overall"]["parse_rate"], 0.25)
        self.assertEqual(result["overall"]["cost"]["generated_tokens"]["sum"], 40)

    def test_paired_bootstrap_groups_images_and_is_reproducible(self):
        left = score_predictions(self.manifest, [{"id": r["id"], "completion": ""} for r in self.manifest])
        right = score_predictions(self.manifest, [{"id": r["id"], "completion": "<answer>100</answer>"} for r in self.manifest])
        comparison = paired_cluster_comparison(left, right, samples=100)
        self.assertEqual(comparison["image_group_count"], 2)
        self.assertEqual(comparison["accuracy_difference"], 1)
        self.assertEqual(comparison["bootstrap_95_percent_interval"], [1, 1])
        self.assertEqual(comparison, paired_cluster_comparison(left, right, samples=100))
        right[0]["target"] = "200"
        with self.assertRaises(ValueError):
            paired_cluster_comparison(left, right, samples=100)


if __name__ == "__main__":
    unittest.main()
