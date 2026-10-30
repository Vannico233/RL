"""GPU evaluation orchestration with complete-coverage scoring and trace export."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from .data import observation_record, read_jsonl, resolve_image, sha256_file, write_jsonl
from .episode import run_episode
from .evaluation import score_predictions, summarize
from .reward import score_completion
from .vision import ImageBudget


def _save_case(directory, trajectory, observation, score, original_path):
    directory.mkdir(parents=True, exist_ok=False)
    with Image.open(original_path) as original:
        original.convert("RGB").save(directory / "original.png")
    serial = []
    image_index = 0
    for message in trajectory["messages"]:
        content = message["content"]
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block["type"] == "image":
                    name = f"image_{image_index}.png"
                    block["image"].save(directory / name)
                    blocks.append({"type": "image", "path": name, "size": list(block["image"].size)})
                    image_index += 1
                else:
                    blocks.append(dict(block))
            content = blocks
        serial.append({"role": message["role"], "content": content})
    payload = {"observation": observation, "original_image": "original.png", "score": score.to_dict(), "messages": serial,
               "prediction": trajectory["prediction"], "turns": trajectory["turns"]}
    (directory / "trajectory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_run(backend, manifest, data_root, output, policy="O4", seed=42, case_limit=10,
                 diagnostic_file=None, provenance=None):
    manifest, data_root, output = Path(manifest), Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh evaluation directory; prior runs must remain immutable")
    if type(case_limit) is not int or case_limit < 0:
        raise ValueError("case_limit must be a nonnegative integer")
    records = read_jsonl(manifest)
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Evaluation requires a non-empty manifest with unique ids")
    diagnostics = {}
    if policy == "D0":
        if not diagnostic_file or any(row["split"] != "val" for row in records):
            raise ValueError("D0 requires a validation-only manifest and a reviewed region file")
        region_rows = read_jsonl(diagnostic_file)
        diagnostics = {row["id"]: row["regions"] for row in region_rows}
        if len(diagnostics) != len(region_rows) or set(diagnostics) != {row["id"] for row in records}:
            raise ValueError("Diagnostic region ids must exactly cover the selected validation manifest")
    output.mkdir(parents=True)
    predictions, saved = [], {True: 0, False: 0}
    quotas = {True: case_limit // 2, False: case_limit - case_limit // 2}
    run_info = {
        "status": "running", "policy": policy, "seed": seed, "manifest_sha256": sha256_file(manifest),
        "question_count": len(records), "provenance": provenance or {},
        "whole_budget": asdict(ImageBudget()), "high_budget": asdict(ImageBudget(max_pixels=2007040)),
        "crop_budget": asdict(ImageBudget(max_pixels=501760)),
        "diagnostic_region_sha256": sha256_file(Path(diagnostic_file)) if diagnostic_file else None,
        "case_selection": "First correct/incorrect cases up to the declared quotas; metrics cover every question.",
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as prediction_file, \
                (output / "turns.jsonl").open("w", encoding="utf-8") as turns_file:
            for record in records:
                image_path = resolve_image(data_root, record["image"])
                if sha256_file(image_path) != record["image_sha256"]:
                    raise ValueError("Evaluation image differs from frozen manifest")
                with Image.open(image_path) as image:
                    trajectory = run_episode(backend, observation_record(record), image, policy=policy, seed=seed,
                                             diagnostic_regions=diagnostics.get(record["id"]))
                prediction = trajectory["prediction"]
                predictions.append(prediction)
                prediction_file.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                prediction_file.flush()
                turns_file.write(json.dumps({"id": record["id"], "turns": trajectory["turns"]}, ensure_ascii=False, allow_nan=False) + "\n")
                turns_file.flush()
                score = score_completion(prediction["completion"], record["answer"])
                if saved[score.relaxed] < quotas[score.relaxed]:
                    case_name = hashlib.sha256(record["id"].encode("utf-8")).hexdigest()[:20]
                    _save_case(output / "cases" / case_name, trajectory, observation_record(record), score, image_path)
                    saved[score.relaxed] += 1
        scored = score_predictions(records, predictions)
        write_jsonl(output / "scored.jsonl", scored)
        summary = summarize(scored)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as error:
        run_info.update(status="failed", completed_questions=len(predictions), error_type=type(error).__name__, error=str(error))
        run_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    run_info.update(status="complete", completed_questions=len(predictions), saved_cases=sum(saved.values()))
    run_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
