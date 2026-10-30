"""Optional protocol warmup from explicitly reviewed training-only regions."""

import json
from pathlib import Path

from PIL import Image

from .crop import CropSession
from .data import read_jsonl, resolve_image, sha256_file, write_jsonl
from .episode import tool_feedback
from .protocol import parse_action, system_prompt, tool_action
from .vision import ImageBudget, display_image


def prepare_reviewed_sft(manifest, data_root, reviews, output, limit=128):
    rows = read_jsonl(manifest)
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("SFT preparation accepts training data only")
    if type(limit) is not int or limit < 1:
        raise ValueError("SFT sample limit must be positive")
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate training ids")
    annotations = read_jsonl(reviews)
    if not annotations or len({row["id"] for row in annotations}) != len(annotations):
        raise ValueError("Reviews must be non-empty with unique ids")
    for review in annotations:
        if review["id"] not in indexed or review.get("reviewed") is not True or not str(review.get("review_note", "")).strip():
            raise ValueError("Every review needs a training id, reviewed=true, and a non-empty review_note")
        if not isinstance(review.get("regions"), list) or len(review["regions"]) > 2:
            raise ValueError("Supply zero, one, or two reviewed regions")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    examples = []
    selected = sorted(annotations, key=lambda row: row["id"])[:limit]
    for index, review in enumerate(selected):
        row = indexed[review["id"]]
        original_path = resolve_image(Path(data_root), row["image"])
        if sha256_file(original_path) != row["image_sha256"]:
            raise ValueError("Training image changed after manifest freeze")
        with Image.open(original_path) as original:
            first = display_image(original, ImageBudget())
            session = CropSession(original, first.size)
        if review.get("reference_size") != list(first.size):
            raise ValueError("Review coordinates must match the actual first displayed image dimensions")
        initial_path = output / "images" / f"{index}_0.png"
        first.save(initial_path)
        images = [str(initial_path)]
        messages = [{"role": "system", "content": system_prompt(first.size, 2)},
                    {"role": "user", "content": "<image>\n" + row["question"]}]
        for number, region in enumerate(review["regions"], 1):
            result = session.execute(region)
            if not result.event["success"]:
                raise ValueError("Invalid reviewed crop: " + result.event["error"])
            crop = display_image(result.image, ImageBudget(max_pixels=501760))
            result.event["origin"] = "model"
            result.event["display_size"] = list(crop.size)
            crop_path = output / "images" / f"{index}_{number}.png"
            crop.save(crop_path)
            images.append(str(crop_path))
            messages.extend([{"role": "assistant", "content": tool_action(region)},
                             {"role": "user", "content": tool_feedback(result, session.remaining) + "\n<image>"}])
        final = "<answer>" + row["answer"] + "</answer>"
        if parse_action(final).kind != "answer":
            raise ValueError("Training target cannot be represented in the fixed answer protocol")
        messages.append({"role": "assistant", "content": final})
        examples.append({"messages": messages, "images": images, "chart_id": row["id"]})
    write_jsonl(output / "sft.jsonl", examples)
    audit = {"count": len(examples), "training_manifest_sha256": sha256_file(Path(manifest)),
             "review_file_sha256": sha256_file(Path(reviews)), "selected_ids": [row["id"] for row in selected],
             "review_notes": {row["id"]: row["review_note"] for row in selected},
             "note": "Supervised protocol demonstrations with reviewed regions and gold training answers; not model-generated reasoning or evidence of tool utility."}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
