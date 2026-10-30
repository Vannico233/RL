"""Export paired B1/B2 datasets and exact initial-image coordinate references."""

from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image

from .data import read_jsonl, resolve_image, sha256_file, write_jsonl
from .protocol import system_prompt
from .vision import ImageBudget, display_image


def export_swift(manifest, data_root, output, whole_budget=ImageBudget(), crop_budget=ImageBudget(max_pixels=501760)):
    manifest, data_root, output = Path(manifest), Path(data_root), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory to preserve the previous configuration")
    records = read_jsonl(manifest)
    if not records or any(row["split"] not in ("train", "val") for row in records):
        raise ValueError("Training exports accept non-empty official train/val manifests only")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate ids in manifest")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    exports = {"B1": [], "B2": []}
    cache = {}
    for row in records:
        original_path = resolve_image(data_root, row["image"])
        if str(original_path) not in cache:
            if sha256_file(original_path) != row["image_sha256"]:
                raise ValueError("Original image hash differs from frozen manifest: " + row["image"])
            with Image.open(original_path) as original:
                if original.size != (row["width"], row["height"]):
                    raise ValueError("Original dimensions differ from frozen manifest")
                initial = display_image(original, whole_budget)
            display_path = output / "images" / (row["image_sha256"] + ".png")
            initial.save(display_path)
            cache[str(original_path)] = (display_path, initial.size)
        display_path, reference_size = cache[str(original_path)]
        if "<image>" in row["question"]:
            raise ValueError("Question contains a reserved image placeholder; inspect it before exporting")
        for setting, max_calls in (("B1", 0), ("B2", 2)):
            exports[setting].append({
                "messages": [{"role": "system", "content": system_prompt(reference_size, max_calls)},
                             {"role": "user", "content": "<image>\n" + row["question"]}],
                "images": [str(display_path)], "solution": row["answer"],
                "chart_id": row["id"], "chart_original_image": str(original_path),
                "chart_image_sha256": row["image_sha256"], "chart_reference_size": list(reference_size),
                "chart_crop_max_pixels": crop_budget.max_pixels, "chart_min_pixels": crop_budget.min_pixels,
            })
    for setting, rows in exports.items():
        write_jsonl(output / f"{setting}.jsonl", rows)
    info = {
        "manifest_sha256": sha256_file(manifest), "count": len(records),
        "whole_budget": asdict(whole_budget), "crop_budget": asdict(crop_budget),
        "output_hashes": {setting: sha256_file(output / f"{setting}.jsonl") for setting in exports},
        "note": "Export paths are absolute and must be accessible to both trainer and rollout server on the same host.",
    }
    (output / "export.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info
