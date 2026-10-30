"""Prepare frozen manifests from the official ChartQA directory layout."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

from PIL import Image


SOURCES = ("human", "augmented")
SPLITS = ("test", "val", "train")  # Keep test first; remove overlap from lower-priority splits.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_jsonl(path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{index}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def resolve_image(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing image or image outside dataset root: {relative_path}")
    return path


def load_official(root: Path):
    root = root.resolve()
    records = []
    input_files = []
    cache = {}
    for split in SPLITS:
        for source in SOURCES:
            path = root / split / f"{split}_{source}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected an array: {path}")
            input_files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "rows": len(rows)})
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object: {path}:{index}")
                name, question, answer = row.get("imgname"), row.get("query"), row.get("label")
                if not isinstance(name, str) or not name or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
                    raise ValueError(f"Invalid image basename: {path}:{index}")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Invalid question: {path}:{index}")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Expected one string label: {path}:{index}")
                relative = f"{split}/png/{name}"
                if relative not in cache:
                    image_path = resolve_image(root, relative)
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        pixels = hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
                        cache[relative] = {
                            "image_sha256": sha256_file(image_path),
                            "pixel_sha256": pixels,
                            "width": rgb.width,
                            "height": rgb.height,
                        }
                identity = {"split": split, "source": source, "image": relative, "question": question.strip(), "answer": answer.strip()}
                records.append({
                    "id": _digest(identity), "image_id": name, **identity, **cache[relative],
                    "group_id": cache[relative]["pixel_sha256"],
                    "source_row": index,
                })
    return records, input_files


def deduplicate(records: list[dict]):
    """Conservative image-level split isolation with explicit removal reasons.

    Repeated generic question text on different images is audited, not removed.
    Duplicate image names across splits are conservatively considered overlap,
    even when their pixels differ. Conflicting labels for the same image/question
    inside a split fail preparation instead of silently choosing an answer.
    """
    kept, removed = [], []
    earlier = {key: {} for key in ("image_id", "image_sha256", "pixel_sha256")}
    question_splits = defaultdict(set)
    for record in records:
        question_splits[" ".join(record["question"].lower().split())].add(record["split"])
    for split in SPLITS:
        within = set()
        labels = {}
        split_kept = []
        for record in (row for row in records if row["split"] == split):
            reasons = [key for key in earlier if record[key] in earlier[key]]
            if reasons:
                removed.append({"id": record["id"], "split": split, "source": record["source"], "image": record["image"],
                                "reason": "cross_split_image_overlap", "matched_by": reasons,
                                "retained_splits": sorted({earlier[key][record[key]] for key in reasons})})
                continue
            pair = (record["pixel_sha256"], " ".join(record["question"].lower().split()))
            if pair in labels and labels[pair] != record["answer"]:
                raise ValueError(f"Conflicting labels inside {split} for image/question: {record['image']}")
            labels[pair] = record["answer"]
            key = (*pair, record["source"])
            if key in within:
                removed.append({"id": record["id"], "split": split, "source": record["source"],
                                "image": record["image"], "reason": "duplicate_image_question_source"})
                continue
            within.add(key)
            split_kept.append(record)
        # Register all source images, including removed aliases. This prevents
        # an overlap chain (different filename/encoding) from re-entering train.
        for record in (row for row in records if row["split"] == split):
            for key in earlier:
                earlier[key].setdefault(record[key], split)
        kept.extend(split_kept)
    return kept, {
        "split_priority": list(SPLITS),
        "removed": removed,
        "cross_split_question_text_count": sum(len(splits) > 1 for splits in question_splits.values()),
        "question_text_note": "Generic question repeats without image overlap are reported, not removed.",
        "pretraining_contamination_checked": False,
    }


def sample_records(records, limit: int | None, seed: int):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Sample limits must be positive integers or None")
    if limit is None or limit >= len(records):
        return sorted(records, key=lambda row: row["id"])
    ordered = {
        source: sorted((row for row in records if row["source"] == source),
                       key=lambda row: _digest([seed, row["id"]]))
        for source in SOURCES
    }
    selected, offsets = [], Counter()
    # Alternate sources; if one is exhausted, use the remainder of the other.
    while len(selected) < limit:
        for source in SOURCES:
            if offsets[source] < len(ordered[source]) and len(selected) < limit:
                selected.append(ordered[source][offsets[source]])
                offsets[source] += 1
    return sorted(selected, key=lambda row: row["id"])


def observation_record(record: dict) -> dict:
    """Allowlist the only manifest fields an inference backend may receive."""
    return {key: record[key] for key in ("id", "image", "question", "width", "height")}


def prepare_chartqa(root, output, train_limit=1024, val_limit=128, test_limit=None, pilot_limit=50, seed=42):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen manifests; choose a new output directory")
    if type(pilot_limit) is not int or pilot_limit < 1:
        raise ValueError("pilot_limit must be positive")
    records, input_files = load_official(root)
    kept, audit = deduplicate(records)
    limits = {"train": train_limit, "val": val_limit, "test": test_limit}
    selected = {split: sample_records([r for r in kept if r["split"] == split], limits[split], seed) for split in SPLITS}
    if any(not rows for rows in selected.values()):
        raise ValueError("A split is empty after deduplication; inspect the source data before proceeding")
    selected["pilot"] = sample_records(selected["val"], pilot_limit, seed + 1)
    audit.update({
        "schema_version": 1, "seed": seed, "input_files": input_files,
        "input_counts": dict(Counter(r["split"] for r in records)),
        "deduplicated_counts": dict(Counter(r["split"] for r in kept)),
        "selected_counts": {key: len(rows) for key, rows in selected.items()},
        "selected_source_counts": {key: dict(Counter(r["source"] for r in rows)) for key, rows in selected.items()},
        "limits": {**limits, "pilot": pilot_limit},
        "test_is_subset": len(selected["test"]) < sum(r["split"] == "test" for r in kept),
        "source": "https://github.com/vis-nlp/ChartQA",
    })
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    audit["manifest_sha256"] = {split: sha256_file(output / f"{split}.jsonl") for split in selected}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return audit
