"""Score complete predictions and compare runs with image-cluster resampling."""

from collections import defaultdict
import math
import random
from statistics import mean

from .reward import SCORING_PROTOCOL, score_completion


COST_FIELDS = ("generated_tokens", "visual_tokens", "image_pixels", "latency_seconds")


def _indexed(records, name):
    result = {}
    for row in records:
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"Missing or duplicate id in {name}: {identity}")
        result[identity] = row
    if not result:
        raise ValueError(f"Empty {name}")
    return result


def _costs(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("cost must be an object")
    result = {}
    for field in COST_FIELDS:
        number = value.get(field)
        if number is not None:
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError(f"Invalid cost {field}: {number}")
            if field != "latency_seconds" and type(number) is not int:
                raise ValueError(f"Cost {field} must be an integer count")
        result[field] = number
    return result


def score_predictions(manifest, predictions):
    gold, outputs = _indexed(manifest, "manifest"), _indexed(predictions, "predictions")
    if gold.keys() != outputs.keys():
        raise ValueError(f"Prediction coverage mismatch: missing={len(gold.keys() - outputs.keys())}, extra={len(outputs.keys() - gold.keys())}")
    scored = []
    for identity, item in gold.items():
        pred = outputs[identity]
        if not isinstance(pred.get("completion"), str):
            raise ValueError(f"Missing text completion for {identity}; failed episodes need an explicit empty completion")
        score = score_completion(pred["completion"], item["answer"])
        events = pred.get("tool_events")
        if events is not None:
            if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
                raise ValueError("tool_events must be a list of objects or omitted")
            if any(type(event.get("success")) is not bool or type(event.get("consumed_budget")) is not bool for event in events):
                raise ValueError("Every tool event must have boolean success and consumed_budget fields")
            attempts = sum(event["consumed_budget"] for event in events)
            if attempts > 2:
                raise ValueError("Recorded trajectory exceeded two tool attempts")
            invalid = sum(not event["success"] for event in events)
            repeated = sum(event.get("repeated_region") is True and event["success"] for event in events)
            successes = sum(event["success"] for event in events)
        else:
            attempts = invalid = repeated = successes = None
        truncated = pred.get("truncated")
        if truncated is not None and type(truncated) is not bool:
            raise ValueError("truncated must be boolean or omitted")
        scored.append({
            "id": identity, "group_id": item["group_id"], "source": item["source"],
            "split": item["split"], "completion": pred["completion"], "target": item["answer"],
            **score.to_dict(), "cost": _costs(pred.get("cost")), "tool_attempts": attempts,
            "tool_requests": len(events) if events is not None else None,
            "invalid_tool_requests": invalid, "successful_crops": successes,
            "repeated_crops": repeated, "truncated": truncated,
        })
    return scored


def summarize(scored):
    if not scored:
        raise ValueError("Cannot summarize an empty run")
    def group_summary(rows):
        result = {
            "count": len(rows), "image_groups": len({row["group_id"] for row in rows}),
            "accuracy": mean(row["relaxed"] for row in rows),
            "strict_accuracy": mean(row["strict"] for row in rows),
            "parse_rate": mean(row["parsed"] for row in rows),
        }
        result["cost"] = {}
        for field in COST_FIELDS:
            values = [row["cost"][field] for row in rows if row["cost"][field] is not None]
            result["cost"][field] = {
                "coverage": len(values) / len(rows), "recorded_count": len(values),
                "mean": mean(values) if values else None,
                "sum": sum(values) if values else None,
            }
        tools = [row for row in rows if row["tool_requests"] is not None]
        requests = sum(row["tool_requests"] for row in tools)
        successes = sum(row["successful_crops"] for row in tools)
        result["tools"] = {
            "coverage": len(tools) / len(rows),
            "call_rate": mean(row["tool_requests"] > 0 for row in tools) if tools else None,
            "mean_attempts": mean(row["tool_attempts"] for row in tools) if tools else None,
            "mean_successful_crops": mean(row["successful_crops"] for row in tools) if tools else None,
            "invalid_request_rate": sum(row["invalid_tool_requests"] for row in tools) / requests if requests else None,
            "repeated_successful_crop_rate": sum(row["repeated_crops"] for row in tools) / successes if successes else None,
        }
        truncation = [row["truncated"] for row in rows if row["truncated"] is not None]
        result["truncation"] = {"coverage": len(truncation) / len(rows), "rate": mean(truncation) if truncation else None}
        return result
    grouped = {source: [row for row in scored if row["source"] == source] for source in sorted({row["source"] for row in scored})}
    source_metrics = {source: group_summary(rows) for source, rows in grouped.items()}
    return {
        "protocol": SCORING_PROTOCOL, "overall": group_summary(scored), "by_source": source_metrics,
        "source_macro_accuracy": mean(value["accuracy"] for value in source_metrics.values()),
        "notes": ["Missing cost observations remain null and lower coverage; they are never replaced with zero.",
                  "All accuracy denominators include parse failures and failed episodes.",
                  "Metrics describe this run only; no multi-training-seed claim is implied."],
    }


def paired_cluster_comparison(left, right, samples=2000, seed=42):
    """right minus left; resample whole image groups, preserve all their QAs."""
    if type(samples) is not int or samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    lhs, rhs = _indexed(left, "left run"), _indexed(right, "right run")
    if lhs.keys() != rhs.keys():
        raise ValueError("Paired comparison requires identical question ids")
    grouped = defaultdict(list)
    for identity, row in lhs.items():
        other = rhs[identity]
        if any(row[key] != other[key] for key in ("group_id", "target", "source", "split", "protocol")):
            raise ValueError(f"Incompatible paired records: {identity}")
        grouped[row["group_id"]].append(int(other["relaxed"]) - int(row["relaxed"]))
    if len(grouped) < 2:
        raise ValueError("At least two independent image groups are required for a confidence interval")
    clusters = [(sum(grouped[key]), len(grouped[key])) for key in sorted(grouped)]
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        deltas.append(sum(item[0] for item in chosen) / sum(item[1] for item in chosen))
    deltas.sort()
    def quantile(probability):
        position = (len(deltas) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        return deltas[lower] + (deltas[upper] - deltas[lower]) * (position - lower)
    return {
        "direction": "right_minus_left", "question_count": len(lhs), "image_group_count": len(clusters),
        "accuracy_difference": sum(total for total, _ in clusters) / len(lhs),
        "bootstrap_95_percent_interval": [quantile(0.025), quantile(0.975)],
        "bootstrap_samples": samples, "seed": seed,
        "note": "Image-cluster bootstrap over one paired evaluation; does not measure training-seed uncertainty.",
    }
