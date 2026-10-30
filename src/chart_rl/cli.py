"""CPU commands; training/inference backends are integrated separately."""

import argparse
import json
from pathlib import Path

from .data import prepare_chartqa, read_jsonl, write_jsonl
from .evaluation import paired_cluster_comparison, score_predictions, summarize
from .swift_data import export_swift


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="chart-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze official ChartQA manifests with image-level split checks")
    prep.add_argument("--data-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--train-limit", type=int, default=1024)
    prep.add_argument("--val-limit", type=int, default=128)
    prep.add_argument("--test-limit", type=int, default=None)
    prep.add_argument("--pilot-limit", type=int, default=50)
    prep.add_argument("--seed", type=int, default=42)
    export = sub.add_parser("export-swift", help="Export paired B1/B2 training inputs with exact image coordinates")
    export.add_argument("--manifest", required=True)
    export.add_argument("--data-root", required=True)
    export.add_argument("--output", required=True)
    scoring = sub.add_parser("score", help="Score every question in a frozen manifest")
    scoring.add_argument("--manifest", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--output", required=True)
    compare = sub.add_parser("compare", help="Paired comparison of scored JSONL files")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--samples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("evaluate", help="Run real Qwen2.5-VL inference and score a complete frozen manifest")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--policy", choices=("O0", "O1", "O2", "O3", "O4", "D0"), default="O4")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--max-new-tokens", type=int, default=256)
    evaluate.add_argument("--case-limit", type=int, default=10)
    evaluate.add_argument("--diagnostic-regions")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        audit = prepare_chartqa(args.data_root, args.output, args.train_limit, args.val_limit,
                                args.test_limit, args.pilot_limit, args.seed)
        print(json.dumps({"output": args.output, "selected_counts": audit["selected_counts"]}, ensure_ascii=False))
    elif args.command == "export-swift":
        print(json.dumps(export_swift(args.manifest, args.data_root, args.output), ensure_ascii=False))
    elif args.command == "score":
        scored = score_predictions(read_jsonl(args.manifest), read_jsonl(args.predictions))
        summary = summarize(scored)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        write_jsonl(output / "scored.jsonl", scored)
        _json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
    elif args.command == "evaluate":
        from .run_evaluation import evaluate_run
        from .transformers_backend import TransformersBackend
        from .provenance import code_identity, model_identity, package_versions
        if Path(args.output).exists():
            raise FileExistsError("Evaluation output already exists")
        provenance = {"model_path": args.model, "adapter_path": args.adapter, "max_new_tokens": args.max_new_tokens,
                      "backend": "transformers", "temperature": 0.0,
                      "model_identity": model_identity(args.model),
                      "adapter_identity": model_identity(args.adapter) if args.adapter else None,
                      "code_identity": code_identity(Path(__file__).resolve().parents[2]), "environment": package_versions()}
        backend = TransformersBackend(args.model, args.adapter, args.device, args.max_new_tokens, args.seed)
        result = evaluate_run(backend, args.manifest, args.data_root, args.output, args.policy, args.seed,
                              args.case_limit, args.diagnostic_regions, provenance)
        print(json.dumps(result, ensure_ascii=False))
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("Refusing to overwrite a comparison report")
        result = paired_cluster_comparison(read_jsonl(args.left), read_jsonl(args.right), args.samples, args.seed)
        _json(output, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
