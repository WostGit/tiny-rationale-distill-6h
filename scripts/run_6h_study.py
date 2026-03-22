#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from contamination_audit import audit
from eval_tiny_distill import evaluate
from logging_utils import as_markdown_table, setup_logger, stable_condition_id, write_json
from metrics_utils import load_eval_metrics, summarize_over_seeds
from plotting_utils import plot_scaling_curves
from train_tiny_distill import train

logger = setup_logger()


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        for supervision in ["answer_only", "short_rationale", "full_rationale"]:
            train_file = f"data/{task}_train_{supervision}.jsonl"
            eval_file = f"data/{task}_eval.jsonl"
            audit_out = output_root / f"audit_{task}_{supervision}.json"
            if not audit_out.exists() or args.overwrite:
                audit(train_file, eval_file, str(audit_out))

    all_condition_rows = []

    for task in args.tasks:
        for supervision in args.supervision_formats:
            train_file = f"data/{task}_train_{supervision}.jsonl"
            eval_file = f"data/{task}_eval.jsonl"
            for budget in args.budgets:
                for seed in args.seeds:
                    cond = stable_condition_id(task, supervision, budget, seed)
                    start = time.time()
                    train_args = argparse.Namespace(
                        task=task,
                        supervision=supervision,
                        train_file=train_file,
                        budget=budget,
                        seed=seed,
                        output_root=str(output_root),
                        model_name=args.model_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        max_length=args.max_length,
                        lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        overwrite=args.overwrite,
                    )
                    train_metrics = train(train_args)
                    eval_args = argparse.Namespace(
                        task=task,
                        supervision=supervision,
                        budget=budget,
                        seed=seed,
                        eval_file=eval_file,
                        output_root=str(output_root),
                        model_name=args.model_name,
                        max_new_tokens=args.max_new_tokens,
                        num_prediction_samples=args.num_prediction_samples,
                        overwrite=args.overwrite,
                    )
                    eval_metrics = evaluate(eval_args)
                    elapsed = time.time() - start
                    all_condition_rows.append(
                        {
                            "condition_id": cond,
                            "task": task,
                            "supervision": supervision,
                            "budget": budget,
                            "seed": seed,
                            "train_time_sec": train_metrics.get("train_time_sec"),
                            "eval_time_sec": eval_metrics.get("eval_time_sec"),
                            "exact_match": eval_metrics.get("exact_match"),
                            "wall_time_sec": round(elapsed, 3),
                        }
                    )
                    logger.info(
                        "COND_DONE task=%s format=%s budget=%s seed=%s train=%.1fs eval=%.1fs em=%.3f",
                        task,
                        supervision,
                        budget,
                        seed,
                        float(train_metrics.get("train_time_sec", 0.0)),
                        float(eval_metrics.get("eval_time_sec", 0.0)),
                        float(eval_metrics.get("exact_match", 0.0)),
                    )

    all_df = pd.DataFrame(all_condition_rows)
    all_df.sort_values(["task", "supervision", "budget", "seed"], inplace=True)
    all_df.to_csv(output_root / "condition_results.csv", index=False)
    write_json(output_root / "condition_results.json", all_df.to_dict(orient="records"))

    eval_df = load_eval_metrics(output_root)
    summary_df = summarize_over_seeds(eval_df)
    summary_df.to_csv(output_root / "summary_over_seeds.csv", index=False)
    write_json(output_root / "summary_over_seeds.json", summary_df.to_dict(orient="records"))

    if not summary_df.empty:
        headers = ["task", "supervision", "budget", "exact_match_mean", "exact_match_std", "n_seeds"]
        rows = [
            [
                r["task"],
                r["supervision"],
                int(r["budget"]),
                f"{r['exact_match_mean']:.3f}",
                f"{r['exact_match_std']:.3f}",
                int(r["n_seeds"]),
            ]
            for _, r in summary_df.iterrows()
        ]
        (output_root / "comparison_table.md").write_text(as_markdown_table(headers, rows), encoding="utf-8")

    plot_scaling_curves(summary_df, output_root)


def main() -> None:
    p = argparse.ArgumentParser(description="Run tiny 6h-style rationale distillation study")
    p.add_argument("--tasks", type=parse_str_csv, default=["arithmetic", "symbolic"])
    p.add_argument("--supervision_formats", type=parse_str_csv, default=["answer_only", "short_rationale", "full_rationale"])
    p.add_argument("--budgets", type=parse_int_csv, default=[16, 32, 64, 128])
    p.add_argument("--seeds", type=parse_int_csv, default=[11, 12, 13, 14, 15])
    p.add_argument("--output_root", default="outputs/metrics")
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--num_prediction_samples", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()
