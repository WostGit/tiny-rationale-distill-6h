#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from logging_utils import read_json, write_json
from metrics_utils import summarize_seed_metrics
from plotting_utils import plot_scaling_curves


def run_cmd(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def condition_id(c: Dict) -> str:
    return f"{c['task']}|{c['format']}|b{c['budget']}|s{c['seed']}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["arithmetic", "symbolic"])
    p.add_argument("--formats", nargs="+", default=["answer_only", "short_rationale", "full_rationale"])
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--output_root", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def aggregate(output_root: Path) -> None:
    run_root = output_root / "runs"
    eval_rows = []
    train_rows = []
    for run_dir in sorted(run_root.glob("*")):
        tpath = run_dir / "train_metrics.json"
        epath = run_dir / "eval_metrics.json"
        if tpath.exists() and epath.exists():
            train_rows.append(read_json(tpath))
            eval_rows.append(read_json(epath))

    agg_rows = []
    tmap = {r["run_id"]: r for r in train_rows}
    for e in eval_rows:
        row = dict(e)
        row["train_runtime_sec"] = tmap.get(e["run_id"], {}).get("train_runtime_sec", None)
        row["train_loss"] = tmap.get(e["run_id"], {}).get("train_loss", None)
        agg_rows.append(row)

    write_json(output_root / "all_condition_metrics.json", {"rows": agg_rows})
    summary = summarize_seed_metrics(agg_rows)
    write_json(output_root / "summary_mean_std.json", {"rows": summary})

    csv_path = output_root / "summary_mean_std.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "format", "budget", "mean_exact_match", "std_exact_match", "n_seeds"])
        w.writeheader()
        w.writerows(summary)

    md_lines = ["| task | format | budget | mean_em | std_em | n_seeds |", "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        md_lines.append(
            f"| {r['task']} | {r['format']} | {r['budget']} | {r['mean_exact_match']:.4f} | {r['std_exact_match']:.4f} | {r['n_seeds']} |"
        )
    (output_root / "comparison_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_scaling_curves(csv_path, output_root)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "scripts/contamination_audit.py", "--data_dir", str(args.data_dir), "--output", str(args.output_root / "contamination_audit.json")])

    conditions = [
        {"task": t, "format": f, "budget": b, "seed": s}
        for t, f, b, s in itertools.product(args.tasks, args.formats, args.budgets, args.seeds)
    ]
    conditions = sorted(conditions, key=condition_id)

    sharded = [c for idx, c in enumerate(conditions) if idx % args.num_shards == args.shard_index]
    print(f"Total conditions: {len(conditions)}, shard {args.shard_index}/{args.num_shards} -> {len(sharded)}")

    for c in sharded:
        common = [
            "--task",
            c["task"],
            "--format",
            c["format"],
            "--budget",
            str(c["budget"]),
            "--seed",
            str(c["seed"]),
            "--model_name",
            args.model_name,
            "--output_root",
            str(args.output_root),
            "--data_dir",
            str(args.data_dir),
        ]
        if args.force:
            common += ["--force"]
        run_cmd([sys.executable, "scripts/train_tiny_distill.py", *common, "--epochs", str(args.epochs)])
        run_cmd([sys.executable, "scripts/eval_tiny_distill.py", *common])
        run_id = f"{c['task']}__{c['format']}__b{c['budget']}__s{c['seed']}"
        em = read_json(args.output_root / "runs" / run_id / "eval_metrics.json")["exact_match"]
        train_t = read_json(args.output_root / "runs" / run_id / "train_metrics.json")["train_runtime_sec"]
        eval_t = read_json(args.output_root / "runs" / run_id / "eval_metrics.json")["eval_runtime_sec"]
        print(
            f"[done] task={c['task']} format={c['format']} budget={c['budget']} seed={c['seed']} "
            f"train_s={train_t:.1f} eval_s={eval_t:.1f} em={em:.4f}"
        )

    aggregate(args.output_root)


if __name__ == "__main__":
    main()
