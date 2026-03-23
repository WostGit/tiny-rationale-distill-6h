"""Orchestrate the full tiny distillation grid and aggregate outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from metrics_utils import load_eval_metrics, save_summary_files, summarize
from plotting_utils import plot_scaling_curves



def parse_csv_ints(v: str):
    return [int(x.strip()) for x in v.split(",") if x.strip()]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="arithmetic,symbolic")
    p.add_argument("--formats", default="answer_only,short_rationale,full_rationale")
    p.add_argument("--budgets", default="16,32,64,128")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--output-root", default="outputs/metrics")
    p.add_argument("--python-bin", default=sys.executable)
    return p.parse_args()



def run_cmd(cmd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)



def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    budgets = parse_csv_ints(args.budgets)
    seeds = parse_csv_ints(args.seeds)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    total = len(tasks) * len(formats) * len(budgets) * len(seeds)

    for task in tasks:
        for supervision in formats:
            for budget in budgets:
                for seed in seeds:
                    run_name = f"{task}__{supervision}__b{budget}__s{seed}"
                    eval_json = out_root / f"eval_{run_name}.json"
                    if eval_json.exists():
                        completed += 1
                        print(f"[resume] ({completed}/{total}) {run_name} already complete")
                        continue

                    train_cmd = [
                        args.python_bin,
                        "scripts/train_tiny_distill.py",
                        "--task",
                        task,
                        "--supervision",
                        supervision,
                        "--budget",
                        str(budget),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--output-root",
                        str(out_root),
                    ]
                    eval_cmd = [
                        args.python_bin,
                        "scripts/eval_tiny_distill.py",
                        "--task",
                        task,
                        "--supervision",
                        supervision,
                        "--budget",
                        str(budget),
                        "--seed",
                        str(seed),
                        "--output-root",
                        str(out_root),
                    ]
                    run_cmd(train_cmd)
                    run_cmd(eval_cmd)
                    completed += 1
                    print(f"[condition-complete] ({completed}/{total}) {run_name}")

    eval_rows = load_eval_metrics(out_root)
    summary_df = summarize(eval_rows)
    saved = save_summary_files(summary_df, out_root)
    plot_scaling_curves(summary_df, out_root)

    print("[summary] saved:", saved)
    print(f"[summary] finished {len(eval_rows)} eval rows")


if __name__ == "__main__":
    main()
