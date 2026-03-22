import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from logging_utils import condition_id, ensure_dir, load_json, print_condition_summary, save_json
from plotting_utils import plot_scaling_curves


def run_cmd(cmd):
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['arithmetic', 'symbolic'])
    ap.add_argument('--formats', nargs='+', default=['answer_only', 'short_rationale', 'full_rationale'])
    ap.add_argument('--budgets', nargs='+', type=int, default=[16, 32, 64, 128])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--model-name', default='roneneldan/TinyStories-33M')
    ap.add_argument('--output-root', default='outputs/metrics')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-conditions', type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)

    conditions = []
    for t in args.tasks:
        for f in args.formats:
            for b in args.budgets:
                for s in args.seeds:
                    conditions.append((t, f, b, s))
    if args.max_conditions:
        conditions = conditions[: args.max_conditions]

    all_rows = []
    for t, f, b, s in conditions:
        cond = condition_id(t, f, b, s)
        train_cmd = [
            sys.executable, 'scripts/train_tiny_distill.py', '--task', t,
            '--supervision-format', f, '--budget', str(b), '--seed', str(s),
            '--epochs', str(args.epochs), '--model-name', args.model_name,
            '--output-root', args.output_root, '--data-dir', args.data_dir
        ]
        if args.resume:
            train_cmd.append('--resume')
        run_cmd(train_cmd)

        eval_cmd = [
            sys.executable, 'scripts/eval_tiny_distill.py', '--task', t,
            '--supervision-format', f, '--budget', str(b), '--seed', str(s),
            '--model-name', args.model_name,
            '--output-root', args.output_root, '--data-dir', args.data_dir
        ]
        if args.resume:
            eval_cmd.append('--resume')
        run_cmd(eval_cmd)

        train_m = load_json(out_root / cond / 'train_metrics.json')
        eval_m = load_json(out_root / cond / 'eval_metrics.json')
        row = {**train_m, **eval_m}
        all_rows.append(row)
        print_condition_summary(row)

    # contamination audit
    run_cmd([sys.executable, 'scripts/contamination_audit.py', '--data-dir', args.data_dir, '--output-root', args.output_root])

    all_path = out_root / 'all_conditions.json'
    save_json(all_path, {'rows': all_rows})

    df = pd.DataFrame(all_rows)
    agg = (
        df.groupby(['task', 'supervision_format', 'budget'], as_index=False)
        .agg(
            exact_match_mean=('exact_match', 'mean'),
            exact_match_std=('exact_match', 'std'),
            train_seconds_mean=('train_seconds', 'mean'),
            eval_seconds_mean=('eval_seconds', 'mean'),
            seeds=('seed', 'count'),
        )
        .fillna(0.0)
    )
    summary_csv = out_root / 'summary_by_budget.csv'
    agg.to_csv(summary_csv, index=False)
    agg_json = out_root / 'summary_by_budget.json'
    save_json(agg_json, {'rows': agg.to_dict(orient='records')})

    # markdown table
    md_path = out_root / 'comparison_table.md'
    with md_path.open('w', encoding='utf-8') as f:
        f.write('| task | format | budget | exact_match_mean | exact_match_std | seeds |\n')
        f.write('|---|---|---:|---:|---:|---:|\n')
        for _, r in agg.sort_values(['task', 'budget', 'supervision_format']).iterrows():
            f.write(
                f"| {r['task']} | {r['supervision_format']} | {int(r['budget'])} | "
                f"{r['exact_match_mean']:.4f} | {r['exact_match_std']:.4f} | {int(r['seeds'])} |\n"
            )

    plot_scaling_curves(str(summary_csv), str(out_root))


if __name__ == '__main__':
    main()
