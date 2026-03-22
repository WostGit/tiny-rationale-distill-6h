"""Publication-style plotting helpers for scaling curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PALETTE = {
    "answer_only": "#1f77b4",
    "short_rationale": "#2ca02c",
    "full_rationale": "#d62728",
}



def plot_scaling_curves(summary_df: pd.DataFrame, output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        return

    for task, task_df in summary_df.groupby("task"):
        plt.figure(figsize=(7.2, 4.6))
        for fmt, fmt_df in task_df.groupby("supervision"):
            fmt_df = fmt_df.sort_values("budget")
            plt.plot(
                fmt_df["budget"],
                fmt_df["exact_match_mean"],
                marker="o",
                label=fmt,
                color=PALETTE.get(fmt),
            )
            plt.fill_between(
                fmt_df["budget"],
                fmt_df["exact_match_mean"] - fmt_df["exact_match_std"],
                fmt_df["exact_match_mean"] + fmt_df["exact_match_std"],
                alpha=0.16,
                color=PALETTE.get(fmt),
            )

        plt.title(f"Tiny Distillation Scaling Curve ({task})")
        plt.xlabel("Training budget (examples)")
        plt.ylabel("Exact match")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend()
        out_path = output_root / f"scaling_curve_{task}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
