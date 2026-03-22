from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scaling_curves(summary_df: pd.DataFrame, out_dir: str | Path) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if summary_df.empty:
        return paths

    formats = ["answer_only", "short_rationale", "full_rationale"]
    colors = {
        "answer_only": "#1f77b4",
        "short_rationale": "#2ca02c",
        "full_rationale": "#d62728",
    }

    for task in sorted(summary_df["task"].unique()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        task_df = summary_df[summary_df["task"] == task]
        for supervision in formats:
            fmt_df = task_df[task_df["supervision"] == supervision].sort_values("budget")
            if fmt_df.empty:
                continue
            x = fmt_df["budget"].tolist()
            y = fmt_df["exact_match_mean"].tolist()
            yerr = fmt_df["exact_match_std"].tolist()
            ax.plot(x, y, marker="o", label=supervision, color=colors.get(supervision))
            ax.fill_between(x, [max(0.0, yi - ei) for yi, ei in zip(y, yerr)], [min(1.0, yi + ei) for yi, ei in zip(y, yerr)], alpha=0.15, color=colors.get(supervision))

        ax.set_title(f"Tiny Distillation Scaling Curve ({task})")
        ax.set_xlabel("Train Budget")
        ax.set_ylabel("Exact Match")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        output_path = out_dir / f"scaling_curve_{task}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        paths.append(str(output_path))
    return paths
