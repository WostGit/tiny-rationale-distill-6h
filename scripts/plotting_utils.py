from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scaling_curves(summary_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)
    for task in sorted(df["task"].unique()):
        tdf = df[df["task"] == task]
        plt.figure(figsize=(7, 4.5))
        for fmt in ["answer_only", "short_rationale", "full_rationale"]:
            fdf = tdf[tdf["format"] == fmt].sort_values("budget")
            if fdf.empty:
                continue
            plt.plot(fdf["budget"], fdf["mean_exact_match"], marker="o", label=fmt)
            plt.fill_between(
                fdf["budget"],
                fdf["mean_exact_match"] - fdf["std_exact_match"],
                fdf["mean_exact_match"] + fdf["std_exact_match"],
                alpha=0.15,
            )
        plt.title(f"{task}: exact-match vs budget")
        plt.xlabel("budget")
        plt.ylabel("exact match")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"scaling_{task}.png", dpi=140)
        plt.close()
