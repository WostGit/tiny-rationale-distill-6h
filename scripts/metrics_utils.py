from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_eval_metrics(metrics_root: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(metrics_root).glob("*/eval_metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["task", "supervision", "budget"], as_index=False)
        .agg(
            exact_match_mean=("exact_match", "mean"),
            exact_match_std=("exact_match", "std"),
            eval_time_sec_mean=("eval_time_sec", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["task", "supervision", "budget"])
    )
    grouped["exact_match_std"] = grouped["exact_match_std"].fillna(0.0)
    return grouped


def to_markdown_rows(summary_df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, r in summary_df.iterrows():
        rows.append(
            [
                r["task"],
                r["supervision"],
                int(r["budget"]),
                f"{r['exact_match_mean']:.3f}",
                f"{r['exact_match_std']:.3f}",
                int(r["n_seeds"]),
            ]
        )
    return rows


def save_records_json(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2)
