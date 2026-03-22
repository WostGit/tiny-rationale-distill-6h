"""Metric aggregation helpers for the tiny distillation study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd



def load_eval_metrics(root: str | Path) -> List[Dict[str, Any]]:
    root = Path(root)
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("eval_*.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows



def summarize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    group_cols = ["task", "supervision", "budget"]
    agg = (
        df.groupby(group_cols)["exact_match"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "exact_match_mean", "std": "exact_match_std"})
    )
    agg["exact_match_std"] = agg["exact_match_std"].fillna(0.0)
    return agg.sort_values(group_cols).reset_index(drop=True)



def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results found."
    table = df.copy()
    table["exact_match_mean"] = table["exact_match_mean"].map(lambda x: f"{x:.4f}")
    table["exact_match_std"] = table["exact_match_std"].map(lambda x: f"{x:.4f}")
    return table.to_markdown(index=False)



def save_summary_files(df: pd.DataFrame, output_root: str | Path) -> Dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "aggregate_summary.csv"
    json_path = output_root / "aggregate_summary.json"
    md_path = output_root / "comparison_table.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    md_path.write_text(to_markdown_table(df), encoding="utf-8")

    best_rows = []
    if not df.empty:
        for task, group in df.groupby("task"):
            idx = group["exact_match_mean"].idxmax()
            best_rows.append(df.loc[idx].to_dict())

    overall_path = output_root / "overall_report.json"
    payload: Dict[str, Any] = {
        "num_aggregated_rows": int(len(df)),
        "best_per_task": best_rows,
        "budgets": sorted(df["budget"].unique().tolist()) if not df.empty else [],
        "formats": sorted(df["supervision"].unique().tolist()) if not df.empty else [],
    }
    overall_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "overall": str(overall_path),
    }
