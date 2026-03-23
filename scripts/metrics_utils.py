from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List


INT_RE = re.compile(r"-?\d+")
SYM_RE = re.compile(r"[a-f](?:\s+[a-f])+")


def normalize_text(x: str) -> str:
    return " ".join(x.strip().lower().split())


def extract_prediction(task: str, generated: str) -> str:
    text = normalize_text(generated)
    if task == "arithmetic":
        ints = INT_RE.findall(text)
        return ints[-1] if ints else ""
    matches = SYM_RE.findall(text)
    return matches[-1] if matches else text


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def summarize_seed_metrics(rows: Iterable[Dict]) -> List[Dict]:
    bucket = defaultdict(list)
    for r in rows:
        key = (r["task"], r["format"], int(r["budget"]))
        bucket[key].append(float(r["exact_match"]))
    out = []
    for (task, fmt, budget), vals in sorted(bucket.items()):
        out.append(
            {
                "task": task,
                "format": fmt,
                "budget": budget,
                "mean_exact_match": mean(vals),
                "std_exact_match": pstdev(vals) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }
        )
    return out
