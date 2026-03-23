"""Contamination audit for train/eval splits with exact and n-gram overlap."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

from logging_utils import read_jsonl, write_json



def char_ngrams(text: str, n: int = 4) -> Set[str]:
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}



def audit_task(task: str, data_dir: Path) -> Dict:
    train_path = data_dir / f"{task}_train_answer_only.jsonl"
    eval_path = data_dir / f"{task}_eval.jsonl"

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    train_inputs = [r["input"] for r in train_rows]
    eval_inputs = [r["input"] for r in eval_rows]

    train_set = set(train_inputs)
    eval_set = set(eval_inputs)
    exact_overlap = sorted(train_set & eval_set)

    train_ngrams = set()
    eval_ngrams = set()
    for t in train_inputs:
        train_ngrams |= char_ngrams(t)
    for t in eval_inputs:
        eval_ngrams |= char_ngrams(t)

    inter = train_ngrams & eval_ngrams
    union = train_ngrams | eval_ngrams
    jaccard = len(inter) / max(1, len(union))

    return {
        "task": task,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "exact_input_overlap_count": len(exact_overlap),
        "exact_input_overlap_examples": exact_overlap[:5],
        "char_4gram_overlap_jaccard": round(jaccard, 6),
        "protocol_note": (
            "Train and eval are split into separate files with disjoint generation ranges. "
            "Arithmetic uses disjoint operand ranges; symbolic uses disjoint symbol alphabets and length ranges."
        ),
    }



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["arithmetic", "symbolic"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-root", default="outputs/metrics")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    audits = [audit_task(task, data_dir) for task in args.tasks]
    write_json(out_root / "contamination_audit.json", {"audits": audits})
    print("[contamination] wrote", out_root / "contamination_audit.json")


if __name__ == "__main__":
    main()
