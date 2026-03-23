#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from logging_utils import write_json


def read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def ngrams(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    toks = text.lower().split()
    return {tuple(toks[i : i + n]) for i in range(max(len(toks) - n + 1, 0))}


def audit_task(task: str, data_dir: Path) -> Dict:
    train_path = data_dir / f"{task}_train_answer_only.jsonl"
    eval_path = data_dir / f"{task}_eval.jsonl"
    train = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    train_inputs = {r["input"].strip() for r in train}
    eval_inputs = {r["input"].strip() for r in eval_rows}

    exact_overlap = sorted(train_inputs.intersection(eval_inputs))

    train_ngrams = set()
    for x in train_inputs:
        train_ngrams |= ngrams(x)

    eval_ngrams = set()
    for x in eval_inputs:
        eval_ngrams |= ngrams(x)

    ng_intersection = train_ngrams.intersection(eval_ngrams)
    ng_ratio = len(ng_intersection) / max(len(eval_ngrams), 1)

    return {
        "task": task,
        "train_examples": len(train),
        "eval_examples": len(eval_rows),
        "exact_input_overlap_count": len(exact_overlap),
        "exact_input_overlap_samples": exact_overlap[:5],
        "ngram_n": 3,
        "eval_ngram_count": len(eval_ngrams),
        "shared_ngram_count": len(ng_intersection),
        "shared_ngram_ratio": ng_ratio,
        "protocol_note": (
            "Train and eval are generated from separate ranges/distributions and stored in separate files. "
            "This audit measures exact prompt overlap and simple 3-gram overlap as a lightweight leakage check."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/contamination_audit.json"))
    args = parser.parse_args()

    report = {"tasks": [audit_task("arithmetic", args.data_dir), audit_task("symbolic", args.data_dir)]}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
