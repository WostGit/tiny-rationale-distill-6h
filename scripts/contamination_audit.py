#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_utils import write_json


def read_inputs(path: str) -> list[str]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["input"] for r in rows]


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = s.lower()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def audit(train_file: str, eval_file: str, out_file: str) -> dict:
    train_inputs = read_inputs(train_file)
    eval_inputs = read_inputs(eval_file)

    train_set = set(train_inputs)
    eval_set = set(eval_inputs)
    exact_overlap = sorted(train_set & eval_set)

    train_grams = set().union(*(char_ngrams(x) for x in train_inputs))
    eval_grams = set().union(*(char_ngrams(x) for x in eval_inputs))
    inter = train_grams & eval_grams
    union = train_grams | eval_grams
    jacc = (len(inter) / len(union)) if union else 0.0

    report = {
        "train_file": train_file,
        "eval_file": eval_file,
        "n_train": len(train_inputs),
        "n_eval": len(eval_inputs),
        "exact_input_overlap_count": len(exact_overlap),
        "exact_input_overlap_examples": exact_overlap[:20],
        "char_3gram_jaccard": jacc,
        "protocol_note": (
            "Train/eval files are generated with disjoint generation regimes: arithmetic uses disjoint operand ranges; "
            "symbolic uses disjoint sequence-length ranges and separate RNG seeds. Audit includes exact input overlap and "
            "coarse 3-gram overlap as a leakage-minimization check."
        ),
    }
    write_json(out_file, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Contamination audit")
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--out_file", required=True)
    args = p.parse_args()
    audit(args.train_file, args.eval_file, args.out_file)


if __name__ == "__main__":
    main()
