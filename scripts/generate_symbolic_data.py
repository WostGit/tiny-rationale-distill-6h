"""Generate symbolic reverse-task data with 3 supervision formats."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from logging_utils import write_jsonl



def build_example(idx: int, rng: random.Random, split: str):
    if split == "train":
        symbols = list("abcde")
        min_len, max_len = 3, 5
    else:
        symbols = list("fghijk")
        min_len, max_len = 5, 7

    seq_len = rng.randint(min_len, max_len)
    seq = [rng.choice(symbols) for _ in range(seq_len)]
    inp = " ".join(seq)
    ans = " ".join(reversed(seq))

    short = f"reverse -> {ans}. answer: {ans}"
    full = (
        f"The rule is to reverse the token order. Original sequence is {inp}. "
        f"Reversed sequence becomes {ans}. Final answer: {ans}"
    )

    base = {
        "id": f"symbolic-{split}-{idx}",
        "input": f"Rule: reverse the token sequence. Input: {inp}",
        "answer": ans,
        "short_rationale": short,
        "full_rationale": full,
    }
    return base



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=220)
    parser.add_argument("--eval-size", type=int, default=140)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train = [build_example(i, rng, "train") for i in range(args.train_size)]
    eval_rows = [build_example(i, rng, "eval") for i in range(args.eval_size)]

    def to_rows(base_rows, fmt: str):
        target_key = {
            "answer_only": "answer",
            "short_rationale": "short_rationale",
            "full_rationale": "full_rationale",
        }[fmt]
        return [
            {
                "id": r["id"],
                "input": r["input"],
                "target": r[target_key],
                "answer": r["answer"],
            }
            for r in base_rows
        ]

    for fmt in ["answer_only", "short_rationale", "full_rationale"]:
        write_jsonl(out_dir / f"symbolic_train_{fmt}.jsonl", to_rows(train, fmt))

    write_jsonl(
        out_dir / "symbolic_eval.jsonl",
        [{"id": r["id"], "input": r["input"], "answer": r["answer"]} for r in eval_rows],
    )


if __name__ == "__main__":
    main()
