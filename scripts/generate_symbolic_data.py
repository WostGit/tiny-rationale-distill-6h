#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from logging_utils import write_jsonl

VOCAB = list("abcdefghi")


def transform(tokens: list[str]) -> list[str]:
    idx = {c: i for i, c in enumerate(VOCAB)}
    mapped = [VOCAB[(idx[t] + 1) % len(VOCAB)] for t in tokens]
    return list(reversed(mapped))


def short_rationale(tokens: list[str], out: list[str]) -> str:
    return f"Shift each symbol by +1 then reverse. Sequence becomes {' '.join(out)}."


def full_rationale(tokens: list[str], out: list[str]) -> str:
    steps = []
    for t in tokens:
        nxt = VOCAB[(VOCAB.index(t) + 1) % len(VOCAB)]
        steps.append(f"{t}->{nxt}")
    mapped = [s.split("->")[1] for s in steps]
    return (
        f"Map each token with +1 over {''.join(VOCAB)} ({', '.join(steps)}). "
        f"Mapped sequence is {' '.join(mapped)}. Reverse it to get {' '.join(out)}."
    )


def build_record(i: int, tokens: list[str], supervision: str) -> dict:
    out_tokens = transform(tokens)
    inp = " ".join(tokens)
    ans = " ".join(out_tokens)
    if supervision == "answer_only":
        target = ans
    elif supervision == "short_rationale":
        target = f"{short_rationale(tokens, out_tokens)} Final answer: {ans}"
    else:
        target = f"{full_rationale(tokens, out_tokens)} Final answer: {ans}"
    return {
        "id": i,
        "task": "symbolic",
        "input": inp,
        "answer": ans,
        "target": target,
        "supervision": supervision,
    }


def make_examples(n: int, seq_len_min: int, seq_len_max: int, seed: int, supervision: str) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        k = rnd.randint(seq_len_min, seq_len_max)
        tokens = [rnd.choice(VOCAB) for _ in range(k)]
        rows.append(build_record(i, tokens, supervision))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for supervision in ["answer_only", "short_rationale", "full_rationale"]:
        train_rows = make_examples(args.train_size, seq_len_min=3, seq_len_max=6, seed=args.seed + 11, supervision=supervision)
        write_jsonl(out_dir / f"symbolic_train_{supervision}.jsonl", train_rows)

    eval_rows = make_examples(args.eval_size, seq_len_min=7, seq_len_max=9, seed=args.seed + 911, supervision="answer_only")
    for r in eval_rows:
        r["target"] = r["answer"]
    write_jsonl(out_dir / "symbolic_eval.jsonl", eval_rows)


if __name__ == "__main__":
    main()
