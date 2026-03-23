#!/usr/bin/env python3
"""Generate arithmetic and symbolic datasets with three supervision formats."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def arithmetic_example(a: int, b: int) -> Tuple[str, str, str]:
    prompt = f"Compute: {a} + {b}."
    answer = str(a + b)
    short = f"{a}+{b}={answer}."
    full = (
        f"Add the ones: {a % 10}+{b % 10}={(a % 10) + (b % 10)}. "
        f"Add the tens/higher values and combine to get {answer}."
    )
    return prompt, answer, short, full


def build_arithmetic(train_n: int, eval_n: int, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    train_pairs = [(rng.randint(10, 79), rng.randint(10, 79)) for _ in range(train_n)]
    eval_pairs = [(rng.randint(120, 299), rng.randint(120, 299)) for _ in range(eval_n)]

    def make_rows(pairs: List[Tuple[int, int]], split: str) -> Dict[str, List[Dict]]:
        rows = {"answer_only": [], "short_rationale": [], "full_rationale": [], "eval": []}
        for a, b in pairs:
            prompt, answer, short, full = arithmetic_example(a, b)
            rows["answer_only"].append({"id": f"arith-{split}-{a}-{b}", "input": prompt, "target": answer})
            rows["short_rationale"].append({"id": f"arith-{split}-{a}-{b}", "input": prompt, "target": f"Reason: {short} Final: {answer}"})
            rows["full_rationale"].append({"id": f"arith-{split}-{a}-{b}", "input": prompt, "target": f"Reason: {full} Final: {answer}"})
            rows["eval"].append({"id": f"arith-{split}-{a}-{b}", "input": prompt, "answer": answer})
        return rows

    train_rows = make_rows(train_pairs, "train")
    eval_rows = make_rows(eval_pairs, "eval")
    return {
        "train_answer_only": train_rows["answer_only"],
        "train_short_rationale": train_rows["short_rationale"],
        "train_full_rationale": train_rows["full_rationale"],
        "eval": eval_rows["eval"],
    }


SYMBOLS = list("abcdef")
OPS = ["reverse", "sort", "shift"]


def apply_rule(tokens: List[str], op: str) -> List[str]:
    if op == "reverse":
        return list(reversed(tokens))
    if op == "sort":
        return sorted(tokens)
    if op == "shift":
        mapping = {"a": "b", "b": "c", "c": "d", "d": "e", "e": "f", "f": "a"}
        return [mapping[t] for t in tokens]
    raise ValueError(op)


def symbolic_example(tokens: List[str], op: str) -> Tuple[str, str, str, str]:
    inp = " ".join(tokens)
    out_tokens = apply_rule(tokens, op)
    answer = " ".join(out_tokens)
    prompt = f"Rule task: apply {op} to sequence [{inp}]"
    short = f"Apply {op} -> {answer}."
    full = f"The rule is {op}. Start from [{inp}]. After transforming each position, the result is [{answer}]."
    return prompt, answer, short, full


def build_symbolic(train_n: int, eval_n: int, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed + 7)

    def sample_seq(length: int) -> List[str]:
        return [rng.choice(SYMBOLS[:4]) for _ in range(length)]

    train_specs = [(sample_seq(rng.randint(4, 6)), rng.choice(OPS)) for _ in range(train_n)]
    eval_specs = [([rng.choice(SYMBOLS[2:]) for _ in range(rng.randint(5, 7))], rng.choice(OPS)) for _ in range(eval_n)]

    def make_rows(specs: List[Tuple[List[str], str]], split: str) -> Dict[str, List[Dict]]:
        rows = {"answer_only": [], "short_rationale": [], "full_rationale": [], "eval": []}
        for i, (tokens, op) in enumerate(specs):
            prompt, answer, short, full = symbolic_example(tokens, op)
            uid = f"sym-{split}-{i}"
            rows["answer_only"].append({"id": uid, "input": prompt, "target": answer})
            rows["short_rationale"].append({"id": uid, "input": prompt, "target": f"Reason: {short} Final: {answer}"})
            rows["full_rationale"].append({"id": uid, "input": prompt, "target": f"Reason: {full} Final: {answer}"})
            rows["eval"].append({"id": uid, "input": prompt, "answer": answer})
        return rows

    train_rows = make_rows(train_specs, "train")
    eval_rows = make_rows(eval_specs, "eval")
    return {
        "train_answer_only": train_rows["answer_only"],
        "train_short_rationale": train_rows["short_rationale"],
        "train_full_rationale": train_rows["full_rationale"],
        "eval": eval_rows["eval"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--train_n", type=int, default=256)
    parser.add_argument("--eval_n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    ar = build_arithmetic(args.train_n, args.eval_n, args.seed)
    sy = build_symbolic(args.train_n, args.eval_n, args.seed)

    write_jsonl(args.out_dir / "arithmetic_train_answer_only.jsonl", ar["train_answer_only"])
    write_jsonl(args.out_dir / "arithmetic_train_short_rationale.jsonl", ar["train_short_rationale"])
    write_jsonl(args.out_dir / "arithmetic_train_full_rationale.jsonl", ar["train_full_rationale"])
    write_jsonl(args.out_dir / "arithmetic_eval.jsonl", ar["eval"])

    write_jsonl(args.out_dir / "symbolic_train_answer_only.jsonl", sy["train_answer_only"])
    write_jsonl(args.out_dir / "symbolic_train_short_rationale.jsonl", sy["train_short_rationale"])
    write_jsonl(args.out_dir / "symbolic_train_full_rationale.jsonl", sy["train_full_rationale"])
    write_jsonl(args.out_dir / "symbolic_eval.jsonl", sy["eval"])

    print("Wrote dataset files to", args.out_dir)


if __name__ == "__main__":
    main()
