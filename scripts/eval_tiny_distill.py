#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import write_json
from metrics_utils import exact_match, extract_prediction


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["arithmetic", "symbolic"])
    parser.add_argument("--format", required=True, choices=["answer_only", "short_rationale", "full_rationale"])
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--eval_max_examples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_id = f"{args.task}__{args.format}__b{args.budget}__s{args.seed}"
    run_dir = args.output_root / "runs" / run_id
    eval_path = run_dir / "eval_metrics.json"
    pred_path = run_dir / "pred_samples.json"

    if eval_path.exists() and pred_path.exists() and not args.force:
        print(f"[skip] eval exists: {run_id}")
        return

    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    base = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base, run_dir / "adapter")
    model.eval()

    rows = read_jsonl(args.data_dir / f"{args.task}_eval.jsonl")[: args.eval_max_examples]
    preds = []
    t0 = time.time()

    for r in rows:
        prompt = f"Input: {r['input']}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_prediction(args.task, decoded.split("Output:")[-1])
        em = exact_match(pred, r["answer"])
        preds.append({"id": r["id"], "input": r["input"], "gold": r["answer"], "pred": pred, "exact_match": em})

    eval_s = time.time() - t0
    accuracy = sum(p["exact_match"] for p in preds) / max(len(preds), 1)
    metrics = {
        "run_id": run_id,
        "task": args.task,
        "format": args.format,
        "budget": args.budget,
        "seed": args.seed,
        "eval_examples": len(preds),
        "exact_match": accuracy,
        "eval_runtime_sec": eval_s,
    }

    write_json(eval_path, metrics)
    write_json(pred_path, {"run_id": run_id, "samples": preds[:20]})
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
