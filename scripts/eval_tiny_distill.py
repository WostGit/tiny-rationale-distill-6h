"""Evaluate one tiny LoRA-distilled condition with robust exact-match extraction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import get_runtime_stats, now_s, read_jsonl, set_seed, write_json
from train_tiny_distill import MODEL_NAME



def extract_answer(task: str, text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    if task == "arithmetic":
        matches = re.findall(r"-?\d+", text)
        return matches[-1] if matches else ""

    text = text.lower().strip()
    m = re.search(r"[a-z](?:\s+[a-z])*", text)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else ""



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["arithmetic", "symbolic"])
    p.add_argument("--supervision", required=True, choices=["answer_only", "short_rationale", "full_rationale"])
    p.add_argument("--budget", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--output-root", default="outputs/metrics")
    p.add_argument("--sample-save-limit", type=int, default=25)
    return p.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_root = Path(args.output_root)
    run_name = f"{args.task}__{args.supervision}__b{args.budget}__s{args.seed}"
    eval_path = out_root / f"eval_{run_name}.json"
    if eval_path.exists():
        print(f"[eval] skip existing run: {run_name}")
        return

    adapter_dir = out_root / "adapters" / run_name
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter does not exist: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    eval_rows = read_jsonl(Path("data") / f"{args.task}_eval.jsonl")

    correct = 0
    preds: List[Dict] = []
    t0 = now_s()

    for i, row in enumerate(eval_rows):
        prompt = f"### Input:\n{row['input']}\n### Response:\n"
        x = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                **x,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        out = tokenizer.decode(gen[0][x["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_answer(args.task, out)
        gold = extract_answer(args.task, row["answer"])
        ok = int(pred == gold)
        correct += ok
        if i < args.sample_save_limit:
            preds.append(
                {
                    "id": row["id"],
                    "input": row["input"],
                    "gold": gold,
                    "pred": pred,
                    "raw_generation": out,
                    "correct": ok,
                }
            )

    eval_time_s = now_s() - t0
    exact_match = correct / max(1, len(eval_rows))

    payload = {
        "run_name": run_name,
        "task": args.task,
        "supervision": args.supervision,
        "budget": args.budget,
        "seed": args.seed,
        "num_eval": len(eval_rows),
        "num_correct": correct,
        "exact_match": round(exact_match, 6),
        "eval_time_s": round(eval_time_s, 3),
        **get_runtime_stats(),
        "prediction_samples": preds,
    }
    write_json(eval_path, payload)
    print(
        f"[eval] done {run_name}: em={payload['exact_match']:.4f}, "
        f"eval_time_s={payload['eval_time_s']:.2f}"
    )


if __name__ == "__main__":
    main()
