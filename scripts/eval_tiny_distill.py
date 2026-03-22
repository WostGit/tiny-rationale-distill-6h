#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import get_peak_rss_mb, now_utc_iso, setup_logger, stable_condition_id, write_json, write_jsonl

logger = setup_logger()

PROMPT_BY_TASK = {
    "arithmetic": "Solve arithmetic. Return final answer as an integer.",
    "symbolic": "Apply symbolic transformation. Return transformed sequence tokens separated by spaces.",
}


def extract_prediction(text: str, task: str) -> str:
    cleaned = text.strip()
    final_match = re.search(r"final answer\s*:\s*(.+)$", cleaned, flags=re.IGNORECASE)
    if final_match:
        cleaned = final_match.group(1).strip()

    if task == "arithmetic":
        num_match = re.findall(r"-?\d+", cleaned)
        return num_match[-1] if num_match else cleaned.splitlines()[0].strip()

    line = cleaned.splitlines()[0].strip()
    tokens = re.findall(r"[a-z]", line.lower())
    return " ".join(tokens)


def evaluate(args: argparse.Namespace) -> dict:
    cond = stable_condition_id(args.task, args.supervision, args.budget, args.seed)
    cond_dir = Path(args.output_root) / cond
    train_metrics_path = cond_dir / "train_metrics.json"
    eval_metrics_path = cond_dir / "eval_metrics.json"

    if eval_metrics_path.exists() and not args.overwrite:
        logger.info("Skipping eval for %s (already exists).", cond)
        return json.loads(eval_metrics_path.read_text(encoding="utf-8"))

    if not train_metrics_path.exists():
        raise FileNotFoundError(f"Missing train metrics at {train_metrics_path}")

    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base_model, cond_dir / "adapter")
    model.to(device)
    model.eval()

    rows = [json.loads(line) for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    exact = 0
    predictions = []
    t0 = time.time()

    for i, row in enumerate(rows):
        prompt = (
            f"Task: {args.task}\nInstruction: {PROMPT_BY_TASK[args.task]}\n"
            f"Input: {row['input']}\nOutput:"
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            gen = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(gen[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_prediction(decoded, args.task)
        gold = row["answer"].strip().lower()
        hit = pred.strip().lower() == gold
        exact += int(hit)
        if i < args.num_prediction_samples:
            predictions.append(
                {
                    "id": row.get("id", i),
                    "input": row["input"],
                    "gold": gold,
                    "prediction": pred,
                    "raw_generation": decoded.strip(),
                    "correct": hit,
                }
            )

    eval_time = time.time() - t0
    em = exact / len(rows)

    write_jsonl(cond_dir / "prediction_samples.jsonl", predictions)

    metrics = {
        "condition_id": cond,
        "task": args.task,
        "supervision": args.supervision,
        "budget": args.budget,
        "seed": args.seed,
        "exact_match": em,
        "n_eval": len(rows),
        "eval_time_sec": round(eval_time, 3),
        "peak_rss_mb": round(get_peak_rss_mb(), 3),
        "timestamp_utc": now_utc_iso(),
        "eval_file": args.eval_file,
        "model_name": args.model_name,
        "train_time_sec": train_metrics.get("train_time_sec"),
        "train_final_loss": train_metrics.get("final_loss"),
    }
    write_json(eval_metrics_path, metrics)
    logger.info(
        "Completed %s | EM=%.3f | train_time=%.1fs | eval_time=%.1fs",
        cond,
        em,
        train_metrics.get("train_time_sec", -1),
        eval_time,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tiny distillation adapter")
    p.add_argument("--task", choices=["arithmetic", "symbolic"], required=True)
    p.add_argument("--supervision", choices=["answer_only", "short_rationale", "full_rationale"], required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_root", default="outputs/metrics")
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--num_prediction_samples", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
