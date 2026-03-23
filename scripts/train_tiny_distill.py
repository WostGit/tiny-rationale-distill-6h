#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from logging_utils import write_json


class SFTDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int = 128):
        self.examples = []
        for r in rows:
            text = f"Input: {r['input']}\nOutput: {r['target']}"
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = item["input_ids"].clone()
            self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"{args.task}__{args.format}__b{args.budget}__s{args.seed}"
    run_dir = args.output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = run_dir / "train_metrics.json"
    adapter_dir = run_dir / "adapter"

    if train_metrics_path.exists() and adapter_dir.exists() and not args.force:
        print(f"[skip] already exists: {run_id}")
        return

    rows = read_jsonl(args.data_dir / f"{args.task}_train_{args.format}.jsonl")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    train_rows = rows[: args.budget]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    ds = SFTDataset(train_rows, tokenizer, max_length=args.max_length)
    targs = TrainingArguments(
        output_dir=str(run_dir / "tmp"),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds)

    start = time.time()
    result = trainer.train()
    train_s = time.time() - start

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(run_dir / "tokenizer")

    metrics = {
        "run_id": run_id,
        "task": args.task,
        "format": args.format,
        "budget": args.budget,
        "seed": args.seed,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_examples": len(train_rows),
        "train_runtime_sec": train_s,
        "train_loss": float(result.training_loss),
        "train_steps": int(result.global_step),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    write_json(train_metrics_path, metrics)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
