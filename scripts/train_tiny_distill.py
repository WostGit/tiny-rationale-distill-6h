#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from logging_utils import get_peak_rss_mb, now_utc_iso, setup_logger, stable_condition_id, write_json

logger = setup_logger()


PROMPT_BY_TASK = {
    "arithmetic": "Solve arithmetic. Return final answer as an integer.",
    "symbolic": "Apply symbolic transformation. Return transformed sequence tokens separated by spaces.",
}


class DistillDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int, task: str):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        prompt = (
            f"Task: {self.task}\nInstruction: {PROMPT_BY_TASK[self.task]}\n"
            f"Input: {row['input']}\nOutput:"
        )
        full = prompt + " " + row["target"]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        input_ids = full_ids
        labels = full_ids.copy()
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    inps, labs, mask = [], [], []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        inps.append(row["input_ids"] + [pad_id] * pad)
        labs.append(row["labels"] + [-100] * pad)
        mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(inps, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    cond = stable_condition_id(args.task, args.supervision, args.budget, args.seed)
    out_dir = Path(args.output_root) / cond
    metrics_path = out_dir / "train_metrics.json"
    adapter_dir = out_dir / "adapter"

    if metrics_path.exists() and adapter_dir.exists() and not args.overwrite:
        logger.info("Skipping training for %s (already exists).", cond)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    dataset = DistillDataset(args.train_file, tokenizer, args.max_length, args.task)
    if len(dataset) < args.budget:
        raise ValueError(f"Budget {args.budget} exceeds dataset size {len(dataset)}")
    subset = torch.utils.data.Subset(dataset, range(args.budget))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = int(0.1 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    loss_history = []
    start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach().cpu().item()))
        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_history.append(mean_epoch_loss)
        logger.info("%s epoch %s/%s loss=%.4f", cond, epoch + 1, args.epochs, mean_epoch_loss)

    train_time = time.time() - start
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")

    metrics = {
        "condition_id": cond,
        "task": args.task,
        "supervision": args.supervision,
        "budget": args.budget,
        "seed": args.seed,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "train_time_sec": round(train_time, 3),
        "peak_rss_mb": round(get_peak_rss_mb(), 3),
        "timestamp_utc": now_utc_iso(),
        "train_file": args.train_file,
    }
    write_json(metrics_path, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tiny rationale distillation adapter")
    p.add_argument("--task", choices=["arithmetic", "symbolic"], required=True)
    p.add_argument("--supervision", choices=["answer_only", "short_rationale", "full_rationale"], required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_root", default="outputs/metrics")
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
