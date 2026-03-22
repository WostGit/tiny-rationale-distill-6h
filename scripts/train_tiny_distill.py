"""Train a tiny student with LoRA adapters for one condition."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import get_runtime_stats, now_s, read_jsonl, set_seed, write_json


MODEL_NAME = "roneneldan/TinyStories-33M"


class DistillDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        prompt = f"### Input:\n{row['input']}\n### Response:\n"
        target = row["target"].strip()

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad = max_len - x["input_ids"].size(0)
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), self.pad_token_id)]))
            attention_mask.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100)]))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["arithmetic", "symbolic"])
    p.add_argument("--supervision", required=True, choices=["answer_only", "short_rationale", "full_rationale"])
    p.add_argument("--budget", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-root", default="outputs/metrics")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.task}__{args.supervision}__b{args.budget}__s{args.seed}"
    train_metrics_path = out_root / f"train_{run_name}.json"
    adapter_dir = out_root / "adapters" / run_name
    if train_metrics_path.exists() and adapter_dir.exists():
        print(f"[train] skip existing run: {run_name}")
        return

    data_path = Path("data") / f"{args.task}_train_{args.supervision}.jsonl"
    rows = read_jsonl(data_path)[: args.budget]
    if len(rows) < args.budget:
        raise ValueError(f"Requested budget={args.budget} but dataset has only {len(rows)} rows")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()

    ds = DistillDataset(rows, tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(tokenizer.pad_token_id))

    optim = AdamW(model.parameters(), lr=args.learning_rate)

    start = now_s()
    losses: List[float] = []
    total_steps = 0
    for epoch in range(args.epochs):
        epoch_losses = []
        bar = tqdm(dl, desc=f"train {run_name} epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in bar:
            optim.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
            losses.append(loss.item())
            total_steps += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"[train] {run_name} epoch={epoch+1} avg_loss={sum(epoch_losses)/max(1,len(epoch_losses)):.4f}")

    train_time_s = now_s() - start
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    stats = {
        "run_name": run_name,
        "task": args.task,
        "supervision": args.supervision,
        "budget": args.budget,
        "seed": args.seed,
        "model_name": MODEL_NAME,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "total_steps": total_steps,
        "final_loss": losses[-1] if losses else math.nan,
        "mean_loss": float(sum(losses) / max(1, len(losses))),
        "train_time_s": round(train_time_s, 3),
        **get_runtime_stats(),
        "adapter_dir": str(adapter_dir),
    }
    write_json(train_metrics_path, stats)
    print(f"[train] done {run_name}: {stats}")


if __name__ == "__main__":
    main()
