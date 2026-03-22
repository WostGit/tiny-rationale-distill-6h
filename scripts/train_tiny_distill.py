import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import condition_id, ensure_dir, get_memory_rss_mb, jsonl_read, save_json


class SupervisedDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=192):
        self.features = []
        for r in rows:
            prompt = f"Input: {r['input']}\nAnswer:" 
            completion = f" {r['supervision']}"
            full = prompt + completion
            tok_full = tokenizer(full, truncation=True, max_length=max_length, return_tensors='pt')
            tok_prompt = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors='pt')
            input_ids = tok_full.input_ids[0]
            attn = tok_full.attention_mask[0]
            labels = input_ids.clone()
            prompt_len = tok_prompt.input_ids.shape[1]
            labels[:prompt_len] = -100
            self.features.append({'input_ids': input_ids, 'attention_mask': attn, 'labels': labels})

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate(batch, pad_id):
    max_len = max(x['input_ids'].shape[0] for x in batch)
    def pad(v, val):
        return torch.cat([v, torch.full((max_len - v.shape[0],), val, dtype=v.dtype)])
    return {
        'input_ids': torch.stack([pad(x['input_ids'], pad_id) for x in batch]),
        'attention_mask': torch.stack([pad(x['attention_mask'], 0) for x in batch]),
        'labels': torch.stack([pad(x['labels'], -100) for x in batch]),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['arithmetic', 'symbolic'])
    ap.add_argument('--supervision-format', required=True, choices=['answer_only', 'short_rationale', 'full_rationale'])
    ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--model-name', default='roneneldan/TinyStories-33M')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--output-root', default='outputs/metrics')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    cond = condition_id(args.task, args.supervision_format, args.budget, args.seed)
    cond_dir = Path(args.output_root) / cond
    train_metrics_path = cond_dir / 'train_metrics.json'
    adapter_dir = cond_dir / 'adapter'

    if args.resume and train_metrics_path.exists() and adapter_dir.exists():
        print(f'[skip] Found existing training outputs for {cond}')
        return

    set_seed(args.seed)
    start = time.time()

    train_path = Path(args.data_dir) / f"{args.task}_train_{args.supervision_format}.jsonl"
    rows = list(jsonl_read(train_path))[: args.budget]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SupervisedDataset(rows, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['c_attn', 'c_proj', 'c_fc'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_trace = []
    steps = 0
    for _ in range(args.epochs):
        for batch in loader:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
            loss_trace.append(float(loss.item()))

    ensure_dir(adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(cond_dir / 'tokenizer')

    train_seconds = time.time() - start
    metrics = {
        'condition_id': cond,
        'task': args.task,
        'supervision_format': args.supervision_format,
        'budget': args.budget,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_examples': len(rows),
        'num_steps': steps,
        'train_loss_final': loss_trace[-1] if loss_trace else None,
        'train_loss_mean': float(np.mean(loss_trace)) if loss_trace else None,
        'train_seconds': train_seconds,
        'max_rss_mb': get_memory_rss_mb(),
        'model_name': args.model_name,
        'lora_config': {'r': 8, 'alpha': 16, 'dropout': 0.05, 'targets': ['c_attn', 'c_proj', 'c_fc']},
    }
    save_json(train_metrics_path, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
