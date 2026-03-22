import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import condition_id, get_memory_rss_mb, jsonl_read, save_json
from metrics_utils import exact_match, extract_final_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['arithmetic', 'symbolic'])
    ap.add_argument('--supervision-format', required=True, choices=['answer_only', 'short_rationale', 'full_rationale'])
    ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--model-name', default='roneneldan/TinyStories-33M')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--output-root', default='outputs/metrics')
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--num-eval', type=int, default=160)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    cond = condition_id(args.task, args.supervision_format, args.budget, args.seed)
    cond_dir = Path(args.output_root) / cond
    out_path = cond_dir / 'eval_metrics.json'
    pred_path = cond_dir / 'prediction_samples.json'

    if args.resume and out_path.exists():
        print(f'[skip] Found existing eval outputs for {cond}')
        return

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cond_dir / 'tokenizer')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base, cond_dir / 'adapter')
    model.eval()

    eval_rows = list(jsonl_read(Path(args.data_dir) / f"{args.task}_eval.jsonl"))[: args.num_eval]

    scores = []
    samples = []
    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            prompt = f"Input: {row['input']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt')
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            out_text = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_final_answer(out_text, args.task)
            gold = extract_final_answer(row['target'], args.task)
            em = exact_match(pred, gold)
            scores.append(em)
            if i < 20:
                samples.append({'input': row['input'], 'gold': gold, 'raw_generation': out_text, 'pred': pred, 'exact_match': em})

    eval_seconds = time.time() - start
    metrics = {
        'condition_id': cond,
        'task': args.task,
        'supervision_format': args.supervision_format,
        'budget': args.budget,
        'seed': args.seed,
        'num_eval': len(eval_rows),
        'exact_match': sum(scores) / len(scores) if scores else 0.0,
        'eval_seconds': eval_seconds,
        'max_rss_mb': get_memory_rss_mb(),
    }

    save_json(out_path, metrics)
    save_json(pred_path, {'condition_id': cond, 'samples': samples})
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
