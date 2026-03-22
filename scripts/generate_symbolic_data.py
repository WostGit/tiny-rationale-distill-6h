"""Generate symbolic rule-following data with three supervision styles."""
import argparse
import json
import random
from pathlib import Path

SYMS = list('ABCDEFGH')


def transform(seq):
    # deterministic composition: reverse then rotate letters by +1
    rev = list(reversed(seq))
    rotated = [SYMS[(SYMS.index(ch) + 1) % len(SYMS)] for ch in rev]
    return rotated


def short_rationale(inp, out):
    return f"Reverse tokens then shift each symbol +1 cyclically. Final answer: {' '.join(out)}"


def full_rationale(inp, out):
    rev = list(reversed(inp))
    step2 = [f"{x}->{SYMS[(SYMS.index(x)+1)%len(SYMS)]}" for x in rev]
    return (
        f"Input sequence is {' '.join(inp)}. "
        f"Step 1 reverse -> {' '.join(rev)}. "
        f"Step 2 shift each symbol by +1 cyclically: {', '.join(step2)}. "
        f"Final answer: {' '.join(out)}"
    )


def make_example(seed, idx, train=True):
    rng = random.Random(seed + idx)
    length = rng.randint(4, 7)
    # keep train/eval separated by symbol subset balance pattern
    allowed = SYMS[:6] if train else SYMS[2:]
    inp = [rng.choice(allowed) for _ in range(length)]
    out = transform(inp)
    return {
        'id': idx,
        'input': ' '.join(inp),
        'target': ' '.join(out),
        'answer_only': ' '.join(out),
        'short_rationale': short_rationale(inp, out),
        'full_rationale': full_rationale(inp, out),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--train-size', type=int, default=256)
    ap.add_argument('--eval-size', type=int, default=160)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    train = [make_example(args.seed, i, train=True) for i in range(args.train_size)]
    train_inputs = {r['input'] for r in train}
    eval_rows = []
    i = 0
    while len(eval_rows) < args.eval_size:
        row = make_example(args.seed + 999, i, train=False)
        if row['input'] not in train_inputs:
            eval_rows.append(row)
        i += 1

    write_jsonl(out_dir / 'symbolic_train_answer_only.jsonl', [
        {'id': r['id'], 'input': r['input'], 'target': r['target'], 'supervision': r['answer_only']}
        for r in train
    ])
    write_jsonl(out_dir / 'symbolic_train_short_rationale.jsonl', [
        {'id': r['id'], 'input': r['input'], 'target': r['target'], 'supervision': r['short_rationale']}
        for r in train
    ])
    write_jsonl(out_dir / 'symbolic_train_full_rationale.jsonl', [
        {'id': r['id'], 'input': r['input'], 'target': r['target'], 'supervision': r['full_rationale']}
        for r in train
    ])
    write_jsonl(out_dir / 'symbolic_eval.jsonl', [
        {'id': r['id'], 'input': r['input'], 'target': r['target']} for r in eval_rows
    ])


if __name__ == '__main__':
    main()
