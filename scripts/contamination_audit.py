import argparse
from collections import Counter
from pathlib import Path

from logging_utils import jsonl_read, save_json


def char_ngrams(text, n=5):
    text = text.replace(' ', '_')
    return [text[i:i+n] for i in range(max(0, len(text)-n+1))]


def audit_task(data_dir: Path, task: str):
    train_inputs = []
    for fmt in ['answer_only', 'short_rationale', 'full_rationale']:
        train_inputs.extend([r['input'] for r in jsonl_read(data_dir / f'{task}_train_{fmt}.jsonl')])
    eval_inputs = [r['input'] for r in jsonl_read(data_dir / f'{task}_eval.jsonl')]

    train_set = set(train_inputs)
    eval_set = set(eval_inputs)
    exact_overlap = sorted(train_set.intersection(eval_set))

    train_ng = Counter()
    for t in train_set:
        train_ng.update(char_ngrams(t))
    eval_ng = Counter()
    for t in eval_set:
        eval_ng.update(char_ngrams(t))

    shared = set(train_ng).intersection(eval_ng)
    denom = max(1, len(set(eval_ng)))
    return {
        'task': task,
        'num_train_inputs_unique': len(train_set),
        'num_eval_inputs_unique': len(eval_set),
        'exact_input_overlap_count': len(exact_overlap),
        'exact_overlap_examples': exact_overlap[:10],
        'char_5gram_overlap_ratio_eval_vocab': len(shared) / denom,
        'protocol_note': (
            'Train/eval are generated from disjoint regimes where possible (arithmetic range split, '
            'symbolic alphabet-pattern split) and stored in separate files. '
            'This audit measures exact input overlaps and approximate character 5-gram overlap.'
        )
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--output-root', default='outputs/metrics')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    result = {
        'arithmetic': audit_task(data_dir, 'arithmetic'),
        'symbolic': audit_task(data_dir, 'symbolic'),
    }
    save_json(Path(args.output_root) / 'contamination_audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
