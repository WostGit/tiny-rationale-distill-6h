import re
from typing import Dict, List


def normalize_answer(text: str) -> str:
    return text.strip()


def extract_final_answer(text: str, task: str) -> str:
    text = text.strip()
    if 'Final answer:' in text:
        text = text.split('Final answer:')[-1].strip()
    if task == 'arithmetic':
        m = re.search(r'-?\d+', text)
        return m.group(0) if m else text.split()[0] if text.split() else ''
    # symbolic
    line = text.splitlines()[0] if text else ''
    return re.sub(r'[^A-Z ]', '', line).strip()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def summarize_scores(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {'mean': 0.0, 'std': 0.0, 'n': 0}
    n = len(scores)
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / n
    return {'mean': mean, 'std': var ** 0.5, 'n': n}
