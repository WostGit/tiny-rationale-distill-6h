import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_ts() -> float:
    return time.time()


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def jsonl_read(path: str | Path):
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def jsonl_write(path: str | Path, rows) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def condition_id(task: str, supervision: str, budget: int, seed: int) -> str:
    return f"{task}__{supervision}__b{budget}__s{seed}"


def get_memory_rss_mb() -> float:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0
    except Exception:
        return -1.0


def print_condition_summary(metrics: Dict[str, Any]) -> None:
    print(
        "[condition-complete] "
        f"task={metrics.get('task')} "
        f"format={metrics.get('supervision_format')} "
        f"budget={metrics.get('budget')} "
        f"seed={metrics.get('seed')} "
        f"train_s={metrics.get('train_seconds')} "
        f"eval_s={metrics.get('eval_seconds')} "
        f"exact_match={metrics.get('exact_match')}"
    )
