"""Utility helpers for deterministic logging, IO, and light system stats."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import numpy as np
except Exception:
    np = None

try:
    import psutil
except Exception:
    psutil = None



def set_seed(seed: int) -> None:
    """Set python/numpy/torch seeds (torch is optional import)."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass



def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)



def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")



def now_s() -> float:
    return time.perf_counter()



def get_runtime_stats() -> Dict[str, Any]:
    if psutil is None:
        return {"rss_memory_mb": None, "cpu_percent_snapshot": None}
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    return {
        "rss_memory_mb": round(mem_mb, 2),
        "cpu_percent_snapshot": psutil.cpu_percent(interval=0.1),
    }
