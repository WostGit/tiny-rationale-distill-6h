# Tiny Rationale Distillation (6h CPU GitHub Actions Study)

This repository is a **tiny reduced reproduction** of rationale-distillation behavior under very small data budgets. It is designed for unattended CPU-only GitHub Actions runs and compares answer-only supervision, short/compressed rationale supervision, and full-rationale supervision on two lightweight tasks. It is intentionally scoped for readability and reproducibility rather than benchmark-level performance.

## Research question
At tiny data budgets, does short/compressed rationale supervision transfer more useful signal to a fixed small student than answer-only or full-rationale supervision under contamination-aware evaluation?

## Supervision formats
- **answer_only**: target contains only the final answer.
- **short_rationale**: target contains a compact reasoning trace plus final answer.
- **full_rationale**: target contains a longer reasoning trace plus final answer.

## Tasks
1. **Arithmetic addition**
   - Train/eval are separated by operand ranges.
   - Train uses smaller numbers; eval uses larger disjoint ranges.
2. **Symbolic rule-following**
   - Deterministic transform rules (`reverse`, `sort`, `shift`) over short symbol sequences.
   - Train/eval are generated from separate token distributions/length ranges.

## Contamination controls
- Train and eval are in separate files for each task.
- `scripts/contamination_audit.py` reports:
  - exact input overlap
  - simple 3-gram overlap
  - protocol note documenting leakage minimization
- Audit outputs are machine-readable JSON at `outputs/metrics/contamination_audit.json`.

## Experimental grid
Default grid used by `scripts/run_6h_study.py`:
- tasks: `arithmetic`, `symbolic`
- formats: `answer_only`, `short_rationale`, `full_rationale`
- budgets: `16, 32, 64, 128`
- seeds: `0..4` (5 seeds)
- epochs: `3`

Total: `2 × 3 × 4 × 5 = 120` conditions.

## What this study proves and does not prove
### This can show
- Whether rationale style affects sample efficiency trends in a small controlled setup.
- Whether short rationales can outperform answer-only and/or full rationales under tiny budgets.

### This does **not** show
- Benchmark-grade absolute performance claims.
- Generalization to larger models, broader datasets, or production settings.
- Causal claims beyond this narrow synthetic setup.

## Predicted results based on prior literature
Prior work indicates chain-of-thought distillation can help small students in some settings, while newer studies suggest rationale quality/length and selection can matter. In this tiny setup, a plausible expectation is:
- short rationales > answer-only at low budgets,
- full rationales may help or hurt depending on verbosity/noise,
- differences shrink as budget increases.

These are hypotheses, not claims; evaluate using produced artifacts.

## Model/training choices
- Student model: `distilgpt2` (small causal LM, CPU-feasible).
- PEFT only: LoRA adapters (`peft`), no full fine-tuning.
- Minimal checkpoints: save adapter + compact metrics.
- Exact-match evaluation with task-specific extraction for integers/symbol sequences.

## How to run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_symbolic_data.py --out_dir data --train_n 256 --eval_n 128 --seed 13
python scripts/run_6h_study.py \
  --tasks arithmetic symbolic \
  --formats answer_only short_rationale full_rationale \
  --budgets 16 32 64 128 \
  --seeds 0 1 2 3 4 \
  --epochs 3
```

## How to run on GitHub Actions
Primary path: `.github/workflows/run-6h-study.yml`
- triggers on `push`, `pull_request`, and `workflow_dispatch`
- uses a shard matrix to spread the 120 conditions across parallel jobs
- uploads per-shard metric artifacts and a merged summary artifact

## How to inspect outputs
All run products are under `outputs/metrics/`:
- per-condition:
  - `runs/<run_id>/train_metrics.json`
  - `runs/<run_id>/eval_metrics.json`
  - `runs/<run_id>/pred_samples.json`
- global:
  - `contamination_audit.json`
  - `all_condition_metrics.json`
  - `summary_mean_std.json`
  - `summary_mean_std.csv`
  - `comparison_table.md`
  - `scaling_arithmetic.png`
  - `scaling_symbolic.png`

Each condition emits a concise completion line including task, format, budget, seed, train/eval time, and exact match.
