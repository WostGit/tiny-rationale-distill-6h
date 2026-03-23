# Tiny Rationale Distillation (6h, CPU, GitHub Actions-first)

This repository is a **tiny reduced reproduction** of rationale-distillation behavior, designed to run unattended for ~6 hours on standard GitHub-hosted `ubuntu-latest` runners with CPU only. It compares answer-only supervision against short/compressed rationale supervision and full-rationale supervision using a fixed tiny student with PEFT/LoRA adapters.

## Research question
At tiny data budgets, does short/compressed rationale supervision transfer more useful signal to a fixed small student than answer-only or full-rationale supervision under contamination-aware evaluation?

## Supervision formats
- `answer_only`: target is only the final answer.
- `short_rationale`: target includes compressed reasoning + final answer.
- `full_rationale`: target includes a longer explanation + final answer.

## Tasks
1. **Arithmetic**
   - Operator mix: `+`, `-`, `*`.
   - Train/eval split has disjoint operand ranges.
2. **Symbolic rule-following**
   - Deterministic rule: reverse token sequence.
   - Train/eval split has disjoint symbol alphabets and length bands.
   - Generator script is checked in at `scripts/generate_symbolic_data.py`.

## Contamination controls
- Train and eval are stored in separate files.
- Arithmetic uses disjoint operand ranges between train and eval.
- Symbolic uses disjoint token alphabets and sequence length ranges.
- `scripts/contamination_audit.py` reports:
  - exact input overlap
  - simple character 4-gram overlap
  - protocol note
- Audit output is JSON at `outputs/metrics/contamination_audit.json`.

## Experimental grid
- Tasks: `arithmetic`, `symbolic`
- Formats: `answer_only`, `short_rationale`, `full_rationale`
- Budgets: `16`, `32`, `64`, `128`
- Seeds: `0,1,2,3,4` (5 seeds)
- Epochs: `3` default

Total: `2 × 3 × 4 × 5 = 120` conditions.

## What this study proves and does not prove
### What it can show
- Relative trends between supervision styles in a tightly controlled tiny setting.
- Whether short rationales appear more sample-efficient than full rationales at very low budgets.

### What it cannot show
- It is **not** a benchmark-grade reproduction of large-scale chain-of-thought distillation literature.
- Results are model/task/setup-specific and should not be overgeneralized.

## Predicted results based on prior literature
Prior work suggests small models can benefit from rationale-style supervision, while newer evidence suggests rationale quality/length selection matters. In this tiny setup, a common prediction is:
- `short_rationale` often improves over `answer_only` at small budgets,
- `full_rationale` can help or hurt depending on verbosity/noise,
- gap size depends strongly on task and seed variance.

## How to run locally
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

# (optional) regenerate symbolic data deterministically
python scripts/generate_symbolic_data.py --out-dir data --train-size 220 --eval-size 140 --seed 13

# contamination audit
python scripts/contamination_audit.py --tasks arithmetic symbolic --output-root outputs/metrics

# full 120-condition run
python scripts/run_6h_study.py \
  --tasks arithmetic,symbolic \
  --formats answer_only,short_rationale,full_rationale \
  --budgets 16,32,64,128 \
  --seeds 0,1,2,3,4 \
  --epochs 3 \
  --output-root outputs/metrics
```

## How to run on GitHub Actions
Primary path: `.github/workflows/run-6h-study.yml`.

Workflow behavior:
- triggers on `push`, `pull_request`, `workflow_dispatch`
- uses matrix sharding over task and seed subsets
- runs contamination audit + study shard
- uploads `outputs/metrics` artifacts for each shard

## How to inspect outputs
All machine-readable outputs land under `outputs/metrics/`:
- `train_*.json`: per-condition train metrics
- `eval_*.json`: per-condition eval metrics (exact match + prediction samples)
- `aggregate_summary.json` and `aggregate_summary.csv`: mean/std over seeds
- `comparison_table.md`: markdown comparison table
- `overall_report.json`: compact top-line summary
- `scaling_curve_arithmetic.png`, `scaling_curve_symbolic.png`: publication-style curves
- `contamination_audit.json`: contamination checks

## Notes on implementation choices
- Fixed student: `roneneldan/TinyStories-33M`
- PEFT/LoRA only (no full-model fine-tuning)
- CPU-only design
- Minimal checkpointing: adapters + compact JSON metrics
- Deterministic seeds and resumable run orchestration
