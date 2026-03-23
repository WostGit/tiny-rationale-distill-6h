# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project intent
This repository is a tiny reduced reproduction of rationale-distillation behavior under strict CPU and runtime constraints.

## Coding style
- Prefer straightforward scripts over deep abstractions.
- Keep JSON outputs machine-readable and stable.
- Preserve deterministic seeding for experiments.

## Reproducibility
- Do not silently change default model, budgets, or seeds without documenting in `README.md`.
- Keep output artifacts under `outputs/metrics/`.
