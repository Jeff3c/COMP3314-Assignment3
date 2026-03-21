# Contributing Guide

## Branching
- Create a feature branch for each task:
  - feature/<short-topic>
  - fix/<short-topic>

## Commit Style
Use clear commit messages:
- feat: add leakage-safe split caching
- fix: prevent redundant ensemble retraining
- docs: update setup and status notes

## Pull Request Checklist
- Code runs locally.
- No dataset or artifact files are committed.
- README and PROJECT_STATUS are updated when behavior changes.
- .gitignore remains consistent with team policy.

## Do Not Commit
- Dataset files: train.csv, test.csv, train_ims/, test_ims/
- Model artifacts: *.joblib, *.cbm
- Feature caches: *.npy
- Environment folders: .venv/

## Suggested Review Focus
- Leakage prevention (split before augmentation).
- Reproducibility and deterministic splits.
- Runtime cost and memory usage.
- Correctness of test-time prediction logic.
