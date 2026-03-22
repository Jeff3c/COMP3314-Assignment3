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

## Documentation
- When updating the pipeline (e.g., new ensemble, meta-learner, OOF stacking, Optuna tuning, logging, or feature extraction changes), update README.md and PROJECT_STATUS.md to reflect the new workflow and requirements.
- Summarize major changes for teammates in the Major Pipeline Updates section of README.md.
