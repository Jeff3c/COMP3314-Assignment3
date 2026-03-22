# Project Status

Last updated: 2026-03-21

## Overall Status
- Stage: Active development
- Pipeline type: Classical ML image classification (no neural networks)
- Main training script: train_classical_strong.py

## What Is Done
- Leakage fix: train/test split is done on dataframe rows before augmentation.
- Augmentation policy: horizontal flip is applied only to train split.
- Safe feature caches are implemented:
  - X_train_safe.npy
  - y_train_safe.npy
  - X_test_safe.npy
  - y_test_safe.npy
- Final prediction uses manual weighted blending of SVM and XGBoost probabilities.
- VotingClassifier double-fit path removed.

## Current Risks / Notes
- Full training is still compute-intensive, but Optuna tuning is now much faster due to subsampling.
- All results and progress can be monitored in training_log.txt.
- Final model is saved as classical_strong_ensemble.joblib and includes all base models and the meta-learner.

## March 2026 Major Pipeline Update
- Pipeline now uses a 5-base-model ensemble (SVM, XGBoost, RandomForest, CatBoost, KNN) with LogisticRegression meta-learner.
- Optuna hyperparameter tuning (15 trials, 15% subsample) for SVM and XGBoost.
- Feature extraction now includes Gabor filter features (mean/std for 4 orientations).
- tqdm progress bars for all feature extraction.
- All logs and progress are saved to training_log.txt for remote monitoring.
- No neural networks are used (classical ML only).

## Next Suggested Tasks
1. Run final full training without interruption and record metrics.
2. Save final chosen blend weights and test report in this file.
3. Add optional evaluation script for reproducible reporting.
4. Add CI lint/check job (optional, if needed for team workflow).

## Teammate Checklist
- Create Python environment and install requirements.
- Download dataset files to project root:
  - train.csv
  - test.csv
  - train_ims/
  - test_ims/
- Run training script.
- Update this status file with latest metrics.
