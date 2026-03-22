# Project Status

Last updated: 2026-03-21

## Overall Status
- Stage: Active development
- Pipeline type: Classical ML image classification (no neural networks)
- Main training script: train_classical_strong.py


## What Is Done
- Leakage fix: train/test split is done on dataframe rows before augmentation.
- Augmentation policy: horizontal flip is applied only to train split.
- Safe feature caches are implemented (current):
  - X_train_safe_gabor_v2.npy
  - y_train_safe_gabor_v2.npy
  - X_val_safe_gabor_v2.npy
  - y_val_safe_gabor_v2.npy
  - X_test_safe_gabor_v2.npy
  - y_test_safe_gabor_v2.npy
  - test_split_names_safe_gabor_v2.npy (optional)
- Final prediction now uses a 5-fold OOF stacked ensemble with calibrated base probabilities.
- Meta-learner is LogisticRegression trained on OOF meta-features.


## Current Risks / Notes
- Full training is still compute-intensive, but Optuna tuning is now much faster due to subsampling.
- All results and progress can be monitored in training_log.txt.
- Final model bundle is saved as stacked_ensemble_v2.joblib.
- All previous .npy/model artifacts not listed above are now deprecated and can be deleted.

## March 2026 Major Pipeline Update
- Pipeline now uses Out-Of-Fold (OOF) stacking with 5-fold StratifiedKFold.
- Base models are SVM, RandomForest, KNN, XGBoost, CatBoost, and LightGBM; each is wrapped by CalibratedClassifierCV(method='sigmoid', cv=3).
- Tree-based models (XGBoost, CatBoost, LightGBM) use SelectFromModel(RandomForest) to keep the top 1,000 features (no PCA for boosting).
- Early stopping with 50 rounds is used for all boosting models.
- Test-Time Augmentation (TTA) averages original and horizontal-flip probabilities before meta-learner prediction.
- Feature extraction includes HOG, LBP, HSV histograms, Gabor statistics, and raw 16x16 grayscale pixels.
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
