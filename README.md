# COMP3314A3 - Classical Image Classification

This repository contains a classical machine learning image classification pipeline using handcrafted features and OOF stacking.

## Scope
- No neural network models.
- Main approach uses 5-fold OOF stacking with calibrated base learners.
- Feature extraction includes HOG, LBP, HSV histograms, Gabor features, and downsampled grayscale raw pixels.

## Repository Contents
- train_classical_strong.py: Main leakage-safe training pipeline.
- train_xgb_gpu.py: XGBoost-focused training script.
- train_catboost_gpu.py: CatBoost-focused training script.
- train_ensemble.py: Additional ensemble experiments.
- finalsolution.ipynb: Notebook work and experimentation.
- PROJECT_STATUS.md: Team status and current progress.

## Data (Not Tracked in Git)
Data files are intentionally ignored and must be placed locally in project root:
- train.csv
- test.csv
- train_ims/
- test_ims/

## Environment Setup (Windows PowerShell)
1. Create virtual environment:
   python -m venv .venv
2. Activate:
   .\.venv\Scripts\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt

## Run Training
Main pipeline:
python train_classical_strong.py


## Outputs
Current pipeline outputs (ignored by Git):
- X_train_safe_gabor_v2.npy
- y_train_safe_gabor_v2.npy
- X_val_safe_gabor_v2.npy
- y_val_safe_gabor_v2.npy
- X_test_safe_gabor_v2.npy
- y_test_safe_gabor_v2.npy
- test_split_names_safe_gabor_v2.npy (optional, auto-generated)
- stacked_ensemble_v2.joblib

Legacy/obsolete outputs (safe to delete):
- X_feat_strong.npy, y_feat_strong.npy
- X_test_hog.npy, X_train_hog.npy, y_train.npy
- X_train_safe.npy, X_val_safe.npy, X_test_safe.npy, y_train_safe.npy, y_val_safe.npy, y_test_safe.npy
- classical_strong_ensemble.joblib, ensemble_model.joblib, xgb_model_gpu.joblib, catboost_model_gpu.cbm
- catboost_info/ (no longer used)

## Team Workflow
1. Pull latest changes.
2. Sync environment with requirements.txt.
3. Run scripts locally with your own downloaded dataset.
4. Update PROJECT_STATUS.md with latest metrics and notes.

## Notes
- XGBoost GPU mode requires a valid local CUDA setup.
- If GPU is unavailable, adjust the script to CPU mode as needed.

## Major Pipeline Updates (March 2026)

- **5-Fold OOF Stacking (v2)**: The main pipeline merges train + validation, then runs StratifiedKFold(n_splits=5) to generate OOF probabilities.
- **Six Calibrated Base Models**: SVM, RandomForest, KNN, XGBoost, CatBoost, and LightGBM are all wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)`.
- **Tree Feature Selection**: XGBoost, CatBoost, and LightGBM use `SelectFromModel(RandomForestClassifier)` to keep the top 1,000 features instead of PCA.
- **Early Stopping**: All boosting models use an early-stopping strategy with 50 rounds.
- **Meta-Learner**: A LogisticRegression model is trained on stacked OOF probabilities.
- **TTA on Test Split**: For each image in the holdout test split, probabilities from the original and horizontally flipped images are averaged before meta prediction.
- **No Neural Networks**: The pipeline only uses handcrafted features and classical ML models.

## How the Pipeline Works
1. **Feature Extraction**: HOG, LBP, HSV histograms, Gabor statistics, and 16x16 grayscale raw pixels are extracted.
2. **Safe Splitting**: Data is split into train/validation/test before augmentation, and only train is augmented.
3. **OOF Base Training**: Six calibrated base models are trained fold-by-fold to produce OOF probabilities.
4. **Tree Selection + Early Stopping**: Boosting models use top-1000 selected features and early stopping with 50 rounds.
5. **Meta Training**: LogisticRegression is trained on concatenated OOF probabilities.
6. **TTA Inference**: For each holdout test image, original and flipped probabilities are averaged for each base model.
7. **Final Prediction and Report**: Meta-learner predicts test labels, prints `classification_report`, and logs final accuracy.

## To Run the Full Pipeline
1. Ensure all dependencies in `requirements.txt` are installed (see below).
2. Place the required data files in the project root.
3. Run:

   python train_classical_strong.py

4. Monitor progress in the terminal or by viewing `training_log.txt`.

## Requirements
- tqdm
- scikit-learn
- xgboost (GPU recommended)
- catboost (GPU recommended)
- lightgbm
- opencv-python
- scikit-image
- numpy, pandas, joblib

## Output Artifact
- Final bundle: `stacked_ensemble_v2.joblib` (all others above are intermediate caches)
