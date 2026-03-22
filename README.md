# COMP3314A3 - Classical Image Classification

This repository contains a classical machine learning image classification pipeline using handcrafted features and ensemble modeling.

## Scope
- No neural network models.
- Main approach uses SVM + XGBoost probability blending.
- Feature extraction includes HOG, LBP, HSV histograms, and downsampled grayscale features.

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
Typical outputs are generated locally and ignored by Git:
- *.joblib
- *.cbm
- *.npy
- catboost_info/

## Team Workflow
1. Pull latest changes.
2. Sync environment with requirements.txt.
3. Run scripts locally with your own downloaded dataset.
4. Update PROJECT_STATUS.md with latest metrics and notes.

## Notes
- XGBoost GPU mode requires a valid local CUDA setup.
- If GPU is unavailable, adjust the script to CPU mode as needed.

## Major Pipeline Updates (March 2026)

- **OOF Ensemble Stacking**: The main pipeline now uses Out-Of-Fold (OOF) stacking with 5-fold StratifiedKFold. Five base models (SVM, XGBoost, RandomForest, CatBoost, KNN) generate OOF meta-features for a LogisticRegression meta-learner. All base models are refit on the full training set before test prediction.
- **Optuna Hyperparameter Tuning**: SVM and XGBoost hyperparameters are tuned using Optuna (15 trials, 15% subsample of training data, 3-fold CV for speed and robustness). Final models are trained on the full training set.
- **Feature Extraction**: In addition to HOG, LBP, HSV histograms, and downsampled grayscale, the pipeline now includes Gabor filter features (mean and std for 4 orientations).
- **Progress and Logging**: All feature extraction uses tqdm progress bars. All terminal/logging output is saved to `training_log.txt` for remote monitoring.
- **No Neural Networks**: All models are classical ML (no deep learning).

## How the Pipeline Works
1. **Feature Extraction**: HOG, LBP, HSV, Gabor, and downsampled grayscale features are extracted for each image.
2. **Safe Splitting**: Data is split into train/val/test before any augmentation. Only the train split is augmented (horizontal flip).
3. **Optuna Tuning**: SVM (C, gamma) and XGBoost (max_depth, learning_rate) are tuned on a 15% subsample of the merged training set using 3-fold cross-validation for 15 trials.
4. **OOF Stacking**: Five base models are trained using 5-fold StratifiedKFold. Their out-of-fold (OOF) probability predictions are stacked to train a LogisticRegression meta-learner. All base models are then refit on the full training set before test prediction.
5. **Test Prediction**: The meta-learner predicts final test classes using stacked base model probabilities from the refit base models.
6. **Logging**: All progress and results are saved to `training_log.txt`.

## To Run the Full Pipeline
1. Ensure all dependencies in `requirements.txt` are installed (see below).
2. Place the required data files in the project root.
3. Run:

   python train_classical_strong.py

4. Monitor progress in the terminal or by viewing `training_log.txt`.

## Requirements
- tqdm
- optuna
- scikit-learn
- xgboost (GPU recommended)
- catboost (GPU recommended)
- opencv-python
- scikit-image
- numpy, pandas, joblib
