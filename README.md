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
