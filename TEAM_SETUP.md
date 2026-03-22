# Team Setup and Sharing

## 1. Clone the Repository
git clone <repo-url>
cd COMP3314A3

## 2. Prepare Python Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## 3. Download and Place Dataset Locally
Place the following in project root (not tracked in Git):
- train.csv
- test.csv
- train_ims/
- test_ims/

## 4. Run the Main Pipeline
python train_classical_strong.py

## 5. Update Team Status
After experiments, update PROJECT_STATUS.md with:
- run date
- script used
- key metrics
- observations

## 6. Typical Collaboration Cycle
1. Pull latest changes.
2. Create branch.
3. Implement and test.
4. Commit and push.
5. Open pull request.

## Pipeline Reference (March 2026)
- The main script now uses a 5-model ensemble (SVM, XGBoost, RandomForest, CatBoost, KNN) with a LogisticRegression meta-learner.
- Optuna is used for SVM/XGBoost hyperparameter tuning (15 trials, 15% subsample for speed).
- All logs and progress are saved to training_log.txt.
- Feature extraction includes HOG, LBP, HSV, Gabor, and downsampled grayscale.
- See README.md for full details and workflow.
