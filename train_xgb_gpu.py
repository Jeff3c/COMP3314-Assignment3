import argparse
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

try:
    from xgboost import XGBClassifier
except Exception as e:
    print("Failed to import xgboost. Ensure it's installed in your environment.", file=sys.stderr)
    raise


def parse_args():
    p = argparse.ArgumentParser(description="Train an XGBClassifier on HOG features using GPU")
    p.add_argument("--X", default="X_train_hog.npy", help="Path to numpy file with features (X)")
    p.add_argument("--y", default="y_train.npy", help="Path to numpy file with labels (y)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-out", default="xgb_model_gpu.joblib", help="Path to save trained model")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Training device")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.X):
        print(f"Feature file not found: {args.X}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.y):
        print(f"Label file not found: {args.y}", file=sys.stderr)
        sys.exit(2)

    X = np.load(args.X)
    y = np.load(args.y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    try:
        stratify = y
    except Exception:
        stratify = None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )

    clf_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "n_jobs": -1,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0,
        "reg_lambda": 1,
        "reg_alpha": 0,
        # XGBoost 3.x GPU setup: use hist with device='cuda'.
        "tree_method": "hist",
        "device": args.device,
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
    }

    print("Training XGBClassifier with parameters:")
    for k, v in clf_kwargs.items():
        print(f"  {k}: {v}")

    clf = XGBClassifier(**clf_kwargs)

    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    except Exception as e:
        print("Training failed:", e, file=sys.stderr)
        print("If this is a GPU error, check that a CUDA-compatible GPU and matching drivers are installed.")
        raise

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"Baseline accuracy (80/20 split): {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_te, y_pred))

    dump(clf, args.model_out)
    print(f"Saved trained model to: {args.model_out}")


if __name__ == "__main__":
    main()
