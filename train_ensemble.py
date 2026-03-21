import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump
import sys

def main():
    # Load data
    print("Loading data...")
    X = np.load("X_train_hog.npy")
    y = np.load("y_train.npy")
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_tr.shape[0]}, Test samples: {X_te.shape[0]}")
    print(f"Features: {X_tr.shape[1]}\n")
    
    # Normalize features for SVM
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    print("Training individual classifiers...")
    
    # Classifier 1: XGBoost (optimized)
    print("  1. XGBoost (optimized)...")
    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    xgb_clf.fit(X_tr, y_tr)
    xgb_acc = accuracy_score(y_te, xgb_clf.predict(X_te))
    print(f"     Accuracy: {xgb_acc:.4f}")
    
    # Classifier 2: CatBoost (optimized)
    print("  2. CatBoost (optimized)...")
    cat_clf = CatBoostClassifier(
        iterations=400,
        max_depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_strength=1,
        task_type="GPU",
        devices="0",
        verbose=False,
        random_state=42,
    )
    cat_clf.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
    cat_acc = accuracy_score(y_te, cat_clf.predict(X_te))
    print(f"     Accuracy: {cat_acc:.4f}")
    
    # Classifier 3: SVM with RBF kernel
    print("  3. SVM (RBF kernel, optimized)...")
    svm_clf = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        decision_function_shape='ovr',
        probability=True,
        random_state=42,
        verbose=0
    )
    svm_clf.fit(X_tr_scaled, y_tr)
    svm_acc = accuracy_score(y_te, svm_clf.predict(X_te_scaled))
    print(f"     Accuracy: {svm_acc:.4f}")
    
    # Create voting ensemble
    print("\nCreating Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('cat', cat_clf),
            ('svm', svm_clf)
        ],
        voting='soft',  # soft voting uses probability predictions
        n_jobs=-1
    )
    
    # For voting to work with mixed transformations, we need custom approach
    # Instead, create weighted ensemble prediction
    print("\nComputing weighted ensemble predictions...")
    
    xgb_pred_proba = xgb_clf.predict_proba(X_te)
    cat_pred_proba = cat_clf.predict_proba(X_te)
    svm_pred_proba = svm_clf.predict_proba(X_te_scaled)
    
    # Favor boosting models slightly; SVM still contributes to hard classes.
    ensemble_proba = (0.4 * xgb_pred_proba) + (0.4 * cat_pred_proba) + (0.2 * svm_pred_proba)
    y_pred_ensemble = np.argmax(ensemble_proba, axis=1)
    
    ensemble_acc = accuracy_score(y_te, y_pred_ensemble)
    
    print(f"\n{'='*60}")
    print("INDIVIDUAL CLASSIFIER ACCURACIES:")
    print(f"{'='*60}")
    print(f"XGBoost:  {xgb_acc:.4f}")
    print(f"CatBoost: {cat_acc:.4f}")
    print(f"SVM:      {svm_acc:.4f}")
    print(f"\nEnsemble (Weighted Average): {ensemble_acc:.4f}")
    print(f"{'='*60}")
    
    if ensemble_acc >= 0.7:
        print(f"\n✓ TARGET ACHIEVED! Accuracy: {ensemble_acc:.4f} >= 0.70")
    else:
        print(f"\n✗ Target not reached. Accuracy: {ensemble_acc:.4f} < 0.70")
        improvement_needed = 0.7 - ensemble_acc
        print(f"   Improvement needed: {improvement_needed:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_te, y_pred_ensemble))
    
    # Save ensemble details
    dump({
        'xgb': xgb_clf,
        'cat': cat_clf,
        'svm': svm_clf,
        'scaler': scaler,
        'accuracies': {
            'xgb': xgb_acc,
            'cat': cat_acc,
            'svm': svm_acc,
            'ensemble': ensemble_acc
        }
    }, "ensemble_model.joblib")
    print("\nEnsemble model saved to: ensemble_model.joblib")

if __name__ == "__main__":
    main()
