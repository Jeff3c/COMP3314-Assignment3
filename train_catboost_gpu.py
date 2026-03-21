import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import sys

def main():
    # Load data
    X = np.load("X_train_hog.npy")
    y = np.load("y_train.npy")
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training CatBoostClassifier with GPU acceleration...")
    print(f"  Training samples: {X_tr.shape[0]}, Test samples: {X_te.shape[0]}")
    print(f"  Features: {X_tr.shape[1]}")
    
    # Train CatBoost with GPU
    clf = CatBoostClassifier(
        iterations=500,
        max_depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=1,
        task_type="GPU",
        devices="0",
        verbose=False,
        random_state=42,
        early_stopping_rounds=50,
    )
    
    try:
        clf.fit(
            X_tr, y_tr,
            eval_set=(X_te, y_te),
            verbose=False
        )
    except Exception as e:
        print(f"GPU training failed: {e}")
        print("Attempting CPU training instead...")
        clf = CatBoostClassifier(
            iterations=500,
            max_depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_strength=1,
            bagging_temperature=1,
            task_type="CPU",
            verbose=False,
            random_state=42,
            early_stopping_rounds=50,
        )
        clf.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
    
    # Evaluate
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    
    print(f"\n{'='*60}")
    print(f"CatBoost Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred))
    
    # Save confusion matrix info
    cm = confusion_matrix(y_te, y_pred)
    print(f"\nConfusion Matrix shape: {cm.shape}")
    print("\nPer-class accuracy:")
    for i in range(10):
        class_acc = cm[i, i] / cm[i].sum()
        print(f"  Class {i}: {class_acc:.4f}")
    
    # Save model
    clf.save_model("catboost_model_gpu.cbm")
    print(f"\nModel saved to: catboost_model_gpu.cbm")

if __name__ == "__main__":
    main()
