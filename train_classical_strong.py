import os
import cv2
import numpy as np
import pandas as pd
from joblib import dump
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


TRAIN_CSV = "train.csv"
TRAIN_DIR = "train_ims"
CACHE_X = "X_feat_strong.npy"
CACHE_Y = "y_feat_strong.npy"


def color_hist_features(img_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    # Convert to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_hsv.shape[:2]
    hs, ws = h // 4, w // 4
    feats = []
    for i in range(4):
        for j in range(4):
            q = img_hsv[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            chans = cv2.split(q)
            for ch in chans:
                hist = cv2.calcHist([ch], [0], None, [bins], [0, 256]).ravel()
                hist = hist / (hist.sum() + 1e-9)
                feats.append(hist)
    return np.concatenate(feats)


def lbp_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    hs, ws = h // 4, w // 4
    feats = []
    for i in range(4):
        for j in range(4):
            q = gray[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            lbp = local_binary_pattern(q, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-9)
            feats.append(hist)
    return np.concatenate(feats)


def hog_features(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_hog = hog_features(gray)
    f_lbp = lbp_features(gray)
    f_hist = color_hist_features(img, bins=16)
    raw_small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    return np.concatenate([f_hog, f_lbp, f_hist, raw_small]).astype(np.float32)


def process_split(df, augment=False):
    X_list = []
    y_list = []
    for i, row in df.iterrows():
        p = os.path.join(TRAIN_DIR, row["im_name"])
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        X_list.append(extract_features(img))
        y_list.append(int(row["label"]))
        if augment:
            img_flip = cv2.flip(img, 1)
            X_list.append(extract_features(img_flip))
            y_list.append(int(row["label"]))
        if (i + 1) % 5000 == 0:
            print(f"Extracted features for {i + 1}/{len(df)} images")
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_or_build_features():
    cache_train = "X_train_safe.npy"
    cache_train_y = "y_train_safe.npy"
    cache_test = "X_test_safe.npy"
    cache_test_y = "y_test_safe.npy"
    if all(os.path.exists(f) for f in [cache_train, cache_train_y, cache_test, cache_test_y]):
        X_tr = np.load(cache_train)
        y_tr = np.load(cache_train_y)
        X_te = np.load(cache_test)
        y_te = np.load(cache_test_y)
        print(f"Loaded cached safe splits: X_tr={X_tr.shape}, X_te={X_te.shape}")
        return X_tr, X_te, y_tr, y_te

    df = pd.read_csv(TRAIN_CSV)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"Extracting features for TRAIN set (with augmentation)...")
    X_tr, y_tr = process_split(df_train, augment=True)
    print(f"Extracting features for TEST set (no augmentation)...")
    X_te, y_te = process_split(df_test, augment=False)
    np.save(cache_train, X_tr)
    np.save(cache_train_y, y_tr)
    np.save(cache_test, X_te)
    np.save(cache_test_y, y_te)
    print(f"Saved safe split feature caches: X_tr={X_tr.shape}, X_te={X_te.shape}")
    return X_tr, X_te, y_tr, y_te



def main():
    X_tr, X_te, y_tr, y_te = load_or_build_features()

    # Validation split for blend weights
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    print("Training calibrated RBF-SVM...")
    svm_base = SVC(C=8, gamma="scale", kernel="rbf", decision_function_shape="ovr", probability=False)
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("cal", CalibratedClassifierCV(svm_base, method="sigmoid", cv=3, n_jobs=-1)),
        ]
    )
    svm.fit(X_fit, y_fit)
    svm_val_proba = svm.predict_proba(X_val)
    svm_val_pred = np.argmax(svm_val_proba, axis=1)
    svm_val_acc = accuracy_score(y_val, svm_val_pred)
    print(f"SVM validation accuracy: {svm_val_acc:.4f}")

    print("Training XGBoost on PCA-denoised features (GPU)...")
    xgb_pre = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=320, random_state=42)),
        ]
    )
    X_fit_xgb = xgb_pre.fit_transform(X_fit)
    X_val_xgb = xgb_pre.transform(X_val)

    xgb = XGBClassifier(
        n_estimators=520,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_lambda=1.5,
        reg_alpha=0.1,
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        num_class=len(np.unique(y_tr)),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_fit_xgb, y_fit, eval_set=[(X_val_xgb, y_val)], verbose=False)
    xgb_val_proba = xgb.predict_proba(X_val_xgb)
    xgb_val_pred = np.argmax(xgb_val_proba, axis=1)
    xgb_val_acc = accuracy_score(y_val, xgb_val_pred)
    print(f"XGBoost validation accuracy: {xgb_val_acc:.4f}")

    print("Selecting blend weights on validation split...")
    candidate_weights = [(0.90, 0.10), (0.85, 0.15), (0.80, 0.20), (0.75, 0.25), (0.70, 0.30)]
    best_w = (0.85, 0.15)
    best_val = -1.0
    for w_svm, w_xgb in candidate_weights:
        val_blend = (w_svm * svm_val_proba) + (w_xgb * xgb_val_proba)
        val_pred = np.argmax(val_blend, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        if val_acc > best_val:
            best_val = val_acc
            best_w = (w_svm, w_xgb)
    print(f"Best validation blend: svm={best_w[0]:.2f}, xgb={best_w[1]:.2f}, acc={best_val:.4f}")

    # Refit on all train data
    print("Refitting SVM and XGBoost on all training data...")
    svm.fit(X_tr, y_tr)
    svm_proba = svm.predict_proba(X_te)

    X_tr_xgb = xgb_pre.fit_transform(X_tr)
    X_te_xgb = xgb_pre.transform(X_te)
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_te_xgb, y_te)], verbose=False)
    xgb_proba = xgb.predict_proba(X_te_xgb)

    print("Manual weighted blend on test set...")
    blend = (best_w[0] * svm_proba) + (best_w[1] * xgb_proba)
    y_pred = np.argmax(blend, axis=1)
    acc = accuracy_score(y_te, y_pred)
    print("=" * 60)
    print(f"Manual blend test accuracy: {acc:.4f}")
    print("=" * 60)
    print(classification_report(y_te, y_pred))

    dump(
        {
            "svm": svm,
            "xgb_pre": xgb_pre,
            "xgb": xgb,
            "blend_weights": best_w,
            "final_acc": acc,
        },
        "classical_strong_ensemble.joblib",
    )
    print("Saved model bundle: classical_strong_ensemble.joblib")


if __name__ == "__main__":
    main()
