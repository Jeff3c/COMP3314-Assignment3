import os
import sys
import logging
import cv2
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


TRAIN_CSV = "train.csv"
TRAIN_DIR = "train_ims"
LOG_FILE = "training_log.txt"
OPTUNA_TRIALS = 50
RANDOM_STATE = 42


class TeeStream:
    """Duplicate stdout/stderr to console and file for remote progress tracking."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging():
    log_file_stream = open(LOG_FILE, "w", buffering=1, encoding="utf-8")
    tee_stream = TeeStream(sys.__stdout__, log_file_stream)
    sys.stdout = tee_stream
    sys.stderr = tee_stream
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Logging initialized. Writing terminal output to %s", LOG_FILE)


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


def gabor_features(gray: np.ndarray) -> np.ndarray:
    feats = []
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for theta in orientations:
        kernel = cv2.getGaborKernel(
            ksize=(9, 9),
            sigma=3.0,
            theta=theta,
            lambd=6.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F,
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        feats.append(np.mean(filtered))
        feats.append(np.std(filtered))
    return np.array(feats, dtype=np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_hog = hog_features(gray)
    f_lbp = lbp_features(gray)
    f_hist = color_hist_features(img, bins=16)
    f_gabor = gabor_features(gray)
    raw_small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    return np.concatenate([f_hog, f_lbp, f_hist, f_gabor, raw_small]).astype(np.float32)


def process_split(df, augment=False):
    X_list = []
    y_list = []
    desc = "Feature extraction (augmented)" if augment else "Feature extraction"
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc, file=sys.__stdout__):
        p = os.path.join(TRAIN_DIR, row.im_name)
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        X_list.append(extract_features(img))
        y_list.append(int(row.label))
        if augment:
            img_flip = cv2.flip(img, 1)
            X_list.append(extract_features(img_flip))
            y_list.append(int(row.label))
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def load_or_build_features():
    cache_train = "X_train_safe_gabor_v2.npy"
    cache_train_y = "y_train_safe_gabor_v2.npy"
    cache_val = "X_val_safe_gabor_v2.npy"
    cache_val_y = "y_val_safe_gabor_v2.npy"
    cache_test = "X_test_safe_gabor_v2.npy"
    cache_test_y = "y_test_safe_gabor_v2.npy"
    if all(os.path.exists(f) for f in [cache_train, cache_train_y, cache_val, cache_val_y, cache_test, cache_test_y]):
        X_tr = np.load(cache_train)
        y_tr = np.load(cache_train_y)
        X_val = np.load(cache_val)
        y_val = np.load(cache_val_y)
        X_te = np.load(cache_test)
        y_te = np.load(cache_test_y)
        logging.info("Loaded cached safe splits: X_tr=%s, X_val=%s, X_te=%s", X_tr.shape, X_val.shape, X_te.shape)
        return X_tr, X_val, X_te, y_tr, y_val, y_te

    df = pd.read_csv(TRAIN_CSV)
    # First split off test set (20%)
    df_trainval, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])
    # Then split trainval into train (70%) and val (10%)
    rel_val = 0.1 / 0.8  # 10% out of the remaining 80%
    df_train, df_val = train_test_split(df_trainval, test_size=rel_val, random_state=RANDOM_STATE, stratify=df_trainval["label"])

    logging.info("Extracting features for TRAIN set (with augmentation)...")
    X_tr, y_tr = process_split(df_train, augment=True)
    logging.info("Extracting features for VALIDATION set (no augmentation)...")
    X_val, y_val = process_split(df_val, augment=False)
    logging.info("Extracting features for TEST set (no augmentation)...")
    X_te, y_te = process_split(df_test, augment=False)

    np.save(cache_train, X_tr)
    np.save(cache_train_y, y_tr)
    np.save(cache_val, X_val)
    np.save(cache_val_y, y_val)
    np.save(cache_test, X_te)
    np.save(cache_test_y, y_te)
    logging.info("Saved safe split feature caches: X_tr=%s, X_val=%s, X_te=%s", X_tr.shape, X_val.shape, X_te.shape)
    return X_tr, X_val, X_te, y_tr, y_val, y_te



def build_xgb(num_class: int, max_depth: int, learning_rate: float, n_estimators: int = 520):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=3,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_lambda=1.5,
        reg_alpha=0.1,
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def objective(trial, X_tr, y_tr, X_val, y_val):
    svm_c = trial.suggest_float("svm_C", 0.1, 100.0, log=True)
    svm_gamma = trial.suggest_float("svm_gamma", 1e-4, 1.0, log=True)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
    xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)

    svm_base = SVC(
        C=svm_c,
        gamma=svm_gamma,
        kernel="rbf",
        decision_function_shape="ovr",
        probability=False,
        verbose=False,
        random_state=RANDOM_STATE,
    )
    svm_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ("cal", CalibratedClassifierCV(svm_base, method="sigmoid", cv=3, n_jobs=-1)),
        ],
        verbose=True,
    )
    svm_model.fit(X_tr, y_tr)
    svm_val_proba = svm_model.predict_proba(X_val)

    xgb_pre = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=320, random_state=RANDOM_STATE)),
        ],
        verbose=True,
    )
    X_tr_xgb = xgb_pre.fit_transform(X_tr)
    X_val_xgb = xgb_pre.transform(X_val)

    xgb_model = build_xgb(
        num_class=len(np.unique(y_tr)),
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        n_estimators=220,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
    xgb_val_proba = xgb_model.predict_proba(X_val_xgb)

    blended_val = 0.5 * svm_val_proba + 0.5 * xgb_val_proba
    y_pred = np.argmax(blended_val, axis=1)
    return accuracy_score(y_val, y_pred)


def main():
    setup_logging()

    logging.info("Loading train/validation/test feature splits...")
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_or_build_features()
    num_class = len(np.unique(y_tr))

    logging.info("Starting Optuna hyperparameter search (%d trials) for SVM and XGBoost...", OPTUNA_TRIALS)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_val, y_val), n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    logging.info("Best Optuna params: %s", best_params)
    logging.info("Best Optuna validation score: %.4f", study.best_value)

    logging.info("Training calibrated RBF-SVM...")
    svm_base = SVC(
        C=best_params["svm_C"],
        gamma=best_params["svm_gamma"],
        kernel="rbf",
        decision_function_shape="ovr",
        probability=False,
        verbose=2,
        random_state=RANDOM_STATE,
    )
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ("cal", CalibratedClassifierCV(svm_base, method="sigmoid", cv=3, n_jobs=-1)),
        ],
        verbose=True,
    )
    svm.fit(X_tr, y_tr)
    svm_val_proba = svm.predict_proba(X_val)
    logging.info("SVM validation accuracy: %.4f", accuracy_score(y_val, np.argmax(svm_val_proba, axis=1)))

    logging.info("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )
    rf.fit(X_tr, y_tr)
    rf_val_proba = rf.predict_proba(X_val)
    logging.info("RF validation accuracy: %.4f", accuracy_score(y_val, np.argmax(rf_val_proba, axis=1)))

    logging.info("Training KNN pipeline...")
    knn = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=50, random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        ],
        verbose=True,
    )
    knn.fit(X_tr, y_tr)
    knn_val_proba = knn.predict_proba(X_val)
    logging.info("KNN validation accuracy: %.4f", accuracy_score(y_val, np.argmax(knn_val_proba, axis=1)))

    logging.info("Training XGBoost on PCA-denoised features (GPU)...")
    xgb_pre = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=320, random_state=RANDOM_STATE)),
        ],
        verbose=True,
    )
    X_tr_xgb = xgb_pre.fit_transform(X_tr)
    X_val_xgb = xgb_pre.transform(X_val)
    X_te_xgb = xgb_pre.transform(X_te)

    xgb = build_xgb(
        num_class=num_class,
        max_depth=best_params["xgb_max_depth"],
        learning_rate=best_params["xgb_learning_rate"],
        n_estimators=520,
    )
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=10)
    xgb_val_proba = xgb.predict_proba(X_val_xgb)
    logging.info("XGBoost validation accuracy: %.4f", accuracy_score(y_val, np.argmax(xgb_val_proba, axis=1)))

    logging.info("Training CatBoost (GPU)...")
    catboost = CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_STATE,
        task_type="GPU",
        verbose=10,
    )
    catboost.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_val_proba = catboost.predict_proba(X_val)
    logging.info("CatBoost validation accuracy: %.4f", accuracy_score(y_val, np.argmax(cat_val_proba, axis=1)))

    logging.info("Training LogisticRegression meta-learner on validation meta-features...")
    X_meta_val = np.hstack([svm_val_proba, xgb_val_proba, rf_val_proba, cat_val_proba, knn_val_proba])
    meta_learner = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    meta_learner.fit(X_meta_val, y_val)

    logging.info("Generating test meta-features and predicting final classes...")
    svm_proba = svm.predict_proba(X_te)
    xgb_proba = xgb.predict_proba(X_te_xgb)
    rf_proba = rf.predict_proba(X_te)
    cat_proba = catboost.predict_proba(X_te)
    knn_proba = knn.predict_proba(X_te)

    X_meta_test = np.hstack([svm_proba, xgb_proba, rf_proba, cat_proba, knn_proba])
    y_pred = meta_learner.predict(X_meta_test)
    acc = accuracy_score(y_te, y_pred)
    logging.info("=" * 60)
    logging.info("Stacked ensemble test accuracy: %.4f", acc)
    logging.info("=" * 60)
    logging.info("\n%s", classification_report(y_te, y_pred))

    dump(
        {
            "svm": svm,
            "xgb_pre": xgb_pre,
            "xgb": xgb,
            "rf": rf,
            "knn": knn,
            "catboost": catboost,
            "meta_learner": meta_learner,
            "optuna_best_params": best_params,
            "final_acc": acc,
        },
        "classical_strong_ensemble.joblib",
    )
    logging.info("Saved model bundle: classical_strong_ensemble.joblib")


if __name__ == "__main__":
    main()
