
import logging
import os
import sys
import time
import gc

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import dump
from lightgbm import LGBMClassifier
from skimage.feature import hog, local_binary_pattern
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier



TRAIN_CSV = "train.csv"
TRAIN_DIR = "train_ims"
LOG_FILE = "training_log.txt"
RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
TOP_TREE_FEATURES = 1000

# If True, delete all .npy feature caches before loading/building features
CLEAR_CACHE = False


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



class EarlyStoppingBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Fit boosting models with an internal validation split for early stopping."""
    _estimator_type = "classifier"

    def __init__(self, model_name: str, model_params: dict, random_state: int = RANDOM_STATE):
        self.model_name = model_name
        self.model_params = model_params
        self.random_state = random_state

    def _build_model(self):
        if self.model_name == "xgb":
            return XGBClassifier(**self.model_params)
        if self.model_name == "cat":
            return CatBoostClassifier(**self.model_params)
        if self.model_name == "lgbm":
            return LGBMClassifier(**self.model_params)
        raise ValueError(f"Unsupported boosting model: {self.model_name}")

    def fit(self, X, y):
        start_time = time.time()
        
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X,
            y,
            test_size=0.15,
            stratify=y,
            random_state=self.random_state,
        )
        self.model_ = self._build_model()
        self.classes_ = np.unique(y)

        if self.model_name == "xgb":
            self.model_.fit(
                X_fit,
                y_fit,
                eval_set=[(X_eval, y_eval)],
                verbose=False,
            )
        elif self.model_name == "cat":
            self.model_.fit(
                X_fit,
                y_fit,
                eval_set=(X_eval, y_eval),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        elif self.model_name == "lgbm":
            from lightgbm import early_stopping, log_evaluation
            eval_set = [(X_eval, y_eval)]
            callbacks = [
                early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                log_evaluation(period=0),  # Keep console output quiet like XGB/CatBoost wrappers
            ]
            self.model_.fit(
                X_fit,
                y_fit,
                eval_set=eval_set,
                eval_metric="multi_logloss",
                callbacks=callbacks,
            )
        
        elapsed = time.time() - start_time
        print(f"  >>> {self.model_name.upper()} training took {elapsed:.2f} seconds.")
        gc.collect()  # Free X_fit/X_eval memory immediately
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


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
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_hsv.shape[:2]
    hs, ws = h // 4, w // 4
    feats = []
    for i in range(4):
        for j in range(4):
            q = img_hsv[i * hs : (i + 1) * hs, j * ws : (j + 1) * ws]
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
            q = gray[i * hs : (i + 1) * hs, j * ws : (j + 1) * ws]
            lbp = local_binary_pattern(q, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-9
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
    # Denoising: Apply 3x3 median blur to input image before feature extraction
    img_denoised = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
    f_hog = hog_features(gray)
    f_lbp = lbp_features(gray)
    f_hist = color_hist_features(img_denoised, bins=16)
    f_gabor = gabor_features(gray)
    raw_small = (
        cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        .ravel()
        / 255.0
    )
    return np.concatenate([f_hog, f_lbp, f_hist, f_gabor, raw_small]).astype(np.float32)


def process_split(df: pd.DataFrame, augment: bool = False):
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


def build_flipped_features(im_names: np.ndarray) -> np.ndarray:
    X_flip = []
    for im_name in tqdm(im_names, total=len(im_names), desc="Building flip features", file=sys.__stdout__):
        p = os.path.join(TRAIN_DIR, str(im_name))
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        X_flip.append(extract_features(cv2.flip(img, 1)))
    return np.vstack(X_flip)


def load_or_build_features():
    cache_train = "X_train_safe_gabor_v2.npy"
    cache_train_y = "y_train_safe_gabor_v2.npy"
    cache_val = "X_val_safe_gabor_v2.npy"
    cache_val_y = "y_val_safe_gabor_v2.npy"
    cache_test = "X_test_safe_gabor_v2.npy"
    cache_test_y = "y_test_safe_gabor_v2.npy"
    cache_test_names = "test_split_names_safe_gabor_v2.npy"


    # Invalidate cache if CLEAR_CACHE is set
    if CLEAR_CACHE:
        for f in [cache_train, cache_train_y, cache_val, cache_val_y, cache_test, cache_test_y, cache_test_names]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logging.info(f"Deleted cache file: {f}")
                except Exception as e:
                    logging.warning(f"Could not delete cache file {f}: {e}")

    has_main_caches = all(
        os.path.exists(f)
        for f in [cache_train, cache_train_y, cache_val, cache_val_y, cache_test, cache_test_y]
    )

    if has_main_caches:
        X_tr = np.load(cache_train)
        y_tr = np.load(cache_train_y)
        X_val = np.load(cache_val)
        y_val = np.load(cache_val_y)
        X_te = np.load(cache_test)
        y_te = np.load(cache_test_y)

        if os.path.exists(cache_test_names):
            test_names = np.load(cache_test_names, allow_pickle=True)
        else:
            df = pd.read_csv(TRAIN_CSV)
            _, df_test = train_test_split(
                df,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=df["label"],
            )
            test_names = df_test["im_name"].to_numpy()
            np.save(cache_test_names, test_names)

        logging.info(
            "Loaded cached safe splits: X_tr=%s, X_val=%s, X_te=%s",
            X_tr.shape,
            X_val.shape,
            X_te.shape,
        )
        return X_tr, X_val, X_te, y_tr, y_val, y_te, test_names

    df = pd.read_csv(TRAIN_CSV)
    df_trainval, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    rel_val = 0.1 / 0.8
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=rel_val,
        random_state=RANDOM_STATE,
        stratify=df_trainval["label"],
    )

    logging.info("Extracting features for TRAIN set (with augmentation)...")
    X_tr, y_tr = process_split(df_train, augment=True)
    logging.info("Extracting features for VALIDATION set (no augmentation)...")
    X_val, y_val = process_split(df_val, augment=False)
    logging.info("Extracting features for TEST set (no augmentation)...")
    X_te, y_te = process_split(df_test, augment=False)
    test_names = df_test["im_name"].to_numpy()

    np.save(cache_train, X_tr)
    np.save(cache_train_y, y_tr)
    np.save(cache_val, X_val)
    np.save(cache_val_y, y_val)
    np.save(cache_test, X_te)
    np.save(cache_test_y, y_te)
    np.save(cache_test_names, test_names)

    logging.info(
        "Saved safe split feature caches: X_tr=%s, X_val=%s, X_te=%s",
        X_tr.shape,
        X_val.shape,
        X_te.shape,
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te, test_names


def build_tree_selector() -> SelectFromModel:
    selector_model = RandomForestClassifier(
        n_estimators=50,           # Reduced for speed/memory
        max_depth=10,              # Shallower trees for less RAM
        random_state=RANDOM_STATE,
        n_jobs=2,                  # Respect CPU limits
        class_weight="balanced_subsample",
    )
    return SelectFromModel(
        estimator=selector_model,
        threshold=-np.inf,
        max_features=TOP_TREE_FEATURES,
    )


def build_base_estimators(num_class: int):
    # ... (Keep SVM pipeline exactly as it is) ...
    svm_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            (
                "svm",
                SVC(
                    C=5.0,
                    gamma=0.0005,
                    kernel="rbf",
                    decision_function_shape="ovr",
                    probability=False,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


    # 1. Constrain the Random Forest and wrap in a pipeline with feature selector
    rf_pipeline = Pipeline(
        steps=[
            ("selector", build_tree_selector()),
            ("rf", RandomForestClassifier(
                n_estimators=400,          # Reduced from 600
                max_depth=20,              # Limit tree depth to save massive RAM
                random_state=RANDOM_STATE,
                n_jobs=2,                  # Only use 2 cores internally, not all of them
                class_weight="balanced_subsample",
            )),
        ]
    )

    # ... (Keep KNN, XGB, CAT, LGBM pipelines exactly as they are) ...
    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=120, random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
        ]
    )

    xgb_pipeline = Pipeline(
        steps=[
            ("selector", build_tree_selector()),
            (
                "boost",
                EarlyStoppingBoostingClassifier(
                    model_name="xgb",
                    model_params={
                        "n_estimators": 1200,
                        "max_depth": 8,
                        "learning_rate": 0.05,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "objective": "multi:softprob",
                        "num_class": num_class,
                        "eval_metric": "mlogloss",
                        "random_state": RANDOM_STATE,
                        "n_jobs": -1,
                        "tree_method": "hist",
                        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                    },
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("selector", build_tree_selector()),
            (
                "boost",
                EarlyStoppingBoostingClassifier(
                    model_name="cat",
                    model_params={
                        "iterations": 1200,
                        "depth": 8,
                        "learning_rate": 0.05,
                        "loss_function": "MultiClass",
                        "eval_metric": "MultiClass",
                        "random_seed": RANDOM_STATE,
                        "od_type": "Iter",
                        "od_wait": EARLY_STOPPING_ROUNDS,
                        "allow_writing_files": False,
                    },
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    lgbm_pipeline = Pipeline(
        steps=[
            ("selector", build_tree_selector()),
            (
                "boost",
                EarlyStoppingBoostingClassifier(
                    model_name="lgbm",
                    model_params={
                        "n_estimators": 1200,
                        "learning_rate": 0.05,
                        "num_leaves": 63,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "objective": "multiclass",
                        "num_class": num_class,
                        "random_state": RANDOM_STATE,
                        "n_jobs": -1,
                        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                        "verbosity": -1,
                    },
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 2. Prevent CalibratedClassifierCV from copying data across all CPU cores
    # Set n_jobs=1 or n_jobs=2 max for the wrappers.
    return {
        "svm": CalibratedClassifierCV(svm_pipeline, method="sigmoid", cv=3, n_jobs=2),
        # RF now uses a pipeline with selector for memory efficiency
        "rf": CalibratedClassifierCV(rf_pipeline, method="sigmoid", cv=3, n_jobs=1),
        "knn": CalibratedClassifierCV(knn_pipeline, method="sigmoid", cv=3, n_jobs=1),
        # These wrappers must be n_jobs=1 for RAM safety
        "xgb": CalibratedClassifierCV(xgb_pipeline, method="sigmoid", cv=3, n_jobs=1),
        "cat": CalibratedClassifierCV(cat_pipeline, method="sigmoid", cv=3, n_jobs=1),
        "lgbm": CalibratedClassifierCV(lgbm_pipeline, method="sigmoid", cv=3, n_jobs=1),
    }


def main():
    setup_logging()
    logging.info("Loading train/validation/test feature splits...")
    X_tr, X_val, X_te, y_tr, y_val, y_te, test_names = load_or_build_features()

    logging.info("Merging train and validation for OOF stacking...")
    X_train_full = np.vstack([X_tr, X_val])
    y_train_full = np.concatenate([y_tr, y_val])
    num_class = len(np.unique(y_train_full))

    logging.info("Preparing flipped test features for TTA...")

    X_te_flip = build_flipped_features(test_names)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model_names = ["svm", "rf", "knn", "xgb", "cat", "lgbm"]

    oof_preds = {
        name: np.zeros((X_train_full.shape[0], num_class), dtype=np.float32)
        for name in model_names
    }
    tta_test_preds = {
        name: np.zeros((X_te.shape[0], num_class, N_SPLITS), dtype=np.float32)
        for name in model_names
    }


    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
        logging.info("Starting OOF fold %d/%d", fold_idx, N_SPLITS)
        fold_start_time = time.time()
        X_fold_train, X_fold_valid = X_train_full[train_idx], X_train_full[valid_idx]
        y_fold_train, y_fold_valid = y_train_full[train_idx], y_train_full[valid_idx]

        estimators = build_base_estimators(num_class=num_class)

        for name in model_names:
            logging.info("Fold %d | Training calibrated %s", fold_idx, name.upper())
            model = estimators[name]
            model.fit(X_fold_train, y_fold_train)

            valid_proba = model.predict_proba(X_fold_valid)
            oof_preds[name][valid_idx] = valid_proba

            test_proba_orig = model.predict_proba(X_te)
            test_proba_flip = model.predict_proba(X_te_flip)
            tta_test_preds[name][:, :, fold_idx - 1] = (test_proba_orig + test_proba_flip) / 2.0

            fold_acc = accuracy_score(y_fold_valid, np.argmax(valid_proba, axis=1))
            logging.info("Fold %d | %s validation accuracy: %.4f", fold_idx, name.upper(), fold_acc)

            # --- ADD THIS LINE TO CLEAR RAM ---
            gc.collect()

        fold_elapsed = time.time() - fold_start_time
        print(f"Fold {fold_idx}/{N_SPLITS} elapsed time: {fold_elapsed:.2f} seconds")

    # After TTA, clear X_te_flip to free memory
    del X_te_flip
    gc.collect()


    logging.info("Training LogisticRegression meta-learner on OOF probabilities...")
    X_meta_train = np.hstack([oof_preds[name] for name in model_names])
    X_meta_test = np.hstack([
        np.mean(tta_test_preds[name], axis=2) for name in model_names
    ])
    gc.collect()  # Free up memory before meta-learner

    # Updated meta-learner: multinomial, lbfgs, l2
    meta_learner = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=1.0,
        max_iter=4000,
        random_state=RANDOM_STATE,
        multi_class='multinomial',
    )
    meta_learner.fit(X_meta_train, y_train_full)

    y_pred = meta_learner.predict(X_meta_test)
    acc = accuracy_score(y_te, y_pred)

    logging.info("=" * 60)
    logging.info("5-Fold OOF stacked ensemble test accuracy: %.4f", acc)
    logging.info("=" * 60)
    logging.info("\n%s", classification_report(y_te, y_pred, digits=4))

    bundle = {
        "model_type": "5fold_oof_stacking_v2",
        "feature_spec": "hog+lbp+color_hist+gabor+raw16x16",
        "base_model_order": model_names,
        "meta_learner": meta_learner,
        "oof_meta_train": X_meta_train,
        "test_meta_features": X_meta_test,
        "test_true_labels": y_te,
        "test_pred_labels": y_pred,
        "final_accuracy": acc,
        "classification_report": classification_report(y_te, y_pred, digits=4),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "tree_feature_top_k": TOP_TREE_FEATURES,
    }
    dump(bundle, "stacked_ensemble_v2.joblib")
    logging.info("Saved model bundle: stacked_ensemble_v2.joblib")


if __name__ == "__main__":
    main()
