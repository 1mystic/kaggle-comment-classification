"""Illustrative reimplementation of the stacked-ensemble training pipeline described in the README.

NOT the original competition notebook: rewritten independently, after the fact,
to demonstrate the approach and structure. Contains no competition data, and the
hyperparameters below are the final values reported in the README, not a tuning
search (that search is not reproduced here).

This module is a structural reference, not a runnable script: it assumes
`X_train_full` / `train_labels` etc. already exist (e.g. built via preprocess.py
and features.py against your own copy of the data).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
import lightgbm as lgb

SEED = 42
N_FOLDS = 3

# Class weights tuned to balance precision/recall per class, not derived
# directly from the raw imbalance ratio (see README).
LGB_CLASS_WEIGHTS = {0: 1.0, 1: 3.5, 2: 1.5, 3: 15.0}

LGB_PARAMS = dict(
    objective="multiclass",
    num_class=4,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=127,
    colsample_bytree=0.5,
    subsample=0.9,
    min_child_samples=10,
    class_weight=LGB_CLASS_WEIGHTS,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

META_LGB_PARAMS = dict(
    objective="multiclass",
    num_class=4,
    num_leaves=16,
    n_estimators=100,
    learning_rate=0.05,
    min_child_samples=200,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


@dataclass
class OOFResult:
    """Out-of-fold predictions for one base model, plus averaged test predictions."""
    oof: np.ndarray
    test: np.ndarray


def train_lightgbm_oof(X_train, y_train, X_test, n_folds: int = N_FOLDS) -> OOFResult:
    """Train the primary LightGBM multiclass model with stratified K-fold OOF generation."""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((X_train.shape[0], 4), dtype=np.float32)
    test_preds = np.zeros((X_test.shape[0], 4), dtype=np.float32)

    for train_idx, val_idx in kfold.split(X_train, y_train):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train[train_idx], y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof[val_idx] = model.predict_proba(X_train[val_idx])
        test_preds += model.predict_proba(X_test) / n_folds

    return OOFResult(oof=oof, test=test_preds)


def train_linear_svc_oof(X_train_text, y_train, X_test_text, n_folds: int = N_FOLDS) -> OOFResult:
    """Calibrated LinearSVC on word-level TF-IDF, contributing max-margin diversity."""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((X_train_text.shape[0], 4), dtype=np.float32)
    test_preds = np.zeros((X_test_text.shape[0], 4), dtype=np.float32)

    for train_idx, val_idx in kfold.split(X_train_text, y_train):
        clf = CalibratedClassifierCV(
            LinearSVC(C=0.184, class_weight="balanced", max_iter=500, random_state=SEED),
            cv=2,
        )
        clf.fit(X_train_text[train_idx], y_train[train_idx])
        oof[val_idx] = clf.predict_proba(X_train_text[val_idx])
        test_preds += clf.predict_proba(X_test_text) / n_folds

    return OOFResult(oof=oof, test=test_preds)


# NOTE: additional base models (LR-word, LR-char, NB-SVM, LGB Class-3-vs-rest,
# LGB Class-3-vs-Class-2) follow the same OOF pattern as above and are omitted
# here for brevity; see README for the full 7-model roster.


def stack_oof_predictions(*results: OOFResult) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate each base model's OOF/test probability outputs into the meta-learner's input."""
    stacked_train = np.hstack([r.oof for r in results]).astype(np.float32)
    stacked_test = np.hstack([r.test for r in results]).astype(np.float32)
    return stacked_train, stacked_test


def train_meta_learner(stacked_train, y_train, n_folds: int = 5) -> np.ndarray:
    """Shallow LightGBM meta-learner over the stacked base-model probabilities."""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED + 99)
    meta_oof = np.zeros((stacked_train.shape[0], 4), dtype=np.float32)

    for train_idx, val_idx in kfold.split(stacked_train, y_train):
        model = lgb.LGBMClassifier(**META_LGB_PARAMS)
        model.fit(stacked_train[train_idx], y_train[train_idx])
        meta_oof[val_idx] = model.predict_proba(stacked_train[val_idx])

    return meta_oof


def fit_class_thresholds(
    meta_oof_probs: np.ndarray,
    y_true: np.ndarray,
    bounds=((0.8, 1.2), (0.7, 2.5), (0.85, 1.1), (0.8, 1.25)),
) -> np.ndarray:
    """Optimize per-class probability multipliers (Nelder-Mead) to correct minority-class recall.

    Applied as: argmax(probs * multipliers). Bounds kept tight to avoid overfitting
    the multipliers themselves to the validation fold.
    """
    def neg_macro_f1(multipliers):
        adjusted = meta_oof_probs * multipliers
        preds = adjusted.argmax(axis=1)
        return -f1_score(y_true, preds, average="macro")

    result = minimize(
        neg_macro_f1,
        x0=np.ones(4),
        method="Nelder-Mead",
        bounds=bounds,
    )
    return result.x


def predict_with_thresholds(probs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    return (probs * multipliers).argmax(axis=1)
