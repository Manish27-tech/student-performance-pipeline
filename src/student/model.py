"""Regression, pass/fail classification, leakage scenarios and transfer tests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from . import config  # noqa: E402
from .clean import clean  # noqa: E402
from .features import (  # noqa: E402
    add_features,
    build_feature_matrix,
    feature_columns,
    pass_target,
)

plt.rcParams["figure.autolayout"] = True


# --------------------------------------------------------------------------- #
# Preprocessing and candidates
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                 sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)],
        remainder="drop", verbose_feature_names_out=True,
    )


def regression_candidates() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            random_state=config.RANDOM_STATE,
        ),
    }


def classification_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            random_state=config.RANDOM_STATE,
        ),
    }


REGRESSION_SPACES = {
    "ridge": {"model__alpha": loguniform(1e-1, 1e3)},
    "random_forest": {
        "model__n_estimators": randint(250, 650),
        "model__max_depth": [None, 10, 16],
        "model__min_samples_leaf": randint(1, 7),
        "model__max_features": [1.0, "sqrt", 0.6],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(150, 450),
        "model__learning_rate": loguniform(0.01, 0.15),
        "model__max_depth": randint(2, 5),
        "model__subsample": [0.7, 0.85, 1.0],
    },
}

CLASSIFICATION_SPACES = {
    "logistic_regression": {"model__C": loguniform(1e-2, 1e2)},
    "random_forest": {
        "model__n_estimators": randint(250, 650),
        "model__max_depth": [None, 10, 16],
        "model__min_samples_leaf": randint(1, 7),
    },
    "gradient_boosting": {
        "model__n_estimators": randint(150, 450),
        "model__learning_rate": loguniform(0.01, 0.15),
        "model__max_depth": randint(2, 5),
    },
}


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, predictions) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 4),
        "mae": round(float(mean_absolute_error(y_true, predictions)), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
    }


def compare_regressors(X_train, y_train,
                       grade_set: bool | str = False) -> list[dict]:
    numeric, categorical = feature_columns(grade_set)
    folds = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error",
               "neg_mae": "neg_mean_absolute_error"}

    rows = []
    for name, estimator in regression_candidates().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=1, error_score="raise")
        rows.append({
            "model": name,
            "r2_mean": round(float(scores["test_r2"].mean()), 4),
            "r2_std": round(float(scores["test_r2"].std()), 4),
            "rmse_mean": round(float(-scores["test_neg_rmse"].mean()), 4),
            "mae_mean": round(float(-scores["test_neg_mae"].mean()), 4),
            "fit_seconds": round(float(scores["fit_time"].mean()), 2),
        })
    return sorted(rows, key=lambda row: row["r2_mean"], reverse=True)


def tune_regressor(name, X_train, y_train,
                   grade_set: bool | str = False) -> tuple[Pipeline, dict]:
    numeric, categorical = feature_columns(grade_set)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", regression_candidates()[name]),
    ])
    search = RandomizedSearchCV(
        pipeline, REGRESSION_SPACES[name], n_iter=config.SEARCH_ITER, scoring="r2",
        cv=KFold(n_splits=config.SEARCH_FOLDS, shuffle=True,
                 random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    params = {key.replace("model__", ""):
              (value.item() if hasattr(value, "item") else value)
              for key, value in search.best_params_.items()}
    return search.best_estimator_, params


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compare_classifiers(X_train, y_train,
                        grade_set: bool | str = False) -> list[dict]:
    numeric, categorical = feature_columns(grade_set)
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "precision": "precision",
               "recall": "recall"}

    rows = []
    for name, estimator in classification_candidates().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=1, error_score="raise")
        rows.append({
            "model": name,
            "roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "f1_mean": round(float(scores["test_f1"].mean()), 4),
            "precision_mean": round(float(scores["test_precision"].mean()), 4),
            "recall_mean": round(float(scores["test_recall"].mean()), 4),
            "fit_seconds": round(float(scores["fit_time"].mean()), 2),
        })
    return sorted(rows, key=lambda row: row["roc_auc_mean"], reverse=True)


def tune_classifier(name, X_train, y_train,
                    grade_set: bool | str = False) -> tuple[Pipeline, dict]:
    numeric, categorical = feature_columns(grade_set)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", classification_candidates()[name]),
    ])
    search = RandomizedSearchCV(
        pipeline, CLASSIFICATION_SPACES[name], n_iter=config.SEARCH_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=config.SEARCH_FOLDS, shuffle=True,
                           random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    params = {key.replace("model__", ""):
              (value.item() if hasattr(value, "item") else value)
              for key, value in search.best_params_.items()}
    return search.best_estimator_, params
