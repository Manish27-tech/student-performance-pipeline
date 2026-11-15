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
