"""Tests for metrics, preprocessing and the leakage scenarios."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from student.features import build_feature_matrix, feature_columns
from student.model import (
    build_preprocessor,
    classification_candidates,
    classification_metrics,
    feature_importance,
    regression_candidates,
    regression_metrics,
)


def test_regression_metrics_on_perfect_predictions() -> None:
    y_true = np.array([10.0, 12.0, 8.0, 15.0])
    metrics = regression_metrics(y_true, y_true)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_classification_metrics_on_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y_true, y_true.astype(float), threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_preprocessor_scales_and_encodes(clean_combined) -> None:
    X, y = build_feature_matrix(clean_combined, include_grades=False)
    numeric, categorical = feature_columns(include_grades=False)
    transformed = build_preprocessor(numeric, categorical).fit_transform(X, y)

    assert transformed.ndim == 2
    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()


def test_early_and_late_pipelines_differ_only_by_grades(clean_combined) -> None:
    early_X, y = build_feature_matrix(clean_combined, include_grades=False)
    late_X, _ = build_feature_matrix(clean_combined, include_grades=True)

    added = set(late_X.columns) - set(early_X.columns)
    assert added == {"G1", "G2", "grade_mean", "G1_to_G2", "grade_trend"}
    assert set(early_X.columns) <= set(late_X.columns)


def test_linear_pipelines_produce_valid_outputs(clean_combined) -> None:
    X, y = build_feature_matrix(clean_combined, include_grades=False)
    numeric, categorical = feature_columns(include_grades=False)
    preprocessor = build_preprocessor(numeric, categorical)

    regressor = Pipeline([("preprocess", preprocessor),
                          ("model", Ridge(alpha=10.0))]).fit(X, y)
    classifier = Pipeline([("preprocess", preprocessor),
                           ("model", LogisticRegression(max_iter=1000))]).fit(
        X, (y >= 10).astype(int))

    predictions = regressor.predict(X)
    probabilities = classifier.predict_proba(X)[:, 1]

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_feature_importance_is_sorted_and_clean(clean_combined) -> None:
    X, y = build_feature_matrix(clean_combined, include_grades=False)
    numeric, categorical = feature_columns(include_grades=False)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", regression_candidates()["gradient_boosting"]),
    ]).fit(X, y)

    importances = feature_importance(pipeline, top_n=6)
    values = [item["importance"] for item in importances]

    assert len(importances) == 6
    assert values == sorted(values, reverse=True)
    assert all("__" not in item["feature"] for item in importances)


def test_candidate_families_are_distinct() -> None:
    regression = regression_candidates()
    classification = classification_candidates()

    assert set(regression) == {"ridge", "random_forest", "gradient_boosting"}
    assert set(classification) == {"logistic_regression", "random_forest",
                                   "gradient_boosting"}
    assert len({type(model).__name__ for model in regression.values()}) == 3
    assert len({type(model).__name__ for model in classification.values()}) == 3


def test_scenario_descriptions_cover_three_settings() -> None:
    from student import config

    assert set(config.SCENARIOS) == {"no_grades", "with_g1", "with_g1_g2"}
    assert all(description for description in config.SCENARIOS.values())
    assert pytest.approx(config.PASS_MARK) == 10
