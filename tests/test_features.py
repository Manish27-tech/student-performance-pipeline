"""Tests for engineered features and the leakage contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from student import config
from student.features import (
    add_features,
    add_grade_features,
    add_indices,
    build_feature_matrix,
    feature_columns,
    pass_target,
)


def test_parent_education_is_the_sum(clean_combined: pd.DataFrame) -> None:
    engineered = add_indices(clean_combined)
    expected = engineered["Medu"] + engineered["Fedu"]
    pd.testing.assert_series_equal(engineered["parent_education"], expected,
                                   check_names=False)


def test_support_and_alcohol_indices(clean_combined: pd.DataFrame) -> None:
    engineered = add_indices(clean_combined)

    assert engineered["support_index"].between(0, 3).all()
    assert engineered["social_index"].between(2, 10).all()
    assert engineered["alcohol_index"].between(2, 10).all()


def test_study_balance_subtracts_social_time(clean_combined: pd.DataFrame) -> None:
    engineered = add_indices(clean_combined)
    expected = engineered["studytime"] - (engineered["freetime"] + engineered["goout"]) / 2
    pd.testing.assert_series_equal(engineered["study_balance"], expected,
                                   check_names=False)


def test_absence_flag_matches_absences(clean_combined: pd.DataFrame) -> None:
    engineered = add_indices(clean_combined)

    assert (engineered["absence_flag"] == (engineered["absences"] > 0).astype(int)).all()
    assert set(engineered["effort_band"]) <= {"low", "medium", "high"}


def test_grade_features_track_the_trajectory(clean_combined: pd.DataFrame) -> None:
    engineered = add_grade_features(clean_combined)
    expected_mean = (engineered["G1"] + engineered["G2"]) / 2

    pd.testing.assert_series_equal(engineered["grade_mean"], expected_mean,
                                   check_names=False)
    assert (engineered["G1_to_G2"] == engineered["G2"] - engineered["G1"]).all()
    assert set(engineered["grade_trend"]) <= {"improving", "stable", "declining"}


def test_feature_columns_switch_grades_off_by_default() -> None:
    default_numeric, _ = feature_columns()
    leaked_numeric, _ = feature_columns(include_grades=True)
    g1_numeric, _ = feature_columns("g1")

    assert "G1" not in default_numeric
    assert "G2" not in default_numeric
    assert "grade_mean" not in default_numeric
    assert {"G1", "G2", "grade_mean", "G1_to_G2"} <= set(leaked_numeric)
    assert len(leaked_numeric) == len(default_numeric) + 4
    # The middle scenario must be genuinely different from both extremes.
    assert "G1" in g1_numeric
    assert "G2" not in g1_numeric
    assert len(g1_numeric) == len(default_numeric) + 1


def test_feature_matrix_respects_the_scenario(clean_combined: pd.DataFrame) -> None:
    early_X, early_y = build_feature_matrix(clean_combined, include_grades=False)
    late_X, late_y = build_feature_matrix(clean_combined, include_grades=True)

    assert "G1" not in early_X.columns
    assert "G1" in late_X.columns
    assert late_X.shape[1] == early_X.shape[1] + 5  # four numeric + grade_trend
    assert early_y.equals(late_y)
    assert not early_X.isna().all(axis=1).any()
    assert np.isfinite(early_X.select_dtypes(include=[np.number]).to_numpy()).all()


def test_pass_target_uses_the_documented_threshold(clean_combined: pd.DataFrame) -> None:
    target = pass_target(clean_combined)

    assert target.dtype == int
    assert (target == (clean_combined[config.TARGET] >= config.PASS_MARK).astype(int)).all()


def test_add_features_is_a_superset_of_both_steps(clean_combined: pd.DataFrame) -> None:
    engineered = add_features(clean_combined)

    assert "parent_education" in engineered.columns
    assert "grade_mean" in engineered.columns


def test_no_grade_derived_feature_leaks_into_the_early_scenario() -> None:
    numeric, categorical = feature_columns()
    combined = set(numeric) | set(categorical)

    assert not {"G1", "G2", "grade_mean", "G1_to_G2", "grade_trend"} & combined
