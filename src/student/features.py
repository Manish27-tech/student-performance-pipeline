"""Feature engineering for the student-performance models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EFFORT_BINS = [0, 1.5, 2.5, 4]
EFFORT_LABELS = ["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Indices
# --------------------------------------------------------------------------- #
def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the many 1-5 Likert columns into interpretable indices.

    The questionnaire contains 13 ordinal items with overlapping meaning. Summing
    them where the documentation groups them (parental education, institutional
    support, social life, alcohol) removes noise without inventing information.
    """
    out = df.copy()

    out["parent_education"] = out["Medu"] + out["Fedu"]
    out["support_index"] = out["schoolsup"] + out["famsup"] + out["paid"]
    out["social_index"] = out["goout"] + out["freetime"]
    out["alcohol_index"] = out["Dalc"] + out["Walc"]
    out["study_balance"] = out["studytime"] - (out["freetime"] + out["goout"]) / 2
    out["absence_flag"] = (out["absences"] > 0).astype(int)
    out["effort_band"] = pd.cut(
        out["studytime"], bins=EFFORT_BINS, labels=EFFORT_LABELS, right=True
    ).astype(str)

    return out


# --------------------------------------------------------------------------- #
# Grade trajectory
# --------------------------------------------------------------------------- #
def add_grade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the G1 -> G2 movement without peeking at the target.

    ``grade_mean`` and ``G1_to_G2`` are legitimate features for a third-period
    forecast. They are kept in a separate step so the leakage scenarios can switch
    them off cleanly.
    """
    out = df.copy()

    out["grade_mean"] = (out["G1"] + out["G2"]) / 2
    out["G1_to_G2"] = out["G2"] - out["G1"]
    out["grade_trend"] = np.select(
        [out["G1_to_G2"] > 0, out["G1_to_G2"] == 0],
        ["improving", "stable"],
        default="declining",
    )
    return out


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature step."""
    return add_grade_features(add_indices(df))


def feature_columns(include_grades: bool | str = False) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) lists for a leakage scenario.

    ``include_grades`` accepts three meaningful values: ``False`` (no prior grades),
    ``"g1"`` (first period only) and ``True``/``"g1_g2"`` (both prior periods plus
    their derived statistics).
    """
    numeric = [*config.NUMERIC_FEATURES, *config.ENGINEERED_NUMERIC]
    categorical = [*config.CATEGORICAL_FEATURES, *config.ENGINEERED_CATEGORICAL]

    if include_grades in (True, "g1_g2"):
        numeric = [*numeric, "G1", "G2", "grade_mean", "G1_to_G2"]
    elif include_grades == "g1":
        numeric = [*numeric, "G1"]

    if include_grades in (True, "g1", "g1_g2"):
        # ``grade_trend`` is built from the previous grades, so it can only join
        # the scenario that is allowed to see them.
        categorical = [*categorical, *config.GRADE_CATEGORICAL]
    return numeric, categorical


def build_feature_matrix(df: pd.DataFrame, include_grades: bool | str = False
                         ) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into model inputs and the G3 grade."""
    engineered = add_features(df)
    numeric, categorical = feature_columns(include_grades)
    features = engineered[numeric + categorical].copy()

    # scikit-learn imputers reject pandas' nullable ``pd.NA``; hand categoricals
    # over as object columns with numpy NaN instead.
    for column in categorical:
        features[column] = (
            features[column].astype(object).where(features[column].notna(), np.nan)
        )
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)

    target = pd.to_numeric(engineered[config.TARGET], errors="coerce").astype(int)
    return features, target


def pass_target(df: pd.DataFrame) -> pd.Series:
    """Binary pass/fail label at the documented 10/20 threshold."""
    grades = pd.to_numeric(df[config.TARGET], errors="coerce")
    return (grades >= config.PASS_MARK).astype(int)
