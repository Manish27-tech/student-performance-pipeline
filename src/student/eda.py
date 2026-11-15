"""Exploratory data analysis for the student-performance dataset."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .features import add_features  # noqa: E402
from .merge import compare_subjects, first_to_final_trajectory, pairing_report  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def compute_summary(frames: dict[str, pd.DataFrame]) -> dict:
    combined = pd.concat(frames.values(), ignore_index=True)
    engineered = add_features(combined)

    per_subject = {}
    for subject, frame in frames.items():
        passed = frame[config.TARGET] >= config.PASS_MARK
        per_subject[subject] = {
            "label": config.SUBJECT_LABELS[subject],
            "rows": int(len(frame)),
            "mean_g3": round(float(frame[config.TARGET].mean()), 3),
            "median_g3": float(frame[config.TARGET].median()),
            "std_g3": round(float(frame[config.TARGET].std()), 3),
            "pass_rate": round(float(passed.mean()), 4),
            "zero_g3": int((frame[config.TARGET] == 0).sum()),
            "corr_g1_g3": round(float(frame["G1"].corr(frame["G3"])), 4),
            "corr_g2_g3": round(float(frame["G2"].corr(frame["G3"])), 4),
            "trajectory": first_to_final_trajectory(frame),
        }

    def mean_by(column: str) -> dict:
        grouped = engineered.groupby(column, observed=True)[config.TARGET].agg(
            ["mean", "size"])
        return {
            str(level): {"mean_g3": round(float(row["mean"]), 2), "students": int(row["size"])}
            for level, row in grouped.iterrows()
        }

    numeric_columns = [
        "age", "Medu", "Fedu", "studytime", "failures", "absences", "famrel",
        "freetime", "goout", "Dalc", "Walc", "health", "parent_education",
        "support_index", "social_index", "alcohol_index", "G1", "G2", config.TARGET,
    ]
    correlations = (
        engineered[numeric_columns].corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET).sort_values(key=lambda series: series.abs(), ascending=False)
    )

    return {
        "rows": int(len(combined)),
        "per_subject": per_subject,
        "mean_g3": round(float(combined[config.TARGET].mean()), 3),
        "pass_rate": round(float((combined[config.TARGET] >= config.PASS_MARK).mean()), 4),
        "zero_g3": int((combined[config.TARGET] == 0).sum()),
        "mean_g3_by_studytime": mean_by("studytime"),
        "mean_g3_by_failures": mean_by("failures"),
        "mean_g3_by_higher": mean_by("higher"),
        "mean_g3_by_schoolsup": mean_by("schoolsup"),
        "mean_g3_by_romantic": mean_by("romantic"),
        "mean_g3_by_goout": mean_by("goout"),
        "mean_g3_by_effort_band": mean_by("effort_band"),
        "mean_g3_by_subject": mean_by("subject"),
        "absences": {
            "mean": round(float(combined["absences"].mean()), 2),
            "median": float(combined["absences"].median()),
            "p95": float(combined["absences"].quantile(0.95)),
            "max": int(combined["absences"].max()),
            "zero_share": round(float((combined["absences"] == 0).mean()), 4),
        },
        "top_correlations_with_g3": [
            {"feature": name, "correlation": round(float(value), 4)}
            for name, value in correlations.head(10).items()
        ],
        "pairing": pairing_report(frames["mat"], frames["por"]),
    }
