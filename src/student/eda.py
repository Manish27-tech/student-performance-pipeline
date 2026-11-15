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


# --------------------------------------------------------------------------- #
# Distribution figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str) -> None:
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figure -> {path.relative_to(config.ROOT)}")


def plot_grade_distributions(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (subject, frame) in zip(axes, frames.items(), strict=True):
        grades = frame[config.TARGET]
        ax.hist(grades, bins=range(0, 22), color=config.SUBJECT_COLOURS[subject],
                edgecolor="white", align="left")
        ax.axvline(config.PASS_MARK, color="black", linestyle="--", linewidth=1.2)
        ax.text(config.PASS_MARK + 0.2, ax.get_ylim()[1] * 0.9,
                f"pass mark = {config.PASS_MARK}", fontsize=9)
        ax.set_title(f"{config.SUBJECT_LABELS[subject]} (n={len(frame)})")
        ax.set_xlabel("Final grade G3 (0–20)")
        ax.set_ylabel("Students")
    fig.suptitle("Portuguese grades are shifted right and piled against 0/20 bounds",
                 fontsize=13)
    _save(fig, "01_grade_distributions.png")


def plot_trajectory(frames: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    periods = ["G1", "G2", "G3"]

    sample = frames["mat"].sample(25, random_state=config.RANDOM_STATE)
    for _, row in sample.iterrows():
        ax.plot(periods, [row["G1"], row["G2"], row["G3"]], color="grey", alpha=0.18,
                linewidth=1)

    for subject, frame in frames.items():
        means = [frame[period].mean() for period in periods]
        ax.plot(periods, means, marker="o", linewidth=2.6,
                color=config.SUBJECT_COLOURS[subject],
                label=f"{config.SUBJECT_LABELS[subject]} mean")
        for period, value in zip(periods, means, strict=True):
            ax.annotate(f"{value:.2f}", (period, value), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

    ax.set_ylabel("Grade (0–20)")
    ax.set_title("Grade path across the three periods (grey = 25 maths students)")
    ax.legend()
    _save(fig, "02_grade_trajectory.png")


def plot_zero_grades(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for subject, frame in frames.items():
        for value in (0, 1):
            rows.append({
                "subject": config.SUBJECT_LABELS[subject],
                "zero": "G3 = 0" if value == 0 else "G3 > 0",
                "students": int((frame[config.TARGET] == 0).sum() if value == 0
                                else (frame[config.TARGET] > 0).sum()),
            })
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=frame, x="subject", y="students", hue="zero", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Students")
    ax.set_title("Zero finals are real observations, not missing data")
    ax.legend(title="")
    _save(fig, "03_zero_grades.png")


# --------------------------------------------------------------------------- #
# Relationship figures
# --------------------------------------------------------------------------- #
def plot_correlation_heatmap(frames: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(frames.values(), ignore_index=True)
    engineered = add_features(combined)
    columns = [
        "age", "Medu", "Fedu", "studytime", "failures", "absences", "goout",
        "Dalc", "Walc", "health", "parent_education", "support_index", "social_index",
        "alcohol_index", "G1", "G2", config.TARGET,
    ]
    correlation = engineered[columns].corr(numeric_only=True)
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 7.5}, ax=ax)
    ax.set_title("Correlation matrix (both subjects pooled)")
    _save(fig, "04_correlation_heatmap.png")


def plot_g3_by_drivers(frames: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(frames.values(), ignore_index=True)
    engineered = add_features(combined)
    panels = [
        ("failures", "Past failures", "count"),
        ("studytime", "Weekly study time (1–4)", "count"),
        ("higher", "Wants higher education", "count"),
        ("goout", "Goes out with friends (1–5)", "count"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (column, label, kind) in zip(axes.ravel(), panels, strict=True):
        if kind == "count":
            sns.boxplot(data=engineered, x=column, y=config.TARGET, ax=ax,
                        color="#B0C4DE", showfliers=False)
            sns.stripplot(data=engineered, x=column, y=config.TARGET, ax=ax,
                          color="#333333", size=1.6, alpha=0.35, jitter=0.18)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Final grade G3")
    fig.suptitle("Habits and history move the final grade; the spread stays wide",
                 fontsize=13)
    _save(fig, "05_g3_by_drivers.png")


def plot_absences(frames: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(frames.values(), ignore_index=True)
    combined = combined.assign(passed=combined[config.TARGET] >= config.PASS_MARK)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    sns.histplot(combined["absences"], bins=40, ax=axes[0], color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("School absences")
    axes[0].set_ylabel("Students (log scale)")
    axes[0].set_title(f"{combined['absences'].eq(0).mean():.0%} have zero absences")

    sns.boxplot(data=combined, x="passed", y="absences", ax=axes[1],
                showfliers=False, hue="passed", legend=False)
    axes[1].set_xticks([0, 1], ["Failed (<10)", "Passed (>=10)"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Absences")
    axes[1].set_title("Absences are right-skewed and weakly separating")
    _save(fig, "06_absences.png")


def plot_pass_balance(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for subject, frame in frames.items():
        passed = int((frame[config.TARGET] >= config.PASS_MARK).sum())
        rows.append({"subject": config.SUBJECT_LABELS[subject], "outcome": "Pass",
                     "students": passed})
        rows.append({"subject": config.SUBJECT_LABELS[subject], "outcome": "Fail",
                     "students": int(len(frame) - passed)})
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=frame, x="subject", y="students", hue="outcome", ax=ax,
                palette=["#55A868", "#C44E52"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Students")
    ax.set_title("Pass rate differs sharply between the two subjects")
    ax.legend(title="")
    _save(fig, "07_pass_balance.png")
