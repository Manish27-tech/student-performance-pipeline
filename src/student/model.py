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


# --------------------------------------------------------------------------- #
# Leakage scenarios
# --------------------------------------------------------------------------- #
def run_scenarios(df: pd.DataFrame, y_regression: pd.Series, y_pass: pd.Series,
                  train_index, test_index) -> dict:
    """Fit regression and classification under each grade-availability scenario."""
    results: dict[str, dict] = {}

    for scenario, description in config.SCENARIOS.items():
        grade_set: bool | str = {"no_grades": False, "with_g1": "g1",
                                 "with_g1_g2": "g1_g2"}[scenario]
        X, _ = build_feature_matrix(df, include_grades=grade_set)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        reg_train, reg_test = y_regression.iloc[train_index], y_regression.iloc[test_index]
        cls_train, cls_test = y_pass.iloc[train_index], y_pass.iloc[test_index]

        reg_comparison = compare_regressors(X_train, reg_train, grade_set)
        best_regressor, reg_params = tune_regressor(
            reg_comparison[0]["model"], X_train, reg_train, grade_set)
        best_regressor.fit(X_train, reg_train)
        reg_predictions = best_regressor.predict(X_test)

        cls_comparison = compare_classifiers(X_train, cls_train, grade_set)
        best_classifier, cls_params = tune_classifier(
            cls_comparison[0]["model"], X_train, cls_train, grade_set)
        best_classifier.fit(X_train, cls_train)
        cls_probabilities = best_classifier.predict_proba(X_test)[:, 1]

        results[scenario] = {
            "description": description,
            "include_grades": grade_set,
            "features": int(X.shape[1]),
            "regression_comparison": reg_comparison,
            "regression_model": reg_comparison[0]["model"],
            "regression_params": reg_params,
            "regression": regression_metrics(reg_test, reg_predictions),
            "classification_comparison": cls_comparison,
            "classification_model": cls_comparison[0]["model"],
            "classification_params": cls_params,
            "classification": classification_metrics(cls_test, cls_probabilities),
            "_y_test": reg_test,
            "_reg_pred": reg_predictions,
            "_cls_test": cls_test,
            "_cls_prob": cls_probabilities,
            "_reg_pipeline": best_regressor,
            "_cls_pipeline": best_classifier,
        }
        print(f"[model] {scenario:<12} R2={results[scenario]['regression']['r2']:.3f} "
              f"RMSE={results[scenario]['regression']['rmse']:.2f} "
              f"| AUC={results[scenario]['classification']['roc_auc']:.3f} "
              f"F1={results[scenario]['classification']['f1']:.3f}")
    return results


# --------------------------------------------------------------------------- #
# Cross-subject transfer
# --------------------------------------------------------------------------- #
def cross_subject_transfer(frames: dict[str, pd.DataFrame]) -> dict:
    """Does a no-grade model trained on one subject work on the other?"""
    transfer: dict[str, dict] = {}
    name = "gradient_boosting"

    for train_subject, test_subject in (("mat", "por"), ("por", "mat")):
        # Clean both sides defensively: yes/no columns must be 0/1 before the
        # indices are summed, otherwise ``support_index`` becomes a string.
        train_frame = clean(frames[train_subject])[0]
        test_frame = clean(frames[test_subject])[0]
        X_train, y_train = build_feature_matrix(train_frame, include_grades=False)
        X_test, y_test = build_feature_matrix(test_frame, include_grades=False)
        pass_train, pass_test = (pass_target(train_frame), pass_target(test_frame))

        regressor = regression_candidates()[name]
        classifier = classification_candidates()[name]
        numeric, categorical = feature_columns(include_grades=False)
        reg_pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", regressor),
        ]).fit(X_train, y_train)
        cls_pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", classifier),
        ]).fit(X_train, pass_train)

        regression = regression_metrics(y_test, reg_pipeline.predict(X_test))
        classification = classification_metrics(
            pass_test, cls_pipeline.predict_proba(X_test)[:, 1])

        transfer[f"{train_subject}_to_{test_subject}"] = {
            "train_subject": train_subject,
            "test_subject": test_subject,
            "regression": regression,
            "classification": classification,
        }
        print(f"[model] transfer {train_subject}->{test_subject}: "
              f"R2={regression['r2']:.3f} AUC={classification['roc_auc']:.3f}")
    return transfer


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(config.FIGURE_DIR / filename, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_leakage(scenarios: dict) -> None:
    frame = pd.DataFrame([
        {"scenario": name, "R2": block["regression"]["r2"],
         "AUC": block["classification"]["roc_auc"]}
        for name, block in scenarios.items()
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric, colour, label in (
        (axes[0], "R2", "#4C72B0", "Regression R² (final grade)"),
        (axes[1], "AUC", "#DD8452", "Classification ROC AUC (pass/fail)"),
    ):
        sns.barplot(data=frame, x="scenario", y=metric, ax=ax, color=colour)
        ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=9)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Performance is almost entirely bought with previous grades",
                 fontsize=13)
    _save(fig, "09_leakage_scenarios.png")


def plot_predicted_vs_actual(scenarios: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, scenario in zip(axes, ("no_grades", "with_g1_g2"), strict=True):
        block = scenarios[scenario]
        truth, prediction = block["_y_test"], block["_reg_pred"]
        ax.scatter(truth, prediction, s=16, alpha=0.5, color="#4C72B0",
                   edgecolor="none")
        ax.plot([0, 20], [0, 20], linestyle="--", color="grey", linewidth=1)
        ax.set_title(f"{scenario} — R² = {block['regression']['r2']:.2f}")
        ax.set_xlabel("Actual G3")
        ax.set_ylabel("Predicted G3")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
    fig.suptitle("Without prior grades the model regresses toward the mean",
                 fontsize=13)
    _save(fig, "10_predicted_vs_actual.png")


def plot_residuals(scenarios: dict) -> None:
    block = scenarios["no_grades"]
    residuals = np.asarray(block["_y_test"]) - np.asarray(block["_reg_pred"])

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    sns.histplot(residuals, bins=25, ax=ax, color="#4C72B0", kde=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_title(f"Errors are wide: MAE = {block['regression']['mae']:.2f} points")
    _save(fig, "11_residuals.png")


def plot_confusion(scenarios: dict) -> None:
    matrix = scenarios["no_grades"]["classification"]["confusion_matrix"]
    array = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(array, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Pass/fail confusion without prior grades")
    _save(fig, "12_confusion.png")


def feature_importance(pipeline: Pipeline, top_n: int = 15) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:  # pragma: no cover
        return []

    order = np.argsort(values)[::-1][:top_n]
    return [{"feature": names[index].split("__", 1)[-1],
             "importance": round(float(values[index]), 5)} for index in order]


def plot_importance(importances: list[dict]) -> None:
    frame = pd.DataFrame(importances).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=frame, x="importance", y="feature", hue="feature", legend=False,
                palette="crest", ax=ax)
    ax.set_title("What the early-warning model leans on")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, "13_feature_importance.png")


def plot_transfer(transfer: dict) -> None:
    frame = pd.DataFrame([
        {"direction": key.replace("_to_", " → "), "R2": block["regression"]["r2"],
         "AUC": block["classification"]["roc_auc"]}
        for key, block in transfer.items()
    ]).melt(id_vars=["direction"], var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    sns.barplot(data=frame, x="direction", y="score", hue="metric", ax=ax,
                palette=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Train → test subject")
    ax.set_title("Cross-subject transfer without grades")
    ax.legend(title="")
    _save(fig, "14_cross_subject.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(df: pd.DataFrame, frames: dict[str, pd.DataFrame] | None = None,
          seed: int = config.RANDOM_STATE) -> dict:
    """Run the full modelling workflow and persist metrics, figures and models."""
    started = time.perf_counter()

    target_regression = pd.to_numeric(df[config.TARGET], errors="coerce").astype(int)
    target_pass = pass_target(df)

    indices = np.arange(len(df))
    train_index, test_index = train_test_split(
        indices, test_size=config.TEST_SIZE, random_state=seed, stratify=target_pass
    )

    print(f"[model] students={len(df):,} train={len(train_index):,} "
          f"test={len(test_index):,}")

    scenarios = run_scenarios(df, target_regression, target_pass, train_index, test_index)
    transfer = cross_subject_transfer(frames) if frames else {}
    importances = feature_importance(scenarios["no_grades"]["_reg_pipeline"])

    # Figures consume the private prediction arrays, so draw before stripping them.
    plot_leakage(scenarios)
    plot_predicted_vs_actual(scenarios)
    plot_residuals(scenarios)
    plot_confusion(scenarios)
    plot_importance(importances)
    if transfer:
        plot_transfer(transfer)

    bundle = {
        "regression": scenarios["no_grades"]["_reg_pipeline"],
        "classifier": scenarios["no_grades"]["_cls_pipeline"],
        "features": feature_columns(include_grades=False),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    public_scenarios = {
        name: {key: value for key, value in block.items() if not key.startswith("_")}
        for name, block in scenarios.items()
    }

    metrics = {
        "generated_at": bundle["trained_at"],
        "dataset": {
            "rows": int(len(df)),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "mean_g3": round(float(target_regression.mean()), 3),
            "pass_rate": round(float(target_pass.mean()), 4),
        },
        "scenarios": public_scenarios,
        "leakage_table": [
            {"scenario": name,
             "regression_r2": block["regression"]["r2"],
             "regression_rmse": block["regression"]["rmse"],
             "classification_auc": block["classification"]["roc_auc"],
             "classification_f1": block["classification"]["f1"]}
            for name, block in public_scenarios.items()
        ],
        "cross_subject_transfer": transfer,
        "primary_regression": public_scenarios["no_grades"]["regression"],
        "primary_classification": public_scenarios["no_grades"]["classification"],
        "feature_importance": importances,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    metrics_path = config.REPORT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle["metrics"] = metrics
    model_path = config.ARTIFACT_DIR / "student_model.joblib"
    joblib.dump(bundle, model_path)

    print(f"[model] primary R2={metrics['primary_regression']['r2']:.3f} "
          f"RMSE={metrics['primary_regression']['rmse']:.2f} "
          f"| AUC={metrics['primary_classification']['roc_auc']:.3f}")
    print(f"[model] metrics -> {metrics_path.relative_to(config.ROOT)}")
    print(f"[model] model   -> {model_path.relative_to(config.ROOT)}")
    return metrics


def load_model(path=None) -> dict:
    path = path or (config.ARTIFACT_DIR / "student_model.joblib")
    return joblib.load(path)


def predict(df: pd.DataFrame, bundle: dict | None = None) -> np.ndarray:
    """Predict the final grade with the saved early-warning model."""
    bundle = bundle or load_model()
    numeric, categorical = bundle["features"]
    features = add_features(df)[numeric + categorical].copy()
    for column in categorical:
        features[column] = features[column].astype(object).where(
            features[column].notna(), np.nan)
    return bundle["regression"].predict(features)
