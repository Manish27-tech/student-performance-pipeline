"""Central configuration: paths, dataset location, merge keys and constants."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
NOTEBOOK_DIR = ROOT / "notebooks"

for _directory in (RAW_DIR, PROCESSED_DIR, REPORT_DIR, FIGURE_DIR, ARTIFACT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# UCI dataset 320 (Cortez & Silva, 2008): secondary-school grades from two
# Portuguese schools, one file per subject.
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
ARCHIVE_NAME = "student.zip"
SUBJECT_FILES = {"mat": "student-mat.csv", "por": "student-por.csv"}
SUBJECT_LABELS = {"mat": "Mathematics", "por": "Portuguese"}

COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3",
]

TARGET = "G3"
PASS_MARK = 10

# The documented pairing keys from `student-merge.R` inside the archive. Using
# *all* shared non-grade columns instead looks safer but collapses the match set
# from 382 to 39 students — see `merge.naive_merge` and the README.
MERGE_KEYS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
]
# Everything the two files share except the three grade columns.
ALL_SHARED_COLUMNS = [column for column in COLUMNS if column not in ("G1", "G2", "G3")]

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences",
]

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic",
    # Present only in the pooled frame; lets the model absorb the level shift
    # between the two subjects instead of pretending they are exchangeable.
    "subject",
]

ENGINEERED_NUMERIC = [
    "parent_education",
    "support_index",
    "social_index",
    "alcohol_index",
    "study_balance",
    "absence_flag",
]
ENGINEERED_CATEGORICAL = ["effort_band"]
# Derived from G1/G2, so it belongs to the grade scenarios only.
GRADE_CATEGORICAL = ["grade_trend"]

# Variables that are only available *after* the first two grading periods. They
# are legitimate for a third-period forecast but catastrophic for an early-warning
# system, so the pipeline quantifies both settings instead of quietly picking one.
GRADE_FEATURES = ["G1", "G2", "grade_mean", "G1_to_G2"]

SCENARIOS = {
    "no_grades": "Demographics, family, habits and absences only (usable early)",
    "with_g1": "Adds the first-period grade (usable after period 1)",
    "with_g1_g2": "Adds both previous grades (strongest, least useful operationally)",
}

# --------------------------------------------------------------------------- #
# Cleaning / validation
# --------------------------------------------------------------------------- #
# Cronbach-style bounded scales used by the dataset documentation.
VALID_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITER = 12
SEARCH_FOLDS = 3

FIG_DPI = 140
SUBJECT_COLOURS = {"mat": "#4C72B0", "por": "#DD8452"}
