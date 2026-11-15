"""Cleaning: type coercion, duplicate policy and an auditable summary."""

from __future__ import annotations

import pandas as pd

from . import config
from .data import YES_NO_COLUMNS


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text, encode yes/no columns and force the documented numeric types.

    Idempotent: binary columns that are already 0/1 pass through untouched.
    """
    out = df.copy()

    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].astype("string").str.strip()

    for column in YES_NO_COLUMNS:
        if out[column].dtype == object or str(out[column].dtype) == "string":
            out[column] = out[column].map({"yes": 1, "no": 0})
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    for column in config.VALID_RANGES:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    for column, (low, high) in config.VALID_RANGES.items():
        out[column] = out[column].clip(lower=low, upper=high)

    return out


# --------------------------------------------------------------------------- #
# Duplicate policy
# --------------------------------------------------------------------------- #
def duplicate_policy() -> dict:
    """Explain why identical rows are *kept* in this dataset.

    Two students can genuinely share every recorded attribute, so dropping
    "duplicates" would delete real people. The pairing step is where multiplicity
    matters, and it is handled explicitly there.
    """
    return {
        "drop": False,
        "reason": "identical attribute vectors can be different students; "
                  "pairing multiplicity is handled in merge.py",
    }


def audit(df: pd.DataFrame) -> dict:
    """Describe what cleaning changed without destroying rows."""
    duplicates = int(df.duplicated().sum())
    key_duplicates = int(df.duplicated(subset=config.MERGE_KEYS).sum())
    zero_grades = int((df[config.TARGET] == 0).sum())
    passed = df[config.TARGET] >= config.PASS_MARK

    by_subject = {}
    if "subject" in df.columns:
        for subject, group in df.groupby("subject", observed=True):
            by_subject[str(subject)] = {
                "rows": int(len(group)),
                "mean_g3": round(float(group[config.TARGET].mean()), 3),
                "median_g3": float(group[config.TARGET].median()),
                "pass_rate": round(float((group[config.TARGET] >= config.PASS_MARK).mean()), 4),
            }

    return {
        "rows_in": int(len(df)),
        "rows_out": int(len(df)),
        "duplicates_kept": duplicates,
        "duplicate_merge_keys": key_duplicates,
        "zero_grades": zero_grades,
        "missing_values": {column: int(count) for column, count in df.isna().sum().items()
                           if count},
        "mean_g3": round(float(df[config.TARGET].mean()), 3),
        "median_g3": float(df[config.TARGET].median()),
        "pass_rate": round(float(passed.mean()), 4),
        "pass_count": int(passed.sum()),
        "by_subject": by_subject,
        "duplicate_policy": duplicate_policy(),
        "clipped_columns": [column for column in config.VALID_RANGES],
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise the frame and attach the audit log."""
    cleaned = normalise(df)
    log = audit(cleaned)
    cleaned.attrs["cleaning_log"] = log
    return cleaned, log


def save_processed(df: pd.DataFrame) -> str:
    destination = config.PROCESSED_DIR / "students_clean.csv"
    df.to_csv(destination, index=False)
    return str(destination)
