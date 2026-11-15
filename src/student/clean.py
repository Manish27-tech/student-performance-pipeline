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
