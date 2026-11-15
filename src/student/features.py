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
