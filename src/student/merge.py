"""Pairing mathematics and Portuguese students and comparing their grades."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def pair_students(mat: pd.DataFrame, por: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two subject files on the documented 13 pairing keys.

    ``student-merge.R`` in the archive is the authority here: it matches on school,
    sex, age, address, family size, parent status, parental education and jobs,
    reason for the school, nursery and internet. Those are the attributes recorded
    once per student, so they identify a person across the two files.
    """
    pairs = mat.merge(por, on=config.MERGE_KEYS, how="inner",
                      suffixes=("_mat", "_por"))
    if pairs.empty:
        raise ValueError("the two subject files share no pairing keys")
    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = np.arange(1, len(pairs) + 1)
    return pairs


def naive_merge(mat: pd.DataFrame, por: pd.DataFrame) -> pd.DataFrame:
    """Join on every shared non-grade column — the tempting mistake.

    Adding *more* equality constraints can only shrink the match set, and here it
    does so dramatically: 39 students instead of 382. Two students with identical
    demographic sheets are not necessarily the same person, and the questionnaire
    items (alcohol, going out, health) drift between the two school years.
    """
    return mat.merge(por, on=config.ALL_SHARED_COLUMNS, how="inner",
                     suffixes=("_mat", "_por"))


def pairing_report(mat: pd.DataFrame, por: pd.DataFrame) -> dict:
    pairs = pair_students(mat, por)
    naive = naive_merge(mat, por)
    return {
        "mat_rows": int(len(mat)),
        "por_rows": int(len(por)),
        "pairs": int(len(pairs)),
        "matched_share_of_mat": round(len(pairs) / len(mat), 4),
        "matched_share_of_por": round(len(pairs) / len(por), 4),
        "naive_merge_rows": int(len(naive)),
        "keys_used": list(config.MERGE_KEYS),
        "all_shared_columns_tried": len(config.ALL_SHARED_COLUMNS),
    }
