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


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare_subjects(pairs: pd.DataFrame) -> dict:
    """Paired comparison of the third-period grade in both subjects.

    The same 382 people sit on both sides of every test, so the analysis is
    within-student: how much better or worse does a person do in Portuguese?
    """
    mat_scores = pd.to_numeric(pairs["G3_mat"], errors="coerce")
    por_scores = pd.to_numeric(pairs["G3_por"], errors="coerce")
    delta = por_scores - mat_scores
    valid = delta.dropna()

    improved = int((valid > 0).sum())
    worsened = int((valid < 0).sum())
    tied = int((valid == 0).sum())
    discordant = improved + worsened

    sign_test = stats.binomtest(improved, discordant, p=0.5) if discordant else None
    wilcoxon = stats.wilcoxon(valid, zero_method="wilcox") if (valid != 0).any() else None
    standardised = float(valid.mean() / valid.std(ddof=1)) if valid.std(ddof=1) else 0.0

    pairs_with_gap = pairs.assign(delta_por_minus_mat=delta).dropna(
        subset=["delta_por_minus_mat"]
    )

    return {
        "pairs": int(len(pairs)),
        "mean_mat": round(float(mat_scores.mean()), 3),
        "mean_por": round(float(por_scores.mean()), 3),
        "mean_delta": round(float(delta.mean()), 3),
        "median_delta": float(delta.median()),
        "std_delta": round(float(delta.std(ddof=1)), 3),
        "improved": improved,
        "worsened": worsened,
        "tied": tied,
        "improved_share": round(improved / len(valid), 4),
        "sign_test_p": float(sign_test.pvalue) if sign_test else None,
        "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon else None,
        "cohen_d_paired": round(standardised, 3),
        "grade_correlation": round(float(mat_scores.corr(por_scores)), 4),
        "pass_rate_mat": round(float((mat_scores >= config.PASS_MARK).mean()), 4),
        "pass_rate_por": round(float((por_scores >= config.PASS_MARK).mean()), 4),
        "largest_gains": pairs_with_gap.nlargest(5, "delta_por_minus_mat")[
            ["pair_id", "G3_mat", "G3_por", "delta_por_minus_mat"]
        ].to_dict("records"),
        "largest_drops": pairs_with_gap.nsmallest(5, "delta_por_minus_mat")[
            ["pair_id", "G3_mat", "G3_por", "delta_por_minus_mat"]
        ].to_dict("records"),
    }


def first_to_final_trajectory(frame: pd.DataFrame) -> dict:
    """Average G1 -> G2 -> G3 path for one subject file."""
    return {
        "rows": int(len(frame)),
        "mean_g1": round(float(frame["G1"].mean()), 3),
        "mean_g2": round(float(frame["G2"].mean()), 3),
        "mean_g3": round(float(frame["G3"].mean()), 3),
        "mean_g1_to_g2": round(float((frame["G2"] - frame["G1"]).mean()), 3),
        "mean_g2_to_g3": round(float((frame["G3"] - frame["G2"]).mean()), 3),
        "zero_g3": int((frame["G3"] == 0).sum()),
    }
