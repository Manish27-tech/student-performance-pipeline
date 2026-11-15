"""Tests for type coercion, clipping and the duplicate policy."""

from __future__ import annotations

import pandas as pd

from student import config
from student.clean import audit, clean, duplicate_policy, normalise


def test_normalise_encodes_yes_no(combined_frame: pd.DataFrame) -> None:
    cleaned = normalise(combined_frame)

    assert set(cleaned["internet"].unique()) <= {0, 1}
    assert set(cleaned["higher"].unique()) <= {0, 1}
    assert str(cleaned["internet"].dtype) == "Int64"


def test_normalise_is_idempotent(combined_frame: pd.DataFrame) -> None:
    once = normalise(combined_frame)
    twice = normalise(once)

    pd.testing.assert_frame_equal(once, twice)


def test_normalise_clips_documented_scales(combined_frame: pd.DataFrame) -> None:
    frame = combined_frame.copy()
    frame.loc[0, "absences"] = 500
    frame.loc[1, "G3"] = 40
    cleaned = normalise(frame)

    assert cleaned.loc[0, "absences"] == config.VALID_RANGES["absences"][1]
    assert cleaned.loc[1, "G3"] == config.VALID_RANGES["G3"][1]


def test_duplicate_policy_keeps_rows() -> None:
    policy = duplicate_policy()

    assert policy["drop"] is False
    assert "different students" in policy["reason"]


def test_audit_keeps_identical_rows(combined_frame: pd.DataFrame) -> None:
    frame = pd.concat([combined_frame, combined_frame.iloc[[0]]], ignore_index=True)
    cleaned = normalise(frame)
    log = audit(cleaned)

    assert log["rows_in"] == log["rows_out"] == len(frame)
    assert log["duplicates_kept"] >= 1


def test_audit_reports_subject_summaries(clean_combined: pd.DataFrame) -> None:
    log = audit(clean_combined)

    assert set(log["by_subject"]) == {"mat", "por"}
    for block in log["by_subject"].values():
        assert 0 <= block["mean_g3"] <= 20
        assert 0 <= block["pass_rate"] <= 1
    assert log["pass_count"] == int((clean_combined[config.TARGET] >= config.PASS_MARK).sum())


def test_clean_attaches_log(combined_frame: pd.DataFrame) -> None:
    cleaned, log = clean(combined_frame)

    assert cleaned.attrs["cleaning_log"] is log
    assert log["missing_values"] == {}
