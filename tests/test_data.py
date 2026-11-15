"""Tests for schema validation and file loading."""

from __future__ import annotations

import pandas as pd
import pytest

from student import config, data


def test_validate_reports_scale_and_pass_rates(combined_frame: pd.DataFrame) -> None:
    report = data.validate(combined_frame)

    assert report["rows"] == len(combined_frame)
    assert set(report["pass_rate"]) == {"mat", "por"}
    assert all(0 < rate < 1 for rate in report["pass_rate"].values())
    assert report["duplicate_rows"] == 0


def test_validate_raises_on_missing_column(mat_frame: pd.DataFrame) -> None:
    with pytest.raises(data.DataValidationError, match="expected_columns"):
        data.validate(mat_frame.drop(columns=["absences"]))


def test_validate_raises_on_grade_out_of_range(mat_frame: pd.DataFrame) -> None:
    frame = mat_frame.copy()
    frame.loc[0, "G3"] = 25
    with pytest.raises(data.DataValidationError, match="G3_range"):
        data.validate(frame)


def test_validate_reports_zero_grades_without_raising(mat_frame: pd.DataFrame) -> None:
    frame = mat_frame.copy()
    frame.loc[0, "G3"] = 0
    report = data.validate(frame)

    zero_check = next(check for check in report["checks"]
                      if check["check"] == "zero_grades_reported")
    assert zero_check["passed"]
    assert report["zero_grades"] >= 1


def test_validate_flags_bad_yes_no_values(mat_frame: pd.DataFrame) -> None:
    frame = mat_frame.copy()
    frame.loc[0, "internet"] = "maybe"
    report = data.validate(frame)

    domain = next(check for check in report["checks"]
                  if check["check"] == "yes_no_domain")
    assert not domain["passed"]


def test_load_subject_reads_semicolon_files(tmp_path) -> None:
    source = pd.DataFrame({
        column: ["x"] if column not in ("G1", "G2", "G3") else [10]
        for column in config.COLUMNS
    })
    path = tmp_path / "student-mat.csv"
    source.to_csv(path, sep=";", index=False, quoting=1)

    frame = pd.read_csv(path, sep=";")
    assert len(frame) == 1
    assert frame.loc[0, "G3"] == 10
    assert list(frame.columns) == config.COLUMNS


def test_combine_keeps_subject_labels(mat_frame, por_frame) -> None:
    combined = data.combine({"mat": mat_frame, "por": por_frame})

    assert len(combined) == len(mat_frame) + len(por_frame)
    assert set(combined["subject"]) == {"mat", "por"}
