"""Tests for the pairing logic and the paired comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from student import config
from student.merge import (
    compare_subjects,
    first_to_final_trajectory,
    naive_merge,
    pair_students,
    pairing_report,
)


def test_pairing_uses_the_documented_keys(mat_frame, por_frame) -> None:
    pairs = pair_students(mat_frame, por_frame)

    assert 0 < len(pairs) <= len(por_frame)
    assert "G3_mat" in pairs.columns
    assert "G3_por" in pairs.columns
    assert pairs["pair_id"].is_unique
    assert set(pairs["pair_id"]) == set(range(1, len(pairs) + 1))


def test_naive_merge_matches_far_fewer_students(mat_frame, por_frame) -> None:
    pairs = pair_students(mat_frame, por_frame)
    naive = naive_merge(mat_frame, por_frame)

    # Adding equality constraints can only shrink an inner join.
    assert len(naive) <= len(pairs)
    assert len(config.ALL_SHARED_COLUMNS) > len(config.MERGE_KEYS)


def test_pairing_report_documents_both_attempts(mat_frame, por_frame) -> None:
    report = pairing_report(mat_frame, por_frame)

    assert report["mat_rows"] == len(mat_frame)
    assert report["por_rows"] == len(por_frame)
    assert report["pairs"] == len(pair_students(mat_frame, por_frame))
    assert report["naive_merge_rows"] == len(naive_merge(mat_frame, por_frame))
    assert report["keys_used"] == config.MERGE_KEYS


def test_comparison_partitions_every_pair(mat_frame, por_frame) -> None:
    pairs = pair_students(mat_frame, por_frame)
    comparison = compare_subjects(pairs)

    assert comparison["improved"] + comparison["worsened"] + comparison["tied"] == len(pairs)
    assert comparison["improved_share"] == pytest.approx(
        comparison["improved"] / len(pairs))
    assert 0.0 <= comparison["sign_test_p"] <= 1.0
    assert 0.0 <= comparison["wilcoxon_p"] <= 1.0
    assert -1.0 <= comparison["grade_correlation"] <= 1.0


def test_comparison_is_symmetric_in_its_means(mat_frame, por_frame) -> None:
    pairs = pair_students(mat_frame, por_frame)
    comparison = compare_subjects(pairs)

    recomputed = (pairs["G3_por"] - pairs["G3_mat"]).mean()
    assert comparison["mean_delta"] == pytest.approx(recomputed, abs=1e-3)
    assert comparison["mean_por"] - comparison["mean_mat"] == pytest.approx(
        comparison["mean_delta"], abs=1e-3)


def test_largest_movers_are_reported(mat_frame, por_frame) -> None:
    comparison = compare_subjects(pair_students(mat_frame, por_frame))

    assert len(comparison["largest_gains"]) <= 5
    assert len(comparison["largest_drops"]) <= 5
    if comparison["largest_gains"]:
        top = comparison["largest_gains"][0]
        assert top["delta_por_minus_mat"] >= 0


def test_trajectory_summary(mat_frame: pd.DataFrame) -> None:
    trajectory = first_to_final_trajectory(mat_frame)

    assert trajectory["rows"] == len(mat_frame)
    assert 0 <= trajectory["mean_g3"] <= 20
    assert trajectory["zero_g3"] == int((mat_frame["G3"] == 0).sum())
