"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

YES_NO = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher",
          "internet", "romantic"]


def _frame(size: int, seed: int, offset: int = 0) -> pd.DataFrame:
    """Build a schema-complete frame whose 13 pairing keys are unique per row."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(size):
        code = index + offset
        row = {
            "school": "GP",
            "sex": "M" if code % 2 else "F",
            "age": 15 + (code // 2) % 8,
            "address": "U" if (code // 16) % 2 else "R",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": (code // 32) % 5,
            "Fedu": (code // 160) % 5,
            "Mjob": "other",
            "Fjob": "other",
            "reason": "course",
            "guardian": "mother",
            "traveltime": int(rng.integers(1, 5)),
            "studytime": int(rng.integers(1, 5)),
            "failures": int(rng.choice([0, 0, 0, 1, 2])),
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": int(rng.integers(1, 6)),
            "freetime": int(rng.integers(1, 6)),
            "goout": int(rng.integers(1, 6)),
            "Dalc": int(rng.integers(1, 6)),
            "Walc": int(rng.integers(1, 6)),
            "health": int(rng.integers(1, 6)),
            "absences": int(rng.integers(0, 30)),
        }
        row["G1"] = int(np.clip(rng.integers(5, 19), 0, 20))
        row["G2"] = int(np.clip(row["G1"] + rng.integers(-3, 4), 0, 20))
        row["G3"] = int(np.clip(row["G2"] + rng.integers(-4, 6), 0, 20))
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in YES_NO:
        if column in frame:
            frame[column] = frame[column].astype(object)
    return frame


@pytest.fixture()
def mat_frame() -> pd.DataFrame:
    frame = _frame(120, seed=1)
    frame["subject"] = "mat"
    return frame


@pytest.fixture()
def por_frame() -> pd.DataFrame:
    frame = _frame(100, seed=2)
    frame["subject"] = "por"
    # Same 100 students as maths, different grades: Portuguese is bumped by 2.
    for column in ("G1", "G2", "G3"):
        frame[column] = (frame[column] + 1).clip(upper=20)
    return frame


@pytest.fixture()
def combined_frame(mat_frame: pd.DataFrame, por_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([mat_frame, por_frame], ignore_index=True)


@pytest.fixture()
def clean_combined(combined_frame: pd.DataFrame) -> pd.DataFrame:
    from student.clean import clean

    cleaned, _ = clean(combined_frame)
    return cleaned
