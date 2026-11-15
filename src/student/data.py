"""Dataset acquisition, loading and validation."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config

YES_NO_COLUMNS = [
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def download(force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Fetch the archive once and extract both subject files."""
    archive = config.RAW_DIR / config.ARCHIVE_NAME
    targets = {subject: config.RAW_DIR / name
               for subject, name in config.SUBJECT_FILES.items()}

    if all(path.exists() for path in targets.values()) and not force:
        if verbose:
            print(f"[data] cached: {', '.join(path.name for path in targets.values())}")
        return targets

    if force or not archive.exists():
        if verbose:
            print(f"[data] downloading {config.DATASET_URL}")
        request = urllib.request.Request(
            config.DATASET_URL, headers={"User-Agent": "student-performance-pipeline/0.1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if len(payload) < 20_000:
            raise RuntimeError(f"archive looks truncated ({len(payload)} bytes)")
        archive.write_bytes(payload)
        if verbose:
            print(f"[data] saved {archive.name} (sha256={_sha256(archive)[:12]}...)")

    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        for subject, filename in config.SUBJECT_FILES.items():
            if filename not in available:
                raise RuntimeError(f"{filename} missing from the archive")
            bundle.extract(filename, config.RAW_DIR)
            if verbose:
                print(f"[data] extracted {filename} -> {targets[subject]}")

    if not archive.exists():  # pragma: no cover - defensive
        raise RuntimeError("download failed and no cached archive exists")
    return targets


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_subject(subject: str) -> pd.DataFrame:
    """Load one subject file (semicolon separated, quoted strings)."""
    if subject not in config.SUBJECT_FILES:
        raise ValueError(f"unknown subject '{subject}'")
    path = config.RAW_DIR / config.SUBJECT_FILES[subject]
    if not path.exists():
        download(verbose=False)

    frame = pd.read_csv(path, sep=";")
    frame.columns = [column.strip() for column in frame.columns]
    frame["subject"] = subject
    return frame


def load_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Return both subject frames, keyed by ``mat`` and ``por``."""
    frames = {subject: load_subject(subject) for subject in config.SUBJECT_FILES}
    if verbose:
        for subject, frame in frames.items():
            print(f"[data] {config.SUBJECT_LABELS[subject]:<12} {len(frame):,} students")
        print(f"[data] combined {sum(len(f) for f in frames.values()):,} rows")
    return frames


def combine(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    frames = frames or load_all(verbose=False)
    return pd.concat(frames.values(), ignore_index=True)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class DataValidationError(ValueError):
    """Raised when a subject file does not satisfy the expected schema."""


def validate(df: pd.DataFrame) -> dict:
    """Run schema, scale and domain checks, raising on critical failures."""
    checks: list[dict] = []
    critical: list[str] = []

    def record(name: str, passed: bool, detail: str, is_critical: bool = False) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail,
                       "critical": is_critical})
        if not passed and is_critical:
            critical.append(name)

    missing = [column for column in config.COLUMNS if column not in df.columns]
    record("expected_columns", not missing,
           "all 33 columns present" if not missing else f"missing: {missing}",
           is_critical=True)
    if critical:
        raise DataValidationError("failed critical validation: " + ", ".join(critical))

    record("non_empty", len(df) > 0, f"{len(df):,} rows", is_critical=True)
    record("expected_scale", len(df) >= 350,
           f"{len(df):,} rows (files carry 395 and 649 students)")

    for column in ("G1", "G2", "G3"):
        low, high = config.VALID_RANGES[column]
        values = pd.to_numeric(df[column], errors="coerce")
        inside = bool(values.between(low, high).all())
        record(f"{column}_range", inside and values.notna().all(),
               f"{column} within {low}-{high}", is_critical=True)

    bad_scales = []
    for column, (low, high) in config.VALID_RANGES.items():
        if column in ("G1", "G2", "G3") or column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if not values.between(low, high).all():
            bad_scales.append(column)
    record("bounded_scales", not bad_scales,
           "documented 1-5 / 0-4 scales respected" if not bad_scales
           else f"outside documented range: {bad_scales}")

    odd_yes_no = {}
    for column in YES_NO_COLUMNS:
        levels = set(df[column].dropna().unique())
        if not levels <= {"yes", "no"}:
            odd_yes_no[column] = sorted(levels)
    record("yes_no_domain", not odd_yes_no,
           "binary columns are yes/no" if not odd_yes_no else f"unexpected: {odd_yes_no}")

    zero_grades = int((pd.to_numeric(df[config.TARGET]) == 0).sum())
    record("zero_grades_reported", True,
           f"{zero_grades} students scored 0 — reported, never silently dropped")

    if "subject" in df.columns:
        rates = (df.assign(passed=pd.to_numeric(df[config.TARGET]) >= config.PASS_MARK)
                 .groupby("subject")["passed"].mean().round(3))
        record("pass_rate_plausible", bool(rates.between(0.4, 0.99).all()),
               f"pass rate by subject: {rates.to_dict()}")
        positive = rates.to_dict()
    else:
        rate = float((pd.to_numeric(df[config.TARGET]) >= config.PASS_MARK).mean())
        record("pass_rate_plausible", 0.4 <= rate <= 0.99, f"pass rate {rate:.3f}")
        positive = {"all": round(rate, 3)}

    duplicates = int(df.drop(columns=["subject"]).duplicated().sum()
                     if "subject" in df.columns else df.duplicated().sum())
    key_duplicates = int(df.duplicated(subset=config.MERGE_KEYS).sum())
    record("duplicates_reported", True,
           f"{duplicates} identical rows, {key_duplicates} duplicate pairing keys "
           "(kept: two students can share every attribute)")

    report = {
        "rows": int(len(df)),
        "columns": len(config.COLUMNS),
        "pass_rate": positive,
        "zero_grades": zero_grades,
        "duplicate_rows": duplicates,
        "duplicate_merge_keys": key_duplicates,
        "mean_g3": round(float(pd.to_numeric(df[config.TARGET]).mean()), 3),
        "checks": checks,
    }
    if critical:
        failed = [check["check"] for check in checks
                  if check["critical"] and not check["passed"]]
        raise DataValidationError("failed critical validation: " + ", ".join(failed))
    df.attrs["validation"] = report
    return report
