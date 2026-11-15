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
