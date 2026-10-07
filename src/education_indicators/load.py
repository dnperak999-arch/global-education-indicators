"""Read the local education extract without interpreting missing values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from education_indicators.validate import ValidationError


def project_root() -> Path:
    """Return the ``global-education-indicators`` directory."""
    return Path(__file__).resolve().parents[2]


def default_raw_csv_path() -> Path:
    """Path of the local education extract. The file is not fetched."""
    return project_root() / "data" / "raw" / "world_bank_education.csv"


def load_education_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the extract as text.

    Blank cells and the ``..`` marker stay as text so validation can reject
    any other non-numeric token instead of turning it into a missing value.
    """
    csv_path = default_raw_csv_path() if path is None else Path(path)
    if not csv_path.is_file():
        raise ValidationError(
            "Raw education CSV not found.\n"
            f"Expected file:\n{csv_path}"
        )
    return pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
