"""Structural checks for the wide education extract.

These checks confirm the file can be reshaped. They do not require any
particular year to be empty, and they do not apply a coverage threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from education_indicators.codes import (
    ID_COLUMNS,
    INDICATORS,
    parse_year_label,
)

MISSING_MARKERS = frozenset({"", ".."})


class ValidationError(ValueError):
    """The extract failed a structural or integrity check."""


@dataclass(frozen=True)
class ValidationReport:
    """Facts established while validating one wide extract."""

    raw_rows: int
    raw_columns: int
    duplicate_rows_collapsed: int
    n_entities: int
    n_indicator_codes: int
    years: tuple[int, ...]
    warnings: tuple[str, ...]


def as_text(value: object) -> str:
    """Normalise a cell to stripped text. Missing numeric NaN becomes blank."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_observation(
    raw: object,
    *,
    country_code: str,
    series_code: str,
    column: str,
) -> float | None:
    """Return a number, or None for a blank or ``..`` cell.

    Any other token raises. Nothing is coerced to missing.
    """
    text = as_text(raw)
    if text in MISSING_MARKERS:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValidationError(
            "Unexpected non-numeric value "
            f"{text!r} for {country_code} / {series_code} in {column}."
        ) from None
    if not math.isfinite(number):
        raise ValidationError(
            "Unexpected non-numeric value "
            f"{text!r} for {country_code} / {series_code} in {column}."
        )
    return number


def validate_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Check the wide extract and collapse identical duplicate source rows.

    The returned frame still has one row per country and series. Conflicting
    duplicates raise. Year cells are still text; :func:`to_tidy` parses them.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValidationError("Education extract is empty.")

    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValidationError(f"Duplicate column names: {duplicates}")

    for column in frame.columns:
        frame[column] = frame[column].map(as_text)

    missing_ids = [column for column in ID_COLUMNS if column not in frame.columns]
    if missing_ids:
        raise ValidationError(f"Required columns are missing: {missing_ids}")

    year_labels = _year_columns(frame)
    allowed = set(ID_COLUMNS) | set(year_labels)
    unexpected = [column for column in frame.columns if column not in allowed]
    if unexpected:
        raise ValidationError(
            "Unexpected columns. Year columns must look like "
            f"'2010 [YR2010]'. Got: {unexpected}"
        )

    frame["Country Code"] = frame["Country Code"].map(_valid_country_code)
    frame["Series Code"] = frame["Series Code"].map(_valid_series_code)
    _require_labels(frame, "Country Name", "Country Code")
    _require_labels(frame, "Series", "Series Code")
    _validate_year_tokens(frame, year_labels)

    cleaned, collapsed = _collapse_duplicates(frame, year_labels)
    _require_all_indicators(cleaned)
    _require_one_label(cleaned, "Country Code", "Country Name", "country code")
    _require_one_label(cleaned, "Series Code", "Series", "series code")

    cleaned = cleaned.loc[:, [*ID_COLUMNS, *year_labels]].reset_index(drop=True)
    years = tuple(parse_year_label(label) for label in year_labels)
    warnings: list[str] = []
    if collapsed:
        warnings.append(
            "Collapsed "
            f"{collapsed} identical duplicate source row(s) "
            "with the same country code and series code."
        )

    report = ValidationReport(
        raw_rows=int(df.shape[0]),
        raw_columns=int(df.shape[1]),
        duplicate_rows_collapsed=collapsed,
        n_entities=int(cleaned["Country Code"].nunique()),
        n_indicator_codes=int(cleaned["Series Code"].nunique()),
        years=years,
        warnings=tuple(warnings),
    )
    return cleaned, report


def _year_columns(frame: pd.DataFrame) -> list[str]:
    parsed: list[tuple[int, str]] = []
    for column in frame.columns:
        year = parse_year_label(column)
        if year is not None:
            parsed.append((year, column))
    if not parsed:
        raise ValidationError(
            "No year columns found. Expected columns like '2010 [YR2010]'."
        )
    years = [year for year, _ in parsed]
    if len(years) != len(set(years)):
        raise ValidationError(f"Duplicate year columns: {years}")
    parsed.sort(key=lambda item: item[0])
    return [label for _, label in parsed]


def _valid_country_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha() or not code.isascii():
        raise ValidationError(f"Invalid Country Code: {value!r}")
    return code


def _valid_series_code(value: str) -> str:
    code = value.strip()
    if code not in INDICATORS:
        expected = ", ".join(INDICATORS)
        raise ValidationError(
            f"Unexpected series code {code!r}. Expected one of: {expected}"
        )
    return code


def _require_labels(frame: pd.DataFrame, label_column: str, key_column: str) -> None:
    missing = frame[label_column] == ""
    if missing.any():
        key = frame.loc[missing, key_column].iloc[0]
        raise ValidationError(f"Blank {label_column} for {key}.")


def _validate_year_tokens(frame: pd.DataFrame, year_labels: list[str]) -> None:
    for record in frame.to_dict(orient="records"):
        for label in year_labels:
            parse_observation(
                record[label],
                country_code=record["Country Code"],
                series_code=record["Series Code"],
                column=label,
            )


def _collapse_duplicates(
    frame: pd.DataFrame,
    year_labels: list[str],
) -> tuple[pd.DataFrame, int]:
    keep_indexes: list[int] = []
    collapsed = 0
    grouped = frame.groupby(["Country Code", "Series Code"], sort=False)
    for (country_code, series_code), group in grouped:
        records = group.to_dict(orient="records")
        signatures = [_row_signature(record, year_labels) for record in records]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValidationError(
                "Conflicting duplicate rows for country code "
                f"{country_code} and series code {series_code}."
            )
        keep_indexes.append(int(group.index[0]))
        collapsed += len(group) - 1
    return frame.loc[keep_indexes].copy(), collapsed


def _row_signature(
    record: dict[str, str],
    year_labels: list[str],
) -> tuple[object, ...]:
    observed = tuple(
        parse_observation(
            record[label],
            country_code=record["Country Code"],
            series_code=record["Series Code"],
            column=label,
        )
        for label in year_labels
    )
    return (record["Country Name"], record["Series"], observed)


def _require_all_indicators(frame: pd.DataFrame) -> None:
    present = set(frame["Series Code"])
    missing = [code for code in INDICATORS if code not in present]
    if missing:
        raise ValidationError(f"Required series codes are missing: {missing}")


def _require_one_label(
    frame: pd.DataFrame,
    key_column: str,
    label_column: str,
    key_name: str,
) -> None:
    for key, labels in frame.groupby(key_column, sort=False)[label_column]:
        unique_labels = sorted(set(labels))
        if len(unique_labels) > 1:
            raise ValidationError(
                f"{key_name} {key} has multiple labels: {unique_labels}"
            )
