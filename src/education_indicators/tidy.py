"""Reshape a validated wide education extract to one row per entity-indicator-year."""

from __future__ import annotations

import pandas as pd

from education_indicators.codes import (
    ID_COLUMNS,
    INDICATORS,
    entity_type_for_code,
    parse_year_label,
)
from education_indicators.validate import ValidationError, parse_observation

TIDY_COLUMNS: tuple[str, ...] = (
    "country_code",
    "country_name",
    "entity_type",
    "indicator_code",
    "indicator_name",
    "year",
    "value",
)


def to_tidy(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt year columns into ``year`` and ``value``.

    ``entity_type`` comes from the country code. The series name is kept as
    ``indicator_name`` and is not used to choose the indicator.
    """
    frame = wide.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing_ids = [column for column in ID_COLUMNS if column not in frame.columns]
    if missing_ids:
        raise ValidationError(f"Required columns are missing: {missing_ids}")

    year_labels = [
        column for column in frame.columns if parse_year_label(column) is not None
    ]
    if not year_labels:
        raise ValidationError(
            "No year columns found. Expected columns like '2010 [YR2010]'."
        )

    long = frame.melt(
        id_vars=list(ID_COLUMNS),
        value_vars=year_labels,
        var_name="year_label",
        value_name="raw_value",
    )
    years: list[int] = []
    values: list[float | None] = []
    for raw_value, country_code, series_code, year_label in zip(
        long["raw_value"],
        long["Country Code"],
        long["Series Code"],
        long["year_label"],
    ):
        year = parse_year_label(str(year_label))
        if year is None:
            raise ValidationError(f"Could not parse year column {year_label!r}.")
        years.append(year)
        values.append(
            parse_observation(
                raw_value,
                country_code=str(country_code),
                series_code=str(series_code),
                column=str(year_label),
            )
        )

    tidy = pd.DataFrame(
        {
            "country_code": long["Country Code"].map(lambda value: str(value).strip().upper()),
            "country_name": long["Country Name"].map(lambda value: str(value).strip()),
            "entity_type": long["Country Code"].map(
                lambda value: entity_type_for_code(str(value).strip().upper())
            ),
            "indicator_code": long["Series Code"].map(lambda value: str(value).strip()),
            "indicator_name": long["Series"].map(lambda value: str(value).strip()),
            "year": pd.Series(years, dtype="int64"),
            "value": pd.Series(values, dtype="Float64"),
        }
    )
    tidy = tidy.sort_values(
        ["country_code", "indicator_code", "year"],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_tidy(tidy)
    return tidy


def validate_tidy(tidy: pd.DataFrame) -> None:
    """Fail if the tidy table is not uniquely keyed or uses an unknown code."""
    if list(tidy.columns) != list(TIDY_COLUMNS):
        raise ValidationError(
            f"Tidy columns are {list(tidy.columns)}; expected {list(TIDY_COLUMNS)}."
        )
    if not pd.api.types.is_integer_dtype(tidy["year"]):
        raise ValidationError(f"Year column is {tidy['year'].dtype}, expected integer.")
    if tidy["year"].isna().any():
        raise ValidationError("Year column contains missing values.")

    duplicate = tidy.duplicated(["country_code", "indicator_code", "year"])
    if duplicate.any():
        sample = tidy.loc[duplicate, ["country_code", "indicator_code", "year"]].head(3)
        raise ValidationError(
            "Tidy key country_code, indicator_code, year is not unique. "
            f"Examples:\n{sample.to_string(index=False)}"
        )

    unknown_types = sorted(set(tidy["entity_type"]) - {"country", "aggregate"})
    if unknown_types:
        raise ValidationError(f"Unknown entity_type values: {unknown_types}")

    unknown_codes = sorted(set(tidy["indicator_code"]) - set(INDICATORS))
    if unknown_codes:
        raise ValidationError(f"Unknown indicator codes in tidy data: {unknown_codes}")

    if not set(INDICATORS).issubset(set(tidy["indicator_code"])):
        missing = [code for code in INDICATORS if code not in set(tidy["indicator_code"])]
        raise ValidationError(f"Required series codes are missing: {missing}")
