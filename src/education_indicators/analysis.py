"""Reporting coverage for country and territory entities.

Counts describe how many non-aggregate entities have a value. They do not
decide whether a year is suitable for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from education_indicators.codes import INDICATORS
from education_indicators.load import project_root
from education_indicators.validate import ValidationError

COVERAGE_COLUMNS: tuple[str, ...] = (
    "indicator_code",
    "year",
    "reporting_entities",
    "total_country_territory_entities",
    "missing_entities",
    "coverage_percent",
)

_REQUIRED_TIDY_COLUMNS = (
    "country_code",
    "entity_type",
    "indicator_code",
    "year",
    "value",
)


@dataclass(frozen=True)
class IndicatorCoverage:
    """Observed reporting span for one indicator. Ties keep every matching year."""

    indicator_code: str
    first_reported_year: int | None
    last_reported_year: int | None
    max_coverage_years: tuple[int, ...]
    max_reporting_entities: int
    max_coverage_percent: float


@dataclass(frozen=True)
class CoverageDiagnostics:
    total_country_territory_entities: int
    aggregates_excluded: int
    indicator_year_combinations: int
    zero_reporting_combinations: int
    by_indicator: tuple[IndicatorCoverage, ...]


def coverage_output_path() -> Path:
    return project_root() / "outputs" / "coverage_by_indicator_year.csv"


def coverage_by_indicator_year(tidy: pd.DataFrame) -> pd.DataFrame:
    """Count non-missing values by indicator and year.

    The denominator is every country/territory entity in the tidy table.
    Aggregate rows are excluded. Every expected indicator appears for every
    year present in ``tidy``, including combinations with no reported values.
    """
    _require_tidy_columns(tidy)
    if tidy.empty:
        raise ValidationError("Tidy table is empty.")

    unknown_types = sorted(set(tidy["entity_type"]) - {"country", "aggregate"})
    if unknown_types:
        raise ValidationError(f"Unknown entity_type values: {unknown_types}")

    unknown_codes = sorted(set(tidy["indicator_code"]) - set(INDICATORS))
    if unknown_codes:
        raise ValidationError(f"Unknown indicator codes in tidy data: {unknown_codes}")

    type_counts = tidy.groupby("country_code", sort=False)["entity_type"].nunique()
    if (type_counts > 1).any():
        raise ValidationError("A country code has more than one entity_type.")

    countries = tidy.loc[tidy["entity_type"] == "country"]
    total = int(countries["country_code"].nunique())
    if total == 0:
        raise ValidationError("No country/territory entities in the tidy table.")

    years = tuple(sorted(int(year) for year in tidy["year"].unique()))
    reported = _reporting_counts(countries)

    rows: list[dict[str, object]] = []
    for indicator_code in INDICATORS:
        for year in years:
            reporting = reported.get((indicator_code, year), 0)
            rows.append(
                {
                    "indicator_code": indicator_code,
                    "year": year,
                    "reporting_entities": reporting,
                    "total_country_territory_entities": total,
                    "missing_entities": total - reporting,
                    "coverage_percent": reporting / total * 100,
                }
            )

    coverage = pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))
    coverage["year"] = coverage["year"].astype("int64")
    coverage["reporting_entities"] = coverage["reporting_entities"].astype("int64")
    coverage["total_country_territory_entities"] = coverage[
        "total_country_territory_entities"
    ].astype("int64")
    coverage["missing_entities"] = coverage["missing_entities"].astype("int64")
    _check_coverage_table(coverage, years)
    return coverage


def coverage_diagnostics(tidy: pd.DataFrame, coverage: pd.DataFrame) -> CoverageDiagnostics:
    """Summarise the coverage table without judging whether coverage is enough."""
    _check_coverage_table(
        coverage,
        tuple(sorted(int(year) for year in coverage["year"].unique())),
    )
    by_indicator = tuple(
        _indicator_coverage(coverage, indicator_code) for indicator_code in INDICATORS
    )
    zero_rows = coverage["reporting_entities"] == 0
    return CoverageDiagnostics(
        total_country_territory_entities=int(
            coverage["total_country_territory_entities"].iloc[0]
        ),
        aggregates_excluded=int(
            tidy.loc[tidy["entity_type"] == "aggregate", "country_code"].nunique()
        ),
        indicator_year_combinations=int(len(coverage)),
        zero_reporting_combinations=int(zero_rows.sum()),
        by_indicator=by_indicator,
    )


def write_coverage(coverage: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the coverage table. This does not write the tidy or raw extract."""
    out_path = coverage_output_path() if path is None else Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out_path, index=False)
    return out_path


def _reporting_counts(countries: pd.DataFrame) -> dict[tuple[str, int], int]:
    observed = countries.loc[countries["value"].notna(), ["indicator_code", "year", "country_code"]]
    if observed.empty:
        return {}
    counts = (
        observed.groupby(["indicator_code", "year"], sort=False)["country_code"]
        .nunique()
    )
    return {
        (str(indicator_code), int(year)): int(count)
        for (indicator_code, year), count in counts.items()
    }


def _indicator_coverage(coverage: pd.DataFrame, indicator_code: str) -> IndicatorCoverage:
    subset = coverage.loc[coverage["indicator_code"] == indicator_code].sort_values("year")
    if subset.empty:
        raise ValidationError(f"Coverage is missing indicator {indicator_code}.")
    reported = subset.loc[subset["reporting_entities"] > 0]
    if reported.empty:
        first_year = None
        last_year = None
    else:
        first_year = int(reported["year"].iloc[0])
        last_year = int(reported["year"].iloc[-1])
    maximum = int(subset["reporting_entities"].max())
    at_maximum = tuple(
        int(year)
        for year in subset.loc[subset["reporting_entities"] == maximum, "year"]
    )
    percent = float(
        subset.loc[subset["year"] == at_maximum[0], "coverage_percent"].iloc[0]
    )
    return IndicatorCoverage(
        indicator_code=indicator_code,
        first_reported_year=first_year,
        last_reported_year=last_year,
        max_coverage_years=at_maximum,
        max_reporting_entities=maximum,
        max_coverage_percent=percent,
    )


def _require_tidy_columns(tidy: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_TIDY_COLUMNS if column not in tidy.columns]
    if missing:
        raise ValidationError(f"Tidy table is missing columns: {missing}")


def _check_coverage_table(coverage: pd.DataFrame, years: tuple[int, ...]) -> None:
    if list(coverage.columns) != list(COVERAGE_COLUMNS):
        raise ValidationError(
            f"Coverage columns are {list(coverage.columns)}; expected {list(COVERAGE_COLUMNS)}."
        )
    expected = {(code, year) for code in INDICATORS for year in years}
    actual = set(zip(coverage["indicator_code"], coverage["year"].astype(int)))
    if actual != expected:
        raise ValidationError(
            "Coverage table does not contain every indicator and year combination."
        )
    if coverage.duplicated(["indicator_code", "year"]).any():
        raise ValidationError("Coverage table has duplicate indicator_code and year rows.")
    totals = coverage["total_country_territory_entities"]
    if totals.nunique() != 1:
        raise ValidationError("Coverage denominator is not constant.")
    imbalance = (
        coverage["reporting_entities"] + coverage["missing_entities"] != totals
    )
    if imbalance.any():
        raise ValidationError(
            "reporting_entities + missing_entities does not equal "
            "total_country_territory_entities."
        )
