"""Validate the local education extract, write coverage and ratio tables, and draw three figures."""

from __future__ import annotations

import sys

import pandas as pd

from education_indicators.analysis import (
    CoverageDiagnostics,
    coverage_by_indicator_year,
    coverage_diagnostics,
    write_coverage,
)
from education_indicators.load import default_raw_csv_path, load_education_csv
from education_indicators.metrics import derive_metrics, render_metric_summary, write_metrics
from education_indicators.plots import (
    literacy_coverage_summary,
    primary_enrolment_scale_table,
    render_literacy_summary,
    render_scale_table,
    write_figures,
    write_literacy_table,
    write_scale_table,
)
from education_indicators.tidy import to_tidy, validate_tidy
from education_indicators.validate import ValidationError, ValidationReport, validate_raw


def main() -> int:
    path = default_raw_csv_path()
    try:
        raw = load_education_csv(path)
        cleaned, report = validate_raw(raw)
        tidy = to_tidy(cleaned)
        validate_tidy(tidy)
        coverage = coverage_by_indicator_year(tidy)
        diagnostics = coverage_diagnostics(tidy, coverage)
        coverage_path = write_coverage(coverage)
        metrics = derive_metrics(tidy)
        metric_paths = write_metrics(metrics)
        figure_paths = write_figures(coverage, metrics)
        literacy = literacy_coverage_summary(coverage)
        literacy_path = write_literacy_table(literacy)
        scale = primary_enrolment_scale_table(tidy, coverage)
        scale_path = write_scale_table(scale)
    except ValidationError as exc:
        print(f"Validation failed:\n{exc}", file=sys.stderr)
        return 1

    _print_summary(tidy, report)
    _print_coverage(coverage, diagnostics, coverage_path)
    print(render_metric_summary(metrics))
    print()
    print(f"Wrote {metric_paths['teachers']}")
    print(f"Wrote {metric_paths['expenditure']}")
    print(f"Wrote {metric_paths['paired']}")
    print()
    print(render_literacy_summary(literacy))
    print(f"Wrote {literacy_path}")
    print()
    print(render_scale_table(scale))
    print(f"Wrote {scale_path}")
    print()
    print(f"Wrote {figure_paths['coverage']}")
    print(f"Wrote {figure_paths['teachers']}")
    print(f"Wrote {figure_paths['expenditure']}")
    return 0


def _print_summary(tidy: pd.DataFrame, report: ValidationReport) -> None:
    entities = tidy.loc[:, ["country_code", "entity_type"]].drop_duplicates()
    n_country = int((entities["entity_type"] == "country").sum())
    n_aggregate = int((entities["entity_type"] == "aggregate").sum())
    n_missing = int(tidy["value"].isna().sum())
    n_present = int(len(tidy) - n_missing)
    first_year = report.years[0]
    last_year = report.years[-1]

    print(f"Raw shape: {report.raw_rows} rows x {report.raw_columns} columns")
    print(f"Duplicate source rows collapsed: {report.duplicate_rows_collapsed}")
    print(
        f"Entities: {report.n_entities} "
        f"({n_country} country/territory, {n_aggregate} aggregate)"
    )
    print(f"Indicator codes: {report.n_indicator_codes}")
    print(f"Years: {first_year}-{last_year} ({len(report.years)} year columns)")
    print(f"Tidy shape: {tidy.shape[0]} rows x {tidy.shape[1]} columns")
    print(f"Non-null values: {n_present}")
    print(f"Missing values: {n_missing}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")


def _print_coverage(
    coverage: pd.DataFrame,
    diagnostics: CoverageDiagnostics,
    coverage_path,
) -> None:
    print()
    print("Coverage (country/territory entities only; aggregates excluded)")
    print(
        "Country/territory entities: "
        f"{diagnostics.total_country_territory_entities}"
    )
    print(f"Aggregates excluded: {diagnostics.aggregates_excluded}")
    print(f"Indicator-year combinations: {diagnostics.indicator_year_combinations}")
    print(
        "Indicator-year combinations with zero reporting entities: "
        f"{diagnostics.zero_reporting_combinations}"
    )
    print(f"Wrote {coverage_path}")
    print()
    print("By indicator")
    for item in diagnostics.by_indicator:
        if item.first_reported_year is None:
            span = "no reported values"
        else:
            span = (
                f"first reported year {item.first_reported_year}; "
                f"last reported year {item.last_reported_year}"
            )
        years = ", ".join(str(year) for year in item.max_coverage_years)
        print(
            f"{item.indicator_code}: {span}; "
            f"maximum {item.max_reporting_entities} reporting entities "
            f"({item.max_coverage_percent:.2f}%) in {years}"
        )

    literacy_code = "SE.ADT.LITR.ZS"
    print()
    print(f"{literacy_code} by year")
    literacy = coverage.loc[coverage["indicator_code"] == literacy_code]
    for row in literacy.itertuples(index=False):
        print(
            f"{int(row.year)}: {int(row.reporting_entities)} reporting entities, "
            f"{row.coverage_percent:.2f}%"
        )


if __name__ == "__main__":
    sys.exit(main())
