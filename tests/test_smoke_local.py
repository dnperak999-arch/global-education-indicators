"""Smoke check against the copied local extract. Separate from the synthetic tests."""

from __future__ import annotations

import pandas as pd
import pytest

from education_indicators.analysis import coverage_by_indicator_year
from education_indicators.metrics import USD_MILLIONS_TO_USD, derive_metrics
from education_indicators.codes import INDICATORS
from education_indicators.load import default_raw_csv_path, load_education_csv
from education_indicators.tidy import TIDY_COLUMNS, to_tidy, validate_tidy
from education_indicators.validate import validate_raw


def test_local_education_extract() -> None:
    path = default_raw_csv_path()
    if not path.is_file():
        pytest.skip(f"Local extract is not present: {path}")

    raw = load_education_csv(path)
    cleaned, report = validate_raw(raw)
    tidy = to_tidy(cleaned)
    validate_tidy(tidy)

    assert list(tidy.columns) == list(TIDY_COLUMNS)
    assert set(tidy["indicator_code"]) == set(INDICATORS)
    assert report.n_indicator_codes == 10
    assert report.years[0] == 2010
    assert report.years[-1] == 2024
    assert pd.api.types.is_integer_dtype(tidy["year"])
    assert not tidy.duplicated(["country_code", "indicator_code", "year"]).any()
    assert int(tidy["value"].isna().sum() + tidy["value"].notna().sum()) == len(tidy)

    types = tidy.drop_duplicates("country_code").set_index("country_code")["entity_type"]
    expected = {
        "WLD": "aggregate",
        "ARB": "aggregate",
        "CHI": "aggregate",
        "FTI": "aggregate",
        "LDC": "aggregate",
        "PRE": "aggregate",
        "ZAF": "country",
        "EGY": "country",
        "CAF": "country",
        "SAU": "country",
        "ARE": "country",
        "SYR": "country",
        "PRI": "country",
        "ASM": "country",
    }
    for code, entity_type in expected.items():
        assert types[code] == entity_type

    coverage = coverage_by_indicator_year(tidy)
    assert len(coverage) == len(INDICATORS) * tidy["year"].nunique()
    assert not coverage.duplicated(["indicator_code", "year"]).any()
    assert (
        coverage["reporting_entities"] + coverage["missing_entities"]
        == coverage["total_country_territory_entities"]
    ).all()
    assert "WLD" not in set(tidy.loc[tidy["entity_type"] == "country", "country_code"])

    metrics = derive_metrics(tidy)
    assert set(metrics.teachers["education_level"]).issubset({"pre_primary", "primary", "secondary"})
    assert "WLD" not in set(metrics.teachers["country_code"])
    assert "WLD" not in set(metrics.expenditure["country_code"])
    assert not metrics.teachers.duplicated(["country_code", "year", "education_level"]).any()
    assert not metrics.expenditure.duplicated(["country_code", "year", "education_level"]).any()
    assert (metrics.teachers["enrolment"] > 0).all()
    assert (metrics.expenditure["enrolment"] > 0).all()
    expected_spend = (
        metrics.expenditure["expenditure_usd_millions"] * USD_MILLIONS_TO_USD
        / metrics.expenditure["enrolment"]
    )
    assert metrics.expenditure["expenditure_per_enrolled_student_usd"].equals(expected_spend)
