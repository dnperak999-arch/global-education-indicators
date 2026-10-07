"""Coverage counts on synthetic tidy tables."""

from __future__ import annotations

import pandas as pd
import pytest

from education_indicators.analysis import coverage_by_indicator_year, coverage_diagnostics
from education_indicators.codes import INDICATORS
from education_indicators.validate import ValidationError


def _tidy(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["year"] = frame["year"].astype("int64")
    frame["value"] = pd.Series(frame["value"].tolist(), dtype="Float64")
    return frame


def _row(
    country_code: str,
    entity_type: str,
    indicator_code: str,
    year: int,
    value: float | None,
) -> dict[str, object]:
    return {
        "country_code": country_code,
        "country_name": country_code,
        "entity_type": entity_type,
        "indicator_code": indicator_code,
        "indicator_name": indicator_code,
        "year": year,
        "value": value,
    }


def _sample() -> pd.DataFrame:
    """Two country/territory entities, one aggregate, two years, two indicators with values."""
    rows: list[dict[str, object]] = []
    for year, finland, sweden in ((2010, 10.0, None), (2011, None, None)):
        rows.append(_row("FIN", "country", "SE.PRM.TCHR", year, finland))
        rows.append(_row("SWE", "country", "SE.PRM.TCHR", year, sweden))
        rows.append(_row("WLD", "aggregate", "SE.PRM.TCHR", year, 999.0))
        rows.append(_row("FIN", "country", "SE.ADT.LITR.ZS", year, finland))
        rows.append(_row("SWE", "country", "SE.ADT.LITR.ZS", year, sweden))
        rows.append(_row("WLD", "aggregate", "SE.ADT.LITR.ZS", year, 999.0))
    return _tidy(rows)


def test_aggregates_are_excluded_and_missing_values_do_not_count() -> None:
    coverage = coverage_by_indicator_year(_sample())
    teachers_2010 = coverage.loc[
        (coverage["indicator_code"] == "SE.PRM.TCHR") & (coverage["year"] == 2010)
    ].iloc[0]

    assert teachers_2010["reporting_entities"] == 1
    assert teachers_2010["total_country_territory_entities"] == 2
    assert teachers_2010["missing_entities"] == 1
    assert teachers_2010["coverage_percent"] == 50


def test_zero_reporting_year_is_kept() -> None:
    coverage = coverage_by_indicator_year(_sample())
    teachers_2011 = coverage.loc[
        (coverage["indicator_code"] == "SE.PRM.TCHR") & (coverage["year"] == 2011)
    ].iloc[0]

    assert teachers_2011["reporting_entities"] == 0
    assert teachers_2011["coverage_percent"] == 0
    assert teachers_2011["missing_entities"] == 2


def test_every_indicator_year_combination_is_present_once() -> None:
    coverage = coverage_by_indicator_year(_sample())

    assert len(coverage) == len(INDICATORS) * 2
    assert set(coverage["indicator_code"]) == set(INDICATORS)
    assert set(coverage["year"]) == {2010, 2011}
    assert not coverage.duplicated(["indicator_code", "year"]).any()
    assert (
        coverage["reporting_entities"] + coverage["missing_entities"]
        == coverage["total_country_territory_entities"]
    ).all()


def test_absent_indicator_year_is_zero_rather_than_omitted() -> None:
    coverage = coverage_by_indicator_year(_sample())
    enrolment = coverage.loc[coverage["indicator_code"] == "SE.PRM.ENRL"]

    assert enrolment["reporting_entities"].tolist() == [0, 0]
    assert enrolment["coverage_percent"].tolist() == [0, 0]


def test_literacy_uses_the_same_count_as_any_other_indicator() -> None:
    coverage = coverage_by_indicator_year(_sample())
    literacy = coverage.loc[coverage["indicator_code"] == "SE.ADT.LITR.ZS"].reset_index(drop=True)
    teachers = coverage.loc[coverage["indicator_code"] == "SE.PRM.TCHR"].reset_index(drop=True)

    assert literacy["reporting_entities"].tolist() == teachers["reporting_entities"].tolist()
    assert literacy["coverage_percent"].tolist() == teachers["coverage_percent"].tolist()
    assert literacy["missing_entities"].tolist() == teachers["missing_entities"].tolist()


def test_coverage_does_not_depend_on_row_order() -> None:
    tidy = _sample()
    reversed_rows = tidy.iloc[::-1].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        coverage_by_indicator_year(tidy),
        coverage_by_indicator_year(reversed_rows),
    )


def test_tied_maximum_keeps_every_year() -> None:
    rows = [
        _row("FIN", "country", "SE.PRM.TCHR", 2010, 1.0),
        _row("FIN", "country", "SE.PRM.TCHR", 2011, 1.0),
        _row("FIN", "country", "SE.PRM.TCHR", 2012, None),
    ]
    diagnostics = coverage_diagnostics(_tidy(rows), coverage_by_indicator_year(_tidy(rows)))
    teachers = diagnostics.by_indicator[0]

    assert teachers.indicator_code == "SE.PRM.TCHR"
    assert teachers.first_reported_year == 2010
    assert teachers.last_reported_year == 2011
    assert teachers.max_reporting_entities == 1
    assert teachers.max_coverage_years == (2010, 2011)
    assert teachers.max_coverage_percent == 100
    assert diagnostics.aggregates_excluded == 0
    assert diagnostics.zero_reporting_combinations == len(INDICATORS) * 3 - 2


def test_indicator_with_no_reported_values_has_no_reported_span() -> None:
    rows = [_row("FIN", "country", "SE.PRM.TCHR", 2010, None)]
    diagnostics = coverage_diagnostics(_tidy(rows), coverage_by_indicator_year(_tidy(rows)))
    teachers = next(item for item in diagnostics.by_indicator if item.indicator_code == "SE.PRM.TCHR")

    assert teachers.first_reported_year is None
    assert teachers.last_reported_year is None
    assert teachers.max_reporting_entities == 0
    assert teachers.max_coverage_percent == 0
    assert teachers.max_coverage_years == (2010,)


def test_unknown_entity_type_raises() -> None:
    rows = [_row("FIN", "region", "SE.PRM.TCHR", 2010, 1.0)]

    with pytest.raises(ValidationError, match="Unknown entity_type"):
        coverage_by_indicator_year(_tidy(rows))
