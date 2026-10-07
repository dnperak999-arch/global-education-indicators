"""Tidy reshape on synthetic wide tables."""

from __future__ import annotations

import pandas as pd
import pytest

from education_indicators.codes import INDICATORS, entity_type_for_code
from education_indicators.tidy import TIDY_COLUMNS, to_tidy, validate_tidy
from education_indicators.validate import validate_raw
from tests.builders import YEAR_2010, YEAR_2011, frame_from, rows_for


def _tidy_for(name: str, code: str, fill: str = "1") -> pd.DataFrame:
    cleaned, _report = validate_raw(frame_from(rows_for(name, code, fill=fill)))
    return to_tidy(cleaned)


def test_tidy_columns_year_type_and_unique_key() -> None:
    tidy = _tidy_for("Finland", "FIN")

    assert list(tidy.columns) == list(TIDY_COLUMNS)
    assert pd.api.types.is_integer_dtype(tidy["year"])
    assert set(tidy["year"]) == {2010, 2011}
    assert all(isinstance(year, int) for year in tidy["year"].tolist())
    assert not tidy.duplicated(["country_code", "indicator_code", "year"]).any()
    assert tidy["indicator_code"].nunique() == len(INDICATORS)
    validate_tidy(tidy)


def test_double_dot_and_blank_become_missing() -> None:
    rows = rows_for("Finland", "FIN")
    target = "SE.PRM.ENRL"
    for row in rows:
        if row["Series Code"] == target:
            row[YEAR_2010] = ".."
            row[YEAR_2011] = "   "

    cleaned, _report = validate_raw(frame_from(rows))
    tidy = to_tidy(cleaned)
    observed = tidy.loc[tidy["indicator_code"] == target, ["year", "value"]]

    assert observed["value"].isna().all()
    assert observed["year"].tolist() == [2010, 2011]


def test_identical_duplicates_do_not_produce_two_tidy_rows() -> None:
    rows = rows_for("Finland", "FIN")
    rows.append(dict(rows[0]))
    cleaned, report = validate_raw(frame_from(rows))
    tidy = to_tidy(cleaned)

    assert report.duplicate_rows_collapsed == 1
    assert not tidy.duplicated(["country_code", "indicator_code", "year"]).any()
    assert len(tidy) == len(INDICATORS) * 2


def test_classification_uses_code_not_country_name() -> None:
    south_africa = _tidy_for("World", "ZAF")
    named_like_a_region = _tidy_for("Central African Republic", "CAF")
    egypt = _tidy_for("Egypt, Arab Rep.", "EGY")
    code_overrides_name = _tidy_for("Finland", "WLD")

    assert set(south_africa["entity_type"]) == {"country"}
    assert set(named_like_a_region["entity_type"]) == {"country"}
    assert set(egypt["entity_type"]) == {"country"}
    assert set(code_overrides_name["entity_type"]) == {"aggregate"}
    assert entity_type_for_code("ZAF") == "country"
    assert entity_type_for_code("WLD") == "aggregate"


def test_indicator_code_not_series_label_decides_the_series() -> None:
    rows = rows_for("Finland", "FIN")
    for row in rows:
        if row["Series Code"] == "UIS.X.US.1.FSGOV":
            row["Series"] = "Government expenditure on pre-primary education, US$ (millions)"

    cleaned, _report = validate_raw(frame_from(rows))
    tidy = to_tidy(cleaned)
    observed = tidy.loc[tidy["indicator_code"] == "UIS.X.US.1.FSGOV"].iloc[0]

    assert observed["indicator_name"].startswith("Government expenditure on pre-primary")
    assert INDICATORS[observed["indicator_code"]].level == "primary"
    assert INDICATORS["UIS.X.US.02.FSGOV"].level == "pre_primary"


def test_reversed_year_columns_still_sort_by_year() -> None:
    rows = rows_for("Finland", "FIN")
    wide = frame_from(rows).loc[
        :,
        ["Country Name", "Country Code", "Series", "Series Code", YEAR_2011, YEAR_2010],
    ]
    cleaned, _report = validate_raw(wide)
    tidy = to_tidy(cleaned)
    years = tidy.loc[tidy["indicator_code"] == "SE.PRM.TCHR", "year"].tolist()

    assert years == [2010, 2011]
