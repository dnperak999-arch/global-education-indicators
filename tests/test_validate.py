"""Structural validation on synthetic wide tables."""

from __future__ import annotations

import pandas as pd
import pytest

from education_indicators.codes import AGGREGATE_CODES, INDICATORS
from education_indicators.load import load_education_csv
from education_indicators.validate import ValidationError, validate_raw
from tests.builders import YEAR_2010, YEAR_2011, frame_from, rows_for


def test_valid_frame_keeps_required_columns_and_codes() -> None:
    cleaned, report = validate_raw(frame_from(rows_for("Finland", "FIN")))

    assert list(cleaned.columns[:4]) == [
        "Country Name",
        "Country Code",
        "Series",
        "Series Code",
    ]
    assert list(cleaned.columns[4:]) == [YEAR_2010, YEAR_2011]
    assert set(cleaned["Series Code"]) == set(INDICATORS)
    assert report.n_indicator_codes == 10
    assert report.years == (2010, 2011)
    assert report.duplicate_rows_collapsed == 0
    assert report.warnings == ()


def test_missing_required_column_raises() -> None:
    frame = frame_from(rows_for("Finland", "FIN")).drop(columns=["Country Code"])

    with pytest.raises(ValidationError, match="Required columns"):
        validate_raw(frame)


def test_year_columns_are_parsed_and_ordered() -> None:
    rows = rows_for("Finland", "FIN")
    frame = frame_from(rows).loc[:, ["Country Name", "Country Code", "Series", "Series Code", YEAR_2011, YEAR_2010]]

    cleaned, report = validate_raw(frame)

    assert list(cleaned.columns[4:]) == [YEAR_2010, YEAR_2011]
    assert report.years == (2010, 2011)


def test_malformed_year_column_raises() -> None:
    frame = frame_from(rows_for("Finland", "FIN"))
    frame = frame.rename(columns={YEAR_2011: "2010 [YR2011]"})

    with pytest.raises(ValidationError, match="Unexpected columns"):
        validate_raw(frame)


def test_unexpected_series_code_raises() -> None:
    rows = rows_for("Finland", "FIN")
    rows.append(
        {
            "Country Name": "Finland",
            "Country Code": "FIN",
            "Series": "Not an indicator",
            "Series Code": "NOT.A.CODE",
            YEAR_2010: "1",
            YEAR_2011: "1",
        }
    )

    with pytest.raises(ValidationError, match="Unexpected series code"):
        validate_raw(frame_from(rows))


def test_missing_required_series_code_raises() -> None:
    rows = [row for row in rows_for("Finland", "FIN") if row["Series Code"] != "SE.PRM.ENRL"]

    with pytest.raises(ValidationError, match="SE.PRM.ENRL"):
        validate_raw(frame_from(rows))


def test_blank_and_double_dot_are_accepted_as_missing_markers() -> None:
    rows = rows_for("Finland", "FIN")
    rows[0][YEAR_2010] = ".."
    rows[0][YEAR_2011] = ""

    cleaned, _report = validate_raw(frame_from(rows))
    primary = cleaned.loc[cleaned["Series Code"] == rows[0]["Series Code"]].iloc[0]

    assert primary[YEAR_2010] == ".."
    assert primary[YEAR_2011] == ""


def test_unexpected_non_numeric_token_raises() -> None:
    rows = rows_for("Finland", "FIN")
    rows[0][YEAR_2010] = "n/a"

    with pytest.raises(ValidationError, match="Unexpected non-numeric value 'n/a'"):
        validate_raw(frame_from(rows))


def test_identical_duplicate_rows_collapse() -> None:
    rows = rows_for("Finland", "FIN")
    rows.append(dict(rows[0]))

    cleaned, report = validate_raw(frame_from(rows))

    assert report.duplicate_rows_collapsed == 1
    assert len(cleaned) == len(INDICATORS)
    assert "Collapsed 1 identical duplicate" in report.warnings[0]


def test_equivalent_numeric_text_is_not_a_conflict() -> None:
    rows = rows_for("Finland", "FIN", fill="1")
    copy = dict(rows[0])
    copy[YEAR_2010] = "1.0"
    copy[YEAR_2011] = "1.00"
    rows.append(copy)

    _cleaned, report = validate_raw(frame_from(rows))

    assert report.duplicate_rows_collapsed == 1


def test_missing_markers_match_each_other_when_collapsing() -> None:
    rows = rows_for("Finland", "FIN", fill="..")
    copy = dict(rows[0])
    copy[YEAR_2010] = ""
    copy[YEAR_2011] = ".."
    rows.append(copy)

    _cleaned, report = validate_raw(frame_from(rows))

    assert report.duplicate_rows_collapsed == 1


def test_conflicting_duplicate_rows_raise() -> None:
    rows = rows_for("Finland", "FIN")
    copy = dict(rows[0])
    copy[YEAR_2010] = "99"
    rows.append(copy)

    with pytest.raises(ValidationError, match="Conflicting duplicate rows"):
        validate_raw(frame_from(rows))


def test_primary_and_pre_primary_codes_stay_distinct() -> None:
    primary = INDICATORS["UIS.X.US.1.FSGOV"]
    pre_primary = INDICATORS["UIS.X.US.02.FSGOV"]

    assert primary.code != pre_primary.code
    assert primary.level == "primary"
    assert pre_primary.level == "pre_primary"
    assert primary.measure == pre_primary.measure == "expenditure_usd_millions"
    assert INDICATORS["SE.PRM.TCHR"].level == "primary"
    assert INDICATORS["SE.PRE.TCHR"].level == "pre_primary"


def test_aggregate_list_excludes_name_traps() -> None:
    for code in ["ZAF", "EGY", "CAF", "SAU", "ARE", "SYR", "TTO", "ASM", "PRI"]:
        assert code not in AGGREGATE_CODES
    assert "PRE" in AGGREGATE_CODES
    assert "WLD" in AGGREGATE_CODES
    assert all(len(code) == 3 and code.isalpha() for code in AGGREGATE_CODES)


def test_missing_csv_names_the_expected_path(tmp_path) -> None:
    missing = tmp_path / "world_bank_education.csv"

    with pytest.raises(ValidationError, match=str(missing)):
        load_education_csv(missing)


def test_invalid_country_code_raises() -> None:
    rows = rows_for("Finland", "FIN")
    for row in rows:
        row["Country Code"] = "F1"

    with pytest.raises(ValidationError, match="Invalid Country Code"):
        validate_raw(frame_from(rows))
