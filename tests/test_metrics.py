"""Same-level ratio tests on synthetic tidy tables."""

from __future__ import annotations

import pandas as pd
import pytest

from education_indicators.metrics import (
    EXPENDITURE_PAIRS,
    TEACHER_PAIRS,
    USD_MILLIONS_TO_USD,
    derive_metrics,
)
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
    rows = [
        _row("FIN", "country", "SE.PRM.TCHR", 2010, 10),
        _row("FIN", "country", "SE.PRM.ENRL", 2010, 100),
        _row("FIN", "country", "SE.PRE.TCHR", 2010, 999),
        _row("FIN", "country", "SE.PRE.ENRL", 2010, 1),
        _row("FIN", "country", "UIS.X.US.1.FSGOV", 2010, 2),
        _row("FIN", "country", "UIS.X.US.02.FSGOV", 2010, 50),
        _row("FIN", "country", "SE.SEC.ENRL", 2010, 40),
        _row("SWE", "country", "SE.PRM.TCHR", 2010, 5),
        _row("WLD", "aggregate", "SE.PRM.TCHR", 2010, 1000),
        _row("WLD", "aggregate", "SE.PRM.ENRL", 2010, 10),
        _row("FIN", "country", "SE.PRM.TCHR", 2011, None),
        _row("FIN", "country", "SE.PRM.ENRL", 2011, None),
        _row("SWE", "country", "SE.PRM.TCHR", 2011, None),
    ]
    return _tidy(rows)


def test_exact_level_pairs_are_hard_coded() -> None:
    assert TEACHER_PAIRS["pre_primary"] == ("SE.PRE.TCHR", "SE.PRE.ENRL")
    assert TEACHER_PAIRS["primary"] == ("SE.PRM.TCHR", "SE.PRM.ENRL")
    assert TEACHER_PAIRS["secondary"] == ("SE.SEC.TCHR", "SE.SEC.ENRL")
    assert EXPENDITURE_PAIRS["pre_primary"] == ("UIS.X.US.02.FSGOV", "SE.PRE.ENRL")
    assert EXPENDITURE_PAIRS["primary"] == ("UIS.X.US.1.FSGOV", "SE.PRM.ENRL")
    assert EXPENDITURE_PAIRS["secondary"] == ("UIS.X.US.2T3.FSGOV", "SE.SEC.ENRL")
    assert EXPENDITURE_PAIRS["primary"][0] != TEACHER_PAIRS["pre_primary"][0]


def test_primary_teacher_ratio_ignores_pre_primary_and_aggregates() -> None:
    metrics = derive_metrics(_sample())
    primary = metrics.teachers.loc[metrics.teachers["education_level"] == "primary"]

    assert primary["country_code"].tolist() == ["FIN"]
    assert primary["teachers"].tolist() == [10]
    assert primary["enrolment"].tolist() == [100]
    assert primary["teachers_per_enrolled_student"].tolist() == [0.1]
    assert "WLD" not in set(metrics.teachers["country_code"])
    assert "SWE" not in set(primary["country_code"])

    pre_primary = metrics.teachers.loc[metrics.teachers["education_level"] == "pre_primary"].iloc[0]
    assert pre_primary["teachers"] == 999
    assert pre_primary["enrolment"] == 1
    assert pre_primary["teachers_per_enrolled_student"] == 999


def test_primary_expenditure_uses_primary_millions_times_one_million() -> None:
    metrics = derive_metrics(_sample())
    primary = metrics.expenditure.loc[metrics.expenditure["education_level"] == "primary"].iloc[0]
    pre_primary = metrics.expenditure.loc[
        metrics.expenditure["education_level"] == "pre_primary"
    ].iloc[0]

    assert primary["expenditure_usd_millions"] == 2
    assert primary["enrolment"] == 100
    assert primary["expenditure_per_enrolled_student_usd"] == 2 * USD_MILLIONS_TO_USD / 100
    assert pre_primary["expenditure_usd_millions"] == 50
    assert pre_primary["expenditure_per_enrolled_student_usd"] == 50 * USD_MILLIONS_TO_USD / 1
    assert "WLD" not in set(metrics.expenditure["country_code"])


def test_missing_either_side_does_not_create_a_ratio() -> None:
    metrics = derive_metrics(_sample())
    secondary = metrics.teachers.loc[metrics.teachers["education_level"] == "secondary"]

    assert secondary.empty
    assert "SWE" not in set(metrics.teachers["country_code"])


def test_zero_enrolment_is_excluded_and_counted() -> None:
    rows = [
        _row("FIN", "country", "SE.PRM.TCHR", 2010, 4),
        _row("FIN", "country", "SE.PRM.ENRL", 2010, 0),
    ]
    metrics = derive_metrics(_tidy(rows))

    assert metrics.teachers.empty
    assert metrics.zero_enrolment_teacher_rows == 1
    paired = metrics.paired_coverage.loc[
        (metrics.paired_coverage["metric"] == "teachers_per_enrolled_student")
        & (metrics.paired_coverage["education_level"] == "primary")
    ].iloc[0]
    assert paired["numerator_reporting_entities"] == 1
    assert paired["denominator_reporting_entities"] == 1
    assert paired["paired_reporting_entities"] == 0


def test_negative_enrolment_fails() -> None:
    rows = [
        _row("FIN", "country", "SE.PRM.TCHR", 2010, 4),
        _row("FIN", "country", "SE.PRM.ENRL", 2010, -1),
    ]
    with pytest.raises(ValidationError, match="Negative"):
        derive_metrics(_tidy(rows))


def test_negative_teacher_count_fails() -> None:
    rows = [
        _row("FIN", "country", "SE.SEC.TCHR", 2010, -2),
        _row("FIN", "country", "SE.SEC.ENRL", 2010, 10),
    ]
    with pytest.raises(ValidationError, match="Negative"):
        derive_metrics(_tidy(rows))


def test_negative_expenditure_fails() -> None:
    rows = [
        _row("FIN", "country", "UIS.X.US.2T3.FSGOV", 2010, -3),
        _row("FIN", "country", "SE.SEC.ENRL", 2010, 10),
    ]
    with pytest.raises(ValidationError, match="Negative"):
        derive_metrics(_tidy(rows))


def test_paired_coverage_counts_only_complete_valid_pairs() -> None:
    metrics = derive_metrics(_sample())
    primary_2010 = metrics.paired_coverage.loc[
        (metrics.paired_coverage["metric"] == "teachers_per_enrolled_student")
        & (metrics.paired_coverage["education_level"] == "primary")
        & (metrics.paired_coverage["year"] == 2010)
    ].iloc[0]
    primary_2011 = metrics.paired_coverage.loc[
        (metrics.paired_coverage["metric"] == "teachers_per_enrolled_student")
        & (metrics.paired_coverage["education_level"] == "primary")
        & (metrics.paired_coverage["year"] == 2011)
    ].iloc[0]

    assert primary_2010["total_country_territory_entities"] == 2
    assert primary_2010["numerator_reporting_entities"] == 2
    assert primary_2010["denominator_reporting_entities"] == 1
    assert primary_2010["paired_reporting_entities"] == 1
    assert primary_2010["paired_coverage_percent"] == 50
    assert primary_2011["paired_reporting_entities"] == 0
    assert primary_2011["paired_coverage_percent"] == 0


def test_outputs_do_not_depend_on_row_order_and_keys_are_unique() -> None:
    tidy = _sample()
    forward = derive_metrics(tidy)
    backward = derive_metrics(tidy.iloc[::-1].reset_index(drop=True))

    pd.testing.assert_frame_equal(forward.teachers, backward.teachers)
    pd.testing.assert_frame_equal(forward.expenditure, backward.expenditure)
    pd.testing.assert_frame_equal(forward.paired_coverage, backward.paired_coverage)
    assert not forward.teachers.duplicated(["country_code", "year", "education_level"]).any()
    assert not forward.expenditure.duplicated(["country_code", "year", "education_level"]).any()


def test_large_positive_ratio_is_kept() -> None:
    rows = [
        _row("FIN", "country", "SE.PRM.TCHR", 2010, 1000),
        _row("FIN", "country", "SE.PRM.ENRL", 2010, 1),
    ]
    metrics = derive_metrics(_tidy(rows))
    ratio = metrics.teachers["teachers_per_enrolled_student"].iloc[0]

    assert ratio == 1000
