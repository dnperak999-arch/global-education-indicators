"""Figures and the primary-enrolment scale table. Synthetic frames only."""

from __future__ import annotations

import pandas as pd
import pytest

from education_indicators.codes import INDICATORS
from education_indicators.plots import (
    EXPENDITURE_YEAR,
    EXPENDITURE_YSCALE,
    FIGURE_COVERAGE,
    FIGURE_EXPENDITURE,
    FIGURE_TEACHERS,
    SCALE_TITLE,
    TEACHER_YEAR,
    coverage_matrix,
    coverage_subtitle,
    distribution_slice,
    distribution_tick_labels,
    format_median_usd,
    level_medians,
    literacy_coverage_summary,
    plot_expenditure_per_student,
    plot_reporting_coverage,
    plot_teacher_capacity,
    primary_enrolment_scale_table,
    select_primary_enrolment_year,
    teacher_display_frame,
    teacher_display_values,
    write_literacy_table,
    write_scale_table,
)
from education_indicators.validate import ValidationError

YEARS = list(range(2010, 2025))


def _coverage(reporting: dict[tuple[str, int], int] | None = None, total: int = 223) -> pd.DataFrame:
    rows = []
    for code in INDICATORS:
        for year in YEARS:
            count = 0 if reporting is None else reporting.get((code, year), 0)
            rows.append(
                {
                    "indicator_code": code,
                    "year": year,
                    "reporting_entities": count,
                    "total_country_territory_entities": total,
                    "missing_entities": total - count,
                    "coverage_percent": round(100 * count / total, 2),
                }
            )
    return pd.DataFrame(rows)


def _teachers() -> pd.DataFrame:
    rows = []
    for level, count, ratio in (
        ("pre_primary", 2, 0.230769),
        ("primary", 3, 0.05),
        ("secondary", 1, 0.04),
    ):
        for index in range(count):
            rows.append(
                {
                    "country_code": f"{level[:3].upper()}{index}",
                    "country_name": f"{level} {index}",
                    "year": TEACHER_YEAR,
                    "education_level": level,
                    "teachers": 10,
                    "enrolment": 100,
                    "teachers_per_enrolled_student": ratio,
                }
            )
    rows.append(
        {
            "country_code": "OLD",
            "country_name": "Other year",
            "year": 2015,
            "education_level": "primary",
            "teachers": 1,
            "enrolment": 10,
            "teachers_per_enrolled_student": 0.1,
        }
    )
    return pd.DataFrame(rows)


def _expenditure() -> pd.DataFrame:
    rows = []
    for level, values in (
        ("pre_primary", [0.0, 0.0, 1200.0]),
        ("primary", [800.0, 900.0]),
        ("secondary", [1500.0]),
    ):
        for index, value in enumerate(values):
            rows.append(
                {
                    "country_code": f"{level[:3].upper()}{index}",
                    "country_name": f"{level} {index}",
                    "year": EXPENDITURE_YEAR,
                    "education_level": level,
                    "expenditure_usd_millions": value,
                    "enrolment": 100,
                    "expenditure_per_enrolled_student_usd": value,
                }
            )
    rows.append(
        {
            "country_code": "OLD",
            "country_name": "Other year",
            "year": 2010,
            "education_level": "primary",
            "expenditure_usd_millions": 5.0,
            "enrolment": 100,
            "expenditure_per_enrolled_student_usd": 50000.0,
        }
    )
    return pd.DataFrame(rows)


def _tidy() -> pd.DataFrame:
    rows = [
        ("USA", "United States", "country", 2010, 500.0),
        ("IND", "India", "country", 2010, 900.0),
        ("CHN", "China", "country", 2010, 1000.0),
        ("WLD", "World", "aggregate", 2010, 999999.0),
        ("USA", "United States", "country", 2012, 50.0),
        ("FRA", "France", "country", 2012, 40.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["country_code", "country_name", "entity_type", "year", "value"],
    ).assign(indicator_code="SE.PRM.ENRL")


def test_coverage_matrix_keeps_ten_indicators_and_all_years_including_zeros():
    coverage = _coverage({("SE.PRM.ENRL", 2012): 4, ("SE.ADT.LITR.ZS", 2024): 0})
    matrix = coverage_matrix(coverage)
    assert matrix.shape == (10, 15)
    assert list(matrix.columns) == YEARS
    assert len(matrix.index) == len(INDICATORS)
    assert int(matrix.loc["Primary enrolment", 2012]) == 4
    assert int(matrix.loc["Adult literacy rate", 2024]) == 0
    assert (matrix.loc[:, 2021] == 0).all()


def test_coverage_matrix_rejects_a_dropped_year():
    coverage = _coverage().drop(index=_coverage().index[0])
    with pytest.raises(ValidationError, match="missing at least one year"):
        coverage_matrix(coverage)


def test_teacher_slice_is_2016_and_counts_match_labels():
    teachers = _teachers()
    subset = distribution_slice(teachers, TEACHER_YEAR, "teachers_per_enrolled_student")
    assert set(subset["year"]) == {TEACHER_YEAR}
    assert set(subset["education_level"]) == {"pre_primary", "primary", "secondary"}
    labels = distribution_tick_labels(subset)
    assert labels == ["Pre-primary\n(n=2)", "Primary\n(n=3)", "Secondary\n(n=1)"]
    assert (subset["teachers_per_enrolled_student"] == 0.230769).any()


def test_expenditure_slice_is_2011_and_keeps_zeros():
    expenditure = _expenditure()
    subset = distribution_slice(expenditure, EXPENDITURE_YEAR, "expenditure_per_enrolled_student_usd")
    assert set(subset["year"]) == {EXPENDITURE_YEAR}
    assert set(subset["education_level"]) == {"pre_primary", "primary", "secondary"}
    counts = subset.groupby("education_level").size().to_dict()
    assert counts == {"pre_primary": 3, "primary": 2, "secondary": 1}
    assert labels_match_rows(subset)
    zeros = subset.loc[subset["expenditure_per_enrolled_student_usd"] == 0]
    assert len(zeros) == 2
    assert set(zeros["education_level"]) == {"pre_primary"}


def labels_match_rows(subset: pd.DataFrame) -> bool:
    labels = distribution_tick_labels(subset)
    return labels == [
        "Pre-primary\n(n=3)",
        "Primary\n(n=2)",
        "Secondary\n(n=1)",
    ]


def test_scale_year_is_the_earliest_maximum():
    coverage = _coverage()
    coverage.loc[
        (coverage["indicator_code"] == "SE.PRM.ENRL") & (coverage["year"] == 2010),
        "reporting_entities",
    ] = 5
    coverage.loc[
        (coverage["indicator_code"] == "SE.PRM.ENRL") & (coverage["year"] == 2012),
        "reporting_entities",
    ] = 5
    coverage.loc[
        (coverage["indicator_code"] == "SE.PRM.ENRL") & (coverage["year"] == 2014),
        "reporting_entities",
    ] = 4
    assert select_primary_enrolment_year(coverage) == 2010
    assert select_primary_enrolment_year(coverage.copy()) == 2010


def test_scale_table_excludes_aggregates_and_sorts_descending(tmp_path):
    coverage = _coverage()
    coverage.loc[
        (coverage["indicator_code"] == "SE.PRM.ENRL") & (coverage["year"] == 2010),
        "reporting_entities",
    ] = 9
    table = primary_enrolment_scale_table(_tidy(), coverage)
    assert set(table["year"]) == {2010}
    assert "WLD" not in set(table["country_code"])
    assert list(table["country_code"]) == ["CHN", "IND", "USA"]
    assert list(table["primary_enrolment"]) == [1000, 900, 500]
    path = write_scale_table(table, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert SCALE_TITLE in text
    assert path.name == "primary_enrolment_scale_top10.csv"


def test_literacy_summary_keeps_every_year_from_2010_to_2019(tmp_path):
    coverage = _coverage({("SE.ADT.LITR.ZS", 2018): 79, ("SE.ADT.LITR.ZS", 2019): 1})
    summary = literacy_coverage_summary(coverage)
    assert list(summary["year"]) == list(range(2010, 2020))
    assert list(summary.columns) == ["year", "reporting_entities", "coverage_percent"]
    assert int(summary.loc[summary["year"] == 2018, "reporting_entities"].iloc[0]) == 79
    assert int(summary.loc[summary["year"] == 2019, "reporting_entities"].iloc[0]) == 1
    path = write_literacy_table(summary, tmp_path)
    assert "not an average literacy rate" in path.read_text(encoding="utf-8")


def test_coverage_subtitle_stays_short_and_matrix_is_unchanged():
    coverage = _coverage({("SE.PRM.ENRL", 2012): 4})
    before = coverage.copy(deep=True)
    matrix = coverage_matrix(coverage)
    assert coverage_subtitle(223) == (
        "Country/territory entities reporting a numeric value; maximum = 223."
    )
    assert matrix.shape == (10, 15)
    assert int(matrix.loc["Primary enrolment", 2012]) == 4
    pd.testing.assert_frame_equal(coverage, before)


def test_teacher_display_is_ratio_times_100_and_leaves_the_ratio_column():
    teachers = _teachers()
    before = teachers.copy(deep=True)
    ratios = teachers["teachers_per_enrolled_student"]
    displayed = teacher_display_values(ratios)
    pd.testing.assert_series_equal(displayed, ratios * 100)
    pd.testing.assert_series_equal(teachers["teachers_per_enrolled_student"], ratios)
    frame = teacher_display_frame(teachers)
    assert set(frame["year"]) == {TEACHER_YEAR}
    assert frame.groupby("education_level").size().to_dict() == {
        "pre_primary": 2,
        "primary": 3,
        "secondary": 1,
    }
    scaled = frame["teachers_per_enrolled_student"] * 100
    assert frame["teachers_per_100_enrolled_students"].tolist() == scaled.tolist()
    pre_primary = frame.loc[frame["education_level"] == "pre_primary", "teachers_per_100_enrolled_students"]
    assert pre_primary.max() == pytest.approx(23.0769)
    pd.testing.assert_frame_equal(teachers, before)


def test_expenditure_medians_use_the_selected_rows_including_zeros():
    expenditure = _expenditure()
    before = expenditure.copy(deep=True)
    subset = distribution_slice(expenditure, EXPENDITURE_YEAR, "expenditure_per_enrolled_student_usd")
    medians = level_medians(subset, "expenditure_per_enrolled_student_usd")
    assert medians["pre_primary"] == 0.0
    assert medians["primary"] == 850.0
    assert medians["secondary"] == 1500.0
    assert int((subset["expenditure_per_enrolled_student_usd"] == 0).sum()) == 2
    assert format_median_usd(870.933423) == "Median: $871"
    assert format_median_usd(1035.551733) == "Median: $1,036"
    assert format_median_usd(1244.758845) == "Median: $1,245"
    notes = [format_median_usd(medians[level]) for level in ("pre_primary", "primary", "secondary")]
    assert distribution_tick_labels(subset, notes) == [
        "Pre-primary\n(n=3)\nMedian: $0",
        "Primary\n(n=2)\nMedian: $850",
        "Secondary\n(n=1)\nMedian: $1,500",
    ]
    assert EXPENDITURE_YSCALE == "linear"
    pd.testing.assert_frame_equal(expenditure, before)


def test_figure_calls_keep_years_scale_and_medians(tmp_path, monkeypatch):
    captured = {}

    def fake_draw(subset, value_column, path, title, subtitle, ylabel, median_labels=None, yscale="linear"):
        captured[path.name] = {
            "years": set(subset["year"]),
            "column": value_column,
            "values": subset[value_column].astype(float).tolist(),
            "counts": subset.groupby("education_level").size().to_dict(),
            "medians": median_labels,
            "yscale": yscale,
            "zeros": int((subset[value_column] == 0).sum()),
        }
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("education_indicators.plots._draw_distribution", fake_draw)
    teachers = _teachers()
    expenditure = _expenditure()
    plot_teacher_capacity(teachers, tmp_path / FIGURE_TEACHERS)
    plot_expenditure_per_student(expenditure, tmp_path / FIGURE_EXPENDITURE)

    teacher = captured[FIGURE_TEACHERS]
    assert teacher["years"] == {2016}
    assert teacher["column"] == "teachers_per_100_enrolled_students"
    assert teacher["counts"] == {"pre_primary": 2, "primary": 3, "secondary": 1}
    assert teacher["yscale"] == "linear"
    assert teacher["values"] == pytest.approx([value * 100 for value in (
        teachers.loc[teachers["year"] == 2016, "teachers_per_enrolled_student"].tolist()
    )])

    spend = captured[FIGURE_EXPENDITURE]
    assert spend["years"] == {2011}
    assert spend["yscale"] == "linear"
    assert spend["zeros"] == 2
    assert spend["counts"] == {"pre_primary": 3, "primary": 2, "secondary": 1}
    assert spend["medians"] == ["Median: $0", "Median: $850", "Median: $1,500"]
    assert 0.0 in spend["values"]


def test_plotting_does_not_modify_inputs(tmp_path):
    coverage = _coverage({("SE.PRM.ENRL", 2012): 10})
    teachers = _teachers()
    expenditure = _expenditure()
    coverage_before = coverage.copy(deep=True)
    teachers_before = teachers.copy(deep=True)
    expenditure_before = expenditure.copy(deep=True)

    coverage_path = plot_reporting_coverage(coverage, tmp_path / FIGURE_COVERAGE)
    teacher_path = plot_teacher_capacity(teachers, tmp_path / FIGURE_TEACHERS)
    expenditure_path = plot_expenditure_per_student(expenditure, tmp_path / FIGURE_EXPENDITURE)

    pd.testing.assert_frame_equal(coverage, coverage_before)
    pd.testing.assert_frame_equal(teachers, teachers_before)
    pd.testing.assert_frame_equal(expenditure, expenditure_before)
    for path in (coverage_path, teacher_path, expenditure_path):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
