"""Three figures and two compact tables from the validated coverage and ratio outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from education_indicators.analysis import COVERAGE_COLUMNS
from education_indicators.codes import INDICATORS
from education_indicators.load import project_root
from education_indicators.metrics import EDUCATION_LEVELS
from education_indicators.validate import ValidationError

TEACHER_YEAR = 2016
EXPENDITURE_YEAR = 2011
TEACHER_DISPLAY_SCALE = 100
EXPENDITURE_YSCALE = "linear"
PRIMARY_ENROLMENT_CODE = "SE.PRM.ENRL"
LITERACY_CODE = "SE.ADT.LITR.ZS"
LITERACY_YEARS = tuple(range(2010, 2020))
SCALE_TITLE = "Largest reported primary education systems by enrolment"
SCALE_RULE = (
    "Year is the earliest year with the most country/territory entities "
    "reporting primary enrolment (SE.PRM.ENRL)."
)

FIGURE_COVERAGE = "reporting_coverage_heatmap.png"
FIGURE_TEACHERS = "teacher_capacity_2016.png"
FIGURE_EXPENDITURE = "expenditure_per_student_2011.png"
SCALE_TABLE_NAME = "primary_enrolment_scale_top10.csv"
LITERACY_TABLE_NAME = "literacy_coverage_2010_2019.csv"

# Human labels, grouped by measure. Series codes stay in INDICATORS.
INDICATOR_LABELS: tuple[tuple[str, str], ...] = (
    ("SE.PRE.TCHR", "Pre-primary teachers"),
    ("SE.PRM.TCHR", "Primary teachers"),
    ("SE.SEC.TCHR", "Secondary teachers"),
    ("SE.PRE.ENRL", "Pre-primary enrolment"),
    ("SE.PRM.ENRL", "Primary enrolment"),
    ("SE.SEC.ENRL", "Secondary enrolment"),
    ("UIS.X.US.02.FSGOV", "Pre-primary expenditure"),
    ("UIS.X.US.1.FSGOV", "Primary expenditure"),
    ("UIS.X.US.2T3.FSGOV", "Secondary expenditure"),
    ("SE.ADT.LITR.ZS", "Adult literacy rate"),
)

LEVEL_LABELS = {
    "pre_primary": "Pre-primary",
    "primary": "Primary",
    "secondary": "Secondary",
}

_INK = "#1B3A4B"
_BOX = "#D6E2EA"
_POINT = "#2F5D7C"


def coverage_matrix(coverage: pd.DataFrame) -> pd.DataFrame:
    """Pivot reporting counts. Every indicator-year stays, including zeros."""

    frame = coverage.copy(deep=True)
    _require_columns(frame, COVERAGE_COLUMNS, "coverage")
    codes = [code for code, _label in INDICATOR_LABELS]
    if set(codes) != set(INDICATORS):
        raise ValidationError("The heatmap label list does not match the ten indicators.")
    years = sorted(int(year) for year in frame["year"].unique())
    if not years:
        raise ValidationError("Coverage has no years to plot.")
    totals = frame["total_country_territory_entities"].drop_duplicates()
    if len(totals) != 1:
        raise ValidationError("Coverage has more than one entity denominator.")
    rows = []
    for code, _label in INDICATOR_LABELS:
        subset = frame.loc[frame["indicator_code"] == code]
        by_year = {
            int(year): int(count)
            for year, count in zip(subset["year"], subset["reporting_entities"], strict=True)
        }
        if set(by_year) != set(years):
            raise ValidationError(
                f"{code} is missing at least one year. Zero-reporting years must stay in the table."
            )
        rows.append([by_year[year] for year in years])
    labels = [label for _code, label in INDICATOR_LABELS]
    matrix = pd.DataFrame(rows, index=labels, columns=years)
    matrix.index.name = "indicator"
    matrix.columns.name = "year"
    return matrix


def distribution_slice(frame: pd.DataFrame, year: int, value_column: str) -> pd.DataFrame:
    """Copy the rows for one year. High values and exact zeros stay."""

    copied = frame.copy(deep=True)
    _require_columns(
        copied,
        ["country_code", "education_level", "year", value_column],
        value_column,
    )
    subset = copied.loc[copied["year"] == year].copy()
    present = set(subset["education_level"])
    if present != set(EDUCATION_LEVELS):
        raise ValidationError(
            f"Year {year} must contain pre-primary, primary, and secondary rows."
        )
    return subset.reset_index(drop=True)


def level_counts(subset: pd.DataFrame) -> dict[str, int]:
    return {
        level: int((subset["education_level"] == level).sum())
        for level in EDUCATION_LEVELS
    }


def distribution_tick_labels(
    subset: pd.DataFrame,
    median_labels: list[str] | None = None,
) -> list[str]:
    counts = level_counts(subset)
    labels = [f"{LEVEL_LABELS[level]}\n(n={counts[level]})" for level in EDUCATION_LEVELS]
    if median_labels:
        labels = [
            f"{label}\n{note}"
            for label, note in zip(labels, median_labels, strict=True)
        ]
    return labels


def select_primary_enrolment_year(coverage: pd.DataFrame) -> int:
    """Earliest year with the most entities reporting primary enrolment."""

    frame = coverage.copy(deep=True)
    subset = frame.loc[frame["indicator_code"] == PRIMARY_ENROLMENT_CODE]
    if subset.empty:
        raise ValidationError("Coverage has no primary enrolment rows.")
    maximum = subset["reporting_entities"].max()
    years = subset.loc[subset["reporting_entities"] == maximum, "year"]
    return int(years.min())


def primary_enrolment_scale_table(tidy: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Ten largest primary enrolment counts in the best-covered reporting year."""

    observations = tidy.copy(deep=True)
    year = select_primary_enrolment_year(coverage)
    rows = observations.loc[
        (observations["entity_type"] == "country")
        & (observations["indicator_code"] == PRIMARY_ENROLMENT_CODE)
        & (observations["year"] == year)
        & observations["value"].notna()
    ].copy()
    table = pd.DataFrame(
        {
            "country_code": rows["country_code"].astype(str),
            "country_name": rows["country_name"].astype(str),
            "year": rows["year"].astype(int),
            "primary_enrolment": rows["value"].astype(float),
        }
    )
    table = table.sort_values(
        ["primary_enrolment", "country_code"],
        ascending=[False, True],
        kind="mergesort",
    ).head(10)
    table = table.reset_index(drop=True)
    if (table["primary_enrolment"] % 1 == 0).all():
        table["primary_enrolment"] = table["primary_enrolment"].astype("int64")
    return table


def literacy_coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    """Adult literacy reporting counts for 2010–2019. Not a literacy average."""

    frame = coverage.copy(deep=True)
    subset = frame.loc[frame["indicator_code"] == LITERACY_CODE].copy()
    subset = subset.loc[subset["year"].isin(LITERACY_YEARS)]
    found = sorted(int(year) for year in subset["year"])
    if found != list(LITERACY_YEARS):
        raise ValidationError("Literacy coverage is missing a year from 2010–2019.")
    summary = subset.loc[:, ["year", "reporting_entities", "coverage_percent"]].copy()
    summary = summary.sort_values("year", kind="mergesort").reset_index(drop=True)
    summary["year"] = summary["year"].astype(int)
    summary["reporting_entities"] = summary["reporting_entities"].astype(int)
    return summary


def coverage_subtitle(total: int) -> str:
    return f"Country/territory entities reporting a numeric value; maximum = {total}."


def teacher_display_values(ratios: pd.Series) -> pd.Series:
    """Presentation scale only. The stored ratio is teachers per enrolled student."""

    return ratios.astype(float) * TEACHER_DISPLAY_SCALE


def teacher_display_frame(teachers: pd.DataFrame) -> pd.DataFrame:
    subset = distribution_slice(teachers, TEACHER_YEAR, "teachers_per_enrolled_student")
    display = subset.copy()
    display["teachers_per_100_enrolled_students"] = teacher_display_values(
        display["teachers_per_enrolled_student"]
    )
    return display


def level_medians(subset: pd.DataFrame, value_column: str) -> dict[str, float]:
    medians = {}
    for level in EDUCATION_LEVELS:
        values = subset.loc[subset["education_level"] == level, value_column].astype(float)
        medians[level] = float(values.median())
    return medians


def format_median_usd(value: float) -> str:
    return f"Median: ${round(value):,}"


def plot_reporting_coverage(coverage: pd.DataFrame, path: Path) -> Path:
    matrix = coverage_matrix(coverage)
    total = int(coverage["total_country_territory_entities"].iloc[0])
    values = matrix.to_numpy()
    with _style():
        figure, axis = plt.subplots(figsize=(12.4, 7.4), constrained_layout=True)
        image = axis.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=total, interpolation="nearest")
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels([str(year) for year in matrix.columns])
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(list(matrix.index))
        axis.set_xlabel("Year")
        axis.set_ylabel("Indicator")
        midpoint = total / 2
        for row_index, row in enumerate(values):
            for column_index, count in enumerate(row):
                axis.text(
                    column_index,
                    row_index,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if count > midpoint else _INK,
                )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
        colorbar.set_label(f"Reporting entities (0 to {total})")
        figure.suptitle("Reporting coverage by indicator and year", fontsize=14)
        axis.set_title(coverage_subtitle(total), fontsize=10, color="#333333", pad=8)
        _save(figure, path)
    return path


def plot_teacher_capacity(teachers: pd.DataFrame, path: Path) -> Path:
    display = teacher_display_frame(teachers)
    _draw_distribution(
        display,
        value_column="teachers_per_100_enrolled_students",
        path=path,
        title=f"Reported teachers per 100 enrolled students, {TEACHER_YEAR}",
        subtitle="Country/territory entities with both teacher and enrolment data.",
        ylabel="Reported teachers per 100 enrolled students",
    )
    return path


def plot_expenditure_per_student(expenditure: pd.DataFrame, path: Path) -> Path:
    subset = distribution_slice(expenditure, EXPENDITURE_YEAR, "expenditure_per_enrolled_student_usd")
    medians = level_medians(subset, "expenditure_per_enrolled_student_usd")
    _draw_distribution(
        subset,
        value_column="expenditure_per_enrolled_student_usd",
        path=path,
        title=f"Nominal expenditure per enrolled student, {EXPENDITURE_YEAR}",
        subtitle="Current US dollars, not adjusted for inflation. Reported zeros are retained.",
        ylabel="Current US$ per enrolled student",
        median_labels=[format_median_usd(medians[level]) for level in EDUCATION_LEVELS],
        yscale=EXPENDITURE_YSCALE,
    )
    return path


def write_figures(coverage: pd.DataFrame, metrics, output_dir: Path | None = None) -> dict[str, Path]:
    directory = _output_dir(output_dir)
    coverage_path = directory / FIGURE_COVERAGE
    teacher_path = directory / FIGURE_TEACHERS
    expenditure_path = directory / FIGURE_EXPENDITURE
    plot_reporting_coverage(coverage, coverage_path)
    plot_teacher_capacity(metrics.teachers, teacher_path)
    plot_expenditure_per_student(metrics.expenditure, expenditure_path)
    return {
        "coverage": coverage_path,
        "teachers": teacher_path,
        "expenditure": expenditure_path,
    }


def write_scale_table(table: pd.DataFrame, output_dir: Path | None = None) -> Path:
    path = _output_dir(output_dir) / SCALE_TABLE_NAME
    _write_commented_csv(path, (SCALE_TITLE, SCALE_RULE), table)
    return path


def write_literacy_table(summary: pd.DataFrame, output_dir: Path | None = None) -> Path:
    path = _output_dir(output_dir) / LITERACY_TABLE_NAME
    note = (
        "Adult literacy reporting coverage, 2010-2019. "
        "Counts are country/territory entities with a numeric value. "
        "This is not an average literacy rate."
    )
    _write_commented_csv(path, (note,), summary)
    return path


def render_literacy_summary(summary: pd.DataFrame) -> str:
    lines = [
        "Adult literacy reporting coverage, 2010-2019",
        "Counts are entities with a numeric value. This is not an average literacy rate.",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"  {int(row.year)}: {int(row.reporting_entities)} entities ({row.coverage_percent:.2f}%)"
        )
    return "\n".join(lines)


def render_scale_table(table: pd.DataFrame) -> str:
    lines = [SCALE_TITLE, SCALE_RULE, "Absolute enrolment describes system scale, not performance."]
    lines.append(table.to_string(index=False))
    return "\n".join(lines)


def _draw_distribution(
    subset: pd.DataFrame,
    value_column: str,
    path: Path,
    title: str,
    subtitle: str,
    ylabel: str,
    median_labels: list[str] | None = None,
    yscale: str = "linear",
) -> None:
    labels = distribution_tick_labels(subset, median_labels)
    series = [
        subset.loc[subset["education_level"] == level, value_column].astype(float).tolist()
        for level in EDUCATION_LEVELS
    ]
    with _style():
        figure, axis = plt.subplots(figsize=(8.8, 6.4), constrained_layout=True)
        boxes = axis.boxplot(
            series,
            tick_labels=labels,
            showfliers=False,
            widths=0.55,
            patch_artist=True,
            medianprops={"color": _INK, "linewidth": 1.4},
            boxprops={"facecolor": _BOX, "edgecolor": _INK},
            whiskerprops={"color": _INK},
            capprops={"color": _INK},
        )
        del boxes
        for position, level in enumerate(EDUCATION_LEVELS, start=1):
            level_rows = subset.loc[subset["education_level"] == level]
            offsets = _jitter(level_rows["country_code"].astype(str).tolist())
            axis.scatter(
                [position + offset for offset in offsets],
                level_rows[value_column].astype(float).tolist(),
                s=16,
                color=_POINT,
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )
        axis.set_ylabel(ylabel)
        axis.set_yscale(yscale)
        axis.set_ylim(bottom=0)
        axis.yaxis.grid(True, color="#E6EEF2", linewidth=0.8)
        axis.set_axisbelow(True)
        figure.suptitle(title, fontsize=14)
        axis.set_title(subtitle, fontsize=10, color="#333333", pad=8)
        _save(figure, path)


def _jitter(codes: list[str]) -> list[float]:
    offsets = []
    for code in codes:
        spread = sum(ord(character) for character in code) % 100
        offsets.append((spread / 99 - 0.5) * 0.28)
    return offsets


def _style():
    return plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": _INK,
            "text.color": _INK,
            "axes.labelcolor": _INK,
            "xtick.color": _INK,
            "ytick.color": _INK,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_commented_csv(path: Path, comments: tuple[str, ...], table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for comment in comments:
            handle.write(f"# {comment}\n")
        table.to_csv(handle, index=False)


def _output_dir(output_dir: Path | None) -> Path:
    return project_root() / "outputs" if output_dir is None else Path(output_dir)


def _require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{label} is missing columns: {', '.join(missing)}")
