"""Same-level teacher and expenditure ratios for country/territory entities.

Pairs are exact series codes. Aggregates are excluded. A ratio is reported
teachers divided by reported enrolment, or current US dollars per enrolled
student from the extract's US$ millions series. Neither ratio is a quality
measure, and expenditure is not inflation-adjusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from education_indicators.codes import INDICATORS
from education_indicators.load import project_root
from education_indicators.validate import ValidationError

EDUCATION_LEVELS: tuple[str, ...] = ("pre_primary", "primary", "secondary")

TEACHER_PAIRS: dict[str, tuple[str, str]] = {
    "pre_primary": ("SE.PRE.TCHR", "SE.PRE.ENRL"),
    "primary": ("SE.PRM.TCHR", "SE.PRM.ENRL"),
    "secondary": ("SE.SEC.TCHR", "SE.SEC.ENRL"),
}

EXPENDITURE_PAIRS: dict[str, tuple[str, str]] = {
    "pre_primary": ("UIS.X.US.02.FSGOV", "SE.PRE.ENRL"),
    "primary": ("UIS.X.US.1.FSGOV", "SE.PRM.ENRL"),
    "secondary": ("UIS.X.US.2T3.FSGOV", "SE.SEC.ENRL"),
}

USD_MILLIONS_TO_USD = 1_000_000

TEACHER_COLUMNS: tuple[str, ...] = (
    "country_code",
    "country_name",
    "year",
    "education_level",
    "teachers",
    "enrolment",
    "teachers_per_enrolled_student",
)

EXPENDITURE_COLUMNS: tuple[str, ...] = (
    "country_code",
    "country_name",
    "year",
    "education_level",
    "expenditure_usd_millions",
    "enrolment",
    "expenditure_per_enrolled_student_usd",
)

PAIRED_COLUMNS: tuple[str, ...] = (
    "metric",
    "education_level",
    "year",
    "total_country_territory_entities",
    "numerator_reporting_entities",
    "denominator_reporting_entities",
    "paired_reporting_entities",
    "paired_coverage_percent",
)

TEACHER_METRIC = "teachers_per_enrolled_student"
EXPENDITURE_METRIC = "expenditure_per_enrolled_student_usd"

_RATIO_INPUT_CODES = frozenset(
    code for pair in (*TEACHER_PAIRS.values(), *EXPENDITURE_PAIRS.values()) for code in pair
)


@dataclass(frozen=True)
class DerivedMetrics:
    teachers: pd.DataFrame
    expenditure: pd.DataFrame
    paired_coverage: pd.DataFrame
    zero_enrolment_teacher_rows: int
    zero_enrolment_expenditure_rows: int


def derive_metrics(tidy: pd.DataFrame) -> DerivedMetrics:
    """Build both ratio tables and the paired-coverage grid."""
    _check_pair_definitions()
    countries = _country_rows(tidy)
    _reject_negative_inputs(countries)
    total = int(countries["country_code"].nunique())
    if total == 0:
        raise ValidationError("No country/territory entities in the tidy table.")
    years = tuple(sorted(int(year) for year in tidy["year"].unique()))

    teacher_frames: list[pd.DataFrame] = []
    expenditure_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    zero_teacher = 0
    zero_expenditure = 0

    for level in EDUCATION_LEVELS:
        teacher_code, enrolment_code = TEACHER_PAIRS[level]
        paired_teachers = _pair_values(
            countries,
            numerator_code=teacher_code,
            denominator_code=enrolment_code,
            numerator_name="teachers",
        )
        teacher_rows, teacher_zeros = _teacher_ratios(paired_teachers, level)
        teacher_frames.append(teacher_rows)
        zero_teacher += teacher_zeros
        coverage_frames.append(
            _paired_coverage(paired_teachers, TEACHER_METRIC, level, years, total, "teachers")
        )

        spend_code, spend_enrolment_code = EXPENDITURE_PAIRS[level]
        if spend_enrolment_code != enrolment_code:
            raise ValidationError(
                f"{level} expenditure does not use the same enrolment series as teachers."
            )
        paired_spend = _pair_values(
            countries,
            numerator_code=spend_code,
            denominator_code=spend_enrolment_code,
            numerator_name="expenditure_usd_millions",
        )
        spend_rows, spend_zeros = _expenditure_ratios(paired_spend, level)
        expenditure_frames.append(spend_rows)
        zero_expenditure += spend_zeros
        coverage_frames.append(
            _paired_coverage(
                paired_spend,
                EXPENDITURE_METRIC,
                level,
                years,
                total,
                "expenditure_usd_millions",
            )
        )

    teachers = _concat(teacher_frames, TEACHER_COLUMNS)
    expenditure = _concat(expenditure_frames, EXPENDITURE_COLUMNS)
    paired = _concat(coverage_frames, PAIRED_COLUMNS)
    _require_unique(teachers, "teacher capacity")
    _require_unique(expenditure, "expenditure per enrolled student")
    return DerivedMetrics(
        teachers=teachers,
        expenditure=expenditure,
        paired_coverage=paired,
        zero_enrolment_teacher_rows=zero_teacher,
        zero_enrolment_expenditure_rows=zero_expenditure,
    )


def write_metrics(metrics: DerivedMetrics) -> dict[str, Path]:
    """Write the three compact derived tables. The tidy extract is not written."""
    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "teachers": output_dir / "teacher_capacity.csv",
        "expenditure": output_dir / "expenditure_per_student.csv",
        "paired": output_dir / "paired_coverage.csv",
    }
    metrics.teachers.to_csv(paths["teachers"], index=False)
    metrics.expenditure.to_csv(paths["expenditure"], index=False)
    metrics.paired_coverage.to_csv(paths["paired"], index=False)
    return paths


def render_metric_summary(metrics: DerivedMetrics) -> str:
    """Text summary of paired coverage, distributions, and the largest ratios."""
    lines = [
        "",
        "Same-level ratios (country/territory entities only; aggregates excluded)",
        (
            "Zero-enrolment rows excluded from teacher ratios: "
            f"{metrics.zero_enrolment_teacher_rows}"
        ),
        (
            "Zero-enrolment rows excluded from expenditure ratios: "
            f"{metrics.zero_enrolment_expenditure_rows}"
        ),
        (
            "Expenditure per enrolled student is nominal current US$: "
            "US$ millions from the extract, multiplied by 1,000,000, divided by enrolment. "
            "It is not inflation-adjusted."
        ),
        (
            "Teachers per enrolled student is the reported teacher count divided by "
            "the reported enrolment count."
        ),
    ]
    lines.extend(_span_lines(metrics.paired_coverage, TEACHER_METRIC))
    lines.extend(_span_lines(metrics.paired_coverage, EXPENDITURE_METRIC))
    lines.append("")
    lines.append("Paired reporting entities by year")
    for metric in (TEACHER_METRIC, EXPENDITURE_METRIC):
        lines.append(metric)
        lines.append(_paired_grid(metrics.paired_coverage, metric))
    lines.append("")
    lines.append("teachers_per_enrolled_student distribution (min, median, max)")
    lines.append(_distribution(metrics.teachers, metrics.paired_coverage, TEACHER_METRIC, "teachers_per_enrolled_student"))
    lines.append("")
    lines.append(
        "expenditure_per_enrolled_student_usd distribution "
        "(min, median, max; nominal current US$ per enrolled student)"
    )
    lines.append(
        _distribution(
            metrics.expenditure,
            metrics.paired_coverage,
            EXPENDITURE_METRIC,
            "expenditure_per_enrolled_student_usd",
        )
    )
    lines.append("")
    lines.append("Largest ratios in the whole period (diagnostic, not a ranking)")
    lines.extend(_top_lines(metrics.teachers, "teachers_per_enrolled_student", "teachers"))
    lines.extend(
        _top_lines(
            metrics.expenditure,
            "expenditure_per_enrolled_student_usd",
            "expenditure_usd_millions",
        )
    )
    return "\n".join(lines)


def _check_pair_definitions() -> None:
    for level, (teacher_code, enrolment_code) in TEACHER_PAIRS.items():
        _expect(teacher_code, "teachers", level)
        _expect(enrolment_code, "enrolment", level)
    for level, (spend_code, enrolment_code) in EXPENDITURE_PAIRS.items():
        _expect(spend_code, "expenditure_usd_millions", level)
        _expect(enrolment_code, "enrolment", level)
        if enrolment_code != TEACHER_PAIRS[level][1]:
            raise ValidationError(
                f"{level} expenditure enrolment code does not match the teacher pair."
            )


def _expect(code: str, measure: str, level: str) -> None:
    spec = INDICATORS[code]
    if spec.measure != measure or spec.level != level:
        raise ValidationError(
            f"{code} is {spec.measure}/{spec.level}, expected {measure}/{level}."
        )


def _country_rows(tidy: pd.DataFrame) -> pd.DataFrame:
    required = {"country_code", "country_name", "entity_type", "indicator_code", "year", "value"}
    missing = sorted(required - set(tidy.columns))
    if missing:
        raise ValidationError(f"Tidy table is missing columns: {missing}")
    unknown = sorted(set(tidy["entity_type"]) - {"country", "aggregate"})
    if unknown:
        raise ValidationError(f"Unknown entity_type values: {unknown}")
    countries = tidy.loc[tidy["entity_type"] == "country"].copy()
    duplicated = countries.duplicated(["country_code", "indicator_code", "year"])
    if duplicated.any():
        raise ValidationError("Country/territory indicator-year rows are not unique.")
    return countries.sort_values(
        ["country_code", "indicator_code", "year"],
        kind="mergesort",
    ).reset_index(drop=True)


def _reject_negative_inputs(countries: pd.DataFrame) -> None:
    relevant = countries.loc[countries["indicator_code"].isin(_RATIO_INPUT_CODES)]
    negative = relevant.loc[relevant["value"].notna() & (relevant["value"] < 0)]
    if negative.empty:
        return
    sample = negative.loc[:, ["country_code", "indicator_code", "year", "value"]].head(5)
    raise ValidationError(
        "Negative teacher, enrolment, or expenditure values cannot be used in ratios.\n"
        f"{sample.to_string(index=False)}"
    )


def _pair_values(
    countries: pd.DataFrame,
    *,
    numerator_code: str,
    denominator_code: str,
    numerator_name: str,
) -> pd.DataFrame:
    left = _one_series(countries, numerator_code).rename(
        columns={"value": numerator_name, "country_name": "name_left"}
    )
    right = _one_series(countries, denominator_code).rename(
        columns={"value": "enrolment", "country_name": "name_right"}
    )
    merged = left.merge(right, on=["country_code", "year"], how="outer")
    both_names = merged["name_left"].notna() & merged["name_right"].notna()
    conflict = both_names & (merged["name_left"] != merged["name_right"])
    if conflict.any():
        raise ValidationError("A country code has more than one country name in a ratio pair.")
    merged["country_name"] = merged["name_left"].fillna(merged["name_right"])
    return merged.loc[:, ["country_code", "country_name", "year", numerator_name, "enrolment"]]


def _one_series(countries: pd.DataFrame, code: str) -> pd.DataFrame:
    part = countries.loc[
        countries["indicator_code"] == code,
        ["country_code", "country_name", "year", "value"],
    ].copy()
    if part.duplicated(["country_code", "year"]).any():
        raise ValidationError(f"Duplicate country-year rows for {code}.")
    return part


def _teacher_ratios(paired: pd.DataFrame, level: str) -> tuple[pd.DataFrame, int]:
    complete, excluded = _complete_pairs(paired, "teachers")
    if complete.empty:
        return _empty(TEACHER_COLUMNS), excluded
    complete = complete.copy()
    complete["education_level"] = level
    complete["teachers_per_enrolled_student"] = complete["teachers"] / complete["enrolment"]
    return complete.loc[:, list(TEACHER_COLUMNS)], excluded


def _expenditure_ratios(paired: pd.DataFrame, level: str) -> tuple[pd.DataFrame, int]:
    complete, excluded = _complete_pairs(paired, "expenditure_usd_millions")
    if complete.empty:
        return _empty(EXPENDITURE_COLUMNS), excluded
    complete = complete.copy()
    complete["education_level"] = level
    complete["expenditure_per_enrolled_student_usd"] = (
        complete["expenditure_usd_millions"] * USD_MILLIONS_TO_USD / complete["enrolment"]
    )
    return complete.loc[:, list(EXPENDITURE_COLUMNS)], excluded


def _complete_pairs(paired: pd.DataFrame, numerator_name: str) -> tuple[pd.DataFrame, int]:
    negative_enrolment = paired["enrolment"].notna() & (paired["enrolment"] < 0)
    if negative_enrolment.any():
        raise ValidationError("Negative enrolment cannot be used as a ratio denominator.")
    negative_numerator = paired[numerator_name].notna() & (paired[numerator_name] < 0)
    if negative_numerator.any():
        raise ValidationError(f"Negative {numerator_name} cannot be used in a ratio.")
    zero_enrolment = (
        paired[numerator_name].notna()
        & paired["enrolment"].notna()
        & (paired["enrolment"] == 0)
    )
    complete = paired.loc[
        paired[numerator_name].notna() & paired["enrolment"].notna() & (paired["enrolment"] > 0)
    ].copy()
    return complete, int(zero_enrolment.sum())


def _paired_coverage(
    paired: pd.DataFrame,
    metric: str,
    level: str,
    years: tuple[int, ...],
    total: int,
    numerator_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in years:
        year_rows = paired.loc[paired["year"] == year]
        numerator_n = int(year_rows[numerator_name].notna().sum())
        denominator_n = int(year_rows["enrolment"].notna().sum())
        paired_n = int(
            (
                year_rows[numerator_name].notna()
                & year_rows["enrolment"].notna()
                & (year_rows["enrolment"] > 0)
            ).sum()
        )
        rows.append(
            {
                "metric": metric,
                "education_level": level,
                "year": year,
                "total_country_territory_entities": total,
                "numerator_reporting_entities": numerator_n,
                "denominator_reporting_entities": denominator_n,
                "paired_reporting_entities": paired_n,
                "paired_coverage_percent": paired_n / total * 100,
            }
        )
    return pd.DataFrame(rows, columns=list(PAIRED_COLUMNS))


def _concat(frames: list[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        combined = _empty(columns)
    else:
        combined = pd.concat(populated, ignore_index=True)
    combined = combined.loc[:, list(columns)]
    sort_columns = [column for column in ("metric", "education_level", "country_code", "year") if column in combined.columns]
    combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    if "year" in combined.columns:
        combined["year"] = combined["year"].astype("int64")
    return combined


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _require_unique(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        return
    duplicated = frame.duplicated(["country_code", "year", "education_level"])
    if duplicated.any():
        raise ValidationError(f"{label} country/year/level rows are not unique.")


def _span_lines(paired: pd.DataFrame, metric: str) -> list[str]:
    lines = ["", metric]
    subset = paired.loc[paired["metric"] == metric]
    for level in EDUCATION_LEVELS:
        level_rows = subset.loc[subset["education_level"] == level].sort_values("year")
        reported = level_rows.loc[level_rows["paired_reporting_entities"] > 0]
        if reported.empty:
            lines.append(f"{level}: no paired values")
            continue
        maximum = int(reported["paired_reporting_entities"].max())
        at_maximum = [
            int(year)
            for year in reported.loc[
                reported["paired_reporting_entities"] == maximum, "year"
            ]
        ]
        percent = float(
            reported.loc[reported["year"] == at_maximum[0], "paired_coverage_percent"].iloc[0]
        )
        years = ", ".join(str(year) for year in at_maximum)
        lines.append(
            f"{level}: first paired year {int(reported['year'].iloc[0])}; "
            f"last paired year {int(reported['year'].iloc[-1])}; "
            f"maximum paired reporting {maximum} entities ({percent:.2f}%) in {years}"
        )
    return lines


def _paired_grid(paired: pd.DataFrame, metric: str) -> str:
    subset = paired.loc[paired["metric"] == metric]
    grid = subset.pivot(index="year", columns="education_level", values="paired_reporting_entities")
    grid = grid.loc[:, list(EDUCATION_LEVELS)]
    return grid.to_string()


def _distribution(
    ratios: pd.DataFrame,
    paired: pd.DataFrame,
    metric: str,
    value_column: str,
) -> str:
    base = paired.loc[
        paired["metric"] == metric,
        ["education_level", "year", "paired_reporting_entities"],
    ].copy()
    if ratios.empty:
        stats = pd.DataFrame(columns=["education_level", "year", "minimum", "median", "maximum"])
    else:
        stats = (
            ratios.groupby(["education_level", "year"], sort=False)[value_column]
            .agg(minimum="min", median="median", maximum="max")
            .reset_index()
        )
    table = base.merge(stats, on=["education_level", "year"], how="left")
    table = table.sort_values(["education_level", "year"], kind="mergesort")
    return table.to_string(index=False)


def _top_lines(ratios: pd.DataFrame, value_column: str, numerator_column: str) -> list[str]:
    lines = ["", value_column]
    if ratios.empty:
        lines.append("no ratios")
        return lines
    for level in EDUCATION_LEVELS:
        lines.append(level)
        part = ratios.loc[ratios["education_level"] == level].sort_values(
            [value_column, "country_code", "year"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(5)
        if part.empty:
            lines.append("no ratios")
            continue
        for row in part.itertuples(index=False):
            record = row._asdict()
            lines.append(
                f"{record['country_code']} {record['country_name']} {int(record['year'])}: "
                f"numerator {record[numerator_column]}, "
                f"enrolment {record['enrolment']}, "
                f"ratio {record[value_column]}"
            )
    return lines
