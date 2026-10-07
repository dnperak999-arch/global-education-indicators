"""Builders for synthetic wide tables."""

from __future__ import annotations

import pandas as pd

from education_indicators.codes import INDICATORS

YEAR_2010 = "2010 [YR2010]"
YEAR_2011 = "2011 [YR2011]"


def rows_for(name: str, code: str, fill: str = "1") -> list[dict[str, str]]:
    """One source row for every required series code."""
    rows: list[dict[str, str]] = []
    for spec in INDICATORS.values():
        rows.append(
            {
                "Country Name": name,
                "Country Code": code,
                "Series": f"{spec.measure}:{spec.level}",
                "Series Code": spec.code,
                YEAR_2010: fill,
                YEAR_2011: fill,
            }
        )
    return rows


def frame_from(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
