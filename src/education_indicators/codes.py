"""Exact series codes and the reviewed aggregate-code list.

Indicator selection uses series-code equality. Country names are never used
to decide entity type or which indicator a row belongs to.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class Indicator(NamedTuple):
    """One series in the local education extract.

    ``level`` is the education level the series refers to. Adult literacy uses
    ``adult`` because it is not a school level. No derived metric is defined here.
    """

    code: str
    measure: str
    level: str


INDICATORS: dict[str, Indicator] = {
    "SE.PRM.TCHR": Indicator("SE.PRM.TCHR", "teachers", "primary"),
    "SE.SEC.TCHR": Indicator("SE.SEC.TCHR", "teachers", "secondary"),
    "SE.PRE.TCHR": Indicator("SE.PRE.TCHR", "teachers", "pre_primary"),
    "SE.ADT.LITR.ZS": Indicator("SE.ADT.LITR.ZS", "literacy_percent", "adult"),
    "UIS.X.US.02.FSGOV": Indicator(
        "UIS.X.US.02.FSGOV", "expenditure_usd_millions", "pre_primary"
    ),
    "UIS.X.US.1.FSGOV": Indicator(
        "UIS.X.US.1.FSGOV", "expenditure_usd_millions", "primary"
    ),
    "UIS.X.US.2T3.FSGOV": Indicator(
        "UIS.X.US.2T3.FSGOV", "expenditure_usd_millions", "secondary"
    ),
    "SE.PRE.ENRL": Indicator("SE.PRE.ENRL", "enrolment", "pre_primary"),
    "SE.SEC.ENRL": Indicator("SE.SEC.ENRL", "enrolment", "secondary"),
    "SE.PRM.ENRL": Indicator("SE.PRM.ENRL", "enrolment", "primary"),
}

ID_COLUMNS: tuple[str, ...] = (
    "Country Name",
    "Country Code",
    "Series",
    "Series Code",
)

# ``2010 [YR2010]``. The year inside the bracket must be the same year.
YEAR_COLUMN_RE = re.compile(r"^((?:19|20)\d{2}) \[YR\1\]$")

# Grouping codes reviewed from the local education extract.
# A code is on this list only when the extract's own name is a region, income
# group, organisation, or other multi-country aggregate — not because the name
# contains a word such as "Africa" or "Arab".
# Territories and states that are not on this list are labelled "country".
# That label is not a sovereignty judgement. A future extract that adds a new
# grouping code would be labelled "country" until this set is reviewed again.
AGGREGATE_CODES: frozenset[str] = frozenset(
    {
        "ARB",  # Arab World
        "CEB",  # Central Europe and the Baltics
        "CHI",  # Channel Islands
        "CSS",  # Caribbean small states
        "EAP",  # East Asia & Pacific (excluding high income)
        "EAR",  # Early-demographic dividend
        "EAS",  # East Asia & Pacific
        "ECA",  # Europe & Central Asia (excluding high income)
        "ECS",  # Europe & Central Asia
        "EMU",  # Euro area
        "EUU",  # European Union
        "FCS",  # Fragile and conflict affected situations
        "FTI",  # Global Partnership for Education
        "HIC",  # High income
        "HPC",  # Heavily indebted poor countries (HIPC)
        "IBD",  # IBRD only
        "IBT",  # IDA & IBRD total
        "IDA",  # IDA total
        "IDB",  # IDA blend
        "IDX",  # IDA only
        "LAC",  # Latin America & Caribbean (excluding high income)
        "LCN",  # Latin America & Caribbean
        "LDC",  # Least developed countries: UN classification
        "LIC",  # Low income
        "LMC",  # Lower middle income
        "LMY",  # Low & middle income
        "LNX",  # Lending category not classified
        "LTE",  # Late-demographic dividend
        "MEA",  # Middle East & North Africa
        "MIC",  # Middle income
        "MNA",  # Middle East & North Africa (excluding high income)
        "NAC",  # North America
        "OED",  # OECD members
        "OSS",  # Other small states
        "PRE",  # Pre-demographic dividend
        "PSS",  # Pacific island small states
        "PST",  # Post-demographic dividend
        "SAS",  # South Asia
        "SSA",  # Sub-Saharan Africa (excluding high income)
        "SSF",  # Sub-Saharan Africa
        "SST",  # Small states
        "TEA",  # East Asia & Pacific (IDA & IBRD countries)
        "TEC",  # Europe & Central Asia (IDA & IBRD countries)
        "TLA",  # Latin America & the Caribbean (IDA & IBRD countries)
        "TMN",  # Middle East & North Africa (IDA & IBRD countries)
        "TSA",  # South Asia (IDA & IBRD)
        "TSS",  # Sub-Saharan Africa (IDA & IBRD countries)
        "UMC",  # Upper middle income
        "WLD",  # World
    }
)


def parse_year_label(label: str) -> int | None:
    """Return the year from a ``YYYY [YRYYYY]`` column, or None if it is not one."""
    match = YEAR_COLUMN_RE.fullmatch(str(label).strip())
    if match is None:
        return None
    return int(match.group(1))


def entity_type_for_code(country_code: str) -> str:
    """Classify an entity from its code alone.

    ``country`` includes territories. It means "not on the aggregate list".
    """
    if country_code in AGGREGATE_CODES:
        return "aggregate"
    return "country"
