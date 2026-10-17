"""Reporting periods: "FY2025", "Q2 2026", "H1 2026". Pure, no model, no database.

Mixing periods is the quietest way to get a financial number wrong: an annual figure divided by a
quarterly one still produces a number. So every figure carries its period, and code refuses to
combine figures from different ones.

The model is asked to state a period in one of the three canonical shapes. This module accepts a few
common spellings of them ("fiscal 2025", "Q2'26", "second quarter of 2026") and NOTHING else: a
sentence like "three months ended June 30" is not a period we can compare, and is dropped (None)
rather than guessed. The year is the one the COMPANY calls the period, so a fiscal year that ends in
June 2025 is FY2025, as the filing says.
"""

import re
from dataclasses import dataclass
from typing import Literal

PeriodKind = Literal["year", "quarter", "half"]

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


@dataclass(frozen=True)
class Period:
    kind: PeriodKind
    year: int
    part: int | None = None  # the quarter (1-4) or half (1-2); None for a year

    @property
    def label(self) -> str:
        if self.kind == "year":
            return f"FY{self.year}"
        if self.kind == "quarter":
            return f"Q{self.part} {self.year}"
        return f"H{self.part} {self.year}"

    def __str__(self) -> str:
        return self.label

    def is_year_before(self, other: Period) -> bool:
        """Is `self` the same period of the previous year as `other`? (Q2 2025 vs Q2 2026)"""
        return (
            self.kind == other.kind
            and self.part == other.part
            and self.year == other.year - 1
        )


def _year(text: str) -> int | None:
    value = int(text)
    if len(text) == 2:
        value += 2000
    return value if 1990 <= value <= 2100 else None


_YEAR = r"'?(\d{4}|\d{2})"
_QUARTER = re.compile(rf"Q([1-4])\s*(?:FY|FISCAL)?\s*{_YEAR}", re.IGNORECASE)
_QUARTER_FIRST = re.compile(rf"([1-4])Q\s*{_YEAR}", re.IGNORECASE)
_QUARTER_WORDS = re.compile(
    r"(first|second|third|fourth)\s+quarter(?:\s+of)?\s+(?:fiscal\s+)?(\d{4})",
    re.IGNORECASE,
)
_HALF = re.compile(rf"H([12])\s*(?:FY|FISCAL)?\s*{_YEAR}", re.IGNORECASE)
_FISCAL_YEAR = re.compile(rf"(?:FY|fiscal(?:\s+year)?)\s*{_YEAR}", re.IGNORECASE)
_BARE_YEAR = re.compile(r"(\d{4})")


def parse_period(text: object) -> Period | None:
    """A canonical period, or None. Never raises."""
    if not isinstance(text, str):
        return None
    cleaned = " ".join(
        text.replace("(", " ").replace(")", " ").replace(",", " ").split()
    )
    if not cleaned or len(cleaned) > 40:
        return None
    if m := _QUARTER.fullmatch(cleaned):
        year = _year(m.group(2))
        return Period("quarter", year, int(m.group(1))) if year else None
    if m := _QUARTER_FIRST.fullmatch(cleaned):
        year = _year(m.group(2))
        return Period("quarter", year, int(m.group(1))) if year else None
    if m := _QUARTER_WORDS.fullmatch(cleaned):
        year = _year(m.group(2))
        return Period("quarter", year, _ORDINALS[m.group(1).lower()]) if year else None
    if m := _HALF.fullmatch(cleaned):
        year = _year(m.group(2))
        return Period("half", year, int(m.group(1))) if year else None
    if m := _FISCAL_YEAR.fullmatch(cleaned):
        year = _year(m.group(1))
        return Period("year", year) if year else None
    if m := _BARE_YEAR.fullmatch(cleaned):
        year = _year(m.group(1))
        return Period("year", year) if year else None
    return None


def supported_by(period: Period, text: str) -> bool:
    """Does the text the model was shown actually contain this period's year? A period the text
    never mentions can only have come from the model's memory, and is not kept."""
    return str(period.year) in text or f"'{period.year % 100:02d}" in text


def grounded_period(claimed: object, shown_text: str) -> Period | None:
    """The model's period claim, kept only if it parses AND the shown text mentions its year."""
    period = parse_period(claimed)
    if period is None or not supported_by(period, shown_text):
        return None
    return period
