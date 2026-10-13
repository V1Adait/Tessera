"""Find what a question mentions, in code: which workspace entities, and which periods.

Pure functions, no model calls. Metrics are separate (metrics.py) because naming a metric
needs language understanding; entities and periods are patterns we can match exactly.

Assumptions worth knowing (change them in DECISIONS.md, not silently here):
  * "FY24", "fiscal 2024" and a bare "2024" all mean the year 2024. Fiscal years that do not
    match the calendar are the column spec's job later.
  * "the last three years" means the three COMPLETED calendar years before today's year.
  * "since 2022" runs from 2022 to the last completed year.
"""

import re
from datetime import UTC, date, datetime

from backend.app.db.records import EntityRecord
from backend.app.query.models import EntityRef, Period

_MIN_YEAR, _MAX_YEAR = 1990, 2100
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip


# --- entities ---------------------------------------------------------------------------


def find_entities(question: str, entities: list[EntityRecord]) -> list[EntityRef]:
    """Workspace entities whose name appears in the question, in order of first mention.

    Whole-word, case-insensitive. Longer names are tried first, so 'Visa Europe' is not
    swallowed by an entity called 'Visa'. A name that is a substring of a word ('Meta' in
    'metadata') does not match."""
    found: list[tuple[int, EntityRef]] = []
    taken: list[tuple[int, int]] = []
    for entity in sorted(entities, key=lambda e: len(e.name), reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(entity.name)}(?!\w)", re.IGNORECASE)
        match = pattern.search(question)
        if match is None or any(
            s < match.end() and match.start() < e for s, e in taken
        ):
            continue
        taken.append(match.span())
        found.append((match.start(), EntityRef(id=entity.id, name=entity.name)))
    return [ref for _, ref in sorted(found, key=lambda pair: pair[0])]


# --- periods ----------------------------------------------------------------------------

_YEAR = r"(?:19|20)\d{2}"
_FY = re.compile(
    rf"\b(?:FY|fiscal(?: year)?\s*)\s*'?({_YEAR}|\d{{2}})\b", re.IGNORECASE
)
_QUARTER = re.compile(
    rf"\b(?:Q([1-4])\s*(?:FY)?\s*'?({_YEAR}|\d{{2}})|({_YEAR})\s*Q([1-4]))\b",
    re.IGNORECASE,
)
_RANGE = re.compile(rf"\b({_YEAR})\s*(?:-|–|to|through)\s*({_YEAR})\b", re.IGNORECASE)
_SINCE = re.compile(rf"\bsince\s+(?:FY\s*)?({_YEAR})\b", re.IGNORECASE)
_LAST_N = re.compile(
    r"\b(?:last|past|previous)\s+(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s+years?\b",
    re.IGNORECASE,
)
# A bare year is not part of a bigger number ('$2,024', '2024.5') and not a quantity ('2000 employees').
_BARE_YEAR = re.compile(
    rf"(?<![\d$,.])({_YEAR})(?![\d]|[.,]\d|\s*(?:%|x\b|m\b|k\b|bn\b|million|billion|employees|shares))",
    re.IGNORECASE,
)


def _full_year(text: str) -> int:
    value = int(text)
    return value if value >= 100 else 2000 + value


def _ok(year: int) -> bool:
    return _MIN_YEAR <= year <= _MAX_YEAR


def find_periods(question: str, today: date | None = None) -> list[Period]:
    """Years and quarters named in the question, oldest first, without duplicates.

    A range ('2022-2024', '2022 to 2024') and 'since 2022' expand to every year in between.
    '2022 and 2024' is two years, not a range: the user named exactly those two.
    A question with no period returns []: that means "whatever the documents cover"."""
    today = today or datetime.now(tz=UTC).date()
    last_complete = today.year - 1
    years: set[int] = set()
    quarters: set[tuple[int, int]] = set()
    text = question

    for m in _QUARTER.finditer(text):
        quarter = int(m.group(1) or m.group(4))
        year = _full_year(m.group(2) or m.group(3))
        if _ok(year):
            quarters.add((year, quarter))
    text = _QUARTER.sub(" ", text)

    for m in _RANGE.finditer(text):
        start, end = sorted((int(m.group(1)), int(m.group(2))))
        if _ok(start) and _ok(end) and end - start <= 30:
            years.update(range(start, end + 1))
    text = _RANGE.sub(" ", text)

    for m in _SINCE.finditer(text):
        start = int(m.group(1))
        if _ok(start) and start <= last_complete:
            years.update(range(start, last_complete + 1))
    text = _SINCE.sub(" ", text)

    for m in _LAST_N.finditer(text):
        raw = m.group(1).lower()
        count = _NUMBER_WORDS.get(raw) or int(raw)
        if 1 <= count <= 30:
            years.update(range(last_complete - count + 1, last_complete + 1))
    text = _LAST_N.sub(" ", text)

    for m in _FY.finditer(text):
        year = _full_year(m.group(1))
        if _ok(year):
            years.add(year)
    text = _FY.sub(" ", text)

    for m in _BARE_YEAR.finditer(text):
        if _ok(int(m.group(1))):
            years.add(int(m.group(1)))

    periods = [Period(kind="year", year=y) for y in years]
    periods += [Period(kind="quarter", year=y, quarter=q) for y, q in quarters]
    return sorted(periods, key=lambda p: (p.year, p.quarter or 0))
