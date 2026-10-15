"""Table utilities shared by every parser: reading numbers, finding header rows and units, and
the deterministic checks that tell us whether an extracted table can be trusted."""

import re

# --- numbers ----------------------------------------------------------------------------

_YEAR = re.compile(r"(?:19|20)\d{2}")
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_DASHES = {"-", "–", "—", "‒", "n/a", "N/A", "nm", "NM"}


def parse_number(cell: str) -> float | None:
    """The numeric value of a table cell, or None if it is not a number.

    Understands thousands separators, a currency sign, a trailing %, and accounting negatives:
    '(1,234)' is -1234. A dash or 'n/a' is not a number."""
    text = cell.strip()
    if not text or text in _DASHES:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.removeprefix("$").removeprefix("€").removeprefix("£").strip()
    text = text.removesuffix("%").removesuffix("x").strip()
    if text.startswith("-") and negative:
        return None  # "(-5)" is ambiguous: do not guess
    if not _NUMBER.match(text):
        return None
    value = float(text.replace(",", ""))
    return -value if negative else value


def looks_like_year(cell: str) -> bool:
    return bool(_YEAR.fullmatch(cell.strip()))


def is_data_cell(cell: str) -> bool:
    """A cell holding a figure (a year heading does not count: it labels a column)."""
    return parse_number(cell) is not None and not looks_like_year(cell)


def detect_header_rows(rows: list[list[str]]) -> int:
    """How many leading rows are column headings: rows holding words or years but no figures."""
    count = 0
    for row in rows[:3]:
        if not any(row):
            break
        if any(is_data_cell(cell) for cell in row):
            break
        count += 1
    if count == len(rows[:3]) and len(rows) > count:
        # three text-only rows in a row: probably a text table, whose only header is the first row
        return 1
    return count if count < len(rows) else (1 if len(rows) > 1 else 0)


# --- units ------------------------------------------------------------------------------

_UNITS = re.compile(
    r"\b(?:in|amounts\s+in|dollars\s+in|usd\s+in|\$\s*in)\s+(thousands|millions|billions)\b",
    re.IGNORECASE,
)


def find_units(text: str) -> str | None:
    """'(in millions, except per share amounts)' -> 'in millions'."""
    match = _UNITS.search(text)
    return f"in {match.group(1).lower()}" if match else None


# --- shape ------------------------------------------------------------------------------


def pad_rows(rows: list[list[str]]) -> tuple[list[list[str]], bool]:
    """Make every row as wide as the widest. Returns (rows, True if any row had to be padded)."""
    width = max((len(row) for row in rows), default=0)
    ragged = any(len(row) != width for row in rows)
    return [row + [""] * (width - len(row)) for row in rows], ragged


def drop_empty_edges(rows: list[list[str]]) -> list[list[str]]:
    """Remove rows that are completely empty and columns that are empty in every row."""
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep = [c for c in range(width) if any(row[c] for row in rows)]
    return [[row[c] for c in keep] for row in rows]


# --- checks -----------------------------------------------------------------------------

_TOTAL_LABEL = re.compile(r"^\s*(total|subtotal)\b", re.IGNORECASE)


def _decimals(cell: str) -> int:
    """Digits after the decimal point in a figure as printed ('12.50%' -> 2)."""
    core = cell.strip().strip("()$%x ").replace(",", "")
    return len(core.split(".")[1]) if "." in core else 0


def _is_ratio(cell: str) -> bool:
    """A percentage ('6%', '(4)%') or a multiple ('3.5x'): figures that cannot be added up."""
    text = cell.strip()
    return text.endswith(("%", "x")) and any(ch.isdigit() for ch in text)


def _within(total: float, running: float, terms: int, places: int) -> bool:
    """Rounding: every printed figure may be off by half a unit of its last digit, so k terms can
    legitimately differ from a printed total by up to k/2 units. Anything beyond that is a real
    discrepancy (a misread digit or a lost row)."""
    unit = 10.0**-places
    return abs(running - total) <= max(unit, 0.5 * terms * unit) + 1e-9


def _is_blank(cell: str) -> bool:
    """Nothing printed here (empty, or a dash meaning nil): not a term, and not a barrier."""
    text = cell.strip()
    return not text or text in _DASHES


def _sums_to_run(
    rows: list[list[str]], r: int, c: int, header_rows: int, total: float
) -> bool:
    """Does some run of rows directly above row r add up to `total` in column c? Ratio rows
    (a "percentage of net sales" line between two figures) and blank cells are stepped over;
    a text cell stops the run."""
    running = 0.0
    terms = 0
    places = _decimals(rows[r][c])
    for j in range(r - 1, header_rows - 1, -1):
        cell = rows[j][c] if c < len(rows[j]) else ""
        if _is_ratio(cell) or _is_blank(cell):
            continue
        value = parse_number(cell)
        if value is None:
            return False
        running += value
        terms += 1
        places = max(places, _decimals(cell))
        if _within(total, running, terms, places):
            return True
    return False


def _sums_to_subtotals(
    rows: list[list[str]], r: int, c: int, header_rows: int, total: float
) -> bool:
    """A grand total is often the sum of the SUBTOTALS above it (Total assets = Total current
    assets + Total non-current assets), which are not a contiguous run of rows. Try the last two
    to four rows labelled Total/Subtotal."""
    figures: list[tuple[float, str]] = []
    for j in range(r - 1, header_rows - 1, -1):
        if rows[j] and _TOTAL_LABEL.match(rows[j][0]) and c < len(rows[j]):
            value = parse_number(rows[j][c])
            if value is not None and not _is_ratio(rows[j][c]):
                figures.append((value, rows[j][c]))
    for k in range(2, min(4, len(figures)) + 1):
        chosen = figures[:k]
        places = max([_decimals(rows[r][c])] + [_decimals(text) for _, text in chosen])
        if _within(total, sum(v for v, _ in chosen), k, places):
            return True
    return False


def total_mismatches(
    rows: list[list[str]], header_rows: int = 0
) -> list[tuple[int, int]]:
    """Find 'Total ...' rows whose figures are not explained by the rows above them.

    A total is accepted if SOME run of rows directly above it adds up to it (a statement often
    has a subtotal in between), or if it is the sum of the subtotals above it. Percentage and
    multiple columns ('6%', '3.5x') are never judged: they are not sums. Returns [(row, column)]
    of every total cell that matched none: those cells are suspect (a misread digit, a lost row).
    Some totals cannot be checked this way (a net of two groups, a running balance); they may
    show up here, which is why this only lowers confidence and never rejects a table."""
    problems: list[tuple[int, int]] = []
    for r in range(max(header_rows, 1), len(rows)):
        if not rows[r] or not _TOTAL_LABEL.match(rows[r][0]):
            continue
        for c in range(1, len(rows[r])):
            if _is_ratio(rows[r][c]):
                continue
            total = parse_number(rows[r][c])
            if total is None:
                continue
            if r - 1 < header_rows:
                continue
            if _sums_to_run(rows, r, c, header_rows, total):
                continue
            if _sums_to_subtotals(rows, r, c, header_rows, total):
                continue
            problems.append((r, c))
    return problems
