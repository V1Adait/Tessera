"""Turn the model's answer text into a typed value, in code. Plain functions, no model.

The model returns an answer as a string ("$416,161 million", "September 27, 2025"). Parsing it is
code's job (CLAUDE.md: normalise dates and currency in code), so the same input always gives the
same value, and a bad answer is refused with a reason, not silently stored.

Shape of the stored `value` for each column type:

    text       "Cupertino, California"
    number     {"number": "166000", "unit": null}             unit is "%" for a percentage
    currency   {"amount": "416161000000", "currency": "USD",
                "printed": "$416,161", "scale": "millions", "scale_from": "table units"}
    date       {"date": "2025-09-27", "precision": "day"}     or "2027-03" / "2027"
    select     "Revolver"                                     the canonical allowed value
    yes_no     true / false
    status     {"tag": "On track", "reason": "one line"}

Numbers are kept as decimal strings, never floats, so "0.26" stays 0.26 and nothing is rounded.
"""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from backend.app.matrix.models import ColumnType


class BadValue(ValueError):
    """The answer cannot be turned into this column type. The message is safe to show the model."""


@dataclass(frozen=True)
class Normalised:
    value: Any
    display: str


_SCALE_WORDS = {
    "thousand": ("thousands", Decimal(10) ** 3),
    "thousands": ("thousands", Decimal(10) ** 3),
    "million": ("millions", Decimal(10) ** 6),
    "millions": ("millions", Decimal(10) ** 6),
    "mn": ("millions", Decimal(10) ** 6),
    "billion": ("billions", Decimal(10) ** 9),
    "billions": ("billions", Decimal(10) ** 9),
    "bn": ("billions", Decimal(10) ** 9),
    # a single letter stuck to the number: "$1.2B", "$500M", "$3K", "$2.1T", "$450MM"
    "b": ("billions", Decimal(10) ** 9),
    "mm": ("millions", Decimal(10) ** 6),
    "m": ("millions", Decimal(10) ** 6),
    "k": ("thousands", Decimal(10) ** 3),
    "t": ("trillions", Decimal(10) ** 12),
    "trillion": ("trillions", Decimal(10) ** 12),
    "trillions": ("trillions", Decimal(10) ** 12),
}
_UNITS_HINT = {
    "in thousands": "thousands",
    "in millions": "millions",
    "in billions": "billions",
}
_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_CODES = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR")
_CODE_RE = re.compile(rf"\b({'|'.join(_CODES)})\b")
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_SCALE_AFTER = re.compile(
    r"\s*\(?\s*(thousands?|millions?|billions?|trillions?|mn|bn)\b", re.IGNORECASE
)
# Only when the letter is attached to the number ("$1.2B"): "5 M" or "2 T" with a space is ambiguous.
_SCALE_SUFFIX = re.compile(r"(mm|[bmkt])\b", re.IGNORECASE)
_MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sept", "sep"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_ISO = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")
_DATE_MDY = re.compile(
    rf"^({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})$",
    re.IGNORECASE,
)
_DATE_DMY = re.compile(
    rf"^(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\.?,?\s+(\d{{4}})$",
    re.IGNORECASE,
)
_DATE_MY = re.compile(rf"^({_MONTH_PATTERN})\.?,?\s+(\d{{4}})$", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"^(\d{4})$")


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_PER_UNIT = re.compile(
    r"per[\s-]*(?:common\s+)?share|\beps\b|\bper[\s-]*unit\b|earnings\s+per|dividends?\s+per",
    re.IGNORECASE,
)


_UNITS_STATEMENT = re.compile(
    r"\(\s*(?:in|amounts\s+in|dollars\s+in|usd\s+in|\$\s*in)\s+(?:thousands|millions|billions)\b[^)]*\)",
    re.IGNORECASE,
)


def is_per_unit(*texts: str) -> bool:
    """Is this about a per-share (or per-unit) amount? A table header such as "in millions" scales the
    big totals, not the per-share rows beside them, so a table's units must not be applied to these.

    The units statement itself is ignored when looking for the cue: a header such as "(In millions,
    except number of shares and per-share amounts)" mentions per-share amounts precisely to say
    they are NOT in millions, and says nothing about the row being asked for."""
    return any(_PER_UNIT.search(_UNITS_STATEMENT.sub(" ", t)) for t in texts)


# --- numbers and money ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Figure:
    raw: str  # the digits exactly as printed, commas kept: '416,161', '8.0'
    token: str  # the same digits with commas removed: "416161", "8.0", "0.26"
    amount: Decimal
    negative: bool
    percent: bool
    scale_name: str | None
    scale_factor: Decimal
    symbol: str | None
    code: str | None


def _read_figure(answer: str) -> _Figure:
    """The first figure in the answer: its digits, sign, percent sign, scale word and currency."""
    match = _NUMBER_RE.search(answer)
    if match is None:
        raise BadValue("the answer holds no number")
    token = match.group().replace(",", "")
    try:
        amount = Decimal(token)
    except InvalidOperation as exc:
        raise BadValue("the number could not be read") from exc

    before = answer[: match.start()].rstrip(" $€£")
    negative = before.endswith(("(", "-", "−", "–"))
    after = answer[match.end() :]
    percent = after.lstrip(") ").startswith("%")

    scale_name, factor = None, Decimal(1)
    scale = _SCALE_AFTER.match(after) or _SCALE_SUFFIX.match(after)
    if scale and not percent:
        scale_name, factor = _SCALE_WORDS[scale.group(1).lower()]

    symbol = next((s for s in _SYMBOLS if s in answer), None)
    code_match = _CODE_RE.search(answer)
    return _Figure(
        raw=match.group(),
        token=token,
        amount=amount,
        negative=negative,
        percent=percent,
        scale_name=scale_name,
        scale_factor=factor,
        symbol=symbol,
        code=code_match.group(1) if code_match else _SYMBOLS.get(symbol or ""),
    )


def _plain(amount: Decimal) -> str:
    """A decimal with thousands separators and no exponent: 2500000 -> '2,500,000'."""
    text = format(amount, ",f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _decimal_string(amount: Decimal) -> str:
    text = format(amount, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def normalise_number(answer: str) -> Normalised:
    figure = _read_figure(answer)
    amount = -figure.amount if figure.negative else figure.amount
    amount = amount * figure.scale_factor
    unit = "%" if figure.percent else None
    shown = _plain(amount) + ("%" if figure.percent else "")
    return Normalised({"number": _decimal_string(amount), "unit": unit}, shown)


def normalise_currency(answer: str, units_hint: str | None) -> Normalised:
    figure = _read_figure(answer)
    if figure.percent:
        raise BadValue("the answer is a percentage, but this column holds money")
    if figure.code is None:
        raise BadValue(
            "the answer has no currency sign or code: write it as printed, e.g. $5.2 billion"
        )
    scale_name, factor, scale_from = figure.scale_name, figure.scale_factor, None
    if scale_name is not None:
        scale_from = "answer"
    elif units_hint:
        hinted = _UNITS_HINT.get(units_hint.strip().lower())
        if hinted:
            scale_name = hinted
            factor = _SCALE_WORDS[hinted][1]
            scale_from = "table units"
    amount = (-figure.amount if figure.negative else figure.amount) * factor
    printed_figure = ("-" if figure.negative else "") + figure.raw
    printed = f"{figure.symbol or ''}{printed_figure}"
    display = printed + (f" {scale_name[:-1]}" if scale_name else "")
    if not figure.symbol:
        display = f"{figure.code} {printed_figure}" + (
            f" {scale_name[:-1]}" if scale_name else ""
        )
    return Normalised(
        {
            "amount": _decimal_string(amount),
            "currency": figure.code,
            "printed": printed,
            "scale": scale_name,
            "scale_from": scale_from,
        },
        display,
    )


# --- dates ------------------------------------------------------------------------------------


def _month(name: str) -> int:
    return _MONTHS[name.lower().rstrip(".")]


def normalise_date(answer: str) -> Normalised:
    text = collapse(answer).rstrip(".")
    try:
        if m := _DATE_ISO.match(text):
            year, month, day = int(m.group(1)), int(m.group(2)), m.group(3)
            if day is None:
                date(year, month, 1)
                return Normalised(
                    {"date": f"{year:04d}-{month:02d}", "precision": "month"}, text
                )
            parsed = date(year, month, int(day))
            return Normalised(
                {"date": parsed.isoformat(), "precision": "day"}, parsed.isoformat()
            )
        if m := _DATE_MDY.match(text):
            parsed = date(int(m.group(3)), _month(m.group(1)), int(m.group(2)))
            return Normalised(
                {"date": parsed.isoformat(), "precision": "day"}, parsed.isoformat()
            )
        if m := _DATE_DMY.match(text):
            parsed = date(int(m.group(3)), _month(m.group(2)), int(m.group(1)))
            return Normalised(
                {"date": parsed.isoformat(), "precision": "day"}, parsed.isoformat()
            )
        if m := _DATE_MY.match(text):
            year, month = int(m.group(2)), _month(m.group(1))
            date(year, month, 1)
            return Normalised(
                {"date": f"{year:04d}-{month:02d}", "precision": "month"},
                f"{year:04d}-{month:02d}",
            )
        if m := _YEAR_ONLY.match(text):
            year = int(m.group(1))
            date(year, 1, 1)
            return Normalised(
                {"date": f"{year:04d}", "precision": "year"}, f"{year:04d}"
            )
    except ValueError as exc:  # an impossible date such as 31 February
        raise BadValue(f"{answer!r} is not a real calendar date") from exc
    raise BadValue(
        f"{answer!r} is not a date I can read: write it like 'September 27, 2025', '2025-09-27', 'March 2027' or '2027'"
    )


# --- the rest ---------------------------------------------------------------------------------


def normalise_yes_no(answer: str) -> Normalised:
    word = collapse(answer).lower().rstrip(".")
    if word in ("yes", "y", "true"):
        return Normalised(True, "Yes")
    if word in ("no", "n", "false"):
        return Normalised(False, "No")
    raise BadValue("the answer must be exactly Yes or No")


def normalise_select(answer: str, allowed: list[str]) -> Normalised:
    wanted = collapse(answer).casefold()
    for option in allowed:
        if option.casefold() == wanted:
            return Normalised(option, option)
    raise BadValue(f"{answer!r} is not one of the allowed values: {', '.join(allowed)}")


def normalise_status(answer: str, reason: str | None, allowed: list[str]) -> Normalised:
    tag = collapse(answer)
    if allowed:
        tag = normalise_select(tag, allowed).value
    why = collapse(reason or "")
    if not why:
        raise BadValue("a status needs a one-line reason")
    if len(why) > 200:
        raise BadValue("the reason must fit on one line (200 characters)")
    return Normalised({"tag": tag, "reason": why}, f"{tag}: {why}")


def normalise_text(answer: str) -> Normalised:
    text = collapse(answer)
    if not text:
        raise BadValue("the answer is empty")
    return Normalised(text, text)


def normalise(
    column_type: ColumnType,
    answer: str,
    *,
    reason: str | None = None,
    allowed_values: list[str] | None = None,
    units_hint: str | None = None,
) -> Normalised:
    """Parse `answer` for a column type. Raises BadValue with a reason if it cannot be."""
    allowed = allowed_values or []
    match column_type:
        case "text":
            return normalise_text(answer)
        case "number":
            return normalise_number(answer)
        case "currency":
            return normalise_currency(answer, units_hint)
        case "date":
            return normalise_date(answer)
        case "yes_no":
            return normalise_yes_no(answer)
        case "select":
            return normalise_select(answer, allowed)
        case "status":
            return normalise_status(answer, reason, allowed)


# --- is the value in the quote? ---------------------------------------------------------------


def _factor_after(text: str) -> Decimal:
    """The scale a figure carries in the text that follows its digits: 1 if it has none."""
    scale = _SCALE_AFTER.match(text) or _SCALE_SUFFIX.match(text)
    return _SCALE_WORDS[scale.group(1).lower()][1] if scale else Decimal(1)


def supported_by_quote(
    column_type: ColumnType, answer: str, quotes: list[str]
) -> str | None:
    """Catch the model quoting a true sentence but answering with a different number or date.

    For number, currency and date columns the digits it answered with must appear in the text it
    quoted. Returns a problem message, or None if it holds (or the type cannot be checked this way:
    a yes/no, a choice or free text is a judgement, not a transcription)."""
    joined = " ".join(quotes)
    if column_type in ("number", "currency"):
        figure = _read_figure(answer)
        token = figure.token
        stripped = joined.replace(",", "")
        found = list(re.finditer(rf"(?<![\d.]){re.escape(token)}(?!\d|\.\d)", stripped))
        if not found:
            return f"the number {token} does not appear in the quoted text: answer with a number the quote states, do not calculate"
        # The quote may print the figure with a scale ("$1.2B"). An answer that drops it ("$1.2")
        # would be wrong by a factor of a billion, so it must carry the same scale.
        if not any(
            _factor_after(stripped[m.end() :]) in (Decimal(1), figure.scale_factor)
            for m in found
        ):
            tail = _SCALE_AFTER.match(
                stripped[found[0].end() :]
            ) or _SCALE_SUFFIX.match(stripped[found[0].end() :])
            shown = tail.group(1) if tail else "a scale"
            return (
                f"the quote prints {token} with a scale ('{shown}') that the answer leaves out: "
                f"write the amount as printed, with its scale"
            )
        return None
    if column_type == "date":
        parsed = normalise_date(answer).value
        year, *rest = str(parsed["date"]).split("-")
        if year not in joined:
            return f"the year {year} does not appear in the quoted text"
        if parsed["precision"] == "day":
            day = int(rest[1])
            if not re.search(rf"(?<!\d)0?{day}(?!\d)", joined):
                return f"the day {day} does not appear in the quoted text"
        return None
    return None
