"""Derived values: computed in code from other cells, never by the model.

A derived column is not a question. It names other columns and an operation; code reads the input
cells, checks they can honestly be combined, and computes the answer:

    Operating margin = Operating income / Revenue         (a percentage)
    Net debt         = Total debt - Cash and equivalents  (an amount)
    Net debt / EBITDA = (Total debt - Cash) / Adjusted EBITDA   (a multiple)

The rules, which are the point of doing it in code:
  - an input that is missing, failed or uncertain makes the result say so: no number is invented
  - amounts in different currencies, or an amount and a plain number, are not combined
  - figures from different periods are not combined (an annual figure over a quarterly one)
  - growth needs the same period one year apart
  - dividing by zero is reported, never silently turned into 0 or infinity

Pure functions on plain values: no model, no database, so every rule is a test.
"""

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.finance.periods import Period

Op = Literal["ratio", "difference", "sum", "growth", "net_ratio"]
Display = Literal["percent", "multiple"]

ARITY: dict[str, tuple[int, int]] = {  # (fewest, most) input columns
    "ratio": (2, 2),  # first / second
    "difference": (2, 2),  # first - second
    "sum": (2, 4),
    "growth": (2, 2),  # first (current) over second (prior), minus 1
    "net_ratio": (3, 3),  # (first - second) / third
}
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


class Derivation(BaseModel):
    """What a derived column does. Which columns it reads is the column's `depends_on`, in order."""

    model_config = ConfigDict(populate_by_name=True)

    op: Op
    as_: Display | None = Field(default=None, alias="as")  # how a ratio is shown

    def stored(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


def check_shape(derivation: Derivation, inputs: int) -> None:
    """Raises ValueError (with a message for a person) if the operation cannot take this many inputs
    or needs a display it was not given."""
    low, high = ARITY[derivation.op]
    if not low <= inputs <= high:
        want = f"{low}" if low == high else f"{low} to {high}"
        raise ValueError(f"'{derivation.op}' needs {want} input columns, not {inputs}")
    if derivation.op == "ratio" and derivation.as_ is None:
        raise ValueError("a ratio must say whether it is a percent or a multiple")
    if derivation.op == "growth" and derivation.as_ not in (None, "percent"):
        raise ValueError("growth is always a percent")
    if derivation.op == "net_ratio" and derivation.as_ not in (None, "multiple"):
        raise ValueError("net debt over a measure is always a multiple")


@dataclass(frozen=True)
class Operand:
    """One input cell, as the calculation sees it."""

    name: str
    status: str  # a cell status, or "not_filled" when there is no cell yet
    value: Any = None
    type: str = "text"
    period: Period | None = None


@dataclass(frozen=True)
class Derived:
    status: Literal["answered", "not_found", "low_confidence"]
    value: Any = None
    display: str | None = None
    detail: str | None = None  # why it is not answered
    period: Period | None = None
    notes: list[str] = field(
        default_factory=list
    )  # things worth knowing about an answer


# --- reading the inputs ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    amount: Decimal
    kind: Literal["currency", "number"]
    unit: str | None  # a currency code, "%", or None


def quantity(op: Operand) -> Quantity | str:
    """The input as a number, or why it cannot be one."""
    v = op.value
    try:
        if op.type == "currency" and isinstance(v, dict) and "amount" in v:
            return Quantity(
                Decimal(str(v["amount"])), "currency", str(v.get("currency") or "?")
            )
        if op.type == "number" and isinstance(v, dict) and "number" in v:
            return Quantity(Decimal(str(v["number"])), "number", v.get("unit"))
    except InvalidOperation:
        pass
    return f"'{op.name}' is not a number or an amount"


def _describe(status: str) -> str:
    return {
        "not_found": "not found in the documents",
        "not_filled": "not filled yet",
        "pending_or_error": "unreadable",
        "not_applicable": "does not apply",
        "low_confidence": "uncertain",
        "conflicting": "conflicting",
    }.get(status, status)


# --- the calculation ------------------------------------------------------------------------------------------------------


def compute(derivation: Derivation, operands: list[Operand]) -> Derived:
    """Never raises. An answer only when every input is an answered number that can be combined."""
    try:
        check_shape(derivation, len(operands))
    except ValueError as exc:
        return Derived("low_confidence", detail=str(exc))

    missing = [
        o
        for o in operands
        if o.status in ("not_filled", "not_found", "pending_or_error", "not_applicable")
    ]
    if missing:
        which = "; ".join(f"'{o.name}' is {_describe(o.status)}" for o in missing)
        return Derived("not_found", detail=f"needs its inputs: {which}")
    shaky = [o for o in operands if o.status != "answered"]
    if shaky:
        which = "; ".join(f"'{o.name}' is {_describe(o.status)}" for o in shaky)
        return Derived(
            "low_confidence", detail=f"an input cannot be relied on: {which}"
        )

    quantities: list[Quantity] = []
    for o in operands:
        q = quantity(o)
        if isinstance(q, str):
            return Derived("low_confidence", detail=q)
        quantities.append(q)

    problem = _mismatch(derivation, operands, quantities) or periods_problem(
        derivation, operands
    )
    if problem is not None:
        return Derived("low_confidence", detail=problem)

    period, notes = _period_of(derivation, operands)
    try:
        value, display = _calculate(derivation, quantities)
    except ZeroDivisionError:
        return Derived("low_confidence", detail="cannot divide by zero")
    return Derived("answered", value, display, period=period, notes=notes)


def _mismatch(
    derivation: Derivation, operands: list[Operand], q: list[Quantity]
) -> str | None:
    """Why these inputs cannot be combined by this operation, or None."""
    names = ", ".join(f"'{o.name}'" for o in operands)
    if derivation.op in ("difference", "sum", "growth"):
        if len({x.kind for x in q}) > 1:
            return f"{names} are not the same kind of number (an amount and a plain number)"
        if len({x.unit for x in q}) > 1:
            units = ", ".join(sorted({str(x.unit) for x in q}))
            return f"{names} are in different units ({units})"
    if derivation.op == "ratio" and q[0].kind != q[1].kind:
        return f"{names} are not the same kind of number (an amount and a plain number)"
    if (
        derivation.op in ("ratio", "net_ratio")
        and len({x.unit for x in q if x.kind == "currency"}) > 1
    ):
        units = ", ".join(sorted({str(x.unit) for x in q if x.kind == "currency"}))
        return f"{names} are in different currencies ({units})"
    if derivation.op == "net_ratio" and any(x.kind != "currency" for x in q):
        return f"{names} must all be amounts"
    return None


def _period_of(
    derivation: Derivation, operands: list[Operand]
) -> tuple[Period | None, list[str]]:
    """The period of the result. Raises nothing: a clash is reported by `periods_problem`."""
    stated = [o.period for o in operands if o.period is not None]
    notes: list[str] = []
    if len(stated) < len(operands):
        silent = [o.name for o in operands if o.period is None]
        notes.append("period not stated for " + ", ".join(f"'{n}'" for n in silent))
    if derivation.op == "growth":
        return operands[0].period, notes
    return (stated[0] if stated else None), notes


def periods_problem(derivation: Derivation, operands: list[Operand]) -> str | None:
    """Do the inputs' stated periods allow this operation? None when they do (or none are stated)."""
    if derivation.op == "growth":
        current, prior = operands[0].period, operands[1].period
        if (
            current is not None
            and prior is not None
            and not prior.is_year_before(current)
        ):
            return (
                f"growth needs the same period one year apart, but the inputs are "
                f"{current} and {prior}"
            )
        return None
    stated = {o.period for o in operands if o.period is not None}
    if len(stated) > 1:
        labels = ", ".join(sorted(p.label for p in stated))
        return f"the inputs cover different periods ({labels}); they cannot be combined"
    return None


def _calculate(derivation: Derivation, q: list[Quantity]) -> tuple[Any, str]:
    op = derivation.op
    if op == "ratio":
        return _ratio(q[0].amount, q[1].amount, derivation.as_ or "multiple")
    if op == "net_ratio":
        return _ratio(q[0].amount - q[1].amount, q[2].amount, "multiple")
    if op == "growth":
        if q[1].amount == 0:
            raise ZeroDivisionError
        return _percent((q[0].amount / q[1].amount - 1) * 100)
    total = (
        q[0].amount - q[1].amount
        if op == "difference"
        else sum((x.amount for x in q), Decimal(0))
    )
    return _same_kind(total, q[0])


def _ratio(top: Decimal, bottom: Decimal, how: str) -> tuple[Any, str]:
    if bottom == 0:
        raise ZeroDivisionError
    ratio = top / bottom
    return _percent(ratio * 100) if how == "percent" else _multiple(ratio)


def _percent(points: Decimal) -> tuple[Any, str]:
    shown = points.quantize(Decimal("0.01"), ROUND_HALF_UP)
    return {
        "number": _plain(shown),
        "unit": "%",
    }, f"{_plain(points.quantize(Decimal('0.1'), ROUND_HALF_UP))}%"


def _multiple(ratio: Decimal) -> tuple[Any, str]:
    shown = ratio.quantize(Decimal("0.01"), ROUND_HALF_UP)
    return {"number": _plain(shown), "unit": "x"}, f"{_plain(shown)}x"


def _same_kind(total: Decimal, like: Quantity) -> tuple[Any, str]:
    if like.kind == "currency":
        return (
            {"amount": _plain(total), "currency": like.unit, "derived": True},
            format_amount(total, like.unit or "USD"),
        )
    unit = like.unit
    return {"number": _plain(total), "unit": unit}, _plain(total) + (unit or "")


def _plain(value: Decimal) -> str:
    """A decimal as plain digits: no exponent, no trailing zeros, no floats."""
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "") else text


_SCALES = [
    (Decimal(10) ** 12, "trillion"),
    (Decimal(10) ** 9, "billion"),
    (Decimal(10) ** 6, "million"),
    (Decimal(10) ** 3, "thousand"),
]


def format_amount(amount: Decimal, currency: str) -> str:
    """$1.92 billion, -$300 million, $4,200. Only for display: the stored amount is exact."""
    sign = "-" if amount < 0 else ""
    size = abs(amount)
    symbol = SYMBOLS.get(currency, f"{currency} ")
    for scale, word in _SCALES:
        if size >= scale:
            scaled = (size / scale).quantize(Decimal("0.01"), ROUND_HALF_UP)
            return f"{sign}{symbol}{_plain(scaled)} {word}"
    return f"{sign}{symbol}{size:,.2f}".rstrip("0").rstrip(".")
