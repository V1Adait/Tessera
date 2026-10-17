"""Tie-out checks: identities that must hold between cells, tested in code.

Financial statements are full of identities. If the extracted numbers break one, at least one of
them is wrong (or the company defined a term differently), and a person should look before relying
on either. A check is data (it lives in a domain pack), evaluated by this module:

    difference identity   a - b = result     (Revenue - Cost of revenue = Gross profit)
    ratio identity        a / b = result %   (Gross profit / Revenue = Gross margin)

A check has three outcomes, never two:
    ties           the numbers agree within the tolerance
    does_not_tie   they do not, with the gap shown
    cannot_check   an input is missing or uncertain, in different currencies or periods: not a pass

`cannot_check` matters: a check that cannot run must not look like one that passed.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from backend.app.finance.derive import Operand, Quantity, format_amount, quantity
from backend.app.finance.periods import Period

CheckKind = Literal["difference_identity", "ratio_identity"]
Outcome = Literal["ties", "does_not_tie", "cannot_check"]


class CheckSpec(BaseModel):
    id: str
    name: str
    kind: CheckKind
    a: str  # column names
    b: str
    result: str
    # difference identity: the gap as a share of the larger number (0.005 = 0.5%).
    # ratio identity: the gap in percentage points.
    tolerance: Decimal
    note: str | None = (
        None  # shown with a failed check, e.g. "companies define this differently"
    )


class CheckResult(BaseModel):
    id: str
    name: str
    status: Outcome
    detail: str
    attach_to: str  # the column whose cells show this check

    def stored(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


def applicable(spec: CheckSpec, known: set[str]) -> bool:
    """Only checks whose three columns all exist in the workspace are run: a workspace without a
    'Cost of revenue' column has no gross-profit check, and that is not a failure."""
    return all(n.casefold() in known for n in (spec.a, spec.b, spec.result))


def _cannot(spec: CheckSpec, why: str) -> CheckResult:
    return CheckResult(
        id=spec.id,
        name=spec.name,
        status="cannot_check",
        detail=why,
        attach_to=spec.result,
    )


def evaluate(spec: CheckSpec, cells: dict[str, Operand]) -> CheckResult:
    """`cells` is keyed by lower-cased column name. Never raises."""
    operands = [cells.get(n.casefold()) for n in (spec.a, spec.b, spec.result)]
    if any(o is None for o in operands):
        return _cannot(spec, "a column it needs does not exist")
    inputs = [o for o in operands if o is not None]

    bad = [o for o in inputs if o.status != "answered"]
    if bad:
        which = "; ".join(f"'{o.name}' is {_why(o.status)}" for o in bad)
        return _cannot(spec, f"cannot check: {which}")

    quantities: list[Quantity] = []
    for o in inputs:
        q = quantity(o)
        if isinstance(q, str):
            return _cannot(spec, f"cannot check: {q}")
        quantities.append(q)

    periods = {o.period for o in inputs if o.period is not None}
    if len(periods) > 1:
        labels = ", ".join(sorted(p.label for p in periods))
        return _cannot(
            spec, f"cannot check: the figures cover different periods ({labels})"
        )

    if spec.kind == "difference_identity":
        return _difference(spec, inputs, quantities, next(iter(periods), None))
    return _ratio(spec, inputs, quantities)


def _why(status: str) -> str:
    return {
        "not_found": "not found",
        "pending_or_error": "unreadable",
        "low_confidence": "uncertain",
        "conflicting": "conflicting",
        "not_applicable": "not applicable",
    }.get(status, status)


def _show(amount: Decimal, like: Quantity) -> str:
    if like.kind == "currency":
        return format_amount(amount, like.unit or "USD")
    return f"{amount.normalize():f}{like.unit or ''}"


def _difference(
    spec: CheckSpec, inputs: list[Operand], q: list[Quantity], period: Period | None
) -> CheckResult:
    a, b, actual = q
    if len({x.kind for x in q}) > 1 or len({x.unit for x in q}) > 1:
        return _cannot(
            spec, "cannot check: the figures are in different units or currencies"
        )
    expected = a.amount - b.amount
    gap = abs(expected - actual.amount)
    scale = max(abs(expected), abs(actual.amount))
    relative = Decimal(0) if scale == 0 else gap / scale
    words = f"{inputs[0].name} - {inputs[1].name} = {_show(expected, a)}"
    if relative <= spec.tolerance:
        return CheckResult(
            id=spec.id,
            name=spec.name,
            status="ties",
            detail=f"{words}, and '{inputs[2].name}' agrees ({_show(actual.amount, a)})",
            attach_to=spec.result,
        )
    percent = (relative * 100).quantize(Decimal("0.1"))
    detail = (
        f"{words}, but '{inputs[2].name}' says {_show(actual.amount, a)}: "
        f"a gap of {_show(gap, a)} ({percent.normalize():f}%)"
    )
    if spec.note:
        detail += f". {spec.note}"
    return CheckResult(
        id=spec.id,
        name=spec.name,
        status="does_not_tie",
        detail=detail,
        attach_to=spec.result,
    )


def _ratio(spec: CheckSpec, inputs: list[Operand], q: list[Quantity]) -> CheckResult:
    top, bottom, reported = q
    if (
        top.kind != bottom.kind
        or len({x.unit for x in (top, bottom) if x.kind == "currency"}) > 1
    ):
        return _cannot(
            spec, "cannot check: the figures are in different units or currencies"
        )
    if reported.kind != "number" or reported.unit != "%":
        return _cannot(spec, f"cannot check: '{inputs[2].name}' is not a percentage")
    if bottom.amount == 0:
        return _cannot(spec, "cannot check: cannot divide by zero")
    expected = top.amount / bottom.amount * 100
    gap = abs(expected - reported.amount)
    shown = expected.quantize(Decimal("0.1"))
    words = f"{inputs[0].name} / {inputs[1].name} = {shown.normalize():f}%"
    if gap <= spec.tolerance:
        return CheckResult(
            id=spec.id,
            name=spec.name,
            status="ties",
            detail=f"{words}, and '{inputs[2].name}' agrees ({reported.amount.normalize():f}%)",
            attach_to=spec.result,
        )
    detail = (
        f"{words}, but '{inputs[2].name}' says {reported.amount.normalize():f}%: "
        f"a gap of {gap.quantize(Decimal('0.1')).normalize():f} points"
    )
    if spec.note:
        detail += f". {spec.note}"
    return CheckResult(
        id=spec.id,
        name=spec.name,
        status="does_not_tie",
        detail=detail,
        attach_to=spec.result,
    )


def run_checks(specs: list[CheckSpec], cells: dict[str, Operand]) -> list[CheckResult]:
    """Every check that applies to these cells. `cells` is keyed by lower-cased column name."""
    known = set(cells)
    return [evaluate(s, cells) for s in specs if applicable(s, known)]
