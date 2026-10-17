"""Tie-out checks: ties, does not tie, and, just as important, cannot check. Pure."""

from decimal import Decimal

import pytest

from backend.app.finance.checks import CheckSpec, applicable, evaluate, run_checks
from backend.app.finance.derive import Operand
from backend.app.finance.periods import Period

Q2 = Period("quarter", 2026, 2)

GROSS_PROFIT = CheckSpec(
    id="gross_profit_ties",
    name="Gross profit = revenue - cost of revenue",
    kind="difference_identity",
    a="Revenue",
    b="Cost of revenue",
    result="Gross profit",
    tolerance=Decimal("0.005"),
    note="Companies can define cost of revenue differently.",
)
GROSS_MARGIN = CheckSpec(
    id="gross_margin_ties",
    name="Gross margin = gross profit / revenue",
    kind="ratio_identity",
    a="Gross profit",
    b="Revenue",
    result="Gross margin",
    tolerance=Decimal("0.5"),
)


def money(name, amount, currency="USD", period=Q2, status="answered") -> Operand:
    return Operand(
        name, status, {"amount": str(amount), "currency": currency}, "currency", period
    )


def pct(name, n, period=Q2, status="answered", unit="%") -> Operand:
    return Operand(name, status, {"number": str(n), "unit": unit}, "number", period)


def row(*operands: Operand) -> dict[str, Operand]:
    return {o.name.casefold(): o for o in operands}


def gross_profit_row(revenue=1000, cost=400, gross=600, **kw) -> dict[str, Operand]:
    return row(
        money("Revenue", revenue, **kw),
        money("Cost of revenue", cost, **kw),
        money("Gross profit", gross, **kw),
    )


# --- difference identity -------------------------------------------------------------------------------------------------


def test_an_identity_that_holds_ties():
    result = evaluate(GROSS_PROFIT, gross_profit_row())

    assert result.status == "ties" and result.attach_to == "Gross profit"
    assert "agrees" in result.detail


def test_a_small_rounding_gap_inside_the_tolerance_still_ties():
    assert (
        evaluate(GROSS_PROFIT, gross_profit_row(gross=602)).status == "ties"
    )  # 0.33% off


def test_a_gap_outside_the_tolerance_does_not_tie_and_shows_the_gap():
    result = evaluate(GROSS_PROFIT, gross_profit_row(gross=650))

    assert result.status == "does_not_tie"
    assert "a gap of" in result.detail and "Companies can define" in result.detail


def test_the_gap_is_shown_in_readable_money():
    result = evaluate(
        GROSS_PROFIT,
        gross_profit_row(revenue=2_000_000_000, cost=800_000_000, gross=1_300_000_000),
    )

    assert "$1.2 billion" in result.detail and "$1.3 billion" in result.detail


def test_the_tolerance_edge_is_inclusive():
    spec = GROSS_PROFIT.model_copy(
        update={"tolerance": Decimal("0.1")}
    )  # 10% of the larger number

    assert (
        evaluate(spec, gross_profit_row(revenue=1000, cost=400, gross=540)).status
        == "ties"
    )  # 60/600 = 10%... larger is 600
    assert (
        evaluate(spec, gross_profit_row(revenue=1000, cost=400, gross=500)).status
        == "does_not_tie"
    )


def test_all_zeros_tie():
    assert (
        evaluate(GROSS_PROFIT, gross_profit_row(revenue=0, cost=0, gross=0)).status
        == "ties"
    )


def test_a_negative_result_is_handled():
    assert (
        evaluate(
            GROSS_PROFIT, gross_profit_row(revenue=100, cost=300, gross=-200)
        ).status
        == "ties"
    )


# --- ratio identity --------------------------------------------------------------------------------------------------------


def margin_row(gross=600, revenue=1000, reported=60, **kw):
    return row(
        money("Gross profit", gross, **kw),
        money("Revenue", revenue, **kw),
        pct("Gross margin", reported),
    )


def test_a_ratio_that_matches_the_reported_percentage_ties():
    assert evaluate(GROSS_MARGIN, margin_row()).status == "ties"


def test_a_ratio_within_half_a_point_ties():
    assert evaluate(GROSS_MARGIN, margin_row(reported=Decimal("60.4"))).status == "ties"


def test_a_ratio_off_by_more_does_not_tie_and_says_by_how_many_points():
    result = evaluate(GROSS_MARGIN, margin_row(reported=70))

    assert result.status == "does_not_tie" and "10 points" in result.detail


def test_the_reported_margin_must_actually_be_a_percentage():
    row_ = row(
        money("Gross profit", 600),
        money("Revenue", 1000),
        pct("Gross margin", 60, unit=None),
    )

    result = evaluate(GROSS_MARGIN, row_)

    assert result.status == "cannot_check" and "not a percentage" in result.detail


def test_a_zero_denominator_cannot_be_checked():
    result = evaluate(GROSS_MARGIN, margin_row(revenue=0))

    assert result.status == "cannot_check" and "divide by zero" in result.detail


# --- cannot check: never a pass ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "not_found",
        "pending_or_error",
        "low_confidence",
        "conflicting",
        "not_applicable",
    ],
)
def test_a_missing_or_uncertain_input_means_cannot_check_not_pass(status):
    cells = gross_profit_row()
    cells["cost of revenue"] = money("Cost of revenue", 400, status=status)

    result = evaluate(GROSS_PROFIT, cells)

    assert result.status == "cannot_check" and "'Cost of revenue'" in result.detail


def test_different_periods_cannot_be_checked():
    cells = row(
        money("Revenue", 1000, period=Q2),
        money("Cost of revenue", 400, period=Period("year", 2025)),
        money("Gross profit", 600, period=Q2),
    )

    result = evaluate(GROSS_PROFIT, cells)

    assert result.status == "cannot_check" and "different periods" in result.detail


def test_figures_with_no_stated_period_are_still_checked():
    assert evaluate(GROSS_PROFIT, gross_profit_row(period=None)).status == "ties"


def test_different_currencies_cannot_be_checked():
    cells = row(
        money("Revenue", 1000),
        money("Cost of revenue", 400, "EUR"),
        money("Gross profit", 600),
    )

    assert evaluate(GROSS_PROFIT, cells).status == "cannot_check"


def test_text_where_a_number_is_needed_cannot_be_checked():
    cells = gross_profit_row()
    cells["revenue"] = Operand("Revenue", "answered", "see note 4", "text", Q2)

    assert evaluate(GROSS_PROFIT, cells).status == "cannot_check"


def test_a_check_that_names_a_missing_column_cannot_run():
    assert evaluate(GROSS_PROFIT, row(money("Revenue", 1))).status == "cannot_check"


# --- which checks apply ----------------------------------------------------------------------------------------------------------------


def test_a_check_applies_only_when_all_its_columns_exist():
    assert applicable(GROSS_PROFIT, {"revenue", "cost of revenue", "gross profit"})
    assert not applicable(GROSS_PROFIT, {"revenue", "gross profit"})


def test_run_checks_skips_checks_whose_columns_do_not_exist_instead_of_failing_them():
    cells = gross_profit_row()

    results = run_checks([GROSS_PROFIT, GROSS_MARGIN], cells)

    assert [r.id for r in results] == [
        "gross_profit_ties"
    ]  # no 'Gross margin' column here


def test_run_checks_returns_a_result_per_applicable_check():
    cells = {**gross_profit_row(), "gross margin": pct("Gross margin", 60)}

    results = run_checks([GROSS_PROFIT, GROSS_MARGIN], cells)

    assert [(r.id, r.status) for r in results] == [
        ("gross_profit_ties", "ties"),
        ("gross_margin_ties", "ties"),
    ]


def test_the_stored_form_is_small_and_complete():
    result = evaluate(GROSS_PROFIT, gross_profit_row())

    assert set(result.stored()) == {"id", "name", "status", "detail"}
