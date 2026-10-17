"""Derived values: every rule that decides whether numbers may be combined, and the arithmetic.

Pure, no model and no database. These are the rules that keep a computed number honest.
"""

from decimal import Decimal

import pytest

from backend.app.finance.derive import (
    Derivation,
    Operand,
    check_shape,
    compute,
    format_amount,
)
from backend.app.finance.periods import Period

Q2 = Period("quarter", 2026, 2)
Q2_PRIOR = Period("quarter", 2025, 2)
FY25 = Period("year", 2025)


def money(name, amount, currency="USD", period=Q2, status="answered") -> Operand:
    return Operand(
        name,
        status,
        {"amount": str(amount), "currency": currency},
        "currency",
        period,
    )


def number(name, n, unit=None, period=Q2, status="answered") -> Operand:
    return Operand(name, status, {"number": str(n), "unit": unit}, "number", period)


def d(op, as_=None) -> Derivation:
    return Derivation(op=op, **({"as": as_} if as_ else {}))


# --- the arithmetic ---------------------------------------------------------------------------------------------------------


def test_a_margin_is_a_ratio_shown_as_a_percent():
    result = compute(
        d("ratio", "percent"),
        [money("Operating income", 1_190_000_000), money("Revenue", 1_920_000_000)],
    )

    assert result.status == "answered"
    assert result.value == {"number": "61.98", "unit": "%"}
    assert result.display == "62%"  # one decimal, trailing zero dropped


def test_interest_coverage_is_a_ratio_shown_as_a_multiple():
    result = compute(
        d("ratio", "multiple"),
        [money("Operating income", 900), money("Interest expense", 300)],
    )

    assert result.value == {"number": "3", "unit": "x"} and result.display == "3x"


def test_net_debt_is_a_difference_of_amounts():
    result = compute(
        d("difference"),
        [money("Total debt", 5_000_000_000), money("Cash", 1_800_000_000)],
    )

    assert result.value == {"amount": "3200000000", "currency": "USD", "derived": True}
    assert result.display == "$3.2 billion"


def test_net_debt_over_a_measure_is_a_multiple():
    result = compute(
        d("net_ratio"),
        [money("Total debt", 5_000), money("Cash", 1_000), money("EBITDA", 2_000)],
    )

    assert result.value == {"number": "2", "unit": "x"} and result.display == "2x"


def test_growth_is_current_over_prior_minus_one_as_a_percent():
    result = compute(
        d("growth"),
        [
            money("Revenue", 1_100, period=Q2),
            money("Revenue (prior)", 1_000, period=Q2_PRIOR),
        ],
    )

    assert result.value == {"number": "10", "unit": "%"} and result.period == Q2


def test_a_shrinking_figure_has_negative_growth():
    result = compute(
        d("growth"),
        [
            money("Revenue", 900, period=FY25),
            money("Prior", 1_000, period=Period("year", 2024)),
        ],
    )

    assert result.display == "-10%"


def test_a_sum_adds_up_to_four_inputs():
    result = compute(d("sum"), [money("A", 1), money("B", 2), money("C", 3)])

    assert result.value["amount"] == "6"


def test_plain_numbers_keep_their_unit():
    result = compute(d("difference"), [number("A", 70, "%"), number("B", 62, "%")])

    assert result.value == {"number": "8", "unit": "%"} and result.display == "8%"


def test_money_is_exact_not_a_float():
    result = compute(d("difference"), [money("A", "0.3"), money("B", "0.1")])

    assert result.value["amount"] == "0.2"


# --- an input that cannot be used ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", ["not_found", "not_filled", "pending_or_error", "not_applicable"]
)
def test_a_missing_input_makes_the_result_not_found_and_names_it(status):
    result = compute(
        d("ratio", "percent"),
        [money("Operating income", 5), money("Revenue", 0, status=status)],
    )

    assert result.status == "not_found" and result.value is None
    assert "'Revenue'" in (result.detail or "")


@pytest.mark.parametrize("status", ["low_confidence", "conflicting"])
def test_an_uncertain_input_makes_the_result_uncertain_not_a_fact(status):
    result = compute(
        d("ratio", "percent"), [money("A", 5), money("B", 10, status=status)]
    )

    assert result.status == "low_confidence" and result.value is None


def test_every_missing_input_is_named():
    result = compute(
        d("net_ratio"),
        [
            money("Debt", 1, status="not_found"),
            money("Cash", 1),
            money("EBITDA", 1, status="not_filled"),
        ],
    )

    assert (
        "'Debt'" in result.detail
        and "'EBITDA'" in result.detail
        and "'Cash'" not in result.detail
    )  # type: ignore[operator]


def test_a_text_input_is_not_a_number():
    text = Operand("Notes", "answered", "see page 4", "text", Q2)

    result = compute(d("sum"), [money("A", 1), text])

    assert result.status == "low_confidence" and "'Notes'" in (result.detail or "")


# --- things that cannot be combined ------------------------------------------------------------------------------------------------


def test_amounts_in_different_currencies_are_not_combined():
    result = compute(d("difference"), [money("A", 5, "USD"), money("B", 3, "EUR")])

    assert result.status == "low_confidence" and "different units" in (
        result.detail or ""
    )


def test_a_ratio_of_different_currencies_is_not_taken():
    result = compute(
        d("ratio", "multiple"), [money("A", 5, "USD"), money("B", 3, "EUR")]
    )

    assert result.status == "low_confidence" and "different currencies" in (
        result.detail or ""
    )


def test_an_amount_and_a_plain_number_are_not_combined():
    result = compute(d("difference"), [money("Revenue", 5), number("Headcount", 3)])

    assert result.status == "low_confidence" and "not the same kind" in (
        result.detail or ""
    )


def test_a_percent_and_a_plain_number_are_not_added():
    result = compute(d("sum"), [number("A", 5, "%"), number("B", 3)])

    assert result.status == "low_confidence" and "different units" in (
        result.detail or ""
    )


def test_figures_from_different_periods_are_never_combined():
    result = compute(
        d("ratio", "percent"),
        [money("Income", 5, period=Q2), money("Revenue", 9, period=FY25)],
    )

    assert result.status == "low_confidence"
    assert "different periods" in (result.detail or "") and "FY2025" in (
        result.detail or ""
    )


def test_growth_needs_the_same_period_one_year_apart():
    result = compute(
        d("growth"), [money("A", 5, period=Q2), money("B", 4, period=FY25)]
    )

    assert result.status == "low_confidence" and "one year apart" in (
        result.detail or ""
    )


def test_growth_against_the_wrong_year_is_refused():
    result = compute(
        d("growth"),
        [money("A", 5, period=Q2), money("B", 4, period=Period("quarter", 2024, 2))],
    )

    assert result.status == "low_confidence"


def test_an_input_with_no_stated_period_is_allowed_but_the_answer_says_so():
    result = compute(
        d("ratio", "percent"),
        [money("Income", 5, period=Q2), money("Revenue", 10, period=None)],
    )

    assert result.status == "answered" and result.period == Q2
    assert any("period not stated for 'Revenue'" in n for n in result.notes)


def test_no_stated_periods_at_all_still_computes_with_a_note():
    result = compute(
        d("difference"), [money("A", 5, period=None), money("B", 2, period=None)]
    )

    assert result.status == "answered" and result.period is None and result.notes


def test_dividing_by_zero_is_reported_not_hidden():
    for op in (d("ratio", "percent"), d("ratio", "multiple")):
        result = compute(op, [money("A", 5), money("B", 0)])

        assert result.status == "low_confidence" and "divide by zero" in (
            result.detail or ""
        )


def test_growth_from_a_zero_base_is_reported():
    result = compute(
        d("growth"), [money("A", 5, period=Q2), money("B", 0, period=Q2_PRIOR)]
    )

    assert "divide by zero" in (result.detail or "")


def test_net_ratio_with_a_zero_denominator_is_reported():
    result = compute(d("net_ratio"), [money("D", 5), money("C", 1), money("E", 0)])

    assert "divide by zero" in (result.detail or "")


def test_net_ratio_needs_amounts():
    result = compute(d("net_ratio"), [money("D", 5), money("C", 1), number("E", 2)])

    assert result.status == "low_confidence"


# --- the shape of the operation ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "as_", "inputs", "message"),
    [
        ("ratio", "percent", 3, "needs 2"),
        ("difference", None, 1, "needs 2"),
        ("sum", None, 5, "2 to 4"),
        ("net_ratio", None, 2, "needs 3"),
        ("ratio", None, 2, "percent or a multiple"),
        ("growth", "multiple", 2, "always a percent"),
        ("net_ratio", "percent", 3, "always a multiple"),
    ],
)
def test_a_badly_shaped_operation_is_refused_with_a_reason(op, as_, inputs, message):
    with pytest.raises(ValueError, match=message):
        check_shape(d(op, as_), inputs)


def test_compute_reports_a_bad_shape_instead_of_raising():
    result = compute(d("ratio", "percent"), [money("A", 1)])

    assert result.status == "low_confidence" and "needs 2" in (result.detail or "")


def test_an_unknown_operation_is_not_accepted():
    with pytest.raises(ValueError):
        Derivation.model_validate({"op": "square_root"})


def test_the_stored_form_uses_the_as_key_and_leaves_out_nothing_extra():
    assert d("ratio", "percent").stored() == {"op": "ratio", "as": "percent"}
    assert d("difference").stored() == {"op": "difference"}
    assert (
        Derivation.model_validate({"op": "ratio", "as": "multiple"}).as_ == "multiple"
    )


# --- display ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "currency", "text"),
    [
        (Decimal(1920000000), "USD", "$1.92 billion"),
        (Decimal(78000000), "USD", "$78 million"),
        (Decimal(-300000000), "USD", "-$300 million"),
        (Decimal(2500000000000), "USD", "$2.5 trillion"),
        (Decimal(4200), "USD", "$4.2 thousand"),
        (Decimal(120), "USD", "$120"),
        (Decimal(0), "USD", "$0"),
        (Decimal(1500000), "EUR", "€1.5 million"),
        (Decimal(1500000), "JPY", "JPY 1.5 million"),
    ],
)
def test_amounts_are_shown_with_a_scale_a_person_can_read(amount, currency, text):
    assert format_amount(amount, currency) == text


def test_an_unreadable_input_reads_as_a_sentence():
    """Seen live: "'Total debt' is could not be read"."""
    result = compute(
        d("ratio", "percent"),
        [money("A", 1), money("Total debt", 5, status="pending_or_error")],
    )

    assert "'Total debt' is unreadable" in (result.detail or "")
    assert "is could not" not in (result.detail or "")
