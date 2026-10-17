"""Turning an answer string into a typed value, and checking the value is in the quote."""

import pytest

from backend.app.matrix.models import ColumnDraft
from backend.app.matrix.values import (
    BadValue,
    is_per_unit,
    normalise,
    normalise_currency,
    normalise_date,
    normalise_number,
    supported_by_quote,
)

# --- numbers ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "number", "unit", "display"),
    [
        ("166,000", "166000", None, "166,000"),
        ("approximately 166,000 employees", "166000", None, "166,000"),
        ("0.26", "0.26", None, "0.26"),
        ("15.6%", "15.6", "%", "15.6%"),
        ("(4)%", "-4", "%", "-4%"),
        ("-5", "-5", None, "-5"),
        ("(1,234)", "-1234", None, "-1,234"),
        ("2.5 million", "2500000", None, "2,500,000"),
        ("7", "7", None, "7"),
        ("1,000,000.50", "1000000.5", None, "1,000,000.5"),
    ],
)
def test_numbers(answer, number, unit, display):
    result = normalise_number(answer)

    assert result.value == {"number": number, "unit": unit}
    assert result.display == display


@pytest.mark.parametrize("answer", ["", "none", "n/a", "see note", "—"])
def test_an_answer_without_a_number_is_refused(answer):
    with pytest.raises(BadValue):
        normalise_number(answer)


def test_numbers_are_exact_decimals_not_floats():
    assert normalise_number("0.1").value["number"] == "0.1"
    assert (
        normalise_number("1234567890.123456789").value["number"]
        == "1234567890.123456789"
    )


# --- currency ---------------------------------------------------------------------------------


def test_a_dollar_amount_with_a_scale_word():
    result = normalise_currency("$8.0 billion", None)

    assert result.value == {
        "amount": "8000000000",
        "currency": "USD",
        "printed": "$8.0",
        "scale": "billions",
        "scale_from": "answer",
    }
    assert result.display == "$8.0 billion"


def test_a_table_figure_takes_its_scale_from_the_table_units():
    result = normalise_currency("$416,161", "in millions")

    assert result.value["amount"] == "416161000000"
    assert result.value["scale_from"] == "table units"
    assert result.display == "$416,161 million"


def test_the_answers_own_scale_wins_over_the_table_units():
    result = normalise_currency("$5 billion", "in millions")

    assert (
        result.value["amount"] == "5000000000"
        and result.value["scale_from"] == "answer"
    )


def test_no_scale_anywhere_keeps_the_amount_as_printed():
    result = normalise_currency("$0.26", None)

    assert result.value["amount"] == "0.26" and result.value["scale"] is None


@pytest.mark.parametrize(
    ("answer", "code"),
    [("€1,200", "EUR"), ("£300", "GBP"), ("USD 50", "USD"), ("100 EUR", "EUR")],
)
def test_other_currencies(answer, code):
    assert normalise_currency(answer, None).value["currency"] == code


@pytest.mark.parametrize("answer", ["$(5,820)", "($5,820)", "-$5,820"])
def test_negative_amounts(answer):
    assert normalise_currency(answer, None).value["amount"] == "-5820"


def test_a_figure_with_no_currency_is_refused():
    with pytest.raises(BadValue, match="currency"):
        normalise_currency("416,161", "in millions")


def test_a_percentage_is_not_money():
    with pytest.raises(BadValue, match="percentage"):
        normalise_currency("$15.6%", None)


# --- dates ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "date", "precision"),
    [
        ("September 27, 2025", "2025-09-27", "day"),
        ("Sept 27 2025", "2025-09-27", "day"),
        ("27 September 2025", "2025-09-27", "day"),
        ("27th of", None, None),
        ("2025-09-27", "2025-09-27", "day"),
        ("March 2027", "2027-03", "month"),
        ("2027-03", "2027-03", "month"),
        ("2027", "2027", "year"),
        ("Sep. 28, 2024.", "2024-09-28", "day"),
        ("Feb 29, 2024", "2024-02-29", "day"),
    ],
)
def test_dates(answer, date, precision):
    if date is None:
        with pytest.raises(BadValue):
            normalise_date(answer)
        return
    assert normalise_date(answer).value == {"date": date, "precision": precision}


@pytest.mark.parametrize(
    "answer",
    [
        "February 30, 2025",
        "2025-13-01",
        "Feb 29, 2023",
        "soon",
        "",
        "Q3 2025",
        "13/14/2025",
    ],
)
def test_impossible_or_unreadable_dates_are_refused(answer):
    with pytest.raises(BadValue):
        normalise_date(answer)


# --- the other types --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "value"),
    [("Yes", True), ("yes.", True), ("TRUE", True), ("No", False), ("false", False)],
)
def test_yes_no(answer, value):
    assert normalise("yes_no", answer).value is value


@pytest.mark.parametrize("answer", ["maybe", "Yes, but", "", "not stated"])
def test_yes_no_must_be_exactly_yes_or_no(answer):
    with pytest.raises(BadValue):
        normalise("yes_no", answer)


def test_select_returns_the_canonical_allowed_value():
    result = normalise(
        "select", "  revolver ", allowed_values=["Revolver", "Term loan"]
    )

    assert result.value == "Revolver"


def test_select_refuses_a_value_outside_the_list_and_names_the_options():
    with pytest.raises(BadValue, match="Revolver, Term loan"):
        normalise("select", "Bond", allowed_values=["Revolver", "Term loan"])


def test_status_is_a_tag_and_a_one_line_reason():
    result = normalise(
        "status",
        "On track",
        reason="Covenants met",
        allowed_values=["On track", "Watch"],
    )

    assert result.value == {"tag": "On track", "reason": "Covenants met"}
    assert result.display == "On track: Covenants met"


def test_status_needs_a_reason_and_a_known_tag():
    with pytest.raises(BadValue, match="reason"):
        normalise("status", "On track", reason="  ", allowed_values=["On track"])
    with pytest.raises(BadValue, match="allowed"):
        normalise("status", "Fine", reason="x", allowed_values=["On track"])
    with pytest.raises(BadValue, match="one line"):
        normalise("status", "On track", reason="x" * 201, allowed_values=[])


def test_text_is_whitespace_collapsed_and_must_not_be_empty():
    assert (
        normalise("text", "  Cupertino,\n California ").value == "Cupertino, California"
    )
    with pytest.raises(BadValue):
        normalise("text", "   ")


# --- is the value in the quote? ---------------------------------------------------------------


def test_a_number_that_appears_in_the_quote_is_supported():
    assert (
        supported_by_quote(
            "number",
            "166,000",
            ["approximately 166,000 full-time equivalent employees"],
        )
        is None
    )


def test_the_same_number_written_without_commas_is_still_supported():
    assert supported_by_quote("number", "166000", ["166,000 employees"]) is None


def test_a_number_the_quote_does_not_state_is_refused():
    problem = supported_by_quote(
        "number", "170,000", ["approximately 166,000 employees"]
    )

    assert problem and "170000" in problem and "do not calculate" in problem


def test_a_number_cannot_be_matched_inside_a_bigger_number():
    assert (
        supported_by_quote("number", "5", ["net sales were 15 and 5.5 and 250"])
        is not None
    )
    assert supported_by_quote("number", "8", ["growth of 8.5 percent"]) is not None
    assert supported_by_quote("number", "8", ["growth of 8 percent"]) is None


def test_a_sum_the_model_worked_out_is_refused():
    quote = "Americas $178,353 Europe $111,032"

    assert supported_by_quote("currency", "$289,385", [quote]) is not None


def test_money_is_checked_by_its_digits_not_its_sign():
    assert (
        supported_by_quote(
            "currency", "$8.0 billion", ["$8.0 billion of commercial paper"]
        )
        is None
    )


def test_a_date_needs_its_year_and_day_in_the_quote():
    quote = "For the fiscal year ended September 27, 2025"

    assert supported_by_quote("date", "September 27, 2025", [quote]) is None
    assert "year" in (supported_by_quote("date", "September 27, 2024", [quote]) or "")
    assert "day" in (supported_by_quote("date", "September 28, 2025", [quote]) or "")


def test_a_month_precision_date_does_not_need_a_day():
    assert supported_by_quote("date", "March 2027", ["matures in March 2027"]) is None


@pytest.mark.parametrize("column_type", ["text", "yes_no", "select", "status"])
def test_judgement_types_have_no_digit_check(column_type):
    assert supported_by_quote(column_type, "anything", ["unrelated"]) is None


def test_any_one_of_several_quotes_may_hold_the_number():
    quotes = ["bought back shares", "402 million shares"]

    assert supported_by_quote("number", "402 million", quotes) is None
    # "402" alone would be wrong by a factor of a million: the scale the quote prints must be kept
    assert "leaves out" in (supported_by_quote("number", "402", quotes) or "")


# --- column definitions -----------------------------------------------------------------------


def test_a_select_column_needs_two_allowed_values():
    with pytest.raises(ValueError, match="at least two"):
        ColumnDraft(
            name="Type", prompt="Which?", type="select", allowed_values=["Only"]
        )


def test_allowed_values_only_belong_to_select_and_status():
    with pytest.raises(ValueError, match="only apply"):
        ColumnDraft(
            name="N", prompt="How many?", type="number", allowed_values=["a", "b"]
        )


def test_allowed_values_are_trimmed_and_cannot_repeat_ignoring_case():
    draft = ColumnDraft(
        name="T", prompt="p", type="select", allowed_values=[" A ", "b"]
    )
    assert draft.allowed_values == ["A", "b"]
    with pytest.raises(ValueError, match="repeated"):
        ColumnDraft(name="T", prompt="p", type="select", allowed_values=["A", "a"])


def test_blank_names_and_prompts_are_refused():
    with pytest.raises(ValueError):
        ColumnDraft(name="   ", prompt="p")
    with pytest.raises(ValueError):
        ColumnDraft(name="n", prompt="   ")


def test_a_blank_applicability_means_it_applies_to_every_row():
    assert ColumnDraft(name="n", prompt="p", applicability="   ").applicability is None


@pytest.mark.parametrize(
    "text",
    [
        "Diluted earnings per share",
        "EPS",
        "eps",
        "dividend per share",
        "Dividends per share",
        "per-share amounts",
        "Earnings per common share",
        "per unit",
    ],
)
def test_per_share_wording_is_recognised(text):
    assert is_per_unit(text)


@pytest.mark.parametrize(
    "text",
    [
        "Total net sales",
        "Operating income",
        "Share repurchases",
        "shares outstanding",
        "Net income",
    ],
)
def test_ordinary_amounts_are_not_per_share(text):
    assert not is_per_unit(text)


def test_a_units_header_that_mentions_per_share_amounts_is_not_a_per_share_cue():
    header = "(In millions, except number of shares, which are reflected in thousands, and per-share amounts)"

    assert not is_per_unit("Operating income", "Operating income", header)
    assert is_per_unit("Diluted earnings per share", header)
    assert is_per_unit("Basic", "Earnings per share | $7.49", header)


def test_per_share_wording_outside_the_units_header_still_counts():
    assert is_per_unit("(In millions) Dividends per share")


# --- a scale letter stuck to the number: "$1.2B" must never become $1.2 ---------------------------------------


@pytest.mark.parametrize(
    ("answer", "amount", "scale", "shown"),
    [
        ("$1.2B", "1200000000", "billions", "$1.2 billion"),
        ("$1.2b", "1200000000", "billions", "$1.2 billion"),
        ("$500M", "500000000", "millions", "$500 million"),
        ("$450MM", "450000000", "millions", "$450 million"),
        ("$3K", "3000", "thousands", "$3 thousand"),
        ("$2.1T", "2100000000000", "trillions", "$2.1 trillion"),
        ("$1.2bn", "1200000000", "billions", "$1.2 billion"),
        ("$1.2 billion", "1200000000", "billions", "$1.2 billion"),
    ],
)
def test_a_scale_letter_attached_to_the_number_is_understood(
    answer, amount, scale, shown
):
    result = normalise_currency(answer, None)

    assert result.value["amount"] == amount and result.value["scale"] == scale
    assert result.display == shown


def test_a_scale_letter_on_a_plain_number():
    assert normalise_number("5M users").value == {"number": "5000000", "unit": None}


def test_a_letter_that_is_not_a_scale_does_not_scale_anything():
    assert normalise_currency("$1.2X", None).value["amount"] == "1.2"
    assert (
        normalise_currency("$1.2 B", None).value["amount"] == "1.2"
    )  # a space makes it ambiguous
    assert normalise_number("62%").value == {"number": "62", "unit": "%"}


@pytest.mark.parametrize("column_type", ["currency", "number"])
def test_an_answer_that_drops_the_scale_the_quote_prints_is_refused(column_type):
    problem = supported_by_quote(
        column_type,
        "$1.2",
        ["Q2 2026 adjusted operating income was $1.2B, representing 62%."],
    )

    assert problem is not None and "leaves out" in problem and "'B'" in problem


@pytest.mark.parametrize("answer", ["$1.2B", "$1.2 billion", "$1.2bn"])
def test_an_answer_that_keeps_the_scale_is_accepted(answer):
    assert (
        supported_by_quote("currency", answer, ["income was $1.2B, a margin of 62%"])
        is None
    )


def test_a_quote_with_no_scale_does_not_forbid_a_scaled_answer():
    """A table row '1,200' under an 'in millions' heading: the answer may say $1,200 million."""
    assert (
        supported_by_quote("currency", "$1,200 million", ["Revenue | 1,200 | 1,100"])
        is None
    )


def test_the_same_figure_in_two_quotes_is_fine_if_one_agrees():
    quotes = ["a $5 deposit was made", "total of $5B in assets"]

    assert (
        supported_by_quote("currency", "$5", quotes) is None
    )  # the first quote prints $5 with no scale
