"""Reporting periods: parsing the canonical shapes and refusing everything else."""

import pytest

from backend.app.finance.periods import (
    Period,
    grounded_period,
    parse_period,
    supported_by,
)


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("FY2025", "FY2025"),
        ("FY 2025", "FY2025"),
        ("fy2025", "FY2025"),
        ("fiscal 2025", "FY2025"),
        ("Fiscal Year 2025", "FY2025"),
        ("FY25", "FY2025"),
        ("2025", "FY2025"),
        ("Q2 2026", "Q2 2026"),
        ("q2 2026", "Q2 2026"),
        ("Q2 FY2026", "Q2 2026"),
        ("Q2'26", "Q2 2026"),
        ("Q2 26", "Q2 2026"),
        ("2Q26", "Q2 2026"),
        ("second quarter of 2026", "Q2 2026"),
        ("Fourth quarter 2025", "Q4 2025"),
        ("H1 2026", "H1 2026"),
        ("H2 FY2025", "H2 2025"),
        ("(Q3 2025)", "Q3 2025"),
        ("  FY2025,  ", "FY2025"),
    ],
)
def test_the_common_spellings_become_one_canonical_period(text, label):
    assert parse_period(text) is not None and parse_period(text).label == label  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "three months ended June 30, 2026",  # a sentence, not a period we can compare
        "latest",
        "Q5 2026",
        "Q0 2026",
        "H3 2026",
        "FY1850",
        "FY2500",
        "2025 and 2024",
        "the year 2025",
        "x" * 80,
        None,
        2025,
        ["FY2025"],
    ],
)
def test_anything_else_is_not_a_period_and_never_raises(text):
    assert parse_period(text) is None


def test_the_same_period_spelled_two_ways_is_equal():
    assert parse_period("Q2 FY2026") == parse_period("second quarter of 2026")
    assert parse_period("fiscal 2025") == parse_period("FY2025")


def test_different_periods_are_different():
    assert parse_period("FY2025") != parse_period("Q4 2025")
    assert parse_period("Q1 2026") != parse_period("Q2 2026")


def test_a_period_is_usable_as_a_dictionary_key():
    assert {parse_period("FY2025"): 1}[Period("year", 2025)] == 1


def test_the_label_is_what_is_shown():
    assert str(Period("quarter", 2026, 2)) == "Q2 2026"


def test_the_same_period_of_the_year_before():
    q2_2026, q2_2025 = Period("quarter", 2026, 2), Period("quarter", 2025, 2)

    assert q2_2025.is_year_before(q2_2026)
    assert not q2_2026.is_year_before(q2_2025)  # not backwards
    assert not Period("quarter", 2025, 1).is_year_before(q2_2026)  # a different quarter
    assert not Period("year", 2025).is_year_before(q2_2026)  # a year is not a quarter
    assert Period("year", 2024).is_year_before(Period("year", 2025))
    assert not Period("year", 2023).is_year_before(
        Period("year", 2025)
    )  # two years apart


# --- a period must be in the text, not just in the model's head ---------------------------------------------


def test_a_period_whose_year_is_in_the_text_is_kept():
    text = "Total revenue for the three months ended June 30, 2026 was $1.0 billion."

    assert grounded_period("Q2 2026", text) == Period("quarter", 2026, 2)


def test_a_period_whose_year_the_text_never_mentions_is_dropped():
    """The model 'remembers' FY2025 for a company; the text it was shown only talks about 2026."""
    assert (
        grounded_period(
            "FY2025", "Revenue for the quarter ended June 30, 2026 was $1.0 billion."
        )
        is None
    )


def test_an_unparseable_claim_is_dropped_even_if_the_text_has_years():
    assert grounded_period("last quarter", "revenue in 2026") is None
    assert grounded_period(None, "revenue in 2026") is None


def test_a_two_digit_year_with_an_apostrophe_counts_as_support():
    assert supported_by(Period("quarter", 2026, 2), "Q2 '26 revenue grew")
    assert not supported_by(Period("quarter", 2026, 2), "Q2 revenue grew")
