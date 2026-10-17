"""Deciding which row a file belongs to: pure rules, no database and no model.

Getting this wrong is the expensive mistake: a file in the wrong row quietly poisons that row's
answers. So the rules lean towards asking a person over guessing.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.app.db.records import EntityRecord
from backend.app.intake.match import (
    Decision,
    decide,
    display_name,
    find_matches,
    normalise,
)
from backend.app.intake.models import Identity


def row(
    name: str, ticker: str | None = None, aliases: list[str] | None = None
) -> EntityRecord:
    return EntityRecord(
        id=uuid4(),
        workspace_id=uuid4(),
        name=name,
        position=1,
        created_at=datetime.now(UTC),
        ticker=ticker,
        aliases=aliases or [],
    )


def who(
    company="Palantir Technologies Inc.", ticker="PLTR", confidence=0.95
) -> Identity:
    return Identity(
        company=company,
        ticker=ticker,
        doc_type="10-Q",
        period="Q2 2026",
        confidence=confidence,
    )


# --- names ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Palantir Technologies Inc.", "palantir technologies"),
        ("PALANTIR  TECHNOLOGIES, INC", "palantir technologies"),
        ("Apple Inc", "apple"),
        ("The Coca-Cola Company", "coca cola"),
        ("Meta Platforms, Inc.", "meta platforms"),
        ("AT&T Inc.", "at t"),
        ("Visa", "visa"),
        ("", ""),
    ],
)
def test_names_are_compared_without_case_punctuation_or_legal_suffixes(raw, expected):
    assert normalise(raw) == expected


# --- finding candidates ----------------------------------------------------------------------------


def test_the_ticker_is_the_strongest_match():
    pltr = row("Palantir", ticker="PLTR")

    found = find_matches(who(company="Something Else Entirely"), [pltr, row("Other")])

    assert [(m.how, m.entity) for m in found] == [("ticker", pltr)]


def test_the_same_name_ignoring_case_and_suffix_matches_exactly():
    apple = row("Apple")

    assert find_matches(who("APPLE INC.", ticker=None), [apple])[0].how == "exact"


def test_an_alias_counts_as_the_name():
    pltr = row("Palantir", aliases=["Palantir Technologies"])

    assert (
        find_matches(who("Palantir Technologies Inc.", None), [pltr])[0].how == "exact"
    )


def test_a_shorter_name_that_starts_the_longer_one_is_a_prefix_match():
    pltr = row("Palantir")

    found = find_matches(who("Palantir Technologies Inc.", None), [pltr])

    assert [m.how for m in found] == ["prefix"]


def test_words_in_the_middle_are_not_a_prefix_match():
    assert find_matches(who("Bank of America", None), [row("America")]) == []


def test_unrelated_companies_do_not_match():
    assert (
        find_matches(
            who("Globex Corporation", "GBX"), [row("Palantir", "PLTR"), row("Apple")]
        )
        == []
    )


# --- the decision ---------------------------------------------------------------------------------------


def test_an_exact_match_attaches_to_that_row():
    apple = row("Apple", ticker="AAPL")

    d = decide(who("Apple Inc.", "AAPL"), [apple], min_confidence=0.75)

    assert (d.action, d.entity) == ("assign", apple)


def test_a_prefix_match_attaches_and_remembers_the_new_spelling():
    pltr = row("Palantir")

    d = decide(who(), [pltr], min_confidence=0.75)

    assert (d.action, d.entity) == ("assign", pltr)
    assert d.learn_alias == "Palantir Technologies Inc."
    assert d.learn_ticker == "PLTR"


def test_a_known_company_with_a_new_ticker_is_remembered_not_changed():
    pltr = row("Palantir", ticker="PLTR")

    d = decide(who("Palantir", "PLTR"), [pltr], min_confidence=0.75)

    assert d.learn_ticker is None and d.learn_alias is None  # nothing new to learn


def test_a_company_not_in_the_matrix_gets_a_new_row():
    d = decide(who("Globex Corporation", "GBX"), [row("Apple")], min_confidence=0.75)

    assert d.action == "create"
    assert d.new_name == "Globex" and d.new_ticker == "GBX"


def test_the_first_file_of_all_creates_the_first_row():
    d = decide(who(), [], min_confidence=0.75)

    assert d.action == "create" and d.new_name == "Palantir Technologies"


def test_two_possible_rows_are_a_question_for_a_person_not_a_guess():
    rows = [row("Acme Capital"), row("Acme Holdings")]

    d = decide(who("Acme", None), rows, min_confidence=0.75)

    assert d.action == "review" and set(d.candidates) == {
        "Acme Capital",
        "Acme Holdings",
    }
    assert "more than one" in d.reason


def test_the_same_name_with_a_different_ticker_is_a_question():
    other = row("Delta", ticker="DAL")

    d = decide(who("Delta", "DLTA"), [other], min_confidence=0.75)

    assert d.action == "review" and "ticker" in d.reason


def test_a_matching_ticker_beats_a_different_looking_name():
    meta = row("Facebook", ticker="META")

    d = decide(who("Meta Platforms, Inc.", "META"), [meta], min_confidence=0.75)

    assert (d.action, d.entity) == ("assign", meta)
    assert d.learn_alias == "Meta Platforms, Inc."


def test_similar_but_not_the_same_company_is_a_question():
    d = decide(
        who("Apple Hospitality REIT", "APLE"), [row("Apple")], min_confidence=0.75
    )

    assert d.action == "review" and d.candidates == ["Apple"]


def test_new_row_names_lose_the_legal_suffix_but_keep_their_case():
    assert display_name("Meta Platforms, Inc.") == "Meta Platforms"
    assert display_name("Visa Inc") == "Visa"
    assert display_name("McDonald's Corporation") == "McDonald's"
    assert display_name("Inc") == "Inc"  # never reduces a name to nothing


def test_low_confidence_never_acts():
    d = decide(who(confidence=0.5), [row("Palantir")], min_confidence=0.75)

    assert d.action == "review" and "not sure" in d.reason


def test_no_company_found_never_acts():
    d = decide(
        Identity(company=None, ticker=None, doc_type=None, period=None, confidence=0.9),
        [],
        min_confidence=0.75,
    )

    assert d.action == "review" and "could not tell" in d.reason


def test_a_decision_is_plain_data():
    d = decide(who(), [], min_confidence=0.75)

    assert isinstance(d, Decision)
